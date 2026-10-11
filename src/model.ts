/**
 * The GARCH(1,1) / GJR-GARCH(1,1) model with a constant mean and normal
 * innovations — the specification `arch` fits by default
 * (`arch_model(mean="Constant", vol="GARCH", p=1, o=0|1, q=1, dist="normal")`):
 *
 *     r_t = μ + e_t,   e_t ~ N(0, s_t)
 *     s_t = ω + α·e²_{t−1} + γ·e²_{t−1}·[e_{t−1} < 0] + β·s_{t−1}
 *
 * γ is 0 for the symmetric model. The recursion starts from `arch`'s
 * backcast: an exponentially weighted (0.94) mean of the first 75 squared
 * residuals, standing in for e²_{−1} and s_{−1} (half of it for the
 * asymmetric term, the expected share of down moves).
 */

import { skewConstants, skewDownShare, skewLogDensity } from './skewt.js';
import { digamma, logGamma } from './special.js';

export { digamma, logGamma } from './special.js';

/** `egarch` is the log-variance model in `egarch.ts`: same parameter names, other meanings. */
export type GarchModel = 'garch' | 'gjr' | 'egarch';

/** Model parameters in the units of the returns they were fitted to. */
export type GarchParams = {
  readonly mu: number;
  readonly omega: number;
  readonly alpha: number;
  /** 0 for the symmetric model. */
  readonly gamma: number;
  readonly beta: number;
  /** Degrees of freedom of Student-t or skewed-t innovations (> 2); absent for normal ones. */
  readonly nu?: number;
  /** Skewness of Hansen's skewed-t innovations, in (−1, 1); absent for normal and t. */
  readonly lambda?: number;
  /** GARCH-X: the weight δ ≥ 0 of the exogenous x_{t−1} in s_t; absent without one. */
  readonly delta?: number;
};

/**
 * A GARCH-X driver: x_t ≥ 0 for every return (an implied or realised
 * variance, say), entering s_{t+1} as δ·x_t, and the x_{−1} the recursion
 * starts from (`exogenousBackcast`).
 */
export type Exogenous = { readonly values: ArrayLike<number>; readonly backcast: number };

/** x_{−1}: the 0.94-weighted mean of the first 75 values of x, as `backcast` weighs squared residuals. */
export function exogenousBackcast(values: ArrayLike<number>): number {
  const tau = Math.min(75, values.length);
  let weight = 1;
  let weights = 0;
  let sum = 0;
  for (let i = 0; i < tau; i += 1) {
    sum += weight * values[i];
    weights += weight;
    weight *= 0.94;
  }
  return sum / weights;
}

/** The innovations: normal, Student-t (ν), or Hansen's skewed t (ν, λ) — `arch`'s `skewt`. */
export type Distribution = 'normal' | 't' | 'skewt';

/** Refuses anything but an integer ≥ `min` — a count of steps, rows or windows. */
export function checkCount(name: string, value: number, min = 1): void {
  if (!(Number.isInteger(value) && value >= min))
    throw new RangeError(`${name} must be an integer ≥ ${min}, got ${value}`);
}

/** The mean: a fitted constant μ (`arch`'s default), or zero — μ held at 0, for returns already demeaned. */
export type MeanModel = 'constant' | 'zero';

const LOG_2PI = Math.log(2 * Math.PI);
const LOG_PI = Math.log(Math.PI);

/** The part of a standardised-t log density that depends on ν only. */
function tConstant(nu: number): number {
  return logGamma((nu + 1) / 2) - logGamma(nu / 2) - 0.5 * (LOG_PI + Math.log(nu - 2));
}

/** `arch`'s backcast of the pre-sample variance, from the residuals `values − mean`. */
export function backcast(values: ArrayLike<number>, mean = 0): number {
  const tau = Math.min(75, values.length);
  let weight = 1;
  let weights = 0;
  let sum = 0;
  for (let i = 0; i < tau; i += 1) {
    sum += weight * (values[i] - mean) ** 2;
    weights += weight;
    weight *= 0.94;
  }
  return sum / weights;
}

/**
 * E[z²·1{z < 0}]: the share of a unit shock's variance that comes from down
 * moves — ½ for any symmetric innovation, other for a skewed t.
 */
export function downShare(p: GarchParams): number {
  return p.lambda === undefined || p.nu === undefined ? 0.5 : skewDownShare(p.nu, p.lambda);
}

/** α + κ·γ + β, κ the down share (½ unless skewed): how much of today's variance shock is still there tomorrow. */
export function persistence(p: GarchParams): number {
  return p.alpha + (p.gamma ? downShare(p) * p.gamma : 0) + p.beta;
}

/**
 * Runs the variance recursion. Fills `variance` (length n) and, when given,
 * `contributions` with each observation's log-likelihood; returns their sum.
 */
export function logLikelihood(
  returns: ArrayLike<number>,
  p: GarchParams,
  bc: number,
  variance: Float64Array,
  contributions?: Float64Array,
  x?: Exogenous,
): number {
  return continueLikelihood(
    returns,
    p,
    { e2: bc, neg: 0.5 * bc, s: bc, x: x?.backcast ?? 0 },
    0,
    variance,
    contributions,
    x?.values,
  );
}

/** Where the GARCH / GJR recursion stands after an observation: e², e²·[e < 0], s and x (0 without one). */
export type RecursionState = { readonly e2: number; readonly neg: number; readonly s: number; readonly x: number };

/**
 * `logLikelihood` carried on from a state and a running total: over the
 * returns that follow, it fills their variances and adds their terms to
 * `total` one by one, in the order a run over the whole series would.
 */
export function continueLikelihood(
  returns: ArrayLike<number>,
  p: GarchParams,
  from: RecursionState,
  total: number,
  variance: Float64Array,
  contributions?: Float64Array,
  x?: ArrayLike<number>,
): number {
  let prevE2 = from.e2;
  let prevNeg = from.neg;
  let prevS = from.s;
  let prevX = from.x;
  const delta = p.delta ?? 0;
  const nu = p.nu;
  const skew = nu !== undefined && p.lambda !== undefined ? skewConstants(nu, p.lambda) : null;
  const c = nu === undefined ? 0 : tConstant(nu);
  for (let t = 0; t < returns.length; t += 1) {
    const s = p.omega + p.alpha * prevE2 + p.gamma * prevNeg + p.beta * prevS + delta * prevX;
    const e = returns[t] - p.mu;
    const e2 = e * e;
    variance[t] = s;
    const l =
      nu === undefined
        ? -0.5 * (LOG_2PI + Math.log(s) + e2 / s)
        : skew
          ? skewLogDensity(e / Math.sqrt(s), nu, p.lambda!, skew).value - 0.5 * Math.log(s)
          : c - 0.5 * Math.log(s) - ((nu + 1) / 2) * Math.log1p(e2 / (s * (nu - 2)));
    if (contributions) contributions[t] = l;
    total += l;
    prevE2 = e2;
    prevNeg = e < 0 ? e2 : 0;
    prevS = s;
    if (x) prevX = x[t];
  }
  return total;
}

/**
 * The log-likelihood and its analytic gradient with respect to
 * (μ, ω, α, γ, β) — and ν, a sixth entry, for t and skewed-t innovations, λ
 * a seventh for skewed t; with an exogenous x, δ is the eighth (ν's and λ's
 * entries 0 where they are absent). With `scores`, also each observation's
 * gradient — a row of n per parameter, in the same order — for the robust
 * covariance.
 */
export function logLikelihoodGradient(
  returns: ArrayLike<number>,
  p: GarchParams,
  bc: number,
  scores?: Float64Array[],
  x?: Exogenous,
): { value: number; gradient: number[] } {
  const nu = p.nu;
  const skew = nu !== undefined && p.lambda !== undefined ? skewConstants(nu, p.lambda) : null;
  const g = new Array<number>(x ? 8 : nu === undefined ? 5 : skew ? 7 : 6).fill(0);
  const delta = p.delta ?? 0;
  let prevX = x?.backcast ?? 0;
  let dDelta = 0;
  const c = nu === undefined ? 0 : tConstant(nu);
  const dcNu = nu === undefined ? 0 : 0.5 * digamma((nu + 1) / 2) - 0.5 * digamma(nu / 2) - 0.5 / (nu - 2);
  let total = 0;
  let prevE = 0;
  let prevE2 = bc;
  let prevNeg = 0.5 * bc;
  let prevS = bc;
  // d s_{t−1} / d(μ, ω, α, γ, β); at t = 0 the lagged terms are the backcast (a constant).
  let dMu = 0;
  let dOmega = 0;
  let dAlpha = 0;
  let dGamma = 0;
  let dBeta = 0;
  for (let t = 0; t < returns.length; t += 1) {
    const s = p.omega + p.alpha * prevE2 + p.gamma * prevNeg + p.beta * prevS + delta * prevX;
    if (t === 0) {
      dMu = 0;
      dOmega = 1;
      dAlpha = bc;
      dGamma = 0.5 * bc;
      dBeta = bc;
      dDelta = prevX;
    } else {
      const negative = prevE < 0 ? 1 : 0;
      dMu = -2 * prevE * (p.alpha + p.gamma * negative) + p.beta * dMu;
      dOmega = 1 + p.beta * dOmega;
      dAlpha = prevE2 + p.beta * dAlpha;
      dGamma = prevNeg + p.beta * dGamma;
      dBeta = prevS + p.beta * dBeta;
      dDelta = prevX + p.beta * dDelta;
    }
    const e = returns[t] - p.mu;
    const e2 = e * e;
    let ds: number;
    let de: number; // ∂l/∂μ through e directly
    if (nu === undefined) {
      total += -0.5 * (LOG_2PI + Math.log(s) + e2 / s);
      ds = -0.5 * (1 / s - e2 / (s * s));
      de = e / s;
    } else if (skew) {
      // l = ln f(z) − ½ ln s,  z = e/√s
      const root = Math.sqrt(s);
      const z = e / root;
      const f = skewLogDensity(z, nu, p.lambda!, skew);
      total += f.value - 0.5 * Math.log(s);
      ds = -0.5 / s - (f.dz * z) / (2 * s);
      de = -f.dz / root;
      g[5] += f.dNu;
      g[6] += f.dLambda;
      if (scores && scores.length > 6) {
        scores[5][t] = f.dNu;
        scores[6][t] = f.dLambda;
      }
    } else {
      // q = e² / (s(ν − 2)); l = c(ν) − ½ ln s − (ν+1)/2 · ln(1 + q)
      const q = e2 / (s * (nu - 2));
      total += c - 0.5 * Math.log(s) - ((nu + 1) / 2) * Math.log1p(q);
      ds = -0.5 / s + ((nu + 1) / 2) * (q / (s * (1 + q)));
      de = ((nu + 1) * e) / (s * (nu - 2) * (1 + q));
      const sNu = dcNu - 0.5 * Math.log1p(q) + ((nu + 1) * q) / (2 * (nu - 2) * (1 + q));
      g[5] += sNu;
      if (scores && scores.length > 5) scores[5][t] = sNu;
    }
    const sMu = ds * dMu + de;
    const sOmega = ds * dOmega;
    const sAlpha = ds * dAlpha;
    const sGamma = ds * dGamma;
    const sBeta = ds * dBeta;
    g[0] += sMu;
    g[1] += sOmega;
    g[2] += sAlpha;
    g[3] += sGamma;
    g[4] += sBeta;
    if (scores) {
      scores[0][t] = sMu;
      scores[1][t] = sOmega;
      scores[2][t] = sAlpha;
      scores[3][t] = sGamma;
      scores[4][t] = sBeta;
    }
    if (x) {
      const sDelta = ds * dDelta;
      g[7] += sDelta;
      if (scores && scores.length > 7) scores[7][t] = sDelta;
      prevX = x.values[t];
    }
    prevE = e;
    prevE2 = e2;
    prevNeg = e < 0 ? e2 : 0;
    prevS = s;
  }
  return { value: total, gradient: g };
}
