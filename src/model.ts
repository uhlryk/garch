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
  /** Degrees of freedom of Student-t innovations (> 2); absent for normal ones. */
  readonly nu?: number;
};

export type Distribution = 'normal' | 't';

/** The mean: a fitted constant μ (`arch`'s default), or zero — μ held at 0, for returns already demeaned. */
export type MeanModel = 'constant' | 'zero';

const LOG_2PI = Math.log(2 * Math.PI);
const LOG_PI = Math.log(Math.PI);

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

/** ln Γ(x) for x > 0 — Lanczos (g = 7), relative error ~1e-15. */
export function logGamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  const z = x - 1;
  let a = LANCZOS[0];
  const t = z + 7.5;
  for (let i = 1; i < 9; i += 1) a += LANCZOS[i] / (z + i);
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

/** ψ(x) = d ln Γ(x) / dx for x > 0 — recurrence up to 6, then the asymptotic series. */
export function digamma(x: number): number {
  let r = 0;
  while (x < 6) {
    r -= 1 / x;
    x += 1;
  }
  const f = 1 / (x * x);
  return r + Math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))));
}

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

/** α + γ/2 + β: how much of today's variance shock is still there tomorrow. */
export function persistence(p: GarchParams): number {
  return p.alpha + p.gamma / 2 + p.beta;
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
): number {
  let total = 0;
  let prevE2 = bc;
  let prevNeg = 0.5 * bc;
  let prevS = bc;
  const nu = p.nu;
  const c = nu === undefined ? 0 : tConstant(nu);
  for (let t = 0; t < returns.length; t += 1) {
    const s = p.omega + p.alpha * prevE2 + p.gamma * prevNeg + p.beta * prevS;
    const e = returns[t] - p.mu;
    const e2 = e * e;
    variance[t] = s;
    const l =
      nu === undefined
        ? -0.5 * (LOG_2PI + Math.log(s) + e2 / s)
        : c - 0.5 * Math.log(s) - ((nu + 1) / 2) * Math.log1p(e2 / (s * (nu - 2)));
    if (contributions) contributions[t] = l;
    total += l;
    prevE2 = e2;
    prevNeg = e < 0 ? e2 : 0;
    prevS = s;
  }
  return total;
}

/**
 * The log-likelihood and its analytic gradient with respect to
 * (μ, ω, α, γ, β) — and ν, a sixth entry, for t innovations. With `scores`,
 * also each observation's gradient — a row of n per parameter (a sixth row
 * takes ν's) — for the robust covariance.
 */
export function logLikelihoodGradient(
  returns: ArrayLike<number>,
  p: GarchParams,
  bc: number,
  scores?: Float64Array[],
): { value: number; gradient: number[] } {
  const nu = p.nu;
  const g = nu === undefined ? [0, 0, 0, 0, 0] : [0, 0, 0, 0, 0, 0];
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
    const s = p.omega + p.alpha * prevE2 + p.gamma * prevNeg + p.beta * prevS;
    if (t === 0) {
      dMu = 0;
      dOmega = 1;
      dAlpha = bc;
      dGamma = 0.5 * bc;
      dBeta = bc;
    } else {
      const negative = prevE < 0 ? 1 : 0;
      dMu = -2 * prevE * (p.alpha + p.gamma * negative) + p.beta * dMu;
      dOmega = 1 + p.beta * dOmega;
      dAlpha = prevE2 + p.beta * dAlpha;
      dGamma = prevNeg + p.beta * dGamma;
      dBeta = prevS + p.beta * dBeta;
    }
    const e = returns[t] - p.mu;
    const e2 = e * e;
    let ds: number;
    let de: number; // ∂l/∂μ through e directly
    if (nu === undefined) {
      total += -0.5 * (LOG_2PI + Math.log(s) + e2 / s);
      ds = -0.5 * (1 / s - e2 / (s * s));
      de = e / s;
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
    prevE = e;
    prevE2 = e2;
    prevNeg = e < 0 ? e2 : 0;
    prevS = s;
  }
  return { value: total, gradient: g };
}
