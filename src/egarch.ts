import { normalCdf } from './inference.js';
import type { GarchParams } from './model.js';

/**
 * EGARCH(1,1,1) — Nelson's exponential GARCH with a constant mean and normal
 * innovations, `arch`'s specification (`vol="EGARCH", p=1, o=1, q=1`):
 *
 *     r_t = μ + e_t,   e_t ~ N(0, s_t),   z_t = e_t / √s_t
 *     ln s_t = ω + α·(|z_{t−1}| − √(2/π)) + γ·z_{t−1} + β·ln s_{t−1}
 *
 * The log keeps s_t positive with no sign constraint on α or γ, and the
 * asymmetry is γ itself: γ < 0 means a down move raises tomorrow's variance
 * more than an up move of the same size. The recursion starts from the log of
 * `arch`'s backcast, with no shock term at t = 0 (as `arch`).
 *
 * **Normal innovations only.** The multi-step variance forecast is
 * E[exp(Σ βʲ·g(z))], and for Student-t innovations E[exp(a·|z|)] is infinite
 * for every a > 0 — the expected variance two steps ahead does not exist.
 * `arch` simulates it and gets a finite, unstable number; here a t EGARCH is
 * refused instead.
 */

/** E|z| for a standard normal. */
export const EGARCH_ABS_MEAN = Math.sqrt(2 / Math.PI);

const LOG_2PI = Math.log(2 * Math.PI);

/**
 * Runs the log-variance recursion from `lnBackcast`. Fills `variance` (s_t)
 * and, when given, `contributions`; returns the log-likelihood (−∞ when the
 * recursion leaves the floating-point range).
 */
export function egarchLogLikelihood(
  returns: ArrayLike<number>,
  p: GarchParams,
  lnBackcast: number,
  variance: Float64Array,
  contributions?: Float64Array,
): number {
  let total = 0;
  let prevL = lnBackcast;
  let prevZ = 0;
  for (let t = 0; t < returns.length; t += 1) {
    const L =
      t === 0
        ? p.omega + p.beta * prevL
        : p.omega + p.alpha * (Math.abs(prevZ) - EGARCH_ABS_MEAN) + p.gamma * prevZ + p.beta * prevL;
    const s = Math.exp(L);
    const e = returns[t] - p.mu;
    variance[t] = s;
    const l = -0.5 * (LOG_2PI + L + (e * e) / s);
    if (contributions) contributions[t] = l;
    total += l;
    prevL = L;
    prevZ = e / Math.sqrt(s);
  }
  return Number.isFinite(total) ? total : -Infinity;
}

/**
 * The log-likelihood and its analytic gradient with respect to
 * (μ, ω, α, γ, β); with `scores`, each observation's gradient, a row per
 * parameter.
 *
 * With L = ln s and k = α·sign(z) + γ, the recursion's derivative is
 *     dL_t = ∂_direct + (β − ½·k·z_{t−1})·dL_{t−1} + k·e^{−L_{t−1}/2}·de_{t−1}
 * (de/dμ = −1), and dl/dL = −½(1 − e²/s).
 */
export function egarchLogLikelihoodGradient(
  returns: ArrayLike<number>,
  p: GarchParams,
  lnBackcast: number,
  scores?: Float64Array[],
): { value: number; gradient: number[] } {
  const g = [0, 0, 0, 0, 0];
  let total = 0;
  let prevL = lnBackcast;
  let prevZ = 0;
  let dMu = 0;
  let dOmega = 0;
  let dAlpha = 0;
  let dGamma = 0;
  let dBeta = 0;
  for (let t = 0; t < returns.length; t += 1) {
    let L: number;
    if (t === 0) {
      L = p.omega + p.beta * prevL;
      dMu = 0;
      dOmega = 1;
      dAlpha = 0;
      dGamma = 0;
      dBeta = prevL;
    } else {
      const absZ = Math.abs(prevZ);
      L = p.omega + p.alpha * (absZ - EGARCH_ABS_MEAN) + p.gamma * prevZ + p.beta * prevL;
      const k = p.alpha * Math.sign(prevZ) + p.gamma;
      const carry = p.beta - 0.5 * k * prevZ;
      const inv = Math.exp(-0.5 * prevL);
      dMu = carry * dMu - k * inv;
      dOmega = 1 + carry * dOmega;
      dAlpha = absZ - EGARCH_ABS_MEAN + carry * dAlpha;
      dGamma = prevZ + carry * dGamma;
      dBeta = prevL + carry * dBeta;
    }
    const s = Math.exp(L);
    const e = returns[t] - p.mu;
    const e2s = (e * e) / s;
    total += -0.5 * (LOG_2PI + L + e2s);
    const dl = -0.5 * (1 - e2s);
    const sMu = dl * dMu + e / s;
    const sOmega = dl * dOmega;
    const sAlpha = dl * dAlpha;
    const sGamma = dl * dGamma;
    const sBeta = dl * dBeta;
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
    prevL = L;
    prevZ = e / Math.sqrt(s);
  }
  if (!Number.isFinite(total)) return { value: -Infinity, gradient: g.map(() => NaN) };
  return { value: total, gradient: g };
}

/** E[exp(b·g(z))] for a standard normal z, g(z) = α(|z| − √(2/π)) + γz — closed form. */
function shockFactor(p: GarchParams, b: number): number {
  const a = b * p.alpha;
  const c = b * p.gamma;
  return (
    Math.exp(-a * EGARCH_ABS_MEAN) *
    (Math.exp(((a + c) * (a + c)) / 2) * normalCdf(a + c) + Math.exp(((a - c) * (a - c)) / 2) * normalCdf(a - c))
  );
}

/**
 * s_{T+1} … s_{T+h}, exact under normal innovations: the first step from the
 * observed last shock; then E[s_{T+k}] = exp(m_k)·Π_{j=0}^{k−2} E[exp(βʲ·g(z))],
 * m_k = ω + β·m_{k−1}, m_1 = ln s_{T+1}.
 */
export function egarchForecast(p: GarchParams, lastVariance: number, lastResidual: number, horizon: number): number[] {
  const z = lastResidual / Math.sqrt(lastVariance);
  let m = p.omega + p.alpha * (Math.abs(z) - EGARCH_ABS_MEAN) + p.gamma * z + p.beta * Math.log(lastVariance);
  let product = 1;
  let power = 1; // β^{k−2}
  const out = [Math.exp(m)];
  for (let k = 2; k <= horizon; k += 1) {
    m = p.omega + p.beta * m;
    product *= shockFactor(p, power);
    power *= p.beta;
    out.push(Math.exp(m) * product);
  }
  return out;
}
