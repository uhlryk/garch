import type { GarchFit } from './fit.js';
import { invert } from './inference.js';
import { logGamma } from './special.js';

/**
 * Is anything left for the model to explain? A fitted model's standardised
 * residuals z_t = (r_t − μ) / √s_t should be serially uncorrelated, and so
 * should z²_t — what remains of an ARCH effect shows up there.
 */

/** z_t = (r_t − μ) / √s_t for every observation of the fit. */
export function standardizedResiduals(fit: GarchFit, returns: ArrayLike<number>): Float64Array {
  const s = fit.conditionalVariance;
  if (returns.length !== s.length)
    throw new RangeError(`the fit has ${s.length} variances, the returns are ${returns.length}`);
  const z = new Float64Array(s.length);
  for (let t = 0; t < s.length; t += 1) z[t] = (returns[t] - fit.params.mu) / Math.sqrt(s[t]);
  return z;
}

/** A χ² test's answer: the statistic, its degrees of freedom and the upper-tail p-value. */
export type TestResult = {
  readonly statistic: number;
  /** Degrees of freedom of the χ² it is referred to. */
  readonly df: number;
  readonly pValue: number;
};

/** The regularised lower incomplete gamma P(a, x): its series below a + 1, its continued fraction above. */
function lowerGamma(a: number, x: number): number {
  if (x <= 0) return 0;
  const front = Math.exp(-x + a * Math.log(x) - logGamma(a));
  if (x < a + 1) {
    let term = 1 / a;
    let sum = term;
    for (let n = 1; n < 1000; n += 1) {
      term *= x / (a + n);
      sum += term;
      if (Math.abs(term) < Math.abs(sum) * 1e-16) break;
    }
    return sum * front;
  }
  return 1 - upperFraction(a, x) * front;
}

/** Lentz's continued fraction for Γ(a, x) / (e^{−x} x^a / Γ(a)). */
function upperFraction(a: number, x: number): number {
  const tiny = 1e-300;
  let b = x + 1 - a;
  let c = 1 / tiny;
  let d = 1 / b;
  let h = d;
  for (let i = 1; i < 1000; i += 1) {
    const an = -i * (i - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  return h;
}

/** P(χ²_df > x), computed in the upper tail (no 1 − P cancellation for a large statistic). */
export function chiSquareSurvival(x: number, df: number): number {
  if (!(df > 0)) throw new RangeError(`df must be positive, got ${df}`);
  if (!(x > 0)) return 1;
  const a = df / 2;
  const h = x / 2;
  if (h < a + 1) return 1 - lowerGamma(a, h);
  return upperFraction(a, h) * Math.exp(-h + a * Math.log(h) - logGamma(a));
}

const checkLags = (lags: number, n: number) => {
  if (!(Number.isInteger(lags) && lags >= 1)) throw new RangeError(`lags must be a positive integer, got ${lags}`);
  if (lags >= n) throw new RangeError(`${lags} lags need more than ${n} observations`);
};

/**
 * Ljung–Box Q = n(n + 2)·Σ_{k ≤ lags} ρ̂²_k / (n − k) against χ²(lags − modelDf),
 * as statsmodels' `acorr_ljungbox` (autocorrelations of the demeaned series
 * over its total sum of squares). Run it on z for the mean, on z² for what
 * remains of the variance dynamics.
 */
export function ljungBox(series: ArrayLike<number>, lags: number, modelDf = 0): TestResult {
  const n = series.length;
  checkLags(lags, n);
  if (!(Number.isInteger(modelDf) && modelDf >= 0 && modelDf < lags))
    throw new RangeError(`modelDf must be an integer in [0, lags), got ${modelDf}`);
  let mean = 0;
  for (let t = 0; t < n; t += 1) mean += series[t];
  mean /= n;
  let ss = 0;
  for (let t = 0; t < n; t += 1) ss += (series[t] - mean) ** 2;
  let q = 0;
  for (let k = 1; k <= lags; k += 1) {
    let c = 0;
    for (let t = k; t < n; t += 1) c += (series[t] - mean) * (series[t - k] - mean);
    q += (c / ss) ** 2 / (n - k);
  }
  const statistic = n * (n + 2) * q;
  const df = lags - modelDf;
  return { statistic, df, pValue: chiSquareSurvival(statistic, df) };
}

/**
 * Engle's ARCH-LM test: e²_t regressed on a constant and e²_{t−1} … e²_{t−lags};
 * LM = (n − lags)·R² against χ²(lags), as statsmodels' `het_arch`. On the
 * standardised residuals, a small p-value says the model left ARCH behind.
 */
export function archLm(series: ArrayLike<number>, lags: number): TestResult {
  const n = series.length;
  checkLags(lags, n);
  const m = lags + 1;
  const xtx = Array.from({ length: m }, () => new Array<number>(m).fill(0));
  const xty = new Array<number>(m).fill(0);
  const row = new Array<number>(m);
  let sumY = 0;
  let sumY2 = 0;
  for (let t = lags; t < n; t += 1) {
    const y = series[t] ** 2;
    row[0] = 1;
    for (let k = 1; k <= lags; k += 1) row[k] = series[t - k] ** 2;
    for (let i = 0; i < m; i += 1) {
      xty[i] += row[i] * y;
      for (let j = 0; j <= i; j += 1) xtx[i][j] += row[i] * row[j];
    }
    sumY += y;
    sumY2 += y * y;
  }
  for (let i = 0; i < m; i += 1) for (let j = i + 1; j < m; j += 1) xtx[i][j] = xtx[j][i];
  const inverse = invert(xtx);
  if (!inverse) throw new RangeError('the lagged squares are collinear');
  const beta = inverse.map((r) => r.reduce((a, v, j) => a + v * xty[j], 0));
  const nobs = n - lags;
  // R² = (β'X'y − n·ȳ²) / (y'y − n·ȳ²)
  const explained = beta.reduce((a, b, i) => a + b * xty[i], 0) - (sumY * sumY) / nobs;
  const total = sumY2 - (sumY * sumY) / nobs;
  const statistic = nobs * (explained / total);
  return { statistic, df: lags, pValue: chiSquareSurvival(statistic, lags) };
}
