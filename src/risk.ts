import { chiSquareSurvival } from './diagnostics.js';
import type { GarchModel } from './model.js';
import { logGamma } from './model.js';
import type { Simulation } from './simulate.js';
import { segments, validationWindows, walkForwardSteps } from './validation.js';

/**
 * Value at risk and expected shortfall, as positive losses in the returns'
 * units: VaR is minus the return's `level` quantile, ES minus its mean below
 * that quantile. `level` is the tail probability (0.01 for a 99 % VaR).
 */

export type TailRisk = {
  readonly valueAtRisk: number;
  readonly expectedShortfall: number;
};

const checkLevel = (level: number) => {
  if (!(level > 0 && level < 0.5)) throw new RangeError(`level must lie in (0, 0.5), got ${level}`);
};

/** Φ⁻¹(p) — Acklam's rational approximation, relative error < 1.2e-9. */
export function normalQuantile(p: number): number {
  if (!(p > 0 && p < 1)) throw new RangeError(`p must lie in (0, 1), got ${p}`);
  const a = [
    -39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239,
  ];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [
    -0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968,
    2.938163982698783,
  ];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const low = 0.02425;
  if (p < low || p > 1 - low) {
    const q = Math.sqrt(-2 * Math.log(p < low ? p : 1 - p));
    const x =
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    return p < low ? x : -x;
  }
  const q = p - 0.5;
  const r = q * q;
  return (
    ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  );
}

/** The regularised incomplete beta I_x(a, b) — Lentz's continued fraction, on the side where it converges. */
function incompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  if (x > (a + 1) / (a + b + 2)) return 1 - incompleteBeta(1 - x, b, a);
  const tiny = 1e-300;
  let c = 1;
  let d = 1 - ((a + b) * x) / (a + 1);
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let h = d;
  for (let m = 1; m < 1000; m += 1) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((a + m2 - 1) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  return (front * h) / a;
}

/** P(T ≤ t) for Student's t with ν degrees of freedom (not standardised). */
export function studentCdf(t: number, nu: number): number {
  const tail = 0.5 * incompleteBeta(nu / (nu + t * t), nu / 2, 0.5);
  return t < 0 ? tail : 1 - tail;
}

/** The p-quantile of Student's t with ν degrees of freedom, p < 0.5: bisection on the CDF, to double precision. */
export function studentQuantile(p: number, nu: number): number {
  if (!(p > 0 && p < 0.5)) throw new RangeError(`p must lie in (0, 0.5), got ${p}`);
  if (!(nu > 0)) throw new RangeError(`ν must be positive, got ${nu}`);
  let lo = -1;
  while (studentCdf(lo, nu) > p) lo *= 2;
  let hi = 0;
  for (let i = 0; i < 200 && hi - lo > 1e-15 * Math.max(1, Math.abs(lo)); i += 1) {
    const mid = (lo + hi) / 2;
    if (studentCdf(mid, nu) > p) hi = mid;
    else lo = mid;
  }
  return (lo + hi) / 2;
}

/**
 * VaR and ES of r = μ + √s·z, z standard normal or — with `nu` — Student-t
 * standardised to unit variance, the innovations a fit assumed. For a
 * one-step forecast: `mean` μ, `variance` s_{T+1}.
 */
export function tailRisk(request: {
  readonly mean: number;
  readonly variance: number;
  readonly nu?: number;
  readonly level: number;
}): TailRisk {
  const { mean, variance, nu, level } = request;
  checkLevel(level);
  if (!(variance > 0)) throw new RangeError(`variance must be positive, got ${variance}`);
  const sd = Math.sqrt(variance);
  if (nu === undefined) {
    const q = normalQuantile(level);
    const density = Math.exp(-0.5 * q * q) / Math.sqrt(2 * Math.PI);
    return { valueAtRisk: -(mean + sd * q), expectedShortfall: -(mean - (sd * density) / level) };
  }
  if (!(nu > 2)) throw new RangeError(`ν must exceed 2, got ${nu}`);
  const scale = Math.sqrt((nu - 2) / nu);
  const q = studentQuantile(level, nu);
  const density = Math.exp(
    logGamma((nu + 1) / 2) -
      logGamma(nu / 2) -
      0.5 * Math.log(nu * Math.PI) -
      ((nu + 1) / 2) * Math.log1p((q * q) / nu),
  );
  // E[T | T ≤ q] = −(ν + q²)/(ν − 1)·f(q)/level
  const tailMean = (-(nu + q * q) / (nu - 1)) * (density / level);
  return { valueAtRisk: -(mean + sd * scale * q), expectedShortfall: -(mean + sd * scale * tailMean) };
}

/**
 * VaR and ES of the cumulative return over the first `steps` steps of
 * simulated paths — the multi-step answer, whose distribution under GARCH is
 * not the one-step family. Empirical: the ⌈level·paths⌉-th worst sum and the
 * mean of the sums at or below it.
 */
export function simulatedTailRisk(simulation: Simulation, level: number, steps?: number): TailRisk {
  checkLevel(level);
  const paths = simulation.returns.length;
  const length = simulation.returns[0]?.length ?? 0;
  const h = steps ?? length;
  if (!(Number.isInteger(h) && h >= 1 && h <= length))
    throw new RangeError(`steps must be in [1, ${length}], got ${h}`);
  const k = Math.ceil(level * paths);
  if (k < 1 || paths < 2) throw new RangeError('too few paths for this level');
  const sums = new Float64Array(paths);
  simulation.returns.forEach((r, i) => {
    let s = 0;
    for (let t = 0; t < h; t += 1) s += r[t];
    sums[i] = s;
  });
  sums.sort();
  let tail = 0;
  for (let i = 0; i < k; i += 1) tail += sums[i];
  return { valueAtRisk: -sums[k - 1], expectedShortfall: -tail / k };
}

export type CoverageTest = {
  readonly observations: number;
  readonly violations: number;
  readonly rate: number;
  /** Kupiec's proportion of failures: is the violation rate `level`? χ²(1). */
  readonly unconditional: { readonly statistic: number; readonly pValue: number };
  /** Christoffersen's independence: does a violation make the next more likely? χ²(1). */
  readonly independence: { readonly statistic: number; readonly pValue: number };
  /** Both at once, χ²(2). */
  readonly conditional: { readonly statistic: number; readonly pValue: number };
};

/** n·ln p with 0·ln 0 = 0. */
const xlogy = (n: number, p: number) => (n === 0 ? 0 : n * Math.log(p));

/** Kupiec and Christoffersen likelihood-ratio tests of a VaR's violations (true = the loss exceeded it), in order. */
export function coverageTest(violations: readonly boolean[], level: number): CoverageTest {
  checkLevel(level);
  const n = violations.length;
  if (n < 2) throw new RangeError('a coverage test needs at least two observations');
  const x = violations.filter(Boolean).length;
  const rate = x / n;
  const uc = -2 * (xlogy(n - x, 1 - level) + xlogy(x, level) - xlogy(n - x, 1 - rate) - xlogy(x, rate));
  let n00 = 0;
  let n01 = 0;
  let n10 = 0;
  let n11 = 0;
  for (let t = 1; t < n; t += 1) {
    if (violations[t - 1]) {
      if (violations[t]) n11 += 1;
      else n10 += 1;
    } else if (violations[t]) n01 += 1;
    else n00 += 1;
  }
  const p01 = n00 + n01 ? n01 / (n00 + n01) : 0;
  const p11 = n10 + n11 ? n11 / (n10 + n11) : 0;
  const p = (n01 + n11) / (n - 1);
  const ind =
    -2 *
    (xlogy(n00 + n10, 1 - p) +
      xlogy(n01 + n11, p) -
      xlogy(n00, 1 - p01) -
      xlogy(n01, p01) -
      xlogy(n10, 1 - p11) -
      xlogy(n11, p11));
  const result = (statistic: number, df: number) => ({ statistic, pValue: chiSquareSurvival(statistic, df) });
  return {
    observations: n,
    violations: x,
    rate,
    unconditional: result(uc, 1),
    independence: result(ind, 1),
    conditional: result(uc + ind, 2),
  };
}

export type VarBacktestOptions = {
  readonly model?: GarchModel;
  readonly dist?: 'normal' | 't';
  readonly level: number;
  readonly minTrain?: number;
  readonly step?: number;
};

export type VarBacktest = CoverageTest & {
  /** One per scored origin: the VaR forecast for origin + 1 and the return that came. */
  readonly forecasts: readonly { readonly origin: number; readonly valueAtRisk: number; readonly realized: number }[];
  /** Origins whose fit did not converge. */
  readonly skipped: number;
};

/**
 * One-step VaR walked forward: at each origin, fit on [0, origin], forecast
 * the VaR of return origin + 1 from s_{T+1} and the fit's μ (and ν), and count
 * a violation when the return fell below −VaR. The windows are
 * `validationWindows`' (minTrain 500, step 10 by default; step 1 tests every
 * day, at one fit per day).
 */
export function varBacktest(returns: ArrayLike<number>, options: VarBacktestOptions): VarBacktest {
  const model = options.model ?? 'garch';
  const all = Float64Array.from(returns);
  const windows = validationWindows(all.length, { horizon: 1, minTrain: options.minTrain, step: options.step });
  const steps = segments(windows.map((w) => w.origin)).flatMap((origins) =>
    walkForwardSteps({ returns: all, model, dist: options.dist, horizon: 1, origins }),
  );
  const forecasts: { origin: number; valueAtRisk: number; realized: number }[] = [];
  steps.forEach((s, i) => {
    if (!s) return;
    const origin = windows[i].origin;
    const { valueAtRisk } = tailRisk({
      mean: s.fit.params.mu,
      variance: s.forecast,
      nu: s.fit.params.nu,
      level: options.level,
    });
    forecasts.push({ origin, valueAtRisk, realized: all[origin + 1] });
  });
  return {
    ...coverageTest(
      forecasts.map((f) => f.realized < -f.valueAtRisk),
      options.level,
    ),
    forecasts,
    skipped: steps.length - forecasts.length,
  };
}
