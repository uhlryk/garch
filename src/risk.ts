import { chiSquareSurvival } from './diagnostics.js';
import type { Distribution, GarchModel, MeanModel } from './model.js';
import { logGamma, normalQuantile, studentQuantile } from './special.js';

export { normalQuantile, studentCdf, studentQuantile } from './special.js';

import type { Simulation } from './simulate.js';
import { skewPartialMoments, skewQuantile } from './skewt.js';
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

/**
 * VaR and ES of r = μ + √s·z, z standard normal or — with `nu` — Student-t
 * (skewed t with `lambda` too)
 * standardised to unit variance, the innovations a fit assumed. For a
 * one-step forecast: `mean` μ, `variance` s_{T+1}.
 */
export function tailRisk(request: {
  readonly mean: number;
  readonly variance: number;
  readonly nu?: number;
  /** With `nu`, a skewed t (Hansen's, as fitted with `dist: 'skewt'`). */
  readonly lambda?: number;
  readonly level: number;
}): TailRisk {
  const { mean, variance, nu, lambda, level } = request;
  checkLevel(level);
  if (!(variance > 0)) throw new RangeError(`variance must be positive, got ${variance}`);
  const sd = Math.sqrt(variance);
  if (nu === undefined) {
    const q = normalQuantile(level);
    const density = Math.exp(-0.5 * q * q) / Math.sqrt(2 * Math.PI);
    return { valueAtRisk: -(mean + sd * q), expectedShortfall: -(mean - (sd * density) / level) };
  }
  if (!(nu > 2)) throw new RangeError(`ν must exceed 2, got ${nu}`);
  if (lambda !== undefined) {
    if (!(lambda > -1 && lambda < 1)) throw new RangeError(`λ must lie in (−1, 1), got ${lambda}`);
    const q = skewQuantile(level, nu, lambda);
    // E[z | z ≤ q] = E[z·1{z ≤ q}] / level
    const tailMean = skewPartialMoments(q, nu, lambda)[1] / level;
    return { valueAtRisk: -(mean + sd * q), expectedShortfall: -(mean + sd * tailMean) };
  }
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
  readonly dist?: Distribution;
  /** A fitted constant (default) or zero; see `FitOptions.mean`. */
  readonly mean?: MeanModel;
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
    walkForwardSteps({ returns: all, model, dist: options.dist, mean: options.mean, horizon: 1, origins }),
  );
  const forecasts: { origin: number; valueAtRisk: number; realized: number }[] = [];
  steps.forEach((s, i) => {
    if (!s) return;
    const origin = windows[i].origin;
    const { valueAtRisk } = tailRisk({
      mean: s.fit.params.mu,
      variance: s.forecast,
      nu: s.fit.params.nu,
      lambda: s.fit.params.lambda,
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
