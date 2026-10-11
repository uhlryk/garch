import { studentCdf } from './special.js';

/**
 * Is one forecast's loss lower than another's by more than noise? The
 * Diebold–Mariano test on the per-origin loss differences, with the
 * Harvey–Leybourne–Newbold small-sample correction.
 */

export type DieboldMariano = {
  /** mean(lossA − lossB): below 0, A lost less. */
  readonly meanDifference: number;
  /** The HAC standard error of that mean. */
  readonly standardError: number;
  /** The HLN-corrected statistic, referred to Student's t with `df` degrees of freedom. */
  readonly statistic: number;
  readonly df: number;
  /** Two-sided: is either forecast better? */
  readonly pValue: number;
  readonly observations: number;
  /** Autocovariance lags in the HAC variance: horizon − 1. */
  readonly lags: number;
};

/** How `dieboldMariano` weighs the autocovariances. */
export type DieboldMarianoOptions = {
  /**
   * The HAC weights of the autocovariances: `bartlett` (default), Newey–West's
   * 1 − k/horizon, never negative; `uniform`, Diebold and Mariano's original
   * and R's `dm.test` default, all 1 — it can come out non-positive, and then
   * the test is refused.
   */
  readonly kernel?: 'bartlett' | 'uniform';
};

/**
 * Diebold–Mariano on two loss series of the same forecasts' targets, in
 * order: d_t = lossA_t − lossB_t, its mean over a HAC variance with
 * `horizon` − 1 lags (h-step forecasts' errors overlap that far), times
 * HLN's √((n + 1 − 2h + h(h − 1)/n)/n), against t(n − 1). For forecasts made
 * every `step` origins over `horizon`-long windows, the windows overlap for
 * ⌈horizon/step⌉ origins: pass that as `horizon`.
 */
export function dieboldMariano(
  lossA: readonly number[],
  lossB: readonly number[],
  horizon = 1,
  options: DieboldMarianoOptions = {},
): DieboldMariano {
  const n = lossA.length;
  if (lossB.length !== n) throw new RangeError(`the loss series differ in length: ${n} and ${lossB.length}`);
  if (!(Number.isInteger(horizon) && horizon >= 1))
    throw new RangeError(`horizon must be an integer ≥ 1, got ${horizon}`);
  if (n < horizon + 2) throw new RangeError(`${n} losses are too few for horizon ${horizon}`);
  const d = new Float64Array(n);
  let mean = 0;
  for (let t = 0; t < n; t += 1) {
    d[t] = lossA[t] - lossB[t];
    if (!Number.isFinite(d[t])) throw new RangeError(`loss ${t} is not a finite number`);
    mean += d[t];
  }
  mean /= n;
  const autocovariance = (k: number) => {
    let s = 0;
    for (let t = k; t < n; t += 1) s += (d[t] - mean) * (d[t - k] - mean);
    return s / n;
  };
  const uniform = options.kernel === 'uniform';
  let longRun = autocovariance(0);
  for (let k = 1; k < horizon; k += 1) longRun += 2 * (uniform ? 1 : 1 - k / horizon) * autocovariance(k);
  if (!(longRun > 0)) throw new RangeError('the loss differences have no positive long-run variance');
  const standardError = Math.sqrt(longRun / n);
  const h = horizon;
  const statistic = (mean / standardError) * Math.sqrt((n + 1 - 2 * h + (h * (h - 1)) / n) / n);
  const df = n - 1;
  return {
    meanDifference: mean,
    standardError,
    statistic,
    df,
    pValue: 2 * studentCdf(-Math.abs(statistic), df),
    observations: n,
    lags: horizon - 1,
  };
}
