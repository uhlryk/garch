import { fitGarch, type GarchFit } from './fit.js';
import { forecastVariance } from './forecast.js';
import type { Distribution, GarchModel, GarchParams, MeanModel } from './model.js';

/**
 * Walk-forward, out of sample: is the forecast worth quoting? The same
 * harness `arch` fits were graded with before this package replaced them.
 */

export type Window = {
  /** The last in-sample index; the train window is [0, origin]. */
  readonly origin: number;
  /** [origin + 1, origin + horizon], always exactly `horizon` long. */
  readonly testStart: number;
  readonly testEnd: number;
};

/**
 * Expanding windows over n rows. INVARIANT: every train index < every test
 * index. An origin whose test window would run past the end is not yielded —
 * the last origin is n − 1 − horizon.
 */
export function walkForwardOrigins(n: number, minTrain: number, horizon: number, step = 1): Window[] {
  if (!(minTrain >= 1)) throw new RangeError(`minTrain must be ≥ 1, got ${minTrain}`);
  if (!(horizon >= 1)) throw new RangeError(`horizon must be ≥ 1, got ${horizon}`);
  if (!(step >= 1)) throw new RangeError(`step must be ≥ 1, got ${step}`);
  const out: Window[] = [];
  for (let origin = minTrain - 1; origin <= n - 1 - horizon; origin += step)
    out.push({ origin, testStart: origin + 1, testEnd: origin + horizon });
  return out;
}

/**
 * QLIKE (Patton 2011): mean(r/f − ln(r/f) − 1) over variances. ≥ 0, 0 at
 * r = f, and harder on UNDER-prediction — the dangerous direction for a risk
 * read. A zero realised variance is floored so one flat window cannot blow up
 * the mean.
 */
export function qlike(realized: readonly number[], forecast: readonly number[]): number {
  if (realized.length !== forecast.length) throw new RangeError('qlike: length mismatch');
  if (!realized.length) throw new RangeError('qlike: empty inputs');
  let sum = 0;
  for (let i = 0; i < realized.length; i += 1) {
    if (!(forecast[i] > 0)) throw new RangeError('qlike: forecast variance must be positive');
    sum += qlikeTerm(realized[i], forecast[i]);
  }
  return sum / realized.length;
}

/** One observation's QLIKE, r/f − ln(r/f) − 1, the ratio floored above 0. */
export function qlikeTerm(realized: number, forecast: number): number {
  const ratio = Math.max(realized / forecast, 1e-300);
  return ratio - Math.log(ratio) - 1;
}

/** Realised-variance proxy: the mean squared return (zero mean over a short horizon). */
export function realizedVariance(returns: ArrayLike<number>, from = 0, to = returns.length - 1): number {
  let s = 0;
  for (let i = from; i <= to; i += 1) s += returns[i] * returns[i];
  return s / (to - from + 1);
}

export type Baseline = 'naive' | 'garch';

export type Validation = {
  readonly loss: number | null;
  readonly baselineLoss: number | null;
  /** null when no origin could be scored. */
  readonly beatsBaseline: boolean | null;
  readonly origins: number;
  /** Origins dropped because a fit did not converge or a forecast was not usable. */
  readonly skipped: number;
};

export type ValidationOptions = {
  readonly model: GarchModel;
  readonly dist?: Distribution;
  /** A fitted constant (default) or zero; see `FitOptions.mean`. */
  readonly mean?: MeanModel;
  readonly horizon: number;
  readonly minTrain?: number;
  readonly step?: number;
  /**
   * What the model must beat: `naive` — the trailing `horizon` realised
   * variance (the random-walk vol forecast); `garch` — the symmetric model,
   * for asking whether an asymmetric term earns its keep.
   */
  readonly baseline: Baseline;
};

const mean = (v: readonly number[]) => v.reduce((a, b) => a + b, 0) / v.length;

/**
 * Origins per segment. A walk-forward is cut into segments of this many
 * consecutive origins; each starts cold (the grid alone) and warm-starts from
 * the previous origin inside it. The cut depends on the history only — never
 * on how many cores run it — so the answer is the same in-process and on any
 * number of workers.
 */
export const SEGMENT_ORIGINS = 32;

/** `items` cut into consecutive runs of `size`. */
export function segments<T>(items: readonly T[], size = SEGMENT_ORIGINS): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

export type ForecastsRequest = {
  readonly returns: ArrayLike<number>;
  readonly model: GarchModel;
  readonly dist?: Distribution;
  /** A fitted constant (default) or zero; see `FitOptions.mean`. */
  readonly mean?: MeanModel;
  readonly horizon: number;
  /** Train-end indices, ascending; one segment (the caller cuts with `segments`). */
  readonly origins: readonly number[];
};

/** A fit as a simulation needs it: its parameters and where its recursion stood at the train end. */
export type FitAt = {
  readonly params: GarchParams;
  /** s_end, the conditional variance of the last return in the window. */
  readonly lastVariance: number;
  /** e_end. */
  readonly lastResidual: number;
};

/** One walk-forward origin: the fit on [0, origin] and its mean variance forecast. */
export type WalkForwardStep = {
  readonly fit: FitAt;
  readonly forecast: number;
};

/**
 * One segment of a walk-forward: at each origin, fit on [0, origin] (warm from
 * the previous origin's optimum as well as the grid) and forecast the mean
 * variance of the next `horizon` returns. null where the fit did not converge
 * or could not be made.
 */
export function walkForwardForecasts(request: ForecastsRequest): (number | null)[] {
  return walkForwardSteps(request).map((s) => s?.forecast ?? null);
}

/** `walkForwardForecasts` with each origin's fit kept — what a simulation from every origin needs. */
export function walkForwardSteps(request: ForecastsRequest): (WalkForwardStep | null)[] {
  const all = Float64Array.from(request.returns as ArrayLike<number>);
  let warm: GarchParams | null = null;
  return request.origins.map((origin) => {
    const fit = safeFit(all.subarray(0, origin + 1), request.model, warm, request.dist, request.mean);
    if (!fit?.converged) return null;
    warm = fit.params;
    return {
      fit: { params: fit.params, lastVariance: fit.conditionalVariance[origin], lastResidual: fit.lastResidual },
      forecast: mean(forecastVariance(fit, request.horizon)),
    };
  });
}

/**
 * QLIKE of the model's forecasts and the baseline's against what the windows
 * realised. `baseline` null is the naive forecast: the trailing `horizon`
 * realised variance.
 */
export function scoreWalkForward(
  returns: ArrayLike<number>,
  windows: readonly Window[],
  horizon: number,
  forecasts: readonly (number | null)[],
  baseline: readonly (number | null)[] | null,
): Validation {
  const realized: number[] = [];
  const forecast: number[] = [];
  const base: number[] = [];
  let skipped = 0;
  windows.forEach((w, i) => {
    const vhat = forecasts[i];
    const bl = baseline
      ? baseline[i]
      : realizedVariance(returns, w.origin + 1 - Math.min(horizon, w.origin + 1), w.origin);
    const rv = realizedVariance(returns, w.testStart, w.testEnd);
    if (
      vhat === null ||
      bl === null ||
      !(Number.isFinite(vhat) && vhat > 0 && Number.isFinite(bl) && bl > 0 && Number.isFinite(rv))
    ) {
      skipped += 1;
      return;
    }
    realized.push(rv);
    forecast.push(vhat);
    base.push(bl);
  });
  const loss = realized.length ? qlike(realized, forecast) : null;
  const baselineLoss = realized.length ? qlike(realized, base) : null;
  return {
    loss,
    baselineLoss,
    beatsBaseline: loss !== null && baselineLoss !== null ? loss < baselineLoss : null,
    origins: realized.length,
    skipped,
  };
}

/** The windows a validation with these options scores. */
export function validationWindows(
  n: number,
  options: Pick<ValidationOptions, 'horizon' | 'minTrain' | 'step'>,
): Window[] {
  return walkForwardOrigins(n, options.minTrain ?? 500, options.horizon, options.step ?? 10);
}

/**
 * At every origin: fit on [0, origin], forecast the mean variance of the next
 * `horizon` returns, score it and the baseline by QLIKE against what they
 * realised — segment by segment (`SEGMENT_ORIGINS`), in this thread. The model's
 * segments and the baseline's are independent series of fits; `@krzysztofsztompka/garch/node`
 * runs each on its own worker for the same answer.
 */
export function walkForwardValidation(returns: ArrayLike<number>, options: ValidationOptions): Validation {
  const { horizon, model } = options;
  const windows = validationWindows(returns.length, options);
  const run = (m: GarchModel) =>
    segments(windows.map((w) => w.origin)).flatMap((origins) =>
      walkForwardForecasts({ returns, model: m, dist: options.dist, mean: options.mean, horizon, origins }),
    );
  return scoreWalkForward(returns, windows, horizon, run(model), options.baseline === 'garch' ? run('garch') : null);
}

function safeFit(
  train: ArrayLike<number>,
  model: GarchModel,
  warm: GarchParams | null,
  dist?: Distribution,
  mean?: MeanModel,
): GarchFit | null {
  try {
    return fitGarch(train, { model, dist, mean, warmStart: warm });
  } catch {
    return null;
  }
}
