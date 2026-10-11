import { egarchLogLikelihood } from './egarch.js';
import { fitGarch, type GarchFit } from './fit.js';
import { forecastVariance } from './forecast.js';
import {
  type Distribution,
  exogenousBackcast,
  type GarchModel,
  type GarchParams,
  logLikelihood,
  type MeanModel,
} from './model.js';
import { type FitAt, segments, type Validation, walkForwardValidation } from './validation.js';

export type { FitAt } from './validation.js';

/**
 * The two whole questions a caller asks, as plain data in and out — so they
 * can cross a worker boundary (structured clone) unchanged.
 */

export type VolatilityForecastRequest = {
  readonly returns: ArrayLike<number>;
  readonly model?: GarchModel;
  readonly dist?: Distribution;
  /** A fitted constant (default) or zero; see `FitOptions.mean`. */
  readonly mean?: MeanModel;
  /** Variance targeting (garch and gjr); see `FitOptions.targetVariance`. */
  readonly targetVariance?: boolean;
  /** GARCH-X: x aligned with the returns (garch and gjr); see `FitOptions.exogenous`. Forecasts hold it at its last value. */
  readonly exogenous?: ArrayLike<number>;
  readonly horizon: number;
  readonly minTrain?: number;
  readonly step?: number;
  /** Also return the rival optima within this many log-likelihood units of the best. */
  readonly rivalsWithin?: number;
};

/** `volatilityForecast`'s answer: the as-of fit and forecast, and the model graded walk-forward against its baseline. */
export type VolatilityForecast = {
  readonly model: GarchModel;
  readonly mean: MeanModel;
  readonly converged: boolean;
  readonly params: GarchParams;
  readonly persistence: number;
  readonly logLikelihood: number;
  /** See `GarchFit.varianceTarget`: null unless ω was targeted. */
  readonly varianceTarget: number | null;
  /** See `GarchFit`: −2·ln L + 2k and −2·ln L + k·ln n. */
  readonly aic: number;
  readonly bic: number;
  /** One per return, in the returns' units squared. */
  readonly conditionalVariance: number[];
  /** s_{T+1} … s_{T+h}; null when the fit did not converge. */
  readonly forecast: number[] | null;
  /** p-value of γ (robust); null for the symmetric model. */
  readonly gammaPValue: number | null;
  /** t innovations only: ν held on a bound (see `GarchFit.nuBound`). */
  readonly nuBound: 'lower' | 'upper' | null;
  /**
   * The other optima whose log-likelihood is within `rivalsWithin` of the
   * fit's — the modes the data cannot tell from it. Empty unless asked.
   */
  readonly rivals: readonly RivalMode[];
  /** Against the next-simpler model: naive trailing variance for garch, symmetric garch for gjr. */
  readonly validation: Validation;
};

/** The as-of half of a `VolatilityForecast`: the fit on everything and its forecast, without the walk-forward. */
export type AsOfFit = Omit<VolatilityForecast, 'validation'>;

/** Another optimum of the as-of likelihood, with what it would have answered. */
export type RivalMode = {
  readonly params: GarchParams;
  readonly persistence: number;
  readonly logLikelihood: number;
  readonly conditionalVariance: number[];
  readonly forecast: number[];
};

/** Fit on everything and forecast `horizon` steps — the as-of half of `volatilityForecast`. */
export function asOfFit(
  request: Pick<
    VolatilityForecastRequest,
    'returns' | 'model' | 'dist' | 'mean' | 'targetVariance' | 'exogenous' | 'horizon' | 'rivalsWithin'
  >,
): AsOfFit {
  const model = request.model ?? 'garch';
  const fit = fitGarch(request.returns, {
    model,
    dist: request.dist,
    mean: request.mean,
    targetVariance: request.targetVariance,
    exogenous: request.exogenous,
    inference: model !== 'garch',
  });
  const p = fit.inference?.pValues.gamma;
  const within = request.rivalsWithin;
  return {
    rivals:
      within === undefined
        ? []
        : fit.modes
            .slice(1)
            .filter((m) => m.converged && fit.logLikelihood - m.logLikelihood <= within)
            .map((m) => {
              const rival = refit(fit, m.params, request.returns, request.exogenous);
              return {
                params: m.params,
                persistence: m.persistence,
                logLikelihood: m.logLikelihood,
                conditionalVariance: Array.from(rival.conditionalVariance),
                forecast: forecastVariance(rival, request.horizon),
              };
            }),
    model,
    mean: fit.mean,
    converged: fit.converged,
    params: fit.params,
    persistence: fit.persistence,
    logLikelihood: fit.logLikelihood,
    varianceTarget: fit.varianceTarget,
    aic: fit.aic,
    bic: fit.bic,
    conditionalVariance: Array.from(fit.conditionalVariance),
    forecast: fit.converged ? forecastVariance(fit, request.horizon) : null,
    gammaPValue: p !== undefined && Number.isFinite(p) ? p : null,
    nuBound: fit.nuBound,
  };
}

/** The fit as it would be at other parameters: their variance path over the same returns. */
function refit(
  fit: GarchFit,
  params: GarchParams,
  returns: ArrayLike<number>,
  exogenous?: ArrayLike<number>,
): GarchFit {
  const variance = new Float64Array(returns.length);
  if (fit.model === 'egarch') egarchLogLikelihood(returns, params, fit.backcast, variance);
  else
    logLikelihood(
      returns,
      params,
      fit.backcast,
      variance,
      undefined,
      exogenous ? { values: exogenous, backcast: exogenousBackcast(exogenous) } : undefined,
    );
  return { ...fit, params, conditionalVariance: variance, lastResidual: returns[returns.length - 1] - params.mu };
}

/** The walk-forward's baseline for a model: naive trailing variance for garch, symmetric garch for the asymmetric ones. */
export const baselineOf = (model: GarchModel) => (model === 'garch' ? 'naive' : 'garch') as 'garch' | 'naive';

/** Fit on everything, forecast `horizon` steps, and grade the model walk-forward. */
export function volatilityForecast(request: VolatilityForecastRequest): VolatilityForecast {
  const model = request.model ?? 'garch';
  return {
    ...asOfFit(request),
    validation: walkForwardValidation(request.returns, {
      model,
      dist: request.dist,
      mean: request.mean,
      targetVariance: request.targetVariance,
      exogenous: request.exogenous,
      horizon: request.horizon,
      minTrain: request.minTrain,
      step: request.step,
      baseline: baselineOf(model),
    }),
  };
}

/** `fitsAt`'s question: the returns and the train ends to fit at. */
export type FitsAtRequest = {
  readonly returns: ArrayLike<number>;
  /** Train-end indices: each fit sees returns[0 … end]. */
  readonly ends: readonly number[];
  readonly model?: GarchModel;
  readonly dist?: Distribution;
  /** A fitted constant (default) or zero; see `FitOptions.mean`. */
  readonly mean?: MeanModel;
  /** Variance targeting (garch and gjr); see `FitOptions.targetVariance`. */
  readonly targetVariance?: boolean;
  /** GARCH-X: x aligned with the returns (garch and gjr); see `FitOptions.exogenous`. Forecasts hold it at its last value. */
  readonly exogenous?: ArrayLike<number>;
};

/**
 * One fit per train end — what a simulation needs to run the recursion
 * forward from each window. null where the fit did not converge. The ends are
 * cut into segments (`SEGMENT_ORIGINS`, in the order given); inside a segment each
 * fit also starts from the previous converged one.
 */
export function fitsAt(request: FitsAtRequest): (FitAt | null)[] {
  return segments(request.ends).flatMap((ends) => fitsAtSegment({ ...request, ends }));
}

/** One segment of `fitsAt` — what a worker runs. */
export function fitsAtSegment(request: FitsAtRequest): (FitAt | null)[] {
  const model = request.model ?? 'garch';
  const all = Float64Array.from(request.returns);
  const x = request.exogenous ? Float64Array.from(request.exogenous) : undefined;
  let warm: GarchParams | null = null;
  return request.ends.map((end) => {
    if (!(Number.isInteger(end) && end >= 1 && end < all.length))
      throw new RangeError(`train end ${end} is outside the ${all.length} returns`);
    const fit = fitGarch(all.subarray(0, end + 1), {
      model,
      dist: request.dist,
      mean: request.mean,
      targetVariance: request.targetVariance,
      exogenous: x?.subarray(0, end + 1),
      warmStart: warm,
    });
    if (!fit.converged) return null;
    warm = fit.params;
    return {
      params: fit.params,
      lastVariance: fit.conditionalVariance[end],
      lastResidual: fit.lastResidual,
    };
  });
}
