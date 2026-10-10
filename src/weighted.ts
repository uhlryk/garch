import { type AsOfFit, asOfFit } from './jobs';
import type { Distribution, GarchModel } from './model';
import {
  type FitAt,
  realizedVariance,
  scoreWalkForward,
  segments,
  type Validation,
  validationWindows,
  type WalkForwardStep,
  walkForwardSteps,
  type Window,
} from './validation';

/**
 * Several models weighted by how well each has forecast so far (#1113).
 *
 * #1111 measured the alternative — picking the one model with the lowest past
 * loss — on 15 series: it beat the default on 8 and cost up to +47 % QLIKE
 * where the ranking flipped (VNDA, SMMT). Weighting every model by 1 / its
 * mean past loss averaged −5 % with a worst case of +4 %. At each origin only
 * windows that have already ended count, so no weight has seen the future.
 */

export type ModelSpec = { readonly model: GarchModel; readonly dist: Distribution };

/** {garch, gjr} × {normal, t} — the four #1111 weighed. */
export const WEIGHTED_MODELS: readonly ModelSpec[] = [
  { model: 'garch', dist: 'normal' },
  { model: 'gjr', dist: 'normal' },
  { model: 'garch', dist: 't' },
  { model: 'gjr', dist: 't' },
];

/** Past windows a model needs before its loss decides its weight; until then the weights are equal. */
export const WEIGHT_BURN_IN = 20;

export type WeightedForecastRequest = {
  readonly returns: ArrayLike<number>;
  readonly horizon: number;
  readonly minTrain?: number;
  readonly step?: number;
  readonly models?: readonly ModelSpec[];
  readonly burnIn?: number;
};

export type WeightedModel = {
  readonly spec: ModelSpec;
  /** The fit on the whole history; null when it could not be made. */
  readonly asOf: AsOfFit | null;
  /** Its own walk-forward against the naive baseline. */
  readonly validation: Validation;
  /** Its weight in the as-of forecast (0 when it has no usable forecast). */
  readonly weight: number;
};

/** One walk-forward origin of the combination: each model's fit there and the weights in force. */
export type WeightedOrigin = {
  /** The last in-sample index. */
  readonly origin: number;
  /** One per model, in `models` order: from the windows that ended by `origin`; all 0 when no model has a fit. */
  readonly weights: number[];
  /** One per model; null where its fit did not converge. */
  readonly fits: (FitAt | null)[];
};

export type WeightedForecast = {
  readonly models: readonly WeightedModel[];
  /** Σ wᵢ·s_{T+k} over the models with a forecast; null when none has one. */
  readonly forecast: number[] | null;
  /** Σ wᵢ·s_t with the as-of weights — the path a regime is read from. */
  readonly conditionalVariance: number[] | null;
  /** The weighted forecasts walked forward, against the naive baseline. */
  readonly validation: Validation;
  /** Every origin's fits and weights — what a simulation from each origin draws on. */
  readonly walkForward: readonly WeightedOrigin[];
};

const qlikeTerm = (realized: number, forecast: number) => {
  const ratio = Math.max(realized / forecast, 1e-300);
  return ratio - Math.log(ratio) - 1;
};

const usable = (v: number | null | undefined): v is number => v !== null && v !== undefined && Number.isFinite(v) && v > 0;

/**
 * The combination itself, on forecasts already made: at each origin the
 * models with a forecast there, weighted by 1 / their mean QLIKE over the
 * windows that ended by that origin (equal until each has `burnIn` of them).
 * Also the as-of weights, from every window.
 */
export function weighForecasts(
  returns: ArrayLike<number>,
  windows: readonly Window[],
  perModel: readonly (readonly (number | null)[])[],
  burnIn = WEIGHT_BURN_IN,
): { forecasts: (number | null)[]; originWeights: number[][]; asOfWeights: number[] } {
  const m = perModel.length;
  const realized = windows.map((w) => realizedVariance(returns, w.testStart, w.testEnd));
  const lossSum = new Array<number>(m).fill(0);
  const lossCount = new Array<number>(m).fill(0);
  let scored = 0; // windows already folded into the sums, in order of their end
  const byEnd = windows.map((_, i) => i).sort((a, b) => windows[a].testEnd - windows[b].testEnd);
  const weightsFrom = (available: boolean[]) => {
    const ready = available.every((a, k) => !a || lossCount[k] >= burnIn);
    const raw = available.map((a, k) => (!a ? 0 : ready ? 1 / Math.max(lossSum[k] / lossCount[k], 1e-12) : 1));
    const total = raw.reduce((x, y) => x + y, 0);
    return total > 0 ? raw.map((v) => v / total) : raw;
  };
  const originWeights: number[][] = [];
  const forecasts = windows.map((w, i) => {
    while (scored < byEnd.length && windows[byEnd[scored]].testEnd <= w.origin) {
      const j = byEnd[scored];
      for (let k = 0; k < m; k += 1)
        if (usable(perModel[k][j]) && Number.isFinite(realized[j])) {
          lossSum[k] += qlikeTerm(realized[j], perModel[k][j]!);
          lossCount[k] += 1;
        }
      scored += 1;
    }
    const available = perModel.map((f) => usable(f[i]));
    if (!available.some(Boolean)) {
      originWeights.push(new Array<number>(m).fill(0));
      return null;
    }
    const weights = weightsFrom(available);
    originWeights.push(weights);
    return weights.reduce((s, wt, k) => s + (wt ? wt * perModel[k][i]! : 0), 0);
  });
  while (scored < byEnd.length) {
    const j = byEnd[scored];
    for (let k = 0; k < m; k += 1)
      if (usable(perModel[k][j]) && Number.isFinite(realized[j])) {
        lossSum[k] += qlikeTerm(realized[j], perModel[k][j]!);
        lossCount[k] += 1;
      }
    scored += 1;
  }
  return { forecasts, originWeights, asOfWeights: weightsFrom(perModel.map((_, k) => lossCount[k] > 0)) };
}

/** Put together what the models answered: the as-of combination and every validation. */
export function assembleWeighted(
  returns: ArrayLike<number>,
  request: WeightedForecastRequest,
  specs: readonly ModelSpec[],
  asOf: readonly (AsOfFit | null)[],
  steps: readonly (readonly (WalkForwardStep | null)[])[],
): WeightedForecast {
  const windows = validationWindows(returns.length, request);
  const perModel = steps.map((s) => s.map((x) => x?.forecast ?? null));
  const { forecasts, originWeights, asOfWeights } = weighForecasts(returns, windows, perModel, request.burnIn);
  // a model without an as-of forecast gets no weight; the rest are renormalised
  const raw = specs.map((_, k) => (asOf[k]?.forecast ? asOfWeights[k] : 0));
  const total = raw.reduce((a, b) => a + b, 0);
  const weights = raw.map((w) => (total > 0 ? w / total : 0));
  const mix = (pick: (fit: AsOfFit) => readonly number[]) => {
    if (!(total > 0)) return null;
    let out: number[] | null = null;
    specs.forEach((_, k) => {
      if (!weights[k]) return;
      const v = pick(asOf[k]!);
      out ??= new Array<number>(v.length).fill(0);
      for (let i = 0; i < v.length; i += 1) out[i] += weights[k] * v[i];
    });
    return out;
  };
  return {
    models: specs.map((spec, k) => ({
      spec,
      asOf: asOf[k],
      validation: scoreWalkForward(returns, windows, request.horizon, perModel[k], null),
      weight: weights[k],
    })),
    forecast: mix((f) => f.forecast!),
    conditionalVariance: mix((f) => f.conditionalVariance),
    validation: scoreWalkForward(returns, windows, request.horizon, forecasts, null),
    walkForward: windows.map((w, i) => ({
      origin: w.origin,
      weights: originWeights[i],
      fits: steps.map((s) => s[i]?.fit ?? null),
    })),
  };
}

/** One model's as-of fit, or null when it cannot be made — a worker job, so one model cannot fail the rest. */
export function weightedAsOf(request: { returns: ArrayLike<number>; spec: ModelSpec; horizon: number }): AsOfFit | null {
  return safeAsOf(request.returns, request.spec, request.horizon);
}

function safeAsOf(returns: ArrayLike<number>, spec: ModelSpec, horizon: number): AsOfFit | null {
  try {
    return asOfFit({ returns, model: spec.model, dist: spec.dist, horizon });
  } catch {
    return null;
  }
}

/** The weighted forecast in this thread; `@smi/garch/node` gives the same answer on the pool. */
export function weightedForecast(request: WeightedForecastRequest): WeightedForecast {
  const specs = request.models ?? WEIGHTED_MODELS;
  const returns = Float64Array.from(request.returns);
  const origins = segments(validationWindows(returns.length, request).map((w) => w.origin));
  const steps = specs.map((spec) =>
    origins.flatMap((o) =>
      walkForwardSteps({ returns, model: spec.model, dist: spec.dist, horizon: request.horizon, origins: o }),
    ),
  );
  return assembleWeighted(
    returns,
    request,
    specs,
    specs.map((spec) => safeAsOf(returns, spec, request.horizon)),
    steps,
  );
}
