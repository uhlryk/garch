import { type FitOptions, fitGarch } from './fit.js';
import { asOfFit, fitsAt, fitsAtSegment, volatilityForecast } from './jobs.js';
import { walkForwardForecasts, walkForwardSteps } from './validation.js';
import { weightedAsOf, weightedForecast } from './weighted.js';

/** What a worker can be asked; a segment is the unit `@krzysztofsztompka/garch/node` spreads across cores. */
export const JOBS = {
  volatilityForecast,
  asOfFit,
  walkForwardForecasts,
  walkForwardSteps,
  fitsAt,
  fitsAtSegment,
  weightedForecast,
  weightedAsOf,
  /** One series' fit — a DCC's stage 1 runs one per asset. */
  fitGarch: (request: { readonly returns: ArrayLike<number>; readonly options?: FitOptions }) =>
    fitGarch(request.returns, request.options),
} as const;
export type JobName = keyof typeof JOBS;
export type JobRequest<J extends JobName> = Parameters<(typeof JOBS)[J]>[0];
export type JobResult<J extends JobName> = ReturnType<(typeof JOBS)[J]>;
