import { asOfFit, fitsAt, fitsAtSegment, volatilityForecast } from './jobs';
import { walkForwardForecasts, walkForwardSteps } from './validation';
import { weightedAsOf, weightedForecast } from './weighted';

/** What a worker can be asked; a segment is the unit `@smi/garch/node` spreads across cores. */
export const JOBS = {
  volatilityForecast,
  asOfFit,
  walkForwardForecasts,
  walkForwardSteps,
  fitsAt,
  fitsAtSegment,
  weightedForecast,
  weightedAsOf,
} as const;
export type JobName = keyof typeof JOBS;
export type JobRequest<J extends JobName> = Parameters<(typeof JOBS)[J]>[0];
export type JobResult<J extends JobName> = ReturnType<(typeof JOBS)[J]>;
