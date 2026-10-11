/**
 * Node-only entry (`@krzysztofsztompka/garch/node`): jobs on worker threads, so a
 * walk-forward that takes minutes neither holds the caller's event loop nor
 * runs on one core. A pool of reused workers takes the segments a walk-forward
 * is cut into (`SEGMENT_ORIGINS`) — the answer is the same as in-process, on any
 * number of cores.
 */
import { availableParallelism } from 'node:os';
import { Worker } from 'node:worker_threads';
import type { DccFit, DccOptions } from './dcc.js';
import type { FitAt, VolatilityForecast, VolatilityForecastRequest } from './jobs.js';
import {
  fitDccOnPool,
  fitsAtOnPool,
  type RunOptions,
  runOnPool,
  volatilityForecastOnPool,
  WorkerPool,
  weightedForecastOnPool,
} from './pool.js';
import type { JobName, JobRequest, JobResult } from './registry.js';
import type { WeightedForecast, WeightedForecastRequest } from './weighted.js';
import { workerFile } from './worker-file.js';

export {
  GarchJobFailedError,
  GarchJobInputError,
  GarchJobTimeoutError,
  type PoolTask,
  type RunOptions,
  WorkerPool,
} from './pool.js';

/**
 * Up to `size` worker threads (default: every core but one), started on
 * demand and reused. An idle worker does not keep the process alive.
 */
export class GarchPool extends WorkerPool {
  constructor(size = Math.max(1, availableParallelism() - 1)) {
    super(size, (events) => {
      const worker = new Worker(workerFile);
      worker.on('message', events.answer);
      worker.on('error', (error) => events.lost(error.message));
      worker.on('exit', (code) => events.lost(`the worker exited with code ${code} before answering`));
      return {
        post: (message) => worker.postMessage(message),
        terminate: () => worker.terminate(),
        hold: () => worker.ref(),
        release: () => worker.unref(),
      };
    });
  }
}

let shared: GarchPool | null = null;
/** The process-wide pool, one worker per core but one. */
export function defaultPool(): GarchPool {
  shared ??= new GarchPool();
  return shared;
}

/** One job on one worker of the pool. */
export function runInWorker<J extends JobName>(
  job: J,
  request: JobRequest<J>,
  options: RunOptions = {},
): Promise<JobResult<J>> {
  return runOnPool(options.pool ?? defaultPool(), job, request, options.timeoutMs);
}

/**
 * `volatilityForecast` across the pool: the as-of fit, the model's segments and
 * (for gjr and egarch) the symmetric baseline's segments all at once, scored
 * here. The same answer as the in-process job.
 */
export function volatilityForecastInWorkers(
  request: VolatilityForecastRequest,
  options: RunOptions = {},
): Promise<VolatilityForecast> {
  return volatilityForecastOnPool(options.pool ?? defaultPool(), request, options.timeoutMs);
}

/** `fitsAt` across the pool, segment by segment — the same answer as in-process. */
export function fitsAtInWorkers(request: JobRequest<'fitsAt'>, options: RunOptions = {}): Promise<(FitAt | null)[]> {
  return fitsAtOnPool(options.pool ?? defaultPool(), request, options.timeoutMs);
}

/**
 * `weightedForecast` across the pool: each model's as-of fit and each model's
 * walk-forward segments at once, combined here — the in-process answer.
 */
export function weightedForecastInWorkers(
  request: WeightedForecastRequest,
  options: RunOptions = {},
): Promise<WeightedForecast> {
  return weightedForecastOnPool(options.pool ?? defaultPool(), request, options.timeoutMs);
}

/** `fitDcc` with stage 1 — one fit per asset — across the pool; the in-process answer. */
export function fitDccInWorkers(
  series: readonly ArrayLike<number>[],
  options: DccOptions = {},
  runOptions: RunOptions = {},
): Promise<DccFit> {
  return fitDccOnPool(runOptions.pool ?? defaultPool(), series, options, runOptions.timeoutMs);
}
