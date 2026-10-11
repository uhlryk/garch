/**
 * Browser entry (`@krzysztofsztompka/garch/web`, ES modules only): the pool of
 * `./node` on Web Workers, so a walk-forward does not block the page. The same
 * calls and the same answer to the bit as in-process.
 */

import type { DccFit, DccOptions } from './dcc.js';
import type { FitAt, VolatilityForecast, VolatilityForecastRequest } from './jobs.js';
import {
  fitDccOnPool,
  fitsAtOnPool,
  type PoolWorker,
  type RunOptions,
  runOnPool,
  volatilityForecastOnPool,
  WorkerPool,
  weightedForecastOnPool,
} from './pool.js';

export type { Answer } from './worker-core.js';

import type { JobName, JobRequest, JobResult } from './registry.js';
import { createWebWorker, type WebWorker } from './web-worker-file.js';
import type { WeightedForecast, WeightedForecastRequest } from './weighted.js';
import type { Answer } from './worker-core.js';

export {
  GarchJobFailedError,
  GarchJobInputError,
  GarchJobTimeoutError,
  type PoolTask,
  type PoolWorker,
  type RunOptions,
  type WorkerEvents,
  WorkerPool,
} from './pool.js';
export type { WebWorker } from './web-worker-file.js';

const cores = () =>
  (globalThis as { navigator?: { hardwareConcurrency?: number } }).navigator?.hardwareConcurrency ?? 4;

/**
 * Up to `size` Web Workers (default: every logical core but one), started on
 * demand and reused. `createWorker` replaces how one is started — for a
 * bundler that needs the worker URL its own way, or a test.
 */
export class GarchWebPool extends WorkerPool {
  constructor(size = Math.max(1, cores() - 1), createWorker: () => WebWorker = createWebWorker) {
    super(size, (events): PoolWorker => {
      const worker = createWorker();
      worker.addEventListener('message', (event) => events.answer(event.data as Answer));
      worker.addEventListener('error', (event) => events.lost(event.message ?? 'the worker failed'));
      worker.addEventListener('messageerror', () => events.lost('a message could not be read'));
      return { post: (message) => worker.postMessage(message), terminate: () => worker.terminate() };
    });
  }
}

let shared: GarchWebPool | null = null;
/** The page-wide pool, one worker per logical core but one. */
export function defaultPool(): GarchWebPool {
  shared ??= new GarchWebPool();
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

/** `volatilityForecast` across the pool — the in-process answer. */
export function volatilityForecastInWorkers(
  request: VolatilityForecastRequest,
  options: RunOptions = {},
): Promise<VolatilityForecast> {
  return volatilityForecastOnPool(options.pool ?? defaultPool(), request, options.timeoutMs);
}

/** `fitsAt` across the pool — the in-process answer. */
export function fitsAtInWorkers(request: JobRequest<'fitsAt'>, options: RunOptions = {}): Promise<(FitAt | null)[]> {
  return fitsAtOnPool(options.pool ?? defaultPool(), request, options.timeoutMs);
}

/** `weightedForecast` across the pool — the in-process answer. */
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
