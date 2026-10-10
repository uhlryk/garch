/**
 * Node-only entry (`@smi/garch/node`): jobs on worker threads, so a
 * walk-forward that takes minutes neither holds the caller's event loop nor
 * runs on one core. A pool of reused workers takes the segments a walk-forward
 * is cut into (`SEGMENT_ORIGINS`) — the answer is the same as in-process, on any
 * number of cores.
 */
import { availableParallelism } from 'node:os';
import { join } from 'node:path';
import { Worker } from 'node:worker_threads';
import type { AsOfFit, FitAt, VolatilityForecast, VolatilityForecastRequest } from './jobs';
import { baselineOf } from './jobs';
import type { JobName, JobRequest, JobResult } from './registry';
import {
  assembleWeighted,
  type WeightedForecast,
  type WeightedForecastRequest,
  WEIGHTED_MODELS,
} from './weighted';
import { segments, scoreWalkForward, validationWindows, type WalkForwardStep } from './validation';

/** The job was refused on its input (too little history, a bad number) — the caller's to word. */
export class GarchJobInputError extends Error {}
/** The job did not finish within its deadline; its workers were terminated. */
export class GarchJobTimeoutError extends Error {}
/** The worker failed for any other reason. */
export class GarchJobFailedError extends Error {}

export type RunOptions = {
  readonly timeoutMs?: number;
  /** The pool to run on; the shared default when absent. */
  readonly pool?: GarchPool;
};

/** One job of a call; `weight` is its relative cost. */
export type PoolTask = { readonly job: JobName; readonly request: unknown; readonly weight?: number };

type Answer = { ok: boolean; result?: unknown; input?: boolean; message?: string };

type Task = {
  readonly job: JobName;
  readonly request: unknown;
  readonly batch: Batch;
  readonly resolve: (value: unknown) => void;
};

/** The tasks of one call: they share a deadline and fail together. */
class Batch {
  failed: Error | null = null;
  constructor(readonly reject: (error: Error) => void) {}
}

type Slot = { worker: Worker; task: Task | null };

/**
 * Up to `size` worker threads, started on demand and reused; tasks wait in
 * one queue, first come first served. An idle worker does not keep the
 * process alive.
 */
export class GarchPool {
  private readonly slots: Slot[] = [];
  private readonly queue: Task[] = [];

  constructor(readonly size = Math.max(1, availableParallelism() - 1)) {
    if (!(Number.isInteger(size) && size >= 1)) throw new RangeError(`a pool needs ≥ 1 worker, got ${size}`);
  }

  /**
   * Runs the tasks of one call under one deadline; resolves, in the order
   * given, when all have answered. The heaviest `weight` is dispatched first,
   * so the longest task does not start last.
   */
  run<T>(tasks: readonly PoolTask[], timeoutMs?: number): Promise<T[]> {
    return new Promise<T[]>((resolve, reject) => {
      const results = new Array<unknown>(tasks.length);
      let left = tasks.length;
      let timer: NodeJS.Timeout | undefined;
      const batch = new Batch((error) => {
        if (batch.failed) return;
        batch.failed = error;
        clearTimeout(timer);
        this.cancel(batch);
        reject(error);
      });
      if (!left) return resolve([]);
      if (timeoutMs)
        timer = setTimeout(
          () => batch.reject(new GarchJobTimeoutError(`${tasks[0].job} did not finish within ${timeoutMs} ms`)),
          timeoutMs,
        );
      const order = tasks.map((_, i) => i).sort((a, b) => (tasks[b].weight ?? 0) - (tasks[a].weight ?? 0));
      for (const i of order) {
        const t = tasks[i];
        this.queue.push({
          job: t.job,
          request: t.request,
          batch,
          resolve: (value) => {
            results[i] = value;
            left -= 1;
            if (!left && !batch.failed) {
              clearTimeout(timer);
              resolve(results as T[]);
            }
          },
        });
      }
      this.pump();
    });
  }

  /** Terminates every worker; queued tasks fail. */
  async close(): Promise<void> {
    for (const task of this.queue.splice(0)) task.batch.reject(new GarchJobFailedError('the pool was closed'));
    const slots = this.slots.splice(0);
    await Promise.all(slots.map((s) => s.worker.terminate()));
  }

  private pump(): void {
    while (this.queue.length) {
      let slot = this.slots.find((s) => !s.task);
      if (!slot && this.slots.length < this.size) slot = this.spawn();
      if (!slot) return;
      const task = this.queue.shift()!;
      slot.task = task;
      slot.worker.ref();
      slot.worker.postMessage({ job: task.job, request: task.request });
    }
  }

  private spawn(): Slot {
    const slot: Slot = { worker: new Worker(join(__dirname, 'worker.js')), task: null };
    slot.worker.unref();
    slot.worker.on('message', (answer: Answer) => {
      const task = slot.task;
      slot.task = null;
      slot.worker.unref();
      if (task && !task.batch.failed) {
        if (answer.ok) task.resolve(answer.result);
        else if (answer.input) task.batch.reject(new GarchJobInputError(answer.message));
        else task.batch.reject(new GarchJobFailedError(answer.message));
      }
      this.pump();
    });
    const lost = (message: string) => {
      const i = this.slots.indexOf(slot);
      if (i < 0) return; // terminated on purpose
      this.slots.splice(i, 1);
      slot.task?.batch.reject(new GarchJobFailedError(message));
      slot.task = null;
      this.pump();
    };
    slot.worker.on('error', (error) => lost(error.message));
    slot.worker.on('exit', (code) => lost(`the worker exited with code ${code} before answering`));
    this.slots.push(slot);
    return slot;
  }

  /** Drops a failed call's queued tasks and terminates the workers still running its tasks. */
  private cancel(batch: Batch): void {
    for (let i = this.queue.length - 1; i >= 0; i -= 1) if (this.queue[i].batch === batch) this.queue.splice(i, 1);
    for (const slot of this.slots.filter((s) => s.task?.batch === batch)) {
      this.slots.splice(this.slots.indexOf(slot), 1);
      void slot.worker.terminate();
    }
    this.pump();
  }
}

let shared: GarchPool | null = null;
/** The process-wide pool, one worker per core but one. */
export function defaultPool(): GarchPool {
  shared ??= new GarchPool();
  return shared;
}

/** One job on one worker of the pool. */
export async function runInWorker<J extends JobName>(
  job: J,
  request: JobRequest<J>,
  options: RunOptions = {},
): Promise<JobResult<J>> {
  const [result] = await (options.pool ?? defaultPool()).run<JobResult<J>>([{ job, request }], options.timeoutMs);
  return result;
}

/**
 * `volatilityForecast` across the pool: the as-of fit, the model's segments and
 * (for gjr) the symmetric baseline's segments all at once, scored here. The
 * same answer as the in-process job.
 */
export async function volatilityForecastInWorkers(
  request: VolatilityForecastRequest,
  options: RunOptions = {},
): Promise<VolatilityForecast> {
  const model = request.model ?? 'garch';
  const returns = Float64Array.from(request.returns);
  const windows = validationWindows(returns.length, request);
  const origins = segments(windows.map((w) => w.origin));
  const models = baselineOf(model) === 'garch' ? [model, 'garch' as const] : [model];
  const tasks = [
    {
      job: 'asOfFit' as const,
      request: { returns, model, dist: request.dist, horizon: request.horizon, rivalsWithin: request.rivalsWithin },
      weight: Infinity,
    },
    ...models.flatMap((m) =>
      origins.map((o) => ({
        job: 'walkForwardForecasts' as const,
        request: { returns, model: m, dist: request.dist, horizon: request.horizon, origins: o },
        weight: cost(o),
      })),
    ),
  ];
  const [fit, ...forecasts] = await (options.pool ?? defaultPool()).run<unknown>(tasks, options.timeoutMs);
  const series = models.map((_, k) =>
    (forecasts.slice(k * origins.length, (k + 1) * origins.length) as (number | null)[][]).flat(),
  );
  return {
    ...(fit as AsOfFit),
    validation: scoreWalkForward(returns, windows, request.horizon, series[0], series[1] ?? null),
  };
}

/** A segment's cost: a fit's work grows with its train window. */
const cost = (ends: readonly number[]) => ends.reduce((a, e) => a + e + 1, 0);

/** `fitsAt` across the pool, segment by segment — the same answer as in-process. */
export async function fitsAtInWorkers(
  request: JobRequest<'fitsAt'>,
  options: RunOptions = {},
): Promise<(FitAt | null)[]> {
  const returns = Float64Array.from(request.returns);
  const tasks = segments(request.ends).map((ends) => ({
    job: 'fitsAtSegment' as const,
    request: { returns, ends, model: request.model, dist: request.dist },
    weight: cost(ends) * (request.dist === 't' ? 2 : 1),
  }));
  const out = await (options.pool ?? defaultPool()).run<(FitAt | null)[]>(tasks, options.timeoutMs);
  return out.flat();
}

/**
 * `weightedForecast` across the pool: each model's as-of fit and each model's
 * walk-forward segments at once, combined here — the in-process answer.
 */
export async function weightedForecastInWorkers(
  request: WeightedForecastRequest,
  options: RunOptions = {},
): Promise<WeightedForecast> {
  const specs = request.models ?? WEIGHTED_MODELS;
  const returns = Float64Array.from(request.returns);
  const origins = segments(validationWindows(returns.length, request).map((w) => w.origin));
  const tasks = [
    ...specs.map((spec) => ({
      job: 'weightedAsOf' as const,
      request: { returns, spec, horizon: request.horizon },
      weight: Infinity,
    })),
    ...specs.flatMap((spec) =>
      origins.map((o) => ({
        job: 'walkForwardSteps' as const,
        request: { returns, model: spec.model, dist: spec.dist, horizon: request.horizon, origins: o },
        weight: cost(o) * (spec.dist === 't' ? 2 : 1),
      })),
    ),
  ];
  const out = await (options.pool ?? defaultPool()).run<unknown>(tasks, options.timeoutMs);
  const asOf = out.slice(0, specs.length) as (AsOfFit | null)[];
  const segmentSteps = out.slice(specs.length) as (WalkForwardStep | null)[][];
  const steps = specs.map((_, k) => segmentSteps.slice(k * origins.length, (k + 1) * origins.length).flat());
  return assembleWeighted(returns, request, specs, asOf, steps);
}
