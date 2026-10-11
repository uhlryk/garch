/**
 * The pool, whatever runs its workers: one queue of tasks, a deadline and a
 * failure shared by the tasks of one call, and the calls that cut a
 * walk-forward into segments across it. `./node` gives it worker threads,
 * `./web` Web Workers; the answer is the in-process one on either.
 */
import type { AsOfFit, FitAt, VolatilityForecast, VolatilityForecastRequest } from './jobs.js';
import { baselineOf } from './jobs.js';
import type { JobName, JobRequest, JobResult } from './registry.js';
import { scoreWalkForward, segments, validationWindows, type WalkForwardStep } from './validation.js';
import { assembleWeighted, WEIGHTED_MODELS, type WeightedForecast, type WeightedForecastRequest } from './weighted.js';
import type { Answer } from './worker-core.js';

/** The job was refused on its input (too little history, a bad number) — the caller's to word. */
export class GarchJobInputError extends Error {}
/** The job did not finish within its deadline; its workers were terminated. */
export class GarchJobTimeoutError extends Error {}
/** The worker failed for any other reason. */
export class GarchJobFailedError extends Error {}

export type RunOptions = {
  readonly timeoutMs?: number;
  /** The pool to run on; the environment's shared default when absent. */
  readonly pool?: WorkerPool;
};

/** One job of a call; `weight` is its relative cost. */
export type PoolTask = { readonly job: JobName; readonly request: unknown; readonly weight?: number };

/** One worker as the pool drives it, whatever its runtime. */
export type PoolWorker = {
  post(message: { readonly job: JobName; readonly request: unknown }): void;
  terminate(): unknown;
  /** Busy: keep the process alive (worker threads); idle: let it exit. Optional. */
  hold?(): void;
  release?(): void;
};

/** What a worker reports back: an answer, or that it was lost (crashed, exited) with a reason. */
export type WorkerEvents = {
  readonly answer: (answer: Answer) => void;
  readonly lost: (reason: string) => void;
};

type Task = {
  readonly job: JobName;
  readonly request: unknown;
  readonly batch: Batch;
  readonly resolve: (value: unknown) => void;
};

/** The jobs still running or queued, by name: `walkForwardForecasts (3 of 40 tasks), asOfFit`. */
function unfinished(tasks: readonly PoolTask[], done: readonly boolean[]): string {
  const counts = new Map<JobName, { left: number; all: number }>();
  tasks.forEach((t, i) => {
    const c = counts.get(t.job) ?? { left: 0, all: 0 };
    c.all += 1;
    if (!done[i]) c.left += 1;
    counts.set(t.job, c);
  });
  return [...counts]
    .filter(([, c]) => c.left)
    .map(([job, c]) => (c.all === 1 ? job : `${job} (${c.left} of ${c.all} tasks)`))
    .join(', ');
}

/** The tasks of one call: they share a deadline and fail together. */
class Batch {
  failed: Error | null = null;
  constructor(readonly reject: (error: Error) => void) {}
}

type Slot = { worker: PoolWorker; task: Task | null };

/**
 * Up to `size` workers, started on demand by `create` and reused; tasks wait
 * in one queue, first come first served.
 */
export class WorkerPool {
  private readonly slots: Slot[] = [];
  private readonly queue: Task[] = [];

  constructor(
    readonly size: number,
    private readonly create: (events: WorkerEvents) => PoolWorker,
  ) {
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
      const done = new Array<boolean>(tasks.length).fill(false);
      let left = tasks.length;
      let timer: ReturnType<typeof setTimeout> | undefined;
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
          () =>
            batch.reject(new GarchJobTimeoutError(`${unfinished(tasks, done)} did not finish within ${timeoutMs} ms`)),
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
            done[i] = true;
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
      slot.worker.hold?.();
      slot.worker.post({ job: task.job, request: task.request });
    }
  }

  private spawn(): Slot {
    const slot = { task: null } as Slot;
    slot.worker = this.create({
      answer: (answer) => {
        const task = slot.task;
        slot.task = null;
        slot.worker.release?.();
        if (task && !task.batch.failed) {
          if (answer.ok) task.resolve(answer.result);
          else if (answer.input) task.batch.reject(new GarchJobInputError(answer.message));
          else task.batch.reject(new GarchJobFailedError(answer.message));
        }
        this.pump();
      },
      lost: (reason) => {
        const i = this.slots.indexOf(slot);
        if (i < 0) return; // terminated on purpose
        this.slots.splice(i, 1);
        slot.task?.batch.reject(new GarchJobFailedError(reason));
        slot.task = null;
        this.pump();
      },
    });
    slot.worker.release?.();
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

/** One job on one worker of the pool. */
export async function runOnPool<J extends JobName>(
  pool: WorkerPool,
  job: J,
  request: JobRequest<J>,
  timeoutMs?: number,
): Promise<JobResult<J>> {
  const [result] = await pool.run<JobResult<J>>([{ job, request }], timeoutMs);
  return result;
}

/**
 * `volatilityForecast` across the pool: the as-of fit, the model's segments and
 * (for gjr) the symmetric baseline's segments all at once, scored here. The
 * same answer as the in-process job.
 */
export async function volatilityForecastOnPool(
  pool: WorkerPool,
  request: VolatilityForecastRequest,
  timeoutMs?: number,
): Promise<VolatilityForecast> {
  const model = request.model ?? 'garch';
  const returns = Float64Array.from(request.returns);
  const windows = validationWindows(returns.length, request);
  const origins = segments(windows.map((w) => w.origin));
  const models = baselineOf(model) === 'garch' ? [model, 'garch' as const] : [model];
  const tasks = [
    {
      job: 'asOfFit' as const,
      request: {
        returns,
        model,
        dist: request.dist,
        mean: request.mean,
        horizon: request.horizon,
        rivalsWithin: request.rivalsWithin,
      },
      weight: Infinity,
    },
    ...models.flatMap((m) =>
      origins.map((o) => ({
        job: 'walkForwardForecasts' as const,
        request: { returns, model: m, dist: request.dist, mean: request.mean, horizon: request.horizon, origins: o },
        weight: cost(o),
      })),
    ),
  ];
  const [fit, ...forecasts] = await pool.run<unknown>(tasks, timeoutMs);
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
export async function fitsAtOnPool(
  pool: WorkerPool,
  request: JobRequest<'fitsAt'>,
  timeoutMs?: number,
): Promise<(FitAt | null)[]> {
  const returns = Float64Array.from(request.returns);
  const tasks = segments(request.ends).map((ends) => ({
    job: 'fitsAtSegment' as const,
    request: { returns, ends, model: request.model, dist: request.dist, mean: request.mean },
    weight: cost(ends) * (request.dist && request.dist !== 'normal' ? 2 : 1),
  }));
  const out = await pool.run<(FitAt | null)[]>(tasks, timeoutMs);
  return out.flat();
}

/**
 * `weightedForecast` across the pool: each model's as-of fit and each model's
 * walk-forward segments at once, combined here — the in-process answer.
 */
export async function weightedForecastOnPool(
  pool: WorkerPool,
  request: WeightedForecastRequest,
  timeoutMs?: number,
): Promise<WeightedForecast> {
  const specs = request.models ?? WEIGHTED_MODELS;
  const returns = Float64Array.from(request.returns);
  const origins = segments(validationWindows(returns.length, request).map((w) => w.origin));
  const tasks = [
    ...specs.map((spec) => ({
      job: 'weightedAsOf' as const,
      request: { returns, spec, horizon: request.horizon, mean: request.mean },
      weight: Infinity,
    })),
    ...specs.flatMap((spec) =>
      origins.map((o) => ({
        job: 'walkForwardSteps' as const,
        request: {
          returns,
          model: spec.model,
          dist: spec.dist,
          mean: request.mean,
          horizon: request.horizon,
          origins: o,
        },
        weight: cost(o) * (spec.dist !== 'normal' ? 2 : 1),
      })),
    ),
  ];
  const out = await pool.run<unknown>(tasks, timeoutMs);
  const asOf = out.slice(0, specs.length) as (AsOfFit | null)[];
  const segmentSteps = out.slice(specs.length) as (WalkForwardStep | null)[][];
  const steps = specs.map((_, k) => segmentSteps.slice(k * origins.length, (k + 1) * origins.length).flat());
  return assembleWeighted(returns, request, specs, asOf, steps);
}
