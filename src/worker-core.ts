/** A worker's one job per message, whatever runs it: `./worker` (threads) and `./worker-web` (Web Workers). */
import { GarchInputError } from './fit.js';
import { JOBS, type JobName } from './registry.js';

export type Message = { readonly job: JobName; readonly request: unknown };

/** What a worker answers: the job's result, or why it failed — `input` when the input was refused. */
export type Answer = { ok: boolean; result?: unknown; input?: boolean; message?: string };

export function answer(message: Message): Answer {
  try {
    const run = JOBS[message.job] as (request: unknown) => unknown;
    if (!run) throw new GarchInputError(`no job named ${String(message.job)}`);
    return { ok: true, result: run(message.request) };
  } catch (error) {
    return {
      ok: false,
      input: error instanceof GarchInputError || error instanceof RangeError,
      message: error instanceof Error ? error.message : String(error),
    };
  }
}
