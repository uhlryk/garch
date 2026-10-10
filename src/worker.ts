/**
 * The worker thread's entry: one job per message, answered on the same port.
 * Loaded by `runInWorker` (./node) from the built package.
 */
import { parentPort } from 'node:worker_threads';
import { JOBS, type JobName } from './registry.js';
import { GarchInputError } from './fit.js';

type Message = { readonly job: JobName; readonly request: unknown };

parentPort?.on('message', (message: Message) => {
  try {
    const run = JOBS[message.job] as (request: unknown) => unknown;
    if (!run) throw new GarchInputError(`no job named ${String(message.job)}`);
    parentPort!.postMessage({ ok: true, result: run(message.request) });
  } catch (error) {
    parentPort!.postMessage({
      ok: false,
      input: error instanceof GarchInputError || error instanceof RangeError,
      message: error instanceof Error ? error.message : String(error),
    });
  }
});
