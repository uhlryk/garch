import { join } from 'node:path';
import { Worker } from 'node:worker_threads';
import { simulate as simulatePaths } from '../src/index.js';
import type { GarchParams } from '../src/model.js';
import type { WebWorker } from '../src/web.js';

/**
 * What the test files share: the `arch` fixtures, a seeded path, returns from
 * closes, a relative error, and a Web Worker made of a worker thread.
 */

export { default as fixture } from './fixtures/arch-2026-09-25.json';
export { default as egarchFixture } from './fixtures/arch-egarch-2026-10-01.json';

/** A seeded GARCH / GJR path from the unconditional variance. */
export const simulate = (params: GarchParams, n: number, seed: number) =>
  Array.from(simulatePaths({ model: 'gjr', params, steps: n, paths: 1, seed }).returns[0]);

export const returnsOf = (closes: number[]) => closes.slice(1).map((c, i) => Math.log(c / closes[i]) * 100);
export const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-12);

/**
 * A Web Worker as a page has one, made of a worker thread: \`self\` gets
 * addEventListener / postMessage, then the built web worker entry runs.
 */
export function webWorkerShim(): WebWorker {
  const entry = JSON.stringify(join(__dirname, '../src/worker-web.js'));
  const thread = new Worker(
    `const { parentPort } = require('node:worker_threads');
     globalThis.addEventListener = (type, listener) => {
       if (type === 'message') parentPort.on('message', (data) => listener({ data }));
     };
     globalThis.postMessage = (message) => parentPort.postMessage(message);
     require(${entry});`,
    { eval: true },
  );
  return {
    postMessage: (message) => thread.postMessage(message),
    terminate: () => void thread.terminate(),
    addEventListener: (type: string, listener: (event: { data?: unknown; message?: string }) => void) => {
      if (type === 'message') thread.on('message', (data) => listener({ data }));
      if (type === 'error') thread.on('error', (error) => listener({ message: error.message }));
    },
  };
}
