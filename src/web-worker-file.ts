/**
 * Starts the Web Worker for `./web`. This is the CommonJS build's, which
 * cannot hold `import.meta`; the ES module build replaces it with
 * `web-worker-file.esm.js`, where `new Worker(new URL('./worker-web.js',
 * import.meta.url), { type: 'module' })` is written out literally, so Vite and
 * webpack find the worker and bundle it.
 */

/** The part of a DOM `Worker` the pool uses. */
export type WebWorker = {
  postMessage(message: unknown): void;
  terminate(): void;
  addEventListener(
    type: 'message' | 'error' | 'messageerror',
    listener: (event: { readonly data?: unknown; readonly message?: string }) => void,
  ): void;
};

export function createWebWorker(): WebWorker {
  throw new Error('@krzysztofsztompka/garch/web is ES modules only');
}
