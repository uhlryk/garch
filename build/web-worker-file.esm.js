/** Starts the Web Worker for `./web` — the ES module build's (see src/web-worker-file.ts). */
export function createWebWorker() {
  return new Worker(new URL('./worker-web.js', import.meta.url), { type: 'module' });
}
