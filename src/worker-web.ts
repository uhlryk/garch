/**
 * The Web Worker's entry: one job per message, answered to the page. Loaded
 * by the pool of `./web` from the built package (ES modules).
 */
import { answer, type Message } from './worker-core.js';

type WorkerScope = {
  addEventListener(type: 'message', listener: (event: { data: Message }) => void): void;
  postMessage(message: unknown): void;
};

const scope = globalThis as unknown as WorkerScope;
scope.addEventListener('message', (event) => scope.postMessage(answer(event.data)));
