/**
 * The worker thread's entry: one job per message, answered on the same port.
 * Loaded by the pool of `./node` from the built package.
 */
import { parentPort } from 'node:worker_threads';
import { answer, type Message } from './worker-core.js';

parentPort?.on('message', (message: Message) => parentPort!.postMessage(answer(message)));
