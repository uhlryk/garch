/**
 * Where the worker thread's entry is, next to this file. This is the CommonJS
 * build's; the ES module build replaces it with `worker-file.esm.js`
 * (`new URL('./worker.js', import.meta.url)`), since an ES module has no
 * `__dirname` and CommonJS output cannot contain `import.meta`.
 */
import { join } from 'node:path';

export const workerFile: string | URL = join(__dirname, 'worker.js');
