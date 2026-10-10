// After both tsc builds: mark each half's module format for Node, and give the
// ES module half its own worker locations (src/worker-file.ts and
// src/web-worker-file.ts say why).
import { copyFileSync, rmSync, writeFileSync } from 'node:fs';

writeFileSync('dist/cjs/package.json', `${JSON.stringify({ type: 'commonjs' })}\n`);
writeFileSync('dist/esm/package.json', `${JSON.stringify({ type: 'module' })}\n`);
for (const name of ['worker-file', 'web-worker-file']) {
  copyFileSync(`build/${name}.esm.js`, `dist/esm/${name}.js`);
  rmSync(`dist/esm/${name}.js.map`); // the CommonJS source's map, not this file's
}
