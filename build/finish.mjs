// After both tsc builds: mark each half's module format for Node, and give the
// ES module half its own worker path (src/worker-file.ts says why).
import { copyFileSync, rmSync, writeFileSync } from 'node:fs';

writeFileSync('dist/cjs/package.json', `${JSON.stringify({ type: 'commonjs' })}\n`);
writeFileSync('dist/esm/package.json', `${JSON.stringify({ type: 'module' })}\n`);
copyFileSync('build/worker-file.esm.js', 'dist/esm/worker-file.js');
rmSync('dist/esm/worker-file.js.map'); // the CommonJS source's map, not this file's
