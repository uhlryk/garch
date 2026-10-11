// npm run docs:check — every value an entry point exports has its line in
// README's "API reference" section, so a new feature cannot ship undocumented.
// Runs on the built ES modules (dist/esm).
import { readFileSync } from 'node:fs';

const readme = readFileSync(new URL('../README.md', import.meta.url), 'utf8');
const start = readme.indexOf('\n## API reference\n');
if (start < 0) throw new Error('README has no "## API reference" section');
const end = readme.indexOf('\n## ', start + 1);
const reference = readme.slice(start, end < 0 ? undefined : end);

let missing = 0;
for (const entry of ['index', 'node', 'web']) {
  const exports = Object.keys(await import(`../dist/esm/${entry}.js`)).sort();
  // documented = named in backticks: `name`, `name(…)` or `name` inside a list
  const absent = exports.filter((name) => !new RegExp(`\`${name}\\b`).test(reference));
  for (const name of absent)
    console.error(`README API reference lacks ${entry === 'index' ? '' : `/${entry} `}${name}`);
  missing += absent.length;
  if (!absent.length) console.log(`${entry}: all ${exports.length} exports documented`);
}
if (missing) process.exit(1);
