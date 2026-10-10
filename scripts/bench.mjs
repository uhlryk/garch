// npm run bench [-- --models garch,gjr,egarch] [--weighted] [--workers n]
//
// The timings README quotes: each model's volatilityForecast in this thread and
// on the pool, on SPY's closes from the arch fixture, asserting the two answers
// are the same to the bit. --weighted adds weightedForecast the same way (four
// models, minutes in-process). Runs the built package (dist/cjs).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { availableParallelism, cpus } from 'node:os';
import { parseArgs } from 'node:util';

const require = createRequire(import.meta.url);
const garch = require('../dist/cjs/index.js');
const node = require('../dist/cjs/node.js');

const { values } = parseArgs({
  options: {
    models: { type: 'string', default: 'garch,gjr,egarch' },
    weighted: { type: 'boolean', default: false },
    workers: { type: 'string' },
    horizon: { type: 'string', default: '20' },
  },
});

const fixture = JSON.parse(readFileSync(new URL('../test/fixtures/arch-2026-09-25.json', import.meta.url), 'utf8'));
const closes = fixture.series.spy.closes;
const returns = closes.slice(1).map((c, i) => Math.log(c / closes[i]) * 100);
const horizon = Number(values.horizon);
const pool = new node.GarchPool(values.workers ? Number(values.workers) : undefined);

console.log(`${cpus()[0]?.model.trim()} — ${availableParallelism()} logical cores, pool of ${pool.size} workers`);
console.log(`Node ${process.version}; SPY, ${returns.length} returns, horizon ${horizon}\n`);

const seconds = async (run) => {
  const start = performance.now();
  const answer = await run();
  return { answer, s: (performance.now() - start) / 1000 };
};

const bench = async (label, inProcess, onPool) => {
  const one = await seconds(inProcess);
  const many = await seconds(onPool);
  assert.deepEqual(many.answer, one.answer, `${label}: the pool answered differently`);
  console.log(
    `${label.padEnd(16)} ${one.s.toFixed(1).padStart(7)} s → ${many.s.toFixed(1).padStart(6)} s  ×${(one.s / many.s).toFixed(1)}`,
  );
};

console.log(`${'job'.padEnd(16)} one core →   pool`);
for (const model of values.models.split(',')) {
  const request = { returns, model, horizon };
  await bench(
    model,
    () => garch.volatilityForecast(request),
    () => node.volatilityForecastInWorkers(request, { pool }),
  );
}
if (values.weighted) {
  const request = { returns, horizon };
  await bench(
    'weighted',
    () => garch.weightedForecast(request),
    () => node.weightedForecastInWorkers(request, { pool }),
  );
}
await pool.close();
