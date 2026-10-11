// Paths forward from a GJR-t fit on SPY: the percentiles of the cumulative
// 10-day return (a fan chart's edges), and the 10-day VaR and ES read off them.
//
//   npm run build && node examples/simulate-fan.mjs
import { fitGarch, simulate, simulatedTailRisk } from '@krzysztofsztompka/garch';
import { returns } from './data.mjs';

const fit = fitGarch(returns, { model: 'gjr', dist: 't' });
const steps = 10;
const sim = simulate({
  model: 'gjr',
  params: fit.params,
  lastVariance: fit.conditionalVariance.at(-1),
  lastResidual: fit.lastResidual,
  steps,
  paths: 20_000,
  seed: 1,
});
const sums = sim.returns.map((path) => path.reduce((a, r) => a + r, 0)).sort((a, b) => a - b);
const at = (q) => sums[Math.min(sums.length - 1, Math.floor(q * sums.length))];
console.log(`SPY, cumulative ${steps}-day return (%), from ${sums.length} paths:`);
for (const q of [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
  console.log(`  ${String(q * 100).padStart(2)} %  ${at(q).toFixed(2)}`);
const risk = simulatedTailRisk(sim, 0.01);
console.log(`99 % ${steps}-day VaR ${risk.valueAtRisk.toFixed(2)} %, ES ${risk.expectedShortfall.toFixed(2)} %`);
