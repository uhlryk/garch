// A one-step 99 % value at risk on SPY, walked forward day by day, and graded:
// Kupiec's coverage, Christoffersen's independence, and both — parametric and FHS.
//
//   npm run build && node examples/var-backtest.mjs
import { varBacktest } from '@krzysztofsztompka/garch';
import { returns } from './data.mjs';

const level = 0.01;
const history = returns.slice(-1250);
for (const method of ['parametric', 'fhs']) {
  const out = varBacktest(history, { model: 'gjr', dist: 't', level, minTrain: 750, step: 1, method });
  const pct = (p) => p.toPrecision(2);
  console.log(
    `${method}: ${out.violations} violations in ${out.observations} days (${(out.rate * 100).toFixed(2)} %, ${level * 100} % expected)`,
  );
  console.log(
    `  Kupiec p ${pct(out.unconditional.pValue)}, Christoffersen p ${pct(out.independence.pValue)}, joint p ${pct(out.conditional.pValue)}`,
  );
  const last = out.forecasts.at(-1);
  console.log(`  last VaR ${last.valueAtRisk.toFixed(2)} %, return that came ${last.realized.toFixed(2)} %`);
}
