// Fit GJR-GARCH with Student-t innovations on SPY, forecast 20 days of volatility,
// and say whether the model beat symmetric GARCH walked forward — on the worker pool.
//
//   npm run build && node examples/forecast.mjs
import { volatilityForecastInWorkers } from '@krzysztofsztompka/garch/node';
import { lastDate, returns } from './data.mjs';

const horizon = 20;
const answer = await volatilityForecastInWorkers({ returns, model: 'gjr', dist: 't', horizon }, { timeoutMs: 300_000 });

const { params, forecast, validation } = answer;
console.log(`SPY, ${returns.length} daily returns to ${lastDate}: gjr, Student-t`);
console.log(`  α ${params.alpha.toFixed(3)}  γ ${params.gamma.toFixed(3)} (p ${answer.gammaPValue?.toPrecision(2)})`);
console.log(`  β ${params.beta.toFixed(3)}  ν ${params.nu?.toFixed(1)}  persistence ${answer.persistence.toFixed(4)}`);
if (forecast) {
  // returns are ×100 log returns: √s is a daily volatility in percent
  console.log(`  σ̂ tomorrow ${Math.sqrt(forecast[0]).toFixed(2)} %`);
  console.log(`  σ̂ over ${horizon} days ${Math.sqrt(forecast.reduce((a, s) => a + s, 0)).toFixed(2)} %`);
} else console.log('  the fit did not converge: no forecast');
const dm = validation.dieboldMariano;
console.log(
  `walked forward over ${validation.origins} origins: QLIKE ${validation.loss?.toFixed(4)} vs symmetric GARCH ${validation.baselineLoss?.toFixed(4)}`,
);
console.log(
  `  ${validation.beatsBaseline ? 'beats' : 'does not beat'} its baseline` +
    (dm ? `; Diebold–Mariano p ${dm.pValue.toPrecision(2)}` : ''),
);
