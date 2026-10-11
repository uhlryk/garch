// The web pool in a Vite page: GarchWebPool starts its workers with
// new Worker(new URL('./worker-web.js', import.meta.url), { type: 'module' }),
// which Vite bundles without configuration.
//
//   cd examples/browser && npm install && npm run dev
import { GarchWebPool, volatilityForecastInWorkers } from '@krzysztofsztompka/garch/web';
import spy from '../spy.json';

const returns = spy.closes.slice(1).map((close, i) => Math.log(close / spy.closes[i]) * 100);
const pool = new GarchWebPool();
const out = document.getElementById('out');

document.getElementById('run').addEventListener('click', async () => {
  out.textContent = `walking GJR-GARCH forward on ${returns.length} returns, ${pool.size} workers…`;
  const started = performance.now();
  const answer = await volatilityForecastInWorkers(
    { returns, model: 'gjr', horizon: 20 },
    { pool, timeoutMs: 300_000 },
  );
  const v = answer.validation;
  out.textContent = [
    `SPY to ${spy.last_date}, in ${((performance.now() - started) / 1000).toFixed(1)} s`,
    `σ̂ tomorrow ${Math.sqrt(answer.forecast[0]).toFixed(2)} %`,
    `γ ${answer.params.gamma.toFixed(3)} (p ${answer.gammaPValue?.toPrecision(2)})`,
    `QLIKE ${v.loss.toFixed(4)} vs symmetric GARCH ${v.baselineLoss.toFixed(4)}: ${v.beatsBaseline ? 'beats' : 'does not beat'} it`,
  ].join('\n');
});
