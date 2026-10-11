import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { egarchForecast, egarchLogLikelihood, egarchLogLikelihoodGradient } from '../src/egarch.js';
import { GarchInputError } from '../src/fit.js';
import { backcast, fitGarch, forecastVariance, simulate as simulatePaths, volatilityForecast } from '../src/index.js';
import type { GarchParams } from '../src/model.js';
import { GarchPool, volatilityForecastInWorkers } from '../src/node.js';
import { egarchFixture, fixture, rel, returnsOf } from './helpers.js';

describe('EGARCH', () => {
  const p: GarchParams = { mu: 0.02, omega: 0.01, alpha: 0.15, gamma: -0.12, beta: 0.97 };
  /** An EGARCH path from the unconditional log-variance. */
  const path = (n: number, seed: number) =>
    Array.from(simulatePaths({ model: 'egarch', params: p, steps: n, paths: 1, seed }).returns[0]);

  it('has the analytic gradient its finite difference says', () => {
    const r = path(800, 3);
    const q: GarchParams = { mu: 0.01, omega: 0.02, alpha: 0.1, gamma: -0.05, beta: 0.95 };
    const lnBc = Math.log(backcast(r, r.reduce((a, b) => a + b, 0) / r.length));
    const { gradient } = egarchLogLikelihoodGradient(r, q, lnBc);
    const sig = new Float64Array(r.length);
    (['mu', 'omega', 'alpha', 'gamma', 'beta'] as const).forEach((key, i) => {
      const h = 1e-6;
      const up = egarchLogLikelihood(r, { ...q, [key]: q[key] + h }, lnBc, sig);
      const down = egarchLogLikelihood(r, { ...q, [key]: q[key] - h }, lnBc, sig);
      assert.ok(rel(gradient[i], (up - down) / (2 * h)) < 1e-5, `∂/∂${key}`);
    });
  });

  it('recovers the parameters of a simulated path, the leverage sign included', () => {
    const fit = fitGarch(path(4000, 11), { model: 'egarch', inference: true });
    assert.equal(fit.converged, true);
    assert.ok(Math.abs(fit.params.beta - p.beta) < 0.02, `β ${fit.params.beta}`);
    assert.ok(Math.abs(fit.params.alpha - p.alpha) < 0.05, `α ${fit.params.alpha}`);
    assert.ok(Math.abs(fit.params.gamma - p.gamma) < 0.05, `γ ${fit.params.gamma}`);
    assert.ok(fit.inference!.pValues.gamma < 0.01);
    assert.equal(fit.persistence, fit.params.beta);
  });

  it('forecasts the expected variance exactly — the closed form against its own Monte Carlo', () => {
    const exact = egarchForecast(p, 2.5, -3, 20);
    const sims = 100_000;
    const { variance } = simulatePaths({
      model: 'egarch',
      params: p,
      lastVariance: 2.5,
      lastResidual: -3,
      steps: 20,
      paths: sims,
      seed: 5,
    });
    const mean = exact.map((_, h) => variance.reduce((a, v) => a + v[h], 0) / sims);
    for (const h of [1, 9, 19]) assert.ok(rel(exact[h], mean[h]) < 0.01, `step ${h + 1}: ${exact[h]} vs ${mean[h]}`);
    // the first step is the recursion on the observed shock
    const z0 = -3 / Math.sqrt(2.5);
    assert.ok(
      rel(
        exact[0],
        Math.exp(p.omega + p.alpha * (Math.abs(z0) - Math.sqrt(2 / Math.PI)) + p.gamma * z0 + p.beta * Math.log(2.5)),
      ) < 1e-12,
    );
  });

  it('refuses Student-t innovations: the expected variance two steps ahead is infinite', () => {
    assert.throws(() => fitGarch(path(500, 2), { model: 'egarch', dist: 't' }), GarchInputError);
  });

  describe('against arch 8.0.0', () => {
    for (const name of ['iova', 'spy'] as const) {
      it(`${name}: arch's optimum or a higher one, its γ p-value, and its simulated forecast`, () => {
        const a = egarchFixture.series[name];
        const fit = fitGarch(returnsOf(fixture.series[name].closes), { model: 'egarch', inference: true });
        assert.equal(fit.converged, true);
        assert.ok(fit.logLikelihood >= a.loglik - 1e-3, `ll ${fit.logLikelihood} vs ${a.loglik}`);
        for (const key of ['alpha', 'gamma', 'beta'] as const) assert.ok(rel(fit.params[key], a[key]) < 1e-3, key);
        assert.ok(Math.abs(fit.params.mu - a.mu) < 1e-3 && rel(fit.params.omega, a.omega) < 0.02, 'μ, ω');
        assert.ok(rel(fit.inference!.pValues.gamma, a.gamma_pvalue) < 0.02, 'p(γ)');
        assert.ok(Math.abs(fit.aic - a.aic) < 1e-2 && Math.abs(fit.bic - a.bic) < 1e-2, `aic ${fit.aic} vs ${a.aic}`);
        const forecast = forecastVariance(fit, 20);
        assert.ok(rel(forecast[0], a.variance_1) < 1e-3, 'step 1');
        // arch's 200 000 paths against the closed form: sampling noise only
        for (const h of [9, 19]) assert.ok(rel(forecast[h], a.variance_20_simulated[h]) < 0.005, `step ${h + 1}`);
      });
    }
  });

  it('walks forward against symmetric GARCH, in-process and on the pool alike', async () => {
    const r = returnsOf(fixture.series.spy.closes).slice(-900);
    const request = { returns: r, model: 'egarch' as const, horizon: 20, minTrain: 600, step: 25 };
    const inProcess = volatilityForecast(request);
    assert.ok(inProcess.validation.origins > 0 && inProcess.validation.baselineLoss !== null);
    assert.deepEqual(await volatilityForecastInWorkers(request, { pool: new GarchPool(2) }), inProcess);
  });
});
