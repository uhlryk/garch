import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { GarchInputError } from '../src/fit.js';
import {
  backcast,
  extendFit,
  fitGarch,
  forecastVariance,
  simulate as simulatePaths,
  tailRisk,
  varBacktest,
  volatilityForecast,
} from '../src/index.js';
import { exogenousBackcast, type GarchParams, logLikelihood, logLikelihoodGradient } from '../src/model.js';
import { GarchPool, volatilityForecastInWorkers } from '../src/node.js';
import { fixture, rel, returnsOf } from './helpers.js';

describe('GARCH-X', () => {
  /** x_t: the mean of r² over the last five returns, as the fixtures script computes it. */
  const trailing = (r: number[]) =>
    r.map((_, t) => {
      const w = r.slice(Math.max(0, t - 4), t + 1);
      return w.reduce((a, v) => a + v * v, 0) / w.length;
    });
  // an informative driver: a persistent positive series, and returns whose variance it moves
  const n = 3000;
  const shocks = simulatePaths({
    model: 'garch',
    params: { mu: 0, omega: 1, alpha: 0, gamma: 0, beta: 0 },
    steps: n + 1,
    paths: 1,
    seed: 41,
  }).returns[0];
  const driver = new Array<number>(n + 1);
  let level = 0;
  shocks.forEach((z, t) => {
    level = 0.97 * level + 0.25 * z;
    driver[t] = Math.exp(level);
  });
  const truth: GarchParams = { mu: 0.02, omega: 0.03, alpha: 0.05, gamma: 0.08, beta: 0.8, delta: 0.1 };
  const path = Array.from(
    simulatePaths({ model: 'gjr', params: truth, steps: n, paths: 1, seed: 42, exogenous: driver.slice(0, n) })
      .returns[0],
  );
  // simulate's x_k enters step k + 1's variance; a fit's x_t enters return t + 1's
  const x = driver.slice(1);

  it('has the analytic gradient its finite difference says, δ included', () => {
    const p: GarchParams = { ...truth, delta: 0.07, alpha: 0.06 };
    const bc = backcast(path, path.reduce((a, b) => a + b, 0) / n);
    const exo = { values: x, backcast: exogenousBackcast(x) };
    const { gradient } = logLikelihoodGradient(path, p, bc, undefined, exo);
    const sig = new Float64Array(n);
    (['mu', 'omega', 'alpha', 'gamma', 'beta', 'delta'] as const).forEach((key) => {
      const h = 1e-6;
      const up = logLikelihood(path, { ...p, [key]: p[key]! + h }, bc, sig, undefined, exo);
      const down = logLikelihood(path, { ...p, [key]: p[key]! - h }, bc, sig, undefined, exo);
      const i = key === 'delta' ? 7 : ['mu', 'omega', 'alpha', 'gamma', 'beta'].indexOf(key);
      assert.ok(rel(gradient[i], (up - down) / (2 * h)) < 1e-5, `∂/∂${key}`);
    });
  });

  it('recovers δ on a simulated path, significant, with one parameter more in the count', () => {
    const fit = fitGarch(path, { model: 'gjr', exogenous: x, inference: true });
    assert.ok(fit.converged);
    assert.ok(Math.abs(fit.params.delta! - truth.delta!) < 0.03, `δ ${fit.params.delta}`);
    assert.ok(fit.inference!.pValues.delta! < 1e-3, `p ${fit.inference!.pValues.delta}`);
    assert.equal(fit.parameterCount, 6);
    assert.equal(fit.lastExogenous, x[n - 1]);
    const plain = fitGarch(path, { model: 'gjr' });
    assert.ok(fit.logLikelihood > plain.logLikelihood + 10, 'x carries information');
  });

  it("spy: scipy's optimum of the GARCH-X likelihood, x the trailing five-day mean of r²", () => {
    const r = returnsOf(fixture.series.spy.closes);
    const xs = trailing(r);
    for (const model of ['garch', 'gjr'] as const) {
      const want = fixture.series.spy.arch.garch_x[model];
      const fit = fitGarch(r, { model, exogenous: xs });
      assert.ok(fit.converged, model);
      assert.ok(fit.logLikelihood >= want.loglik - 1e-6, `${model} ll ${fit.logLikelihood} vs ${want.loglik}`);
      for (const k of ['mu', 'omega', 'alpha', 'beta', 'delta'] as const)
        assert.ok(Math.abs(fit.params[k]! - want[k]) < 1e-5, `${model} ${k} ${fit.params[k]} vs ${want[k]}`);
    }
  });

  it('forecasts with x held at its last value or along a given path, and extends with new x', () => {
    const fit = fitGarch(path.slice(0, 2500), { model: 'gjr', exogenous: x.slice(0, 2500) });
    const p = fit.params;
    const held = forecastVariance(fit, 3);
    const along = forecastVariance(fit, 3, [2, 5]);
    const P = p.alpha + p.gamma / 2 + p.beta;
    assert.equal(along[0], held[0]);
    assert.ok(rel(held[1], p.omega + P * held[0] + p.delta! * x[2499]) < 1e-15);
    assert.ok(rel(along[2], p.omega + P * along[1] + p.delta! * 5) < 1e-15);
    const extended = extendFit(fit, path.slice(2500), x.slice(2500));
    const variance = new Float64Array(n);
    const ll = logLikelihood(path, p, fit.backcast, variance, undefined, {
      values: x,
      backcast: exogenousBackcast(x.slice(0, 2500)),
    });
    assert.equal(extended.logLikelihood, ll);
    assert.deepEqual(extended.conditionalVariance, variance);
    assert.equal(extended.lastExogenous, x[n - 1]);
    assert.throws(() => extendFit(fit, path.slice(2500)), GarchInputError);
    const sim = simulatePaths({
      model: 'gjr',
      params: p,
      lastVariance: fit.conditionalVariance[2499],
      lastResidual: fit.lastResidual,
      steps: 3,
      paths: 1,
      seed: 1,
      exogenous: [x[2499], 2, 5],
    });
    assert.ok(rel(sim.variance[0][0], along[0]) < 1e-15);
  });

  it('walks forward with x, the same on the pool, and backtests a VaR with it', async () => {
    const r = path.slice(0, 1000);
    const request = {
      returns: r,
      model: 'garch' as const,
      exogenous: x.slice(0, 1000),
      horizon: 10,
      minTrain: 700,
      step: 20,
    };
    const pool = new GarchPool(2);
    const answer = await volatilityForecastInWorkers(request, { pool });
    await pool.close();
    assert.deepEqual(answer, volatilityForecast(request));
    assert.ok(answer.params.delta! > 0);
    const out = varBacktest(r, { model: 'garch', exogenous: x.slice(0, 1000), level: 0.05, minTrain: 700, step: 20 });
    const f = out.forecasts[0];
    const fit = fitGarch(r.slice(0, f.origin + 1), { exogenous: x.slice(0, f.origin + 1) });
    assert.ok(
      rel(
        f.valueAtRisk,
        tailRisk({ mean: fit.params.mu, variance: forecastVariance(fit, 1)[0], level: 0.05 }).valueAtRisk,
      ) < 1e-9,
    );
  });

  it('refuses what it cannot fit', () => {
    assert.throws(() => fitGarch(path, { exogenous: x.slice(1) }), GarchInputError);
    assert.throws(() => fitGarch(path, { exogenous: x.map((v, i) => (i === 5 ? -1 : v)) }), GarchInputError);
    assert.throws(() => fitGarch(path, { exogenous: x.map(() => 0) }), GarchInputError);
    assert.throws(() => fitGarch(path, { model: 'egarch', exogenous: x }), GarchInputError);
    assert.throws(() => forecastVariance(fitGarch(path), 3, [1, 2]), RangeError);
    assert.throws(() => simulatePaths({ model: 'gjr', params: truth, steps: 3, paths: 1, seed: 1 }), RangeError);
  });

  it('targets the variance with x: ω = V·(1 − P) − δ·mean(x)', () => {
    const fit = fitGarch(path, { model: 'gjr', exogenous: x, targetVariance: true });
    const xMean = x.reduce((a, b) => a + b, 0) / n;
    assert.ok(fit.converged);
    assert.ok(rel(fit.params.omega, fit.varianceTarget! * (1 - fit.persistence) - fit.params.delta! * xMean) < 1e-12);
  });
});
