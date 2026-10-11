import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { egarchLogLikelihood } from '../src/egarch.js';
import { type GarchFit, GarchInputError } from '../src/fit.js';
import { extendFit, fitGarch, forecastVariance, simulate as simulatePaths } from '../src/index.js';
import { type GarchParams, logLikelihood } from '../src/model.js';
import { fixture, rel, returnsOf, simulate } from './helpers.js';

describe('forecastVariance', () => {
  it('decays to the unconditional variance', () => {
    const fit = fitGarch(simulate({ mu: 0, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85 }, 3000, 8));
    const v = forecastVariance(fit, 2000);
    const unconditional = fit.params.omega / (1 - fit.persistence);
    assert.ok(rel(v[v.length - 1], unconditional) < 1e-6);
  });
});

describe('extendFit', () => {
  const r = simulate({ mu: 0.03, omega: 0.05, alpha: 0.05, gamma: 0.1, beta: 0.85 }, 1500, 31);
  const head = r.slice(0, 1200);
  const tail = r.slice(1200);
  const specs = [
    { model: 'garch' },
    { model: 'gjr' },
    { model: 'gjr', dist: 't' },
    { model: 'garch', dist: 'skewt' },
    { model: 'gjr', mean: 'zero' },
    { model: 'egarch' },
    { model: 'egarch', mean: 'zero' },
  ] as const;

  it('runs the recursion on at the same parameters: logLikelihood over the whole series, garch and gjr to the bit', () => {
    for (const spec of specs) {
      const fit = fitGarch(head, spec);
      const extended = extendFit(fit, tail);
      const label = `${spec.model} ${'dist' in spec ? spec.dist : 'normal'} ${'mean' in spec ? spec.mean : ''}`;
      const variance = new Float64Array(r.length);
      const egarch = spec.model === 'egarch';
      const ll = (egarch ? egarchLogLikelihood : logLikelihood)(r, fit.params, fit.backcast, variance);
      assert.equal(extended.extendedBy, tail.length, label);
      assert.deepEqual(extended.params, fit.params);
      assert.equal(extended.lastResidual, r[r.length - 1] - fit.params.mu, label);
      assert.deepEqual(extended.conditionalVariance.subarray(0, head.length), fit.conditionalVariance, label);
      if (egarch) {
        assert.ok(rel(extended.logLikelihood, ll) < 1e-13, label);
        for (let t = 0; t < r.length; t += 1)
          assert.ok(rel(extended.conditionalVariance[t], variance[t]) < 1e-13, `${label}: s_${t}`);
      } else {
        assert.equal(extended.logLikelihood, ll, label);
        assert.deepEqual(extended.conditionalVariance, variance, label);
      }
      const k = fit.parameterCount;
      assert.equal(extended.parameterCount, k);
      assert.equal(extended.bic, -2 * extended.logLikelihood + k * Math.log(r.length));
    }
  });

  it('forecasts from the new last state, and extends in steps as at once', () => {
    const fit = fitGarch(head, { model: 'gjr', dist: 't' });
    const once = extendFit(fit, tail);
    const twice = extendFit(extendFit(fit, tail.slice(0, 100)), tail.slice(100));
    assert.deepEqual(twice, once);
    const s = once.conditionalVariance[r.length - 1];
    const e = once.lastResidual;
    const p = fit.params;
    assert.equal(forecastVariance(once, 1)[0], p.omega + p.alpha * e * e + p.gamma * (e < 0 ? e * e : 0) + p.beta * s);
    const sim = simulatePaths({
      model: 'gjr',
      params: p,
      lastVariance: s,
      lastResidual: e,
      steps: 1,
      paths: 1,
      seed: 1,
    });
    assert.equal(sim.variance[0][0], forecastVariance(once, 1)[0]);
    assert.equal(extendFit(fit, []), fit);
  });

  it("spy: reaches the variance and one-step forecast of arch's fit held fixed over the whole series", () => {
    const a = fixture.series.spy.arch.extend;
    const all = returnsOf(fixture.series.spy.closes);
    const before = all.slice(0, -a.added);
    const params: GarchParams = { mu: a.mu, omega: a.omega, alpha: a.alpha, gamma: a.gamma, beta: a.beta };
    // arch's estimate on the history before, as a fit: its parameters and their variance path there
    const fit = fitGarch(before, { model: 'gjr' });
    const variance = new Float64Array(before.length);
    const ll = logLikelihood(before, params, fit.backcast, variance);
    const archFit = {
      ...fit,
      params,
      conditionalVariance: variance,
      logLikelihood: ll,
      lastResidual: before.at(-1)! - a.mu,
    };
    const extended = extendFit(archFit, all.slice(-a.added));
    assert.ok(rel(extended.conditionalVariance.at(-1)!, a.last_variance) < 1e-10);
    assert.ok(rel(forecastVariance(extended, 1)[0], a.forecast_1) < 1e-10);
    // arch backcasts at the fixed μ, not the sample mean: only the first few terms differ
    assert.ok(Math.abs(extended.logLikelihood - a.loglik) < 1e-3);
  });

  it('refuses a new return that is not a finite number', () => {
    assert.throws(() => extendFit(fitGarch(head), [0.1, NaN]), GarchInputError);
  });
});

describe('simulate', () => {
  const p: GarchParams = { mu: 0.02, omega: 0.05, alpha: 0.05, gamma: 0.1, beta: 0.85 };
  const from = { lastVariance: 2, lastResidual: -1.5 };
  const meanOf = (paths: Float64Array[], k: number) => paths.reduce((a, v) => a + v[k], 0) / paths.length;

  it('gives the same paths for the same seed, and other paths for another', () => {
    const run = (seed: number) => simulatePaths({ model: 'gjr', params: p, steps: 50, paths: 3, seed });
    assert.deepEqual(run(1), run(1));
    assert.notDeepEqual(run(1).returns[0], run(2).returns[0]);
  });

  it('averages to the analytic forecast — gjr normal and t, egarch — from where a fit stands', () => {
    const cases: { model: 'gjr' | 'egarch'; params: GarchParams }[] = [
      { model: 'gjr', params: p },
      { model: 'gjr', params: { ...p, nu: 5 } },
      { model: 'gjr', params: { ...p, nu: 6, lambda: -0.3 } },
      { model: 'egarch', params: { mu: 0, omega: 0.01, alpha: 0.15, gamma: -0.1, beta: 0.97 } },
    ];
    for (const { model, params } of cases) {
      const sim = simulatePaths({ model, params, ...from, steps: 20, paths: 40_000, seed: 3 });
      const fit = { model, params, lastResidual: from.lastResidual, conditionalVariance: Float64Array.of(2) };
      const exact = forecastVariance(fit as unknown as GarchFit, 20);
      assert.ok(rel(sim.variance[0][0], exact[0]) < 1e-12, `${model}: step 1 is the one-step forecast`);
      for (const k of [1, 4, 19]) {
        assert.ok(rel(meanOf(sim.variance, k), exact[k]) < 0.02, `${model} ν=${params.nu}: step ${k + 1}`);
        const e2 = sim.returns.reduce((a, r) => a + (r[k] - params.mu) ** 2, 0) / sim.returns.length;
        assert.ok(rel(e2, meanOf(sim.variance, k)) < 0.04, `${model} ν=${params.nu}: unit-variance innovations`);
      }
    }
  });

  it('starts at the unconditional variance without a fit to start from', () => {
    const sim = simulatePaths({ model: 'garch', params: p, steps: 1, paths: 1, seed: 1 });
    assert.ok(rel(sim.variance[0][0], p.omega / (1 - p.alpha - p.beta)) < 1e-12, 'garch ignores γ');
  });

  it('refuses what it cannot simulate', () => {
    const base = { model: 'gjr' as const, params: p, steps: 10, paths: 1, seed: 1 };
    assert.throws(() => simulatePaths({ ...base, steps: 0 }), RangeError);
    assert.throws(() => simulatePaths({ ...base, paths: 1.5 }), RangeError);
    assert.throws(() => simulatePaths({ ...base, lastVariance: 1 }), RangeError);
    assert.throws(() => simulatePaths({ ...base, params: { ...p, nu: 2 } }), RangeError);
    assert.throws(() => simulatePaths({ ...base, params: { ...p, beta: 0.95 } }), RangeError);
    assert.throws(() => simulatePaths({ ...base, model: 'egarch', params: { ...p, nu: 8 } }), GarchInputError);
  });
});
