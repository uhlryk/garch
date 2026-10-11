import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { GarchInputError } from '../src/fit.js';
import { asOfFit, backcast, fitGarch, quantizedUntil, T_NU_MIN } from '../src/index.js';
import { type GarchParams, logLikelihood, logLikelihoodGradient } from '../src/model.js';
import { fixture, rel, returnsOf, simulate } from './helpers.js';

describe('the likelihood', () => {
  it('has the analytic gradient its finite difference says', () => {
    const r = simulate({ mu: 0.05, omega: 0.1, alpha: 0.08, gamma: 0.1, beta: 0.85 }, 800, 7);
    const p: GarchParams = { mu: 0.03, omega: 0.12, alpha: 0.07, gamma: 0.05, beta: 0.86 };
    const bc = backcast(r.map((v) => v - r.reduce((a, b) => a + b, 0) / r.length));
    const { gradient } = logLikelihoodGradient(r, p, bc);
    const sig = new Float64Array(r.length);
    (['mu', 'omega', 'alpha', 'gamma', 'beta'] as const).forEach((key, i) => {
      const h = 1e-6;
      const up = logLikelihood(r, { ...p, [key]: p[key] + h }, bc, sig);
      const down = logLikelihood(r, { ...p, [key]: p[key] - h }, bc, sig);
      assert.ok(rel(gradient[i], (up - down) / (2 * h)) < 1e-5, `∂/∂${key}`);
    });
  });
});

describe('fitGarch', () => {
  it('recovers a simulated GARCH(1,1)', () => {
    const truth = { mu: 0.02, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85 };
    const fit = fitGarch(simulate(truth, 6000, 1));
    assert.equal(fit.converged, true);
    assert.ok(Math.abs(fit.params.alpha - truth.alpha) < 0.03, `α ${fit.params.alpha}`);
    assert.ok(Math.abs(fit.params.beta - truth.beta) < 0.04, `β ${fit.params.beta}`);
    assert.equal(fit.params.gamma, 0);
  });

  it('finds a real leverage term significant and an absent one not', () => {
    const lev = fitGarch(simulate({ mu: 0, omega: 0.05, alpha: 0.03, gamma: 0.15, beta: 0.85 }, 6000, 2), {
      model: 'gjr',
      inference: true,
    });
    assert.ok(lev.params.gamma > 0.05 && lev.inference!.pValues.gamma < 0.01);
    const sym = fitGarch(simulate({ mu: 0, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85 }, 6000, 3), {
      model: 'gjr',
      inference: true,
    });
    assert.ok(sym.inference!.pValues.gamma > 0.05, `p ${sym.inference!.pValues.gamma}`);
  });

  it('keeps every fit a valid, stationary model', () => {
    for (const seed of [4, 5, 6]) {
      const fit = fitGarch(simulate({ mu: 0, omega: 0.2, alpha: 0.3, gamma: 0.2, beta: 0.5 }, 700, seed), {
        model: 'gjr',
      });
      const p = fit.params;
      assert.ok(p.omega > 0 && p.alpha >= 0 && p.beta >= 0 && p.alpha + p.gamma >= 0 && fit.persistence < 1);
    }
  });

  it('targets the variance under every distribution and mean, and refuses it for EGARCH', () => {
    const r = simulate({ mu: 0.02, omega: 0.05, alpha: 0.05, gamma: 0.1, beta: 0.85, nu: 6, lambda: -0.2 }, 2000, 33);
    for (const options of [
      { model: 'gjr', dist: 't' },
      { model: 'gjr', dist: 'skewt' },
      { model: 'garch', mean: 'zero' },
    ] as const) {
      const fit = fitGarch(r, { ...options, targetVariance: true });
      assert.ok(fit.converged, JSON.stringify(options));
      assert.ok(
        rel(fit.params.omega / (1 - fit.persistence), fit.varianceTarget!) < 1e-12,
        'the unconditional variance',
      );
      assert.ok(fit.logLikelihood <= fitGarch(r, options).logLikelihood + 1e-6);
    }
    const zero = fitGarch(r, { mean: 'zero', targetVariance: true });
    assert.ok(rel(zero.varianceTarget!, r.reduce((a, v) => a + v * v, 0) / r.length) < 1e-12, 'about 0');
    assert.equal(fitGarch(r).varianceTarget, null);
    assert.throws(() => fitGarch(r, { model: 'egarch', targetVariance: true }), GarchInputError);
  });

  it('refuses what it cannot fit', () => {
    assert.throws(() => fitGarch([1, 2, 3]), /at least 10 returns/);
    assert.throws(() => fitGarch(new Array(20).fill(0)), /no variance/);
    assert.throws(() => fitGarch([...new Array(20).fill(1), NaN]), /not a finite number/);
  });
});

describe('the modes of the likelihood', () => {
  const r = returnsOf(fixture.series.iova.closes);
  it('lists every optimum the starts reached, best first, the fit being the first', () => {
    const fit = fitGarch(r, { model: 'gjr' });
    assert.ok(fit.modes.length >= 1);
    assert.deepEqual(fit.modes[0].params, fit.params);
    for (let i = 1; i < fit.modes.length; i += 1)
      assert.ok(fit.modes[i].logLikelihood <= fit.modes[i - 1].logLikelihood);
    assert.equal(
      fit.modes.reduce((a, m) => a + m.starts, 0),
      6,
    );
  });
  it('marks the optimum a warm start reached', () => {
    const cold = fitGarch(r);
    const warm = fitGarch(r, { warmStart: cold.params });
    assert.equal(warm.modes[0].fromWarmStart, true);
    assert.equal(
      cold.modes.some((m) => m.fromWarmStart),
      false,
    );
  });
  it('hands back as rivals only the optima within the asked distance, with their own variance path', () => {
    const sim = simulate({ mu: 0, omega: 0.2, alpha: 0.3, gamma: 0.2, beta: 0.5 }, 700, 12);
    const all = asOfFit({ returns: sim, model: 'gjr', horizon: 5, rivalsWithin: Infinity });
    const fit = fitGarch(sim, { model: 'gjr' });
    assert.equal(all.rivals.length, fit.modes.filter((m, i) => i > 0 && m.converged).length);
    for (const rival of all.rivals) {
      assert.equal(rival.conditionalVariance.length, sim.length);
      assert.equal(rival.forecast.length, 5);
    }
    assert.deepEqual(asOfFit({ returns: sim, model: 'gjr', horizon: 5, rivalsWithin: 0 }).rivals, []);
    assert.deepEqual(asOfFit({ returns: sim, model: 'gjr', horizon: 5 }).rivals, []);
  });
});

describe('Student-t innovations', () => {
  const tPath = (nu: number, n: number, seed: number) =>
    simulate({ mu: 0, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85, nu }, n, seed);
  it('has the analytic gradient its finite difference says, ν included', () => {
    const r = simulate({ mu: 0.05, omega: 0.1, alpha: 0.08, gamma: 0.1, beta: 0.85 }, 800, 14);
    const p: GarchParams = { mu: 0.03, omega: 0.12, alpha: 0.07, gamma: 0.05, beta: 0.86, nu: 6.5 };
    const bc = backcast(r, r.reduce((a, b) => a + b, 0) / r.length);
    const { gradient } = logLikelihoodGradient(r, p, bc);
    const sig = new Float64Array(r.length);
    (['mu', 'omega', 'alpha', 'gamma', 'beta', 'nu'] as const).forEach((key, i) => {
      const h = 1e-6;
      const at = (d: number) => logLikelihood(r, { ...p, [key]: (p[key] as number) + d }, bc, sig);
      assert.ok(rel(gradient[i], (at(h) - at(-h)) / (2 * h)) < 1e-5, `∂/∂${key}`);
    });
  });
  it('recovers ν on a simulated t-GARCH path', () => {
    const r = tPath(5, 6000, 15);
    const fit = fitGarch(r, { dist: 't' });
    assert.equal(fit.converged, true);
    assert.equal(fit.nuBound, null);
    assert.ok(Math.abs(fit.params.nu! - 5) < 1, `ν ${fit.params.nu}`);
    assert.ok(fit.logLikelihood > fitGarch(r).logLikelihood);
  });

  const heavy = tPath(3, 5000, 16);

  it('holds ν on its floor when the data want less, and says so — converged', () => {
    const fit = fitGarch(heavy, { dist: 't' });
    assert.equal(fit.nuBound, 'lower');
    assert.equal(fit.params.nu, T_NU_MIN);
    assert.equal(fit.converged, true);
    assert.equal(fit.parameterCount, 4, 'a held ν is not a free parameter');
    const free = fitGarch(heavy, { dist: 't', nuMin: 2.05 });
    assert.equal(free.nuBound, null);
    assert.equal(free.parameterCount, 5);
    assert.ok(free.params.nu! < T_NU_MIN, `ν ${free.params.nu}`);
    assert.throws(() => fitGarch(heavy, { dist: 't', nuMin: 2 }), RangeError);
  });

  it('gives robust standard errors with ν among the parameters, and none for a held ν', () => {
    const r = simulate({ mu: 0, omega: 0.05, alpha: 0.03, gamma: 0.15, beta: 0.85, nu: 8 }, 4000, 17);
    const fit = fitGarch(r, { model: 'gjr', dist: 't', inference: true });
    assert.equal(fit.nuBound, null);
    assert.ok(Math.abs(fit.params.nu! - 8) < 3, `ν ${fit.params.nu}`);
    assert.ok(fit.inference!.standardErrors.nu! > 0, `se(ν) ${fit.inference!.standardErrors.nu}`);
    assert.ok(fit.inference!.pValues.gamma < 0.01);
    const free = fitGarch(heavy, { model: 'gjr', dist: 't', nuMin: 2.05, inference: true });
    assert.ok(free.inference!.standardErrors.nu! > 0, `se(ν) ${free.inference!.standardErrors.nu}`);
    assert.ok(Number.isNaN(free.inference!.pValues.nu!));
    assert.ok(free.inference!.standardErrors.alpha > 0);
    const held = fitGarch(heavy, { model: 'gjr', dist: 't', inference: true });
    assert.equal(held.nuBound, 'lower');
    assert.ok(Number.isNaN(held.inference!.standardErrors.nu!));
    assert.ok(held.inference!.standardErrors.alpha > 0);
  });
});

describe('quantizedUntil', () => {
  it('ends after the last window with too many exact zeros', () => {
    const coarse = Array.from({ length: 300 }, (_, i) => (i % 3 === 0 ? 1.5 : 0)); // 2/3 zeros
    const fine = simulate({ mu: 0, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85 }, 600, 18);
    const r = [...coarse, ...fine];
    const cut = quantizedUntil(r, { window: 100 });
    assert.ok(cut >= 300 && cut <= 400, `cut ${cut}`); // up to one window past the boundary
    assert.equal(quantizedUntil(fine, { window: 100 }), 0);
    assert.equal(quantizedUntil([0, 0], { window: 100 }), 0);
  });
  it('is where a t fit on real quantized prices stops degenerating (IOVA, cents × a 1:100 reverse split)', () => {
    const r = returnsOf(fixture.series.iova.closes);
    const cut = quantizedUntil(r);
    assert.ok(cut > 800 && cut < 1200, `cut ${cut}`);
    const variance = r.reduce((a, b) => a + b * b, 0) / r.length;
    // it says it converged — at ω ≈ 0, the variance run down onto the zeros
    const degenerate = fitGarch(r.slice(0, 800), { dist: 't' });
    assert.ok(degenerate.params.omega < 1e-6 * variance, `ω ${degenerate.params.omega}`);
    const clean = fitGarch(r.slice(cut), { dist: 't' });
    assert.equal(clean.converged, true);
    assert.ok(clean.params.omega > 1e-3 * variance, `ω ${clean.params.omega}`);
  });
});
