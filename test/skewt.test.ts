import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { GarchInputError } from '../src/fit.js';
import {
  backcast,
  fitGarch,
  simulatedTailRisk,
  simulate as simulatePaths,
  skewDownShare,
  skewPartialMoments,
  skewQuantile,
  tailRisk,
} from '../src/index.js';
import { type GarchParams, logLikelihood, logLikelihoodGradient } from '../src/model.js';
import { fixture, rel, returnsOf, simulate } from './helpers.js';

describe('skewed-t innovations', () => {
  it('is SkewStudent of arch: the same quantiles and partial moments', () => {
    for (const d of fixture.skewt_distribution) {
      for (const [p, q] of Object.entries(d.quantiles))
        assert.ok(Math.abs(skewQuantile(Number(p), d.nu, d.lambda) - q) < 1e-12, `q(${p}; ${d.nu}, ${d.lambda})`);
      for (const m of d.partial_moments)
        assert.ok(
          Math.abs(skewPartialMoments(m.x, d.nu, d.lambda)[m.k] - m.value) < 1e-12,
          `E[z^${m.k}; z < ${m.x}] at (${d.nu}, ${d.lambda})`,
        );
      const [mass, mean, variance] = skewPartialMoments(Infinity, d.nu, d.lambda);
      assert.ok(Math.abs(mass - 1) < 1e-12 && Math.abs(mean) < 1e-12 && Math.abs(variance - 1) < 1e-12);
    }
    assert.equal(skewDownShare(7, 0), 0.5);
  });

  it('has the analytic gradient its finite difference says, ν and λ included', () => {
    const r = simulate({ mu: 0.05, omega: 0.1, alpha: 0.08, gamma: 0.1, beta: 0.85, nu: 6, lambda: -0.3 }, 800, 29);
    const p: GarchParams = { mu: 0.04, omega: 0.12, alpha: 0.07, gamma: 0.09, beta: 0.84, nu: 7, lambda: -0.2 };
    const bc = backcast(r, r.reduce((a, b) => a + b, 0) / r.length);
    const { gradient } = logLikelihoodGradient(r, p, bc);
    const sig = new Float64Array(r.length);
    (['mu', 'omega', 'alpha', 'gamma', 'beta', 'nu', 'lambda'] as const).forEach((key, i) => {
      const h = 1e-6;
      const at = (d: number) => logLikelihood(r, { ...p, [key]: (p[key] as number) + d }, bc, sig);
      assert.ok(rel(gradient[i], (at(h) - at(-h)) / (2 * h)) < 1e-5, `∂/∂${key}`);
    });
  });

  it('recovers ν and the skew of a simulated path, with λ significant', () => {
    const truth: GarchParams = { mu: 0, omega: 0.05, alpha: 0.05, gamma: 0.1, beta: 0.85, nu: 7, lambda: -0.3 };
    const fit = fitGarch(simulate(truth, 6000, 30), { model: 'gjr', dist: 'skewt', inference: true });
    assert.equal(fit.converged, true);
    assert.ok(Math.abs(fit.params.lambda! - truth.lambda!) < 0.08, `λ ${fit.params.lambda}`);
    assert.ok(Math.abs(fit.params.nu! - truth.nu!) < 2.5, `ν ${fit.params.nu}`);
    assert.ok(fit.inference!.pValues.lambda! < 1e-4);
    assert.equal(fit.parameterCount, 7);
    // the reported persistence uses the exact down share, not ½
    assert.ok(
      rel(
        fit.persistence,
        fit.params.alpha + skewDownShare(fit.params.nu!, fit.params.lambda!) * fit.params.gamma + fit.params.beta,
      ) < 1e-12,
    );
  });

  for (const model of ['garch', 'gjr'] as const)
    it(`spy: ${model} as arch's dist='skewt' fits it`, () => {
      const a = fixture.series.spy.arch.skewt[model];
      const fit = fitGarch(returnsOf(fixture.series.spy.closes), { model, dist: 'skewt', inference: true });
      assert.equal(fit.converged, true);
      assert.ok(fit.logLikelihood >= a.loglik - 1e-3, `ll ${fit.logLikelihood} vs ${a.loglik}`);
      for (const k of ['mu', 'omega', 'beta', 'nu', 'lambda'] as const)
        assert.ok(rel(fit.params[k]!, a[k]) < 1e-3, `${k} ${fit.params[k]} vs ${a[k]}`);
      assert.ok(Math.abs(fit.params.alpha - a.alpha) < 1e-5, `α ${fit.params.alpha} vs ${a.alpha}`);
      assert.ok(Math.abs(fit.aic - a.aic) < 1e-2 && Math.abs(fit.bic - a.bic) < 1e-2, 'criteria');
      assert.ok(rel(fit.inference!.pValues.lambda!, a.lambda_pvalue) < 0.02, 'p(λ)');
      if (a.gamma_pvalue !== null) assert.ok(rel(fit.inference!.pValues.gamma, a.gamma_pvalue) < 0.02, 'p(γ)');
    });

  it('gives a skewed VaR and ES that its own paths agree with, and EGARCH refuses it', () => {
    const params: GarchParams = { mu: 0.02, omega: 0.05, alpha: 0.05, gamma: 0.1, beta: 0.85, nu: 6, lambda: -0.3 };
    const sim = simulatePaths({
      model: 'gjr',
      params,
      lastVariance: 2,
      lastResidual: -1.5,
      steps: 1,
      paths: 200_000,
      seed: 31,
    });
    const exact = tailRisk({ mean: params.mu, variance: sim.variance[0][0], nu: 6, lambda: -0.3, level: 0.01 });
    const read = simulatedTailRisk(sim, 0.01);
    assert.ok(rel(read.valueAtRisk, exact.valueAtRisk) < 0.02, `VaR ${read.valueAtRisk} vs ${exact.valueAtRisk}`);
    assert.ok(rel(read.expectedShortfall, exact.expectedShortfall) < 0.03, `ES ${read.expectedShortfall}`);
    const symmetric = tailRisk({ mean: params.mu, variance: sim.variance[0][0], nu: 6, level: 0.01 });
    assert.ok(exact.valueAtRisk > symmetric.valueAtRisk, 'a left skew deepens the loss tail');
    assert.throws(
      () => fitGarch(returnsOf(fixture.series.spy.closes), { model: 'egarch', dist: 'skewt' }),
      GarchInputError,
    );
  });
});
