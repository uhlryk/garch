import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  archLm,
  fitGarch,
  forecastVariance,
  ljungBox,
  standardizedResiduals,
  type TestResult,
  volatilityForecast,
} from '../src/index.js';
import { fixture, rel, returnsOf } from './helpers.js';

/**
 * A replacement for `arch`: where the likelihood has one
 * mode, this IS `arch` — the same parameters, σ̂, γ p-value and walk-forward
 * loss, as `arch` answered them (scripts/arch-fixtures/generate.py).
 */
describe('against arch', () => {
  for (const name of ['iova', 'spy'] as const) {
    const s = fixture.series[name];
    const r = returnsOf(s.closes);
    it(`${name}: garch fit and σ̂`, () => {
      const fit = fitGarch(r);
      const a = s.arch.garch;
      for (const k of ['omega', 'alpha', 'beta'] as const)
        assert.ok(rel(fit.params[k], a[k]) < 1e-3, `${k} ${fit.params[k]} vs ${a[k]}`);
      const v = forecastVariance(fit, 20);
      assert.ok(rel(Math.sqrt(v[0]) / 100, a.sigma_daily) < 1e-4);
      assert.ok(rel(Math.sqrt(v.reduce((x, y) => x + y, 0)) / 100, a.sigma_horizon) < 1e-4);
    });
    it(`${name}: gjr γ and its robust p-value`, () => {
      const fit = fitGarch(r, { model: 'gjr', inference: true });
      const a = s.arch.gjr;
      assert.ok(rel(fit.params.gamma, a.gamma!) < 2e-3, `γ ${fit.params.gamma} vs ${a.gamma}`);
      assert.ok(
        rel(fit.inference!.pValues.gamma, a.gamma_pvalue!) < 0.02,
        `p ${fit.inference!.pValues.gamma} vs ${a.gamma_pvalue}`,
      );
    });
    it(`${name}: Ljung–Box on z and z², ARCH-LM on z — statsmodels' answers on arch's residuals`, () => {
      const fit = fitGarch(r, { model: 'gjr' });
      const z = standardizedResiduals(fit, r);
      const d = s.arch.gjr.diagnostics;
      const check = (label: string, got: TestResult, want: { statistic: number; p_value: number }) => {
        assert.equal(got.df, d.lags);
        assert.ok(rel(got.statistic, want.statistic) < 1e-3, `${label}: ${got.statistic} vs ${want.statistic}`);
        assert.ok(rel(got.pValue, want.p_value) < 1e-3, `${label} p: ${got.pValue} vs ${want.p_value}`);
      };
      check('Q(z)', ljungBox(z, d.lags), d.ljung_box_z);
      check(
        'Q(z²)',
        ljungBox(
          z.map((v) => v * v),
          d.lags,
        ),
        d.ljung_box_z2,
      );
      check('LM', archLm(z, d.lags), d.arch_lm);
    });
    it(`${name}: a zero mean — garch, gjr and egarch, as arch's mean='Zero'`, () => {
      for (const model of ['garch', 'gjr', 'egarch'] as const) {
        const a = s.arch.zero_mean[model];
        const fit = fitGarch(r, { model, mean: 'zero', inference: true });
        assert.equal(fit.mean, 'zero');
        assert.equal(fit.params.mu, 0);
        assert.equal(fit.parameterCount, model === 'garch' ? 3 : 4);
        assert.ok(fit.logLikelihood >= a.loglik - 1e-3, `${model} ll ${fit.logLikelihood} vs ${a.loglik}`);
        for (const k of ['alpha', 'beta'] as const) assert.ok(rel(fit.params[k], a[k]) < 1e-3, `${model} ${k}`);
        assert.ok(Math.abs(fit.aic - a.aic) < 1e-2 && Math.abs(fit.bic - a.bic) < 1e-2, `${model} criteria`);
        assert.ok(Number.isNaN(fit.inference!.standardErrors.mu), `${model}: μ was not fitted`);
        if (a.gamma_pvalue !== null)
          assert.ok(rel(fit.inference!.pValues.gamma, a.gamma_pvalue) < 0.02, `${model} p(γ)`);
      }
    });
    it(`${name}: variance targeting — the optimum scipy finds with ω tied to the sample variance`, () => {
      const a = s.arch.targeting;
      for (const model of ['garch', 'gjr'] as const) {
        const want = a[model];
        const fit = fitGarch(r, { model, targetVariance: true, inference: true });
        assert.ok(fit.converged, model);
        assert.ok(rel(fit.varianceTarget!, a.variance) < 1e-12);
        assert.equal(fit.params.omega, fit.varianceTarget! * (1 - fit.persistence), `${model}: ω = var(r)·(1 − P)`);
        assert.equal(fit.parameterCount, model === 'gjr' ? 4 : 3);
        assert.ok(fit.logLikelihood >= want.loglik - 1e-6, `${model} ll ${fit.logLikelihood} vs ${want.loglik}`);
        for (const k of ['mu', 'alpha', 'beta', 'omega'] as const)
          assert.ok(rel(fit.params[k], want[k]) < 1e-4, `${model} ${k} ${fit.params[k]} vs ${want[k]}`);
        if (want.gamma !== null) assert.ok(rel(fit.params.gamma, want.gamma) < 1e-4, `${model} γ`);
        assert.ok(Number.isNaN(fit.inference!.standardErrors.omega), `${model}: ω was not estimated`);
        assert.ok(fit.inference!.standardErrors.alpha > 0 && fit.inference!.standardErrors.beta > 0);
        assert.ok(fit.logLikelihood <= fitGarch(r, { model }).logLikelihood + 1e-6, 'a constraint cannot raise it');
      }
    });
    it(`${name}: AIC and BIC, garch and gjr`, () => {
      for (const model of ['garch', 'gjr'] as const) {
        const fit = fitGarch(r, { model });
        const a = s.arch[model];
        assert.equal(fit.parameterCount, model === 'gjr' ? 5 : 4);
        assert.ok(Math.abs(fit.aic - a.aic) < 1e-2, `${model} aic ${fit.aic} vs ${a.aic}`);
        assert.ok(Math.abs(fit.bic - a.bic) < 1e-2, `${model} bic ${fit.bic} vs ${a.bic}`);
      }
    });
  }
  it('iova: the walk-forward QLIKE and its naive baseline', () => {
    const s = fixture.series.iova;
    const out = volatilityForecast({ returns: returnsOf(s.closes), horizon: 20 });
    assert.ok(rel(out.validation.loss!, s.arch.garch.qlike!) < 1e-3);
    assert.ok(rel(out.validation.baselineLoss!, s.arch.garch.naive_qlike!) < 1e-9);
    assert.equal(out.validation.beatsBaseline, true);
    // the same test on arch's per-origin losses: forecasts within 1e-3 move it by less than 1 %
    const dm = out.validation.dieboldMariano!;
    const a = s.arch.garch.diebold_mariano!;
    assert.ok(rel(dm.statistic, a.statistic) < 1e-2, `DM ${dm.statistic} vs ${a.statistic}`);
    assert.ok(rel(dm.pValue, a.p_value) < 5e-2, `p ${dm.pValue} vs ${a.p_value}`);
  });
});
