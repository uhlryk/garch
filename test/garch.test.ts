import assert from 'node:assert/strict';
import { join } from 'node:path';
import { describe, it } from 'node:test';
import { Worker } from 'node:worker_threads';
import { egarchForecast, egarchLogLikelihood, egarchLogLikelihoodGradient } from '../src/egarch.js';
import { type GarchFit, GarchInputError } from '../src/fit.js';
import {
  archLm,
  asOfFit,
  backcast,
  chiSquareSurvival,
  compareValidations,
  coverageTest,
  dieboldMariano,
  extendFit,
  fitGarch,
  fitsAt,
  forecastVariance,
  ljungBox,
  normalQuantile,
  qlike,
  quantizedUntil,
  realizedVariance,
  simulatedTailRisk,
  simulate as simulatePaths,
  skewDownShare,
  skewPartialMoments,
  skewQuantile,
  standardizedResiduals,
  studentQuantile,
  T_NU_MIN,
  type TestResult,
  tailRisk,
  validationWindows,
  varBacktest,
  volatilityForecast,
  WEIGHTED_MODELS,
  walkForwardOrigins,
  walkForwardValidation,
  weighForecasts,
  weightedForecast,
} from '../src/index.js';
import { type GarchParams, logLikelihood, logLikelihoodGradient } from '../src/model.js';
import {
  fitsAtInWorkers,
  GarchJobInputError,
  GarchJobTimeoutError,
  GarchPool,
  runInWorker,
  volatilityForecastInWorkers,
  weightedForecastInWorkers,
} from '../src/node.js';
import {
  GarchWebPool,
  volatilityForecastInWorkers as volatilityForecastInWebWorkers,
  type WebWorker,
} from '../src/web.js';
import fixture from './fixtures/arch-2026-09-25.json';
import egarchFixture from './fixtures/arch-egarch-2026-10-01.json';

/** A seeded GARCH / GJR path from the unconditional variance. */
const simulate = (params: GarchParams, n: number, seed: number) =>
  Array.from(simulatePaths({ model: 'gjr', params, steps: n, paths: 1, seed }).returns[0]);

const returnsOf = (closes: number[]) => closes.slice(1).map((c, i) => Math.log(c / closes[i]) * 100);
const rel = (a: number, b: number) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-12);

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

  it('refuses what it cannot fit', () => {
    assert.throws(() => fitGarch([1, 2, 3]), /at least 10 returns/);
    assert.throws(() => fitGarch(new Array(20).fill(0)), /no variance/);
    assert.throws(() => fitGarch([...new Array(20).fill(1), NaN]), /not a finite number/);
  });
});

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

describe('residual diagnostics', () => {
  it('refers a statistic to χ² in its upper tail, as scipy does', () => {
    for (const [x, df, p] of [
      [18.307, 10, 0.05000058909139812],
      [3.841, 1, 0.050013683763956804],
      [0.5, 3, 0.9188914116546758],
      [300, 20, 8.082849629775851e-52],
    ])
      assert.ok(rel(chiSquareSurvival(x, df), p) < 1e-12, `χ²(${df}) > ${x}`);
    assert.equal(chiSquareSurvival(0, 5), 1);
  });

  it('finds the ARCH effect in a GARCH path, and none left in its standardised residuals', () => {
    const r = simulate({ mu: 0, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85 }, 4000, 21);
    assert.ok(archLm(r, 10).pValue < 1e-6);
    assert.ok(
      ljungBox(
        r.map((v) => v * v),
        10,
      ).pValue < 1e-6,
    );
    const z = standardizedResiduals(fitGarch(r), r);
    assert.ok(archLm(z, 10).pValue > 0.01);
    assert.ok(
      ljungBox(
        z.map((v) => v * v),
        10,
      ).pValue > 0.01,
    );
    assert.equal(ljungBox(z, 10, 2).df, 8);
  });

  it('refuses lags it cannot test', () => {
    const r = simulate({ mu: 0, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85 }, 50, 22);
    assert.throws(() => ljungBox(r, 0), RangeError);
    assert.throws(() => ljungBox(r, 50), RangeError);
    assert.throws(() => ljungBox(r, 5, 5), RangeError);
    assert.throws(() => archLm(r, 2.5), RangeError);
    assert.throws(() => standardizedResiduals(fitGarch(r), r.slice(1)), RangeError);
  });
});

describe('value at risk', () => {
  it('has the quantiles scipy has', () => {
    const normal: [number, number][] = [
      [0.001, -3.090232306167813],
      [0.01, -2.3263478740408408],
      [0.3, -0.5244005127080409],
    ];
    for (const [p, q] of normal) assert.ok(rel(normalQuantile(p), q) < 2e-9, `Φ⁻¹(${p})`);
    const student: [number, number, number][] = [
      [0.01, 5, -3.364929998907218],
      [0.05, 3, -2.353363434801824],
      [0.001, 4.5, -6.421242884295792],
      [0.01, 100, -2.3642173662384818],
    ];
    for (const [p, nu, q] of student) assert.ok(rel(studentQuantile(p, nu), q) < 1e-12, `t⁻¹(${p}; ${nu})`);
  });

  it('gives VaR and ES of a normal and a standardised t, as scipy integrates them', () => {
    const normal = tailRisk({ mean: 0.05, variance: 2, level: 0.01 });
    assert.ok(rel(normal.valueAtRisk, 3.239952714266374) < 1e-8);
    assert.ok(rel(normal.expectedShortfall, 3.7191820970422063) < 1e-8);
    const t = tailRisk({ mean: 0.05, variance: 2, nu: 5, level: 0.01 });
    assert.ok(rel(t.valueAtRisk, 3.6360961296546352) < 1e-12);
    assert.ok(rel(t.expectedShortfall, 4.827391720470791) < 1e-12);
  });

  it('reads the same one-step VaR off simulated paths', () => {
    const params: GarchParams = { mu: 0.02, omega: 0.05, alpha: 0.05, gamma: 0.1, beta: 0.85, nu: 6 };
    const from = { lastVariance: 2, lastResidual: -1.5 };
    const sim = simulatePaths({ model: 'gjr', params, ...from, steps: 5, paths: 200_000, seed: 4 });
    const exact = tailRisk({ mean: params.mu, variance: sim.variance[0][0], nu: 6, level: 0.01 });
    const one = simulatedTailRisk(sim, 0.01, 1);
    assert.ok(rel(one.valueAtRisk, exact.valueAtRisk) < 0.02, `VaR ${one.valueAtRisk} vs ${exact.valueAtRisk}`);
    assert.ok(rel(one.expectedShortfall, exact.expectedShortfall) < 0.03, `ES ${one.expectedShortfall}`);
    const five = simulatedTailRisk(sim, 0.01);
    assert.ok(five.valueAtRisk > one.valueAtRisk && five.expectedShortfall > five.valueAtRisk);
  });

  it('tests coverage as Kupiec and Christoffersen do', () => {
    const hits = [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].map(
      Boolean,
    );
    const out = coverageTest(hits, 0.05);
    assert.equal(out.violations, 4);
    assert.ok(rel(out.unconditional.statistic, 3.0726414629314256) < 1e-12);
    assert.ok(rel(out.unconditional.pValue, 0.07962000352763175) < 1e-9);
    assert.ok(rel(out.independence.statistic, 0.4240812862603818) < 1e-12);
    assert.ok(rel(out.conditional.pValue, 0.17405892727585373) < 1e-9);
    assert.equal(coverageTest([false, false, false], 0.01).violations, 0);
  });

  it('walks a one-step VaR forward with the right coverage when the model is the truth', () => {
    const r = simulate({ mu: 0, omega: 0.05, alpha: 0.05, gamma: 0.1, beta: 0.85 }, 3000, 23);
    const out = varBacktest(r, { model: 'gjr', level: 0.05, step: 5 });
    assert.equal(out.observations + out.skipped, validationWindows(r.length, { horizon: 1, step: 5 }).length);
    assert.ok(out.unconditional.pValue > 0.01, `rate ${out.rate}`);
    const f = out.forecasts[0];
    const fit = fitGarch(r.slice(0, f.origin + 1), { model: 'gjr' });
    assert.ok(
      rel(
        f.valueAtRisk,
        tailRisk({ mean: fit.params.mu, variance: forecastVariance(fit, 1)[0], level: 0.05 }).valueAtRisk,
      ) < 1e-9,
    );
    assert.equal(f.realized, r[f.origin + 1]);
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

describe('walk-forward', () => {
  it('refuses a count that is not a whole number', () => {
    const fit = fitGarch(simulate({ mu: 0, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85 }, 300, 28));
    assert.throws(() => forecastVariance(fit, 2.5), /horizon must be an integer/);
    assert.throws(() => walkForwardOrigins(1000, 500.5, 5), /minTrain/);
    assert.throws(() => walkForwardOrigins(1000, 500, 5, 0.5), /step/);
    assert.throws(() => validationWindows(1000, { horizon: 1.5 }), /horizon/);
    assert.throws(() => weighForecasts([1, 2, 3], [], [], 2.5), /burnIn/);
  });

  it('never lets a test window overlap its train window, nor run past the end', () => {
    const w = walkForwardOrigins(100, 50, 5, 3);
    assert.equal(w[0].origin, 49);
    for (const x of w) assert.ok(x.testStart === x.origin + 1 && x.testEnd === x.origin + 5 && x.testEnd <= 99);
    assert.equal(w[w.length - 1].origin <= 94, true);
  });

  it("keeps each scored origin's QLIKE terms, and tests their difference by Diebold–Mariano over the windows' overlap", () => {
    const r = simulate({ mu: 0, omega: 0.05, alpha: 0.03, gamma: 0.15, beta: 0.85 }, 1500, 29);
    const v = walkForwardValidation(r, { model: 'gjr', horizon: 10, step: 5, baseline: 'garch' });
    assert.equal(v.scored.length, v.origins);
    const meanOf = (x: number[]) => x.reduce((a, b) => a + b, 0) / x.length;
    assert.ok(rel(meanOf(v.scored.map((o) => o.loss)), v.loss!) < 1e-12);
    assert.ok(rel(meanOf(v.scored.map((o) => o.baselineLoss)), v.baselineLoss!) < 1e-12);
    const dm = v.dieboldMariano!;
    assert.equal(dm.lags, 1, 'windows 10 long, 5 apart: two overlap');
    assert.deepEqual(
      dm,
      dieboldMariano(
        v.scored.map((o) => o.loss),
        v.scored.map((o) => o.baselineLoss),
        2,
      ),
    );
    assert.ok(rel(dm.meanDifference, v.loss! - v.baselineLoss!) < 1e-9);
    // model against model, on the origins both scored
    const garch = { ...v, scored: v.scored.slice(3).map((o) => ({ ...o, loss: o.baselineLoss })) };
    const both = compareValidations(v, garch, 2);
    assert.equal(both.observations, v.origins - 3);
    assert.deepEqual(
      both,
      dieboldMariano(
        v.scored.slice(3).map((o) => o.loss),
        v.scored.slice(3).map((o) => o.baselineLoss),
        2,
      ),
    );
  });

  it('scores QLIKE at 0 for a perfect forecast and harder for under- than over-prediction', () => {
    assert.equal(qlike([2, 3], [2, 3]), 0);
    assert.ok(qlike([2], [1]) > qlike([2], [4]));
  });
});

describe('Diebold–Mariano', () => {
  it('is the hand computation: HAC variance (Bartlett or uniform), the HLN correction, t(n − 1)', () => {
    for (const c of fixture.diebold_mariano) {
      const kernel = c.kernel as 'bartlett' | 'uniform';
      const dm = dieboldMariano(c.a, c.b, c.h, { kernel });
      assert.equal(dm.observations, c.a.length);
      assert.equal(dm.lags, c.h - 1);
      assert.ok(rel(dm.meanDifference, c.mean) < 1e-12);
      assert.ok(rel(dm.statistic, c.statistic) < 1e-12, `h ${c.h} ${kernel}: ${dm.statistic} vs ${c.statistic}`);
      assert.ok(rel(dm.pValue, c.p_value) < 1e-9, `h ${c.h} ${kernel}: p ${dm.pValue} vs ${c.p_value}`);
    }
  });

  it('refuses what it cannot test', () => {
    assert.throws(() => dieboldMariano([1, 2, 3], [1, 2]), RangeError);
    assert.throws(() => dieboldMariano([1, 2, 3, 4], [1, 2, 3, 4]), /no positive long-run variance/);
    assert.throws(() => dieboldMariano([1, 2, 3, 4], [0, 1, 3, 2], 1.5), /horizon/);
    assert.throws(() => dieboldMariano([1, 2, 3], [0, 1, 3], 3), /too few/);
    assert.throws(() => dieboldMariano([1, NaN, 3, 4], [0, 1, 3, 2]), /not a finite/);
  });
});

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

describe('fitsAt', () => {
  it('fits each train window, and the variance it hands back is that window’s last', () => {
    const r = simulate({ mu: 0, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85 }, 1200, 9);
    const [a, b] = fitsAt({ returns: r, ends: [1199, 799] });
    const whole = fitGarch(r);
    assert.ok(rel(a!.lastVariance, whole.conditionalVariance[1199]) < 1e-3);
    assert.ok(b !== null && b.lastResidual === r[799] - b.params.mu);
  });
});

describe('runInWorker', () => {
  const r = simulate({ mu: 0, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85 }, 700, 10);
  it('answers what the job answers in-process', async () => {
    const inWorker = await runInWorker('fitsAt', { returns: r, ends: [699] });
    assert.deepEqual(inWorker, fitsAt({ returns: r, ends: [699] }));
  });
  it('carries an input refusal as one', async () => {
    await assert.rejects(runInWorker('fitsAt', { returns: r, ends: [5000] }), GarchJobInputError);
  });
  it('terminates a job past its deadline', async () => {
    await assert.rejects(
      runInWorker(
        'volatilityForecast',
        {
          returns: simulate({ mu: 0, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85 }, 20000, 11),
          horizon: 20,
          step: 1,
        },
        { timeoutMs: 50 },
      ),
      GarchJobTimeoutError,
    );
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

/**
 * A Web Worker as a page has one, made of a worker thread: \`self\` gets
 * addEventListener / postMessage, then the built web worker entry runs.
 */
function webWorkerShim(): WebWorker {
  const entry = JSON.stringify(join(__dirname, '../src/worker-web.js'));
  const thread = new Worker(
    `const { parentPort } = require('node:worker_threads');
     globalThis.addEventListener = (type, listener) => {
       if (type === 'message') parentPort.on('message', (data) => listener({ data }));
     };
     globalThis.postMessage = (message) => parentPort.postMessage(message);
     require(${entry});`,
    { eval: true },
  );
  return {
    postMessage: (message) => thread.postMessage(message),
    terminate: () => void thread.terminate(),
    addEventListener: (type: string, listener: (event: { data?: unknown; message?: string }) => void) => {
      if (type === 'message') thread.on('message', (data) => listener({ data }));
      if (type === 'error') thread.on('error', (error) => listener({ message: error.message }));
    },
  };
}

describe('the web pool', () => {
  it('answers the walk-forward exactly as in-process, on Web Workers', async () => {
    const r = simulate({ mu: 0, omega: 0.05, alpha: 0.05, gamma: 0.1, beta: 0.85 }, 900, 25);
    const request = { returns: r, model: 'gjr' as const, horizon: 10, minTrain: 600 };
    const pool = new GarchWebPool(3, webWorkerShim);
    try {
      assert.deepEqual(await volatilityForecastInWebWorkers(request, { pool }), volatilityForecast(request));
    } finally {
      await pool.close();
    }
  });

  it('carries an input refusal and a deadline as the node pool does', async () => {
    const pool = new GarchWebPool(1, webWorkerShim);
    try {
      await assert.rejects(
        volatilityForecastInWebWorkers({ returns: [1, 2, 3], horizon: 5 }, { pool }),
        GarchJobInputError,
      );
      const long = simulate({ mu: 0, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85 }, 3000, 26);
      await assert.rejects(
        volatilityForecastInWebWorkers({ returns: long, model: 'gjr', horizon: 5 }, { pool, timeoutMs: 50 }),
        (error: Error) =>
          error instanceof GarchJobTimeoutError &&
          /walkForwardForecasts \(\d+ of \d+ tasks\) did not finish within 50 ms$/.test(error.message),
      );
      const r = simulate({ mu: 0, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85 }, 700, 27);
      const request = { returns: r, horizon: 5, minTrain: 650 };
      assert.deepEqual(await volatilityForecastInWebWorkers(request, { pool }), volatilityForecast(request));
    } finally {
      await pool.close();
    }
  });
});

describe('the pool', () => {
  const r = returnsOf(fixture.series.iova.closes).slice(0, 1400);
  it('carries a zero mean through the walk-forward, the same on the pool as in-process', async () => {
    const r = simulate({ mu: 0, omega: 0.05, alpha: 0.05, gamma: 0.1, beta: 0.85 }, 900, 24);
    const request = { returns: r, model: 'gjr' as const, mean: 'zero' as const, horizon: 10, minTrain: 600 };
    const pool = new GarchPool(2);
    const answer = await volatilityForecastInWorkers(request, { pool });
    await pool.close();
    assert.deepEqual(answer, volatilityForecast(request));
    assert.equal(answer.mean, 'zero');
    assert.equal(answer.params.mu, 0);
  });

  it('answers the walk-forward exactly as in-process, whatever its size', async () => {
    const request = { returns: r, model: 'gjr' as const, horizon: 20, rivalsWithin: 2 };
    const inProcess = volatilityForecast(request);
    for (const size of [1, 3]) {
      const pool = new GarchPool(size);
      assert.deepEqual(await volatilityForecastInWorkers(request, { pool }), inProcess);
      await pool.close();
    }
  });
  it('answers fitsAt exactly as in-process', async () => {
    const ends = Array.from({ length: 40 }, (_, i) => 300 + i * 10);
    assert.deepEqual(await fitsAtInWorkers({ returns: r, ends }), fitsAt({ returns: r, ends }));
  });
  it('fails the whole call when one of its jobs is refused', async () => {
    await assert.rejects(fitsAtInWorkers({ returns: r, ends: [300, 5000] }), GarchJobInputError);
  });
  it('keeps working after a call ran past its deadline', async () => {
    const pool = new GarchPool(2);
    const long = simulate({ mu: 0, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85 }, 20000, 13);
    await assert.rejects(
      volatilityForecastInWorkers({ returns: long, horizon: 20, step: 1 }, { pool, timeoutMs: 50 }),
      GarchJobTimeoutError,
    );
    const ends = [300, 400];
    assert.deepEqual(await fitsAtInWorkers({ returns: r, ends }, { pool }), fitsAt({ returns: r, ends }));
    await pool.close();
  });
});

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

describe('models weighted by their past loss', () => {
  const r = simulate({ mu: 0, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85 }, 1500, 20);
  const windows = validationWindows(r.length, { horizon: 20, minTrain: 200, step: 10 });
  const truth = windows.map((w) => realizedVariance(r, w.testStart, w.testEnd));
  const flat = windows.map(() => 1);

  it('weighs equally until each model has its burn-in of ended windows', () => {
    const { forecasts } = weighForecasts(r, windows, [truth, flat], 20);
    assert.ok(Math.abs(forecasts[0]! - (truth[0] + 1) / 2) < 1e-12);
  });
  it('then leans on the model that has forecast better', () => {
    const { forecasts, asOfWeights } = weighForecasts(r, windows, [truth, flat], 20);
    const last = forecasts.length - 1;
    assert.ok(Math.abs(forecasts[last]! - truth[last]) < 1e-6 * truth[last]);
    assert.ok(asOfWeights[0] > 0.999 && Math.abs(asOfWeights[0] + asOfWeights[1] - 1) < 1e-12);
  });
  it('never weighs with a window that had not ended at the origin', () => {
    const k = 60;
    const before = weighForecasts(r, windows, [truth, flat], 20).forecasts[k];
    const future = truth.map((v, i) => (windows[i].testEnd > windows[k].origin ? v * 50 : v));
    const after = weighForecasts(r, windows, [future, flat], 20).forecasts[k];
    // only model 0's own forecast at k may differ; its weight must not
    const w0 = (x: number, own: number) => (x - 1) / (own - 1);
    assert.ok(Math.abs(w0(before!, truth[k]) - w0(after!, future[k])) < 1e-9);
  });
  it('answers on the pool exactly as in-process, the as-of weights summing to one', async () => {
    const returns = returnsOf(fixture.series.iova.closes).slice(-1100);
    const request = { returns, horizon: 20 };
    const inProcess = weightedForecast(request);
    assert.deepEqual(await weightedForecastInWorkers(request, { pool: new GarchPool(3) }), inProcess);
    assert.equal(inProcess.models.length, WEIGHTED_MODELS.length);
    assert.ok(Math.abs(inProcess.models.reduce((a, m) => a + m.weight, 0) - 1) < 1e-12);
    const mixed = inProcess.models.reduce((a, m) => a + m.weight * (m.asOf?.forecast?.[0] ?? 0), 0);
    assert.ok(Math.abs(inProcess.forecast![0] - mixed) < 1e-9 * mixed);
    assert.ok(inProcess.validation.origins > 0);
  });
  it('keeps the fits and the weights in force at every origin — what a simulation from there needs', () => {
    const returns = returnsOf(fixture.series.iova.closes).slice(-1100);
    const answer = weightedForecast({ returns, horizon: 20 });
    const origins = validationWindows(returns.length, { horizon: 20 }).map((w) => w.origin);
    assert.deepEqual(
      answer.walkForward.map((o) => o.origin),
      origins,
    );
    for (const o of answer.walkForward)
      if (o.fits.some(Boolean)) assert.ok(Math.abs(o.weights.reduce((a, b) => a + b, 0) - 1) < 1e-12);
    // before the burn-in every model with a fit weighs the same
    assert.deepEqual(
      answer.walkForward[0].weights,
      answer.walkForward[0].fits.map((f, _, all) => (f ? 1 / all.filter(Boolean).length : 0)),
    );
    const k = WEIGHTED_MODELS.findIndex((m) => m.model === 'gjr' && m.dist === 't');
    assert.deepEqual(
      answer.walkForward.map((o) => o.fits[k]),
      fitsAt({ returns, ends: origins, model: 'gjr', dist: 't' }),
    );
  });
});

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
