import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { variancePath } from '../src/fit.js';
import {
  coverageTest,
  filteredTailRisk,
  fitGarch,
  forecastVariance,
  normalQuantile,
  simulatedTailRisk,
  simulate as simulatePaths,
  standardizedResiduals,
  studentQuantile,
  tailRisk,
  validationWindows,
  varBacktest,
} from '../src/index.js';
import type { GarchParams } from '../src/model.js';
import { fixture, rel, returnsOf, simulate } from './helpers.js';

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

describe('filtered historical simulation', () => {
  const spy = returnsOf(fixture.series.spy.closes);
  const a = fixture.series.spy.arch.fhs;
  const params: GarchParams = { mu: a.mu, omega: a.omega, alpha: a.alpha, gamma: a.gamma, beta: a.beta };
  const s = variancePath(spy, 'gjr', params);
  const z = spy.map((r, t) => (r - params.mu) / Math.sqrt(s[t]));

  it("spy: the one-step VaR and ES from the empirical quantile of arch's standardised residuals", () => {
    for (const level of [0.01, 0.05] as const) {
      const want = a.tail[level];
      const got = filteredTailRisk({ mean: params.mu, variance: a.variance_1, residuals: z, level });
      assert.ok(rel(got.valueAtRisk, want.var) < 1e-9, `VaR ${got.valueAtRisk} vs ${want.var}`);
      assert.ok(rel(got.expectedShortfall, want.es) < 1e-9, `ES ${got.expectedShortfall} vs ${want.es}`);
    }
  });

  it("spy: resampled residuals through the recursion — arch's bootstrap forecast", () => {
    const sim = simulatePaths({
      model: 'gjr',
      params,
      lastVariance: s[s.length - 1],
      lastResidual: spy[spy.length - 1] - params.mu,
      steps: a.bootstrap_variance.length,
      paths: a.bootstrap_simulations,
      seed: 5,
      innovations: { bootstrap: z },
    });
    assert.ok(rel(sim.variance[0][0], a.variance_1) < 1e-9, 'step 1 is the one-step forecast');
    a.bootstrap_variance.forEach((want, k) => {
      const got = sim.variance.reduce((x, v) => x + v[k], 0) / sim.variance.length;
      assert.ok(rel(got, want) < 0.02, `step ${k + 1}: ${got} vs ${want}`);
    });
  });

  it('draws only the residuals given, the same for the same seed', () => {
    const pool = [-2, -0.5, 0.1, 0.7, 1.4];
    const run = (seed: number) =>
      simulatePaths({ model: 'garch', params, steps: 200, paths: 2, seed, innovations: { bootstrap: pool } });
    assert.deepEqual(run(7), run(7));
    assert.notDeepEqual(run(7).returns[0], run(8).returns[0]);
    const r = run(7);
    r.returns[0].forEach((x, t) => {
      const drawn = (x - params.mu) / Math.sqrt(r.variance[0][t]);
      assert.ok(
        pool.some((v) => Math.abs(v - drawn) < 1e-9),
        `${drawn}`,
      );
    });
  });

  it('converges to the t VaR and ES when the residuals are t draws', () => {
    const t5: GarchParams = { mu: 0, omega: 1, alpha: 0, gamma: 0, beta: 0, nu: 5 };
    const draws = simulatePaths({ model: 'garch', params: t5, steps: 400_000, paths: 1, seed: 6 }).returns[0];
    for (const level of [0.01, 0.05]) {
      const fhs = filteredTailRisk({ mean: 0.05, variance: 2, residuals: draws, level });
      const t = tailRisk({ mean: 0.05, variance: 2, nu: 5, level });
      assert.ok(rel(fhs.valueAtRisk, t.valueAtRisk) < 0.01, `VaR ${level}: ${fhs.valueAtRisk} vs ${t.valueAtRisk}`);
      assert.ok(rel(fhs.expectedShortfall, t.expectedShortfall) < 0.02, `ES ${level}`);
    }
  });

  it("walks an FHS VaR forward: each origin's VaR from that fit's residuals, covering at its level on a t path", () => {
    const r = simulate({ mu: 0, omega: 0.05, alpha: 0.05, gamma: 0.1, beta: 0.85, nu: 5 }, 2000, 24);
    const out = varBacktest(r, { model: 'gjr', level: 0.05, step: 10, method: 'fhs' });
    assert.ok(out.unconditional.pValue > 0.01, `rate ${out.rate}`);
    const f = out.forecasts[0];
    const train = r.slice(0, f.origin + 1);
    const fit = fitGarch(train, { model: 'gjr' });
    const want = filteredTailRisk({
      mean: fit.params.mu,
      variance: forecastVariance(fit, 1)[0],
      residuals: standardizedResiduals(fit, train),
      level: 0.05,
    });
    assert.ok(rel(f.valueAtRisk, want.valueAtRisk) < 1e-9);
  });

  it('refuses what it cannot resample', () => {
    const base = { model: 'gjr' as const, params, steps: 5, paths: 1, seed: 1 };
    assert.throws(() => simulatePaths({ ...base, innovations: { bootstrap: [] } }), RangeError);
    assert.throws(() => simulatePaths({ ...base, innovations: { bootstrap: [1, NaN] } }), RangeError);
    assert.throws(() => filteredTailRisk({ mean: 0, variance: 1, residuals: [1, 2, 3], level: 0.001 }), /too few/);
    assert.throws(() => filteredTailRisk({ mean: 0, variance: 1, residuals: [1, NaN], level: 0.4 }), RangeError);
  });
});
