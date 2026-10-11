import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { correlationRecursion, fitCorrelation } from '../src/dcc.js';
import { GarchInputError, variancePath } from '../src/fit.js';
import {
  dccForecast,
  fitDcc,
  forecastVariance,
  portfolioTailRisk,
  portfolioVariance,
  simulate as simulatePaths,
} from '../src/index.js';
import type { GarchParams } from '../src/model.js';
import { fitDccInWorkers, GarchPool } from '../src/node.js';
import { fixture, rel, returnsOf } from './helpers.js';

/** A seeded DCC(1,1) path: GARCH(1,1) assets whose standardised shocks have correlation R_t. */
function simulateDcc(params: GarchParams[], a: number, b: number, target: number[][], T: number, seed: number) {
  const N = params.length;
  const eta = simulatePaths({
    model: 'garch',
    params: { mu: 0, omega: 1, alpha: 0, gamma: 0, beta: 0 },
    steps: T * N,
    paths: 1,
    seed,
  }).returns[0];
  let Q = target.map((row) => row.slice());
  const s = params.map((p) => p.omega / (1 - p.alpha - p.beta));
  const series = params.map(() => new Array<number>(T));
  for (let t = 0; t < T; t += 1) {
    const R = Q.map((row, i) => row.map((q, j) => q / Math.sqrt(Q[i][i] * Q[j][j])));
    const L = R.map(() => new Array<number>(N).fill(0));
    for (let i = 0; i < N; i += 1)
      for (let j = 0; j <= i; j += 1) {
        let v = R[i][j];
        for (let k = 0; k < j; k += 1) v -= L[i][k] * L[j][k];
        L[i][j] = i === j ? Math.sqrt(v) : v / L[j][j];
      }
    const z = L.map((row) => row.reduce((acc, l, k) => acc + l * eta[t * N + k], 0));
    params.forEach((p, i) => {
      const e = Math.sqrt(s[i]) * z[i];
      series[i][t] = p.mu + e;
      s[i] = p.omega + p.alpha * e * e + p.beta * s[i];
    });
    Q = Q.map((row, i) => row.map((q, j) => (1 - a - b) * target[i][j] + a * z[i] * z[j] + b * q));
  }
  return series;
}

describe('DCC-GARCH', () => {
  const params: GarchParams[] = [
    { mu: 0.03, omega: 0.05, alpha: 0.08, gamma: 0, beta: 0.88 },
    { mu: 0, omega: 0.1, alpha: 0.05, gamma: 0, beta: 0.9 },
    { mu: 0.01, omega: 0.02, alpha: 0.1, gamma: 0, beta: 0.85 },
  ];
  const target = [
    [1, 0.5, 0.2],
    [0.5, 1, 0.3],
    [0.2, 0.3, 1],
  ];
  const series = simulateDcc(params, 0.04, 0.93, target, 5000, 9);
  const dcc = fitDcc(series);
  const d = fixture.dcc;
  const pair = [fixture.series.iova, fixture.series.spy].map((x) => returnsOf(x.closes).slice(-d.returns));

  it('recovers a and b and the correlation target of a simulated path', () => {
    assert.ok(dcc.converged);
    assert.ok(Math.abs(dcc.a - 0.04) < 0.01, `a ${dcc.a}`);
    assert.ok(Math.abs(dcc.b - 0.93) < 0.02, `b ${dcc.b}`);
    for (let i = 0; i < 3; i += 1)
      for (let j = 0; j < 3; j += 1) assert.ok(Math.abs(dcc.correlationTarget[i][j] - target[i][j]) < 0.05);
    assert.equal(dcc.correlation.length, 5000);
    assert.ok(
      rel(dcc.logLikelihood, dcc.assets.reduce((a, f) => a + f.logLikelihood, 0) + dcc.correlationLogLikelihood) <
        1e-15,
    );
  });

  it("stage 2 on arch's standardised residuals: scipy's optimum of the same likelihood", () => {
    const z = pair.map((r, i) => {
      const p = { ...d.stage1[i], gamma: 0 };
      const s = variancePath(r, 'garch', p);
      return Float64Array.from(r, (x, t) => (x - p.mu) / Math.sqrt(s[t]));
    });
    const fit = fitCorrelation(z);
    assert.ok(rel(fit.a, d.a) < 1e-4, `a ${fit.a} vs ${d.a}`);
    assert.ok(rel(fit.b, d.b) < 1e-4, `b ${fit.b} vs ${d.b}`);
    assert.ok(fit.correlationLogLikelihood >= d.correlation_loglik - 1e-7, `${fit.correlationLogLikelihood}`);
    const atScipy = correlationRecursion(z, fit.correlationTarget, d.a, d.b);
    assert.ok(Math.abs(atScipy.logLikelihood - d.correlation_loglik) < 1e-8, 'the same likelihood at scipy’s (a, b)');
    assert.ok(rel(fit.correlation.at(-1)![1], d.last_correlation[0][1]) < 1e-4);
  });

  it('iova and spy: the whole fit, its forecast and the portfolio, against the fixture', () => {
    const fit = fitDcc(pair);
    assert.ok(rel(fit.a, d.a) < 1e-2 && rel(fit.b, d.b) < 1e-2, `a ${fit.a} b ${fit.b}`);
    assert.ok(Math.abs(fit.logLikelihood - d.loglik) < 0.5, `${fit.logLikelihood} vs ${d.loglik}`);
    const f = dccForecast(fit, d.correlation.length);
    f.correlation.forEach((R, k) => {
      assert.ok(Math.abs(R[0][1] - d.correlation[k][0][1]) < 2e-3, `R step ${k + 1}`);
      for (let i = 0; i < 2; i += 1)
        for (let j = 0; j < 2; j += 1)
          assert.ok(rel(f.covariance[k][i][j], d.covariance[k][i][j]) < 1e-2, `H step ${k + 1}`);
    });
    const pv = portfolioVariance(f.covariance[0], d.weights);
    assert.ok(rel(pv, d.portfolio_variance) < 1e-3, `${pv} vs ${d.portfolio_variance}`);
    const risk = portfolioTailRisk({ dcc: fit, weights: d.weights, level: 0.01 });
    assert.equal(risk.method, 'normal');
    assert.ok(rel(risk.valueAtRisk, d.var_1pct) < 1e-3);
  });

  it('forecasts R exactly one step ahead and reverts to the target after', () => {
    const f = dccForecast(dcc, 400);
    const zT = dcc.lastStandardized;
    const c = 1 - dcc.a - dcc.b;
    const Q = dcc.lastQ.map((row, i) =>
      row.map((q, j) => c * dcc.correlationTarget[i][j] + dcc.a * zT[i] * zT[j] + dcc.b * q),
    );
    assert.ok(rel(f.correlation[0][0][1], Q[0][1] / Math.sqrt(Q[0][0] * Q[1][1])) < 1e-14);
    const Rbar = dcc.correlationTarget[0][1] / Math.sqrt(dcc.correlationTarget[0][0] * dcc.correlationTarget[1][1]);
    assert.ok(Math.abs(f.correlation[399][0][1] - Rbar) < 1e-6);
    assert.deepEqual(f.variance[2], forecastVariance(dcc.assets[2], 400));
  });

  it('simulates the closed form one day ahead, and a seed gives the same answer', () => {
    const w = [0.5, 0.3, 0.2];
    const exact = portfolioTailRisk({ dcc, weights: w, level: 0.01 });
    const sim = portfolioTailRisk({ dcc, weights: w, level: 0.01, method: 'simulated', paths: 200_000, seed: 2 });
    assert.equal(sim.method, 'simulated');
    assert.ok(rel(sim.valueAtRisk, exact.valueAtRisk) < 0.02, `${sim.valueAtRisk} vs ${exact.valueAtRisk}`);
    assert.ok(rel(sim.expectedShortfall, exact.expectedShortfall) < 0.03);
    const ten = { dcc, weights: w, level: 0.01, horizon: 10, paths: 5000, seed: 4 };
    assert.deepEqual(portfolioTailRisk(ten), portfolioTailRisk(ten));
    assert.ok(portfolioTailRisk(ten).valueAtRisk > exact.valueAtRisk);
  });

  it('fits stage 1 on the pool for the in-process answer', async () => {
    const pool = new GarchPool(2);
    const options = { models: [{ model: 'gjr' }, { model: 'garch', dist: 't' }, {}] } as const;
    const onPool = await fitDccInWorkers(series, options, { pool });
    await pool.close();
    assert.deepEqual(onPool, fitDcc(series, options));
    assert.equal(onPool.assets[1].params.nu !== undefined, true);
  });

  it('refuses what it cannot fit', () => {
    assert.throws(() => fitDcc([series[0]]), GarchInputError);
    assert.throws(() => fitDcc([series[0], series[1].slice(1)]), GarchInputError);
    assert.throws(() => fitDcc(series, { models: [{}, {}] }), GarchInputError);
    assert.throws(() => portfolioVariance([[1]], [0.5, 0.5]), RangeError);
    assert.throws(
      () => portfolioTailRisk({ dcc, weights: [1, 0, 0], level: 0.01, horizon: 2, method: 'normal' }),
      RangeError,
    );
  });
});
