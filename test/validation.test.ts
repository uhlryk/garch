import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  compareValidations,
  dieboldMariano,
  fitGarch,
  fitsAt,
  forecastVariance,
  qlike,
  realizedVariance,
  validationWindows,
  WEIGHTED_MODELS,
  walkForwardOrigins,
  walkForwardValidation,
  weighForecasts,
  weightedForecast,
} from '../src/index.js';
import { GarchPool, weightedForecastInWorkers } from '../src/node.js';
import { fixture, rel, returnsOf, simulate } from './helpers.js';

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

describe('fitsAt', () => {
  it('fits each train window, and the variance it hands back is that window’s last', () => {
    const r = simulate({ mu: 0, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85 }, 1200, 9);
    const [a, b] = fitsAt({ returns: r, ends: [1199, 799] });
    const whole = fitGarch(r);
    assert.ok(rel(a!.lastVariance, whole.conditionalVariance[1199]) < 1e-3);
    assert.ok(b !== null && b.lastResidual === r[799] - b.params.mu);
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
