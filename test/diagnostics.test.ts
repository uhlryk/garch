import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { archLm, chiSquareSurvival, fitGarch, ljungBox, standardizedResiduals } from '../src/index.js';
import { rel, simulate } from './helpers.js';

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
