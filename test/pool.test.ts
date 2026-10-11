import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { fitGarch, fitsAt, volatilityForecast, weightedForecast } from '../src/index.js';
import {
  fitsAtInWorkers,
  GarchJobInputError,
  GarchJobTimeoutError,
  GarchPool,
  runInWorker,
  volatilityForecastInWorkers,
  weightedForecastInWorkers,
} from '../src/node.js';
import { GarchWebPool, volatilityForecastInWorkers as volatilityForecastInWebWorkers } from '../src/web.js';
import { fixture, rel, returnsOf, simulate, webWorkerShim } from './helpers.js';

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

  it('carries variance targeting through the walk-forward and the weighted forecast, the same on the pool', async () => {
    const r = simulate({ mu: 0, omega: 0.05, alpha: 0.05, gamma: 0.1, beta: 0.85 }, 900, 25);
    const request = { returns: r, model: 'gjr' as const, targetVariance: true, horizon: 10, minTrain: 600 };
    const models = [
      { model: 'garch', dist: 'normal' },
      { model: 'egarch', dist: 'normal' },
    ] as const;
    const weighted = { returns: r, horizon: 10, minTrain: 700, step: 20, models, targetVariance: true };
    const pool = new GarchPool(2);
    const answer = await volatilityForecastInWorkers(request, { pool });
    const mixed = await weightedForecastInWorkers(weighted, { pool });
    await pool.close();
    assert.deepEqual(answer, volatilityForecast(request));
    assert.equal(answer.varianceTarget, fitGarch(r, { targetVariance: true }).varianceTarget);
    assert.ok(rel(answer.params.omega, answer.varianceTarget! * (1 - answer.persistence)) < 1e-15);
    assert.deepEqual(mixed, weightedForecast(weighted));
    assert.notEqual(mixed.models[0].asOf!.varianceTarget, null, 'garch targeted');
    assert.equal(mixed.models[1].asOf!.varianceTarget, null, 'egarch fitted with ω free');
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
