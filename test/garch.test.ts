import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  realizedVariance,
  validationWindows,
  weighForecasts,
  weightedForecast,
  WEIGHTED_MODELS,
  quantizedUntil,
  T_NU_MIN,
  asOfFit,
  backcast,
  fitGarch,
  fitsAt,
  forecastVariance,
  qlike,
  volatilityForecast,
  walkForwardOrigins,
} from '../src/index.js';
import { logLikelihood, logLikelihoodGradient, type GarchParams } from '../src/model.js';
import {
  fitsAtInWorkers,
  GarchJobInputError,
  GarchJobTimeoutError,
  GarchPool,
  runInWorker,
  volatilityForecastInWorkers,
  weightedForecastInWorkers,
} from '../src/node.js';
import fixture from './fixtures/arch-2026-09-25.json';
import egarchFixture from './fixtures/arch-egarch-2026-10-01.json';
import { egarchForecast, egarchLogLikelihood, egarchLogLikelihoodGradient } from '../src/egarch.js';
import { GarchInputError } from '../src/fit.js';

/** mulberry32 + Box–Muller: a seeded normal stream, so the simulated paths are fixed. */
function normals(seed: number): () => number {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(2 * Math.PI * uniform());
}

function simulate(p: GarchParams, n: number, seed: number): number[] {
  const z = normals(seed);
  const out: number[] = [];
  let s = p.omega / (1 - p.alpha - p.gamma / 2 - p.beta);
  let e = 0;
  for (let t = 0; t < n; t += 1) {
    if (t > 0) s = p.omega + p.alpha * e * e + p.gamma * (e < 0 ? e * e : 0) + p.beta * s;
    e = Math.sqrt(s) * z();
    out.push(p.mu + e);
  }
  return out;
}

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
      const fit = fitGarch(simulate({ mu: 0, omega: 0.2, alpha: 0.3, gamma: 0.2, beta: 0.5 }, 700, seed), { model: 'gjr' });
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

describe('walk-forward', () => {
  it('never lets a test window overlap its train window, nor run past the end', () => {
    const w = walkForwardOrigins(100, 50, 5, 3);
    assert.equal(w[0].origin, 49);
    for (const x of w) assert.ok(x.testStart === x.origin + 1 && x.testEnd === x.origin + 5 && x.testEnd <= 99);
    assert.equal(w[w.length - 1].origin <= 94, true);
  });

  it('scores QLIKE at 0 for a perfect forecast and harder for under- than over-prediction', () => {
    assert.equal(qlike([2, 3], [2, 3]), 0);
    assert.ok(qlike([2], [1]) > qlike([2], [4]));
  });
});

/**
 * The replacement for the `arch` kernel (#1004): where the likelihood has one
 * mode, this IS `arch` — the same parameters, σ̂, γ p-value and walk-forward
 * loss, frozen from `arch`'s own run on 2026-09-25.
 */
describe('against arch (frozen 2026-09-25)', () => {
  for (const name of ['iova', 'spy'] as const) {
    const s = fixture.series[name];
    const r = returnsOf(s.closes);
    it(`${name}: garch fit and σ̂`, () => {
      const fit = fitGarch(r);
      const a = s.arch.garch;
      for (const k of ['omega', 'alpha', 'beta'] as const) assert.ok(rel(fit.params[k], a[k]) < 1e-3, `${k} ${fit.params[k]} vs ${a[k]}`);
      const v = forecastVariance(fit, 20);
      assert.ok(rel(Math.sqrt(v[0]) / 100, a.sigma_daily) < 1e-4);
      assert.ok(rel(Math.sqrt(v.reduce((x, y) => x + y, 0)) / 100, a.sigma_horizon) < 1e-4);
    });
    it(`${name}: gjr γ and its robust p-value`, () => {
      const fit = fitGarch(r, { model: 'gjr', inference: true });
      const a = s.arch.gjr;
      assert.ok(rel(fit.params.gamma, a.gamma!) < 2e-3, `γ ${fit.params.gamma} vs ${a.gamma}`);
      assert.ok(rel(fit.inference!.pValues.gamma, a.gamma_pvalue!) < 0.02, `p ${fit.inference!.pValues.gamma} vs ${a.gamma_pvalue}`);
    });
  }
  it('iova: the walk-forward QLIKE and its naive baseline', () => {
    const s = fixture.series.iova;
    const out = volatilityForecast({ returns: returnsOf(s.closes), horizon: 20 });
    assert.ok(rel(out.validation.loss!, s.arch.garch.qlike!) < 1e-3);
    assert.ok(rel(out.validation.baselineLoss!, s.arch.garch.naive_qlike!) < 1e-9);
    assert.equal(out.validation.beatsBaseline, true);
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
      runInWorker('volatilityForecast', { returns: simulate({ mu: 0, omega: 0.05, alpha: 0.1, gamma: 0, beta: 0.85 }, 20000, 11), horizon: 20, step: 1 }, { timeoutMs: 50 }),
      GarchJobTimeoutError,
    );
  });
});

describe('the modes of the likelihood (#1006)', () => {
  const r = returnsOf(fixture.series.iova.closes);
  it('lists every optimum the starts reached, best first, the fit being the first', () => {
    const fit = fitGarch(r, { model: 'gjr' });
    assert.ok(fit.modes.length >= 1);
    assert.deepEqual(fit.modes[0].params, fit.params);
    for (let i = 1; i < fit.modes.length; i += 1) assert.ok(fit.modes[i].logLikelihood <= fit.modes[i - 1].logLikelihood);
    assert.equal(fit.modes.reduce((a, m) => a + m.starts, 0), 6);
  });
  it('marks the optimum a warm start reached', () => {
    const cold = fitGarch(r);
    const warm = fitGarch(r, { warmStart: cold.params });
    assert.equal(warm.modes[0].fromWarmStart, true);
    assert.equal(cold.modes.some((m) => m.fromWarmStart), false);
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

describe('the pool (#1006)', () => {
  const r = returnsOf(fixture.series.iova.closes).slice(0, 1400);
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
    await assert.rejects(volatilityForecastInWorkers({ returns: long, horizon: 20, step: 1 }, { pool, timeoutMs: 50 }), GarchJobTimeoutError);
    const ends = [300, 400];
    assert.deepEqual(await fitsAtInWorkers({ returns: r, ends }, { pool }), fitsAt({ returns: r, ends }));
    await pool.close();
  });
});

describe('Student-t innovations (#1111)', () => {
  const tDraw = (z: () => number, nu: number) => {
    let c = 0;
    for (let i = 0; i < nu; i += 1) c += z() ** 2;
    return (z() / Math.sqrt(c / nu)) * Math.sqrt((nu - 2) / nu);
  };
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
    const z = normals(15);
    const r: number[] = [];
    let s = 1;
    let e = 0;
    for (let t = 0; t < 6000; t += 1) {
      if (t) s = 0.05 + 0.1 * e * e + 0.85 * s;
      e = Math.sqrt(s) * tDraw(z, 5);
      r.push(e);
    }
    const fit = fitGarch(r, { dist: 't' });
    assert.equal(fit.converged, true);
    assert.equal(fit.nuBound, null);
    assert.ok(Math.abs(fit.params.nu! - 5) < 1, `ν ${fit.params.nu}`);
    assert.ok(fit.logLikelihood > fitGarch(r).logLikelihood);
  });

  const heavy = (() => {
    const z = normals(16);
    const r: number[] = [];
    let s = 1;
    let e = 0;
    for (let t = 0; t < 5000; t += 1) {
      if (t) s = 0.05 + 0.1 * e * e + 0.85 * s;
      e = Math.sqrt(s) * tDraw(z, 3);
      r.push(e);
    }
    return r;
  })();

  it('holds ν on its floor when the data want less, and says so — converged', () => {
    const fit = fitGarch(heavy, { dist: 't' });
    assert.equal(fit.nuBound, 'lower');
    assert.equal(fit.params.nu, T_NU_MIN);
    assert.equal(fit.converged, true);
    const free = fitGarch(heavy, { dist: 't', nuMin: 2.05 });
    assert.equal(free.nuBound, null);
    assert.ok(free.params.nu! < T_NU_MIN, `ν ${free.params.nu}`);
    assert.throws(() => fitGarch(heavy, { dist: 't', nuMin: 2 }), RangeError);
  });

  it('gives robust standard errors with ν among the parameters, and none for a held ν', () => {
    const r = simulate({ mu: 0, omega: 0.05, alpha: 0.03, gamma: 0.15, beta: 0.85 }, 4000, 17);
    const fit = fitGarch(r, { model: 'gjr', dist: 't', inference: true });
    assert.equal(fit.nuBound, null);
    assert.ok(fit.params.nu! > 20, `normal data, ν ${fit.params.nu}`);
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

describe('quantizedUntil (#1111)', () => {
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

describe('models weighted by their past loss (#1113)', () => {
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
    assert.deepEqual(answer.walkForward.map((o) => o.origin), origins);
    for (const o of answer.walkForward)
      if (o.fits.some(Boolean)) assert.ok(Math.abs(o.weights.reduce((a, b) => a + b, 0) - 1) < 1e-12);
    // before the burn-in every model with a fit weighs the same
    assert.deepEqual(answer.walkForward[0].weights, answer.walkForward[0].fits.map((f, _, all) => (f ? 1 / all.filter(Boolean).length : 0)));
    const k = WEIGHTED_MODELS.findIndex((m) => m.model === 'gjr' && m.dist === 't');
    assert.deepEqual(
      answer.walkForward.map((o) => o.fits[k]),
      fitsAt({ returns, ends: origins, model: 'gjr', dist: 't' }),
    );
  });
});


describe('EGARCH (#1111)', () => {
  const p: GarchParams = { mu: 0.02, omega: 0.01, alpha: 0.15, gamma: -0.12, beta: 0.97 };
  /** An EGARCH path from its own recursion. */
  const path = (n: number, seed: number) => {
    const z = normals(seed);
    const out: number[] = [];
    let L = p.omega / (1 - p.beta);
    let prev = 0;
    for (let t = 0; t < n; t += 1) {
      if (t > 0) L = p.omega + p.alpha * (Math.abs(prev) - Math.sqrt(2 / Math.PI)) + p.gamma * prev + p.beta * L;
      prev = z();
      out.push(p.mu + Math.exp(L / 2) * prev);
    }
    return out;
  };

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
    const z = normals(5);
    const sims = 100_000;
    const mean = new Array<number>(20).fill(0);
    for (let k = 0; k < sims; k += 1) {
      let L = Math.log(exact[0]);
      mean[0] += exact[0] / sims;
      for (let h = 1; h < 20; h += 1) {
        const shock = z();
        L = p.omega + p.alpha * (Math.abs(shock) - Math.sqrt(2 / Math.PI)) + p.gamma * shock + p.beta * L;
        mean[h] += Math.exp(L) / sims;
      }
    }
    for (const h of [1, 9, 19]) assert.ok(rel(exact[h], mean[h]) < 0.01, `step ${h + 1}: ${exact[h]} vs ${mean[h]}`);
    // the first step is the recursion on the observed shock
    const z0 = -3 / Math.sqrt(2.5);
    assert.ok(rel(exact[0], Math.exp(p.omega + p.alpha * (Math.abs(z0) - Math.sqrt(2 / Math.PI)) + p.gamma * z0 + p.beta * Math.log(2.5))) < 1e-12);
  });

  it('refuses Student-t innovations: the expected variance two steps ahead is infinite', () => {
    assert.throws(() => fitGarch(path(500, 2), { model: 'egarch', dist: 't' }), GarchInputError);
  });

  describe('against arch 8.0.0 (frozen 2026-10-01)', () => {
    for (const name of ['iova', 'spy'] as const) {
      it(`${name}: arch's optimum or a higher one, its γ p-value, and its simulated forecast`, () => {
        const a = egarchFixture.series[name];
        const fit = fitGarch(returnsOf(fixture.series[name].closes), { model: 'egarch', inference: true });
        assert.equal(fit.converged, true);
        assert.ok(fit.logLikelihood >= a.loglik - 1e-3, `ll ${fit.logLikelihood} vs ${a.loglik}`);
        for (const key of ['alpha', 'gamma', 'beta'] as const) assert.ok(rel(fit.params[key], a[key]) < 1e-3, key);
        assert.ok(Math.abs(fit.params.mu - a.mu) < 1e-3 && rel(fit.params.omega, a.omega) < 0.02, 'μ, ω');
        assert.ok(rel(fit.inference!.pValues.gamma, a.gamma_pvalue) < 0.02, 'p(γ)');
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
