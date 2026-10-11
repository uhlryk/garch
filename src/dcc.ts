import { type FitOptions, fitGarch, type GarchFit, GarchInputError } from './fit.js';
import { forecastVariance } from './forecast.js';
import { bfgs } from './optimize.js';
import { type TailRisk, tailRisk } from './risk.js';
import { innovations, nextVariance } from './simulate.js';

/**
 * DCC-GARCH (Engle 2002): conditional correlations for several assets.
 *
 * Stage 1 fits each asset's own model (`fitGarch`, any model, distribution
 * and mean) and standardises its residuals, z_{i,t} = e_{i,t}/√s_{i,t}.
 * Stage 2 runs DCC(1,1) on them with correlation targeting:
 *
 *     Q_t = (1 − a − b)·Q̄ + a·z_{t−1}z_{t−1}ᵀ + b·Q_{t−1},   Q̄ = (1/T)·Σ z_t z_tᵀ,   Q_1 = Q̄
 *     R_t = diag(Q_t)^{−½}·Q_t·diag(Q_t)^{−½}
 *
 * and (a, b) maximise the correlation part of the Gaussian likelihood,
 * −½·Σ (ln|R_t| + z_tᵀR_t⁻¹z_t − z_tᵀz_t), with a, b ≥ 0 and a + b < 1 by
 * construction. The conditional covariance is H_t = D_t·R_t·D_t, D_t = diag(√s_{i,t}).
 */

/** What stage 1 fits each asset with: `fitGarch`'s specification. */
export type DccAssetSpec = Pick<FitOptions, 'model' | 'dist' | 'mean' | 'targetVariance' | 'nuMin'>;

/** `fitDcc`'s options: how stage 1 fits each asset. */
export type DccOptions = {
  /** One specification for every asset, or one per asset (default: garch, normal, constant mean). */
  readonly models?: DccAssetSpec | readonly DccAssetSpec[];
};

/** A two-stage DCC-GARCH fit: each asset's fit, the DCC(1,1) dynamics and every day's correlation. */
export type DccFit = {
  /** Stage 1: each asset's fit, in the order given. */
  readonly assets: readonly GarchFit[];
  /** The weight of yesterday's shock in Q_t. */
  readonly a: number;
  /** The weight of yesterday's Q in Q_t; a + b < 1. */
  readonly b: number;
  /** Stage 2 converged, and so did every asset's fit. */
  readonly converged: boolean;
  /** Q̄, the standardised residuals' second moment the dynamics revert to (N×N). */
  readonly correlationTarget: number[][];
  /** R_t for every day, each N×N flattened row by row. */
  readonly correlation: Float64Array[];
  /** Q_T, where the correlation recursion stands after the last day (N×N). */
  readonly lastQ: number[][];
  /** z_T, the last day's standardised residuals. */
  readonly lastStandardized: number[];
  /** Stage 2's log-likelihood: the correlation part. */
  readonly correlationLogLikelihood: number;
  /** Stage 1's log-likelihoods plus stage 2's — the joint one when every asset is normal. */
  readonly logLikelihood: number;
};

/** `dccForecast`'s answer: covariance, correlation and each asset's variance, step by step. */
export type DccForecast = {
  /** H_{T+1} … H_{T+h}, each N×N. */
  readonly covariance: number[][][];
  /** R_{T+1} … R_{T+h}, each N×N. */
  readonly correlation: number[][][];
  /** s_{i,T+k} per asset: `forecastVariance` of each. */
  readonly variance: number[][];
};

/** The series as stage 1 takes them: at least two, of one length. */
function checkSeries(series: readonly ArrayLike<number>[]): number {
  if (series.length < 2) throw new GarchInputError(`DCC needs at least two series, got ${series.length}`);
  const T = series[0].length;
  series.forEach((s, i) => {
    if (s.length !== T) throw new GarchInputError(`series ${i} has ${s.length} returns, series 0 has ${T}`);
  });
  return T;
}

/** Stage 1's specification of asset `i`. */
export function assetSpec(options: DccOptions, i: number, count: number): DccAssetSpec {
  const m = options.models;
  if (!Array.isArray(m)) return (m as DccAssetSpec | undefined) ?? {};
  if (m.length !== count) throw new GarchInputError(`${m.length} model specifications for ${count} series`);
  return m[i];
}

/** DCC-GARCH by two-stage maximum likelihood, in this thread. `fitDccInWorkers` runs stage 1 on the pool. */
export function fitDcc(series: readonly ArrayLike<number>[], options: DccOptions = {}): DccFit {
  checkSeries(series);
  const fits = series.map((s, i) => fitGarch(s, assetSpec(options, i, series.length)));
  return dccFromFits(series, fits);
}

/** Stage 2 on stage 1's fits. */
export function dccFromFits(series: readonly ArrayLike<number>[], assets: readonly GarchFit[]): DccFit {
  const T = checkSeries(series);
  const z = series.map((r, i) => {
    const s = assets[i].conditionalVariance;
    const mu = assets[i].params.mu;
    const out = new Float64Array(T);
    for (let t = 0; t < T; t += 1) out[t] = (r[t] - mu) / Math.sqrt(s[t]);
    return out;
  });
  const stage2 = fitCorrelation(z);
  return {
    assets,
    ...stage2,
    converged: stage2.converged && assets.every((f) => f.converged),
    logLikelihood: assets.reduce((a, f) => a + f.logLikelihood, 0) + stage2.correlationLogLikelihood,
  };
}

const logistic = (x: number) => 1 / (1 + Math.exp(-x));
const logit = (p: number) => Math.log(p / (1 - p));

/** (a, b) from the unconstrained x: a + b = σ(x0), a = (a + b)·σ(x1). */
const weightsOf = (x: readonly number[]) => {
  const P = logistic(x[0]);
  const a = P * logistic(x[1]);
  return { a, b: P - a };
};

const START_DCC: readonly [number, number][] = [
  [0.01, 0.97],
  [0.03, 0.95],
  [0.05, 0.9],
  [0.1, 0.8],
];

/**
 * Stage 2 alone, on standardised residuals (one array per asset): (a, b) by
 * BFGS over central-difference gradients from four starts, the best kept.
 */
export function fitCorrelation(z: readonly Float64Array[]): Omit<DccFit, 'assets' | 'logLikelihood'> {
  const N = z.length;
  const T = z[0].length;
  const target = secondMoment(z);
  const objective = (x: readonly number[]) => {
    const { a, b } = weightsOf(x);
    return -correlationRecursion(z, target, a, b).logLikelihood / T;
  };
  const gradient = (x: readonly number[]) => {
    const value = objective(x);
    const g = x.map((v, i) => {
      const h = 1e-6 * Math.max(1, Math.abs(v));
      const up = x.slice();
      const down = x.slice();
      up[i] += h;
      down[i] -= h;
      return (objective(up) - objective(down)) / (2 * h);
    });
    return { value, gradient: g };
  };
  let best: { x: number[]; value: number; converged: boolean } | null = null;
  for (const [a, b] of START_DCC) {
    const run = minimise(gradient, [logit(a + b), logit(a / (a + b))]);
    if (Number.isFinite(run.value) && (!best || run.value < best.value)) best = run;
  }
  if (!best) throw new GarchInputError('no starting point gave a finite DCC likelihood');
  const { a, b } = weightsOf(best.x);
  const run = correlationRecursion(z, target, a, b, true);
  const last = new Array<number>(N);
  for (let i = 0; i < N; i += 1) last[i] = z[i][T - 1];
  return {
    a,
    b,
    converged: best.converged,
    correlationTarget: target,
    correlation: run.correlation!,
    lastQ: run.lastQ,
    lastStandardized: last,
    correlationLogLikelihood: run.logLikelihood,
  };
}

/** Q̄ = (1/T)·Σ z_t z_tᵀ. */
function secondMoment(z: readonly Float64Array[]): number[][] {
  const N = z.length;
  const T = z[0].length;
  const q = Array.from({ length: N }, () => new Array<number>(N).fill(0));
  for (let i = 0; i < N; i += 1)
    for (let j = 0; j <= i; j += 1) {
      let s = 0;
      for (let t = 0; t < T; t += 1) s += z[i][t] * z[j][t];
      q[i][j] = q[j][i] = s / T;
    }
  return q;
}

/**
 * The Q_t recursion over the residuals: the correlation log-likelihood (−∞
 * where an R_t is not positive definite), Q_T, and with `keep` every R_t.
 */
export function correlationRecursion(
  z: readonly Float64Array[],
  target: readonly (readonly number[])[],
  a: number,
  b: number,
  keep = false,
): { logLikelihood: number; lastQ: number[][]; correlation: Float64Array[] | null } {
  const N = z.length;
  const T = z[0].length;
  const c = 1 - a - b;
  const Q = target.map((row) => row.slice());
  const R = new Float64Array(N * N);
  const L = new Float64Array(N * N);
  const y = new Float64Array(N);
  const correlation: Float64Array[] | null = keep ? [] : null;
  let total = 0;
  for (let t = 0; t < T; t += 1) {
    if (t > 0)
      for (let i = 0; i < N; i += 1)
        for (let j = 0; j <= i; j += 1)
          Q[i][j] = Q[j][i] = c * target[i][j] + a * z[i][t - 1] * z[j][t - 1] + b * Q[i][j];
    for (let i = 0; i < N; i += 1)
      for (let j = 0; j < N; j += 1) R[i * N + j] = i === j ? 1 : Q[i][j] / Math.sqrt(Q[i][i] * Q[j][j]);
    correlation?.push(R.slice());
    const logDet = cholesky(R, L, N);
    if (!Number.isFinite(logDet)) return { logLikelihood: -Infinity, lastQ: Q, correlation };
    // zᵀR⁻¹z = |L⁻¹z|²
    let quad = 0;
    let norm = 0;
    for (let i = 0; i < N; i += 1) {
      let v = z[i][t];
      for (let k = 0; k < i; k += 1) v -= L[i * N + k] * y[k];
      y[i] = v / L[i * N + i];
      quad += y[i] * y[i];
      norm += z[i][t] * z[i][t];
    }
    total += -0.5 * (logDet + quad - norm);
  }
  return { logLikelihood: total, lastQ: Q, correlation };
}

/** L with LLᵀ = A (row-major, lower triangle filled); returns ln|A|, NaN when A is not positive definite. */
function cholesky(A: Float64Array, L: Float64Array, N: number): number {
  let logDet = 0;
  for (let i = 0; i < N; i += 1)
    for (let j = 0; j <= i; j += 1) {
      let s = A[i * N + j];
      for (let k = 0; k < j; k += 1) s -= L[i * N + k] * L[j * N + k];
      if (i === j) {
        if (!(s > 0)) return NaN;
        L[i * N + i] = Math.sqrt(s);
        logDet += Math.log(s);
      } else L[i * N + j] = s / L[j * N + j];
    }
  return logDet;
}

/** BFGS, a non-finite value read as a step off the valid region. */
const minimise = (f: (x: readonly number[]) => { value: number; gradient: number[] }, x0: number[]) =>
  bfgs((x) => {
    const out = f(x);
    return Number.isFinite(out.value) ? out : { value: Infinity, gradient: x.map(() => NaN) };
  }, x0);

/** R from Q: diag(Q)^{−½}·Q·diag(Q)^{−½}. */
function normalise(Q: readonly (readonly number[])[]): number[][] {
  return Q.map((row, i) => row.map((q, j) => (i === j ? 1 : q / Math.sqrt(Q[i][i] * Q[j][j]))));
}

/**
 * H_{T+1} … H_{T+h}. R_{T+1} is exact, from z_T and Q_T; further steps use
 * the approximation of Engle and Sheppard (2001) that rmgarch uses,
 * R_{T+k} = (1 − (a+b)^{k−1})·R̄ + (a+b)^{k−1}·R_{T+1}, R̄ the normalised Q̄.
 * Each asset's variance is its own `forecastVariance`.
 */
export function dccForecast(fit: DccFit, horizon: number): DccForecast {
  const variance = fit.assets.map((f) => forecastVariance(f, horizon));
  const N = fit.assets.length;
  const { a, b } = fit;
  const zT = fit.lastStandardized;
  const Q1 = fit.lastQ.map((row, i) =>
    row.map((q, j) => (1 - a - b) * fit.correlationTarget[i][j] + a * zT[i] * zT[j] + b * q),
  );
  const R1 = normalise(Q1);
  const Rbar = normalise(fit.correlationTarget);
  const correlation: number[][][] = [];
  const covariance: number[][][] = [];
  let power = 1; // (a + b)^{k−1}
  for (let k = 0; k < horizon; k += 1) {
    const R = k === 0 ? R1 : R1.map((row, i) => row.map((r, j) => (1 - power) * Rbar[i][j] + power * r));
    correlation.push(R);
    covariance.push(
      Array.from({ length: N }, (_, i) =>
        Array.from({ length: N }, (_, j) => R[i][j] * Math.sqrt(variance[i][k] * variance[j][k])),
      ),
    );
    power *= a + b;
  }
  return { covariance, correlation, variance };
}

/** wᵀ·H·w: the variance of a portfolio with weights w on assets whose covariance is H. */
export function portfolioVariance(covariance: readonly (readonly number[])[], weights: readonly number[]): number {
  const N = covariance.length;
  if (weights.length !== N) throw new RangeError(`${weights.length} weights for ${N} assets`);
  let v = 0;
  for (let i = 0; i < N; i += 1) for (let j = 0; j < N; j += 1) v += weights[i] * covariance[i][j] * weights[j];
  return v;
}

/** What `portfolioTailRisk` reads a portfolio's VaR and ES from. */
export type PortfolioTailRiskRequest = {
  readonly dcc: DccFit;
  /** One per asset: the portfolio return is Σ wᵢ·rᵢ. */
  readonly weights: readonly number[];
  /** Days of the cumulative return (default 1). */
  readonly horizon?: number;
  readonly level: number;
  /** Simulated paths, when the answer is simulated (default 50 000). */
  readonly paths?: number;
  /** The simulation's seed (default 1). */
  readonly seed?: number;
  /** `simulated` simulates even where the closed form applies; `normal` refuses where it does not. */
  readonly method?: 'normal' | 'simulated';
};

/** A portfolio's VaR and ES, and how they were computed. */
export type PortfolioTailRisk = TailRisk & {
  /** `normal`: closed form, one day with every asset normal; `simulated` otherwise. */
  readonly method: 'normal' | 'simulated';
};

/**
 * VaR and ES of the portfolio's cumulative return over `horizon` days, as
 * positive losses. One day with every asset normal: closed form, the
 * portfolio return N(Σwᵢμᵢ, wᵀH_{T+1}w). Otherwise simulated: each day draws
 * every asset's own innovations (unit variance), correlates them through the
 * Cholesky factor of R_t, and runs every recursion forward — each asset's
 * variance and Q. Under non-normal innovations the correlated draws keep their
 * correlation but not exactly each asset's marginal.
 */
export function portfolioTailRisk(request: PortfolioTailRiskRequest): PortfolioTailRisk {
  const { dcc, weights, level } = request;
  const horizon = request.horizon ?? 1;
  const N = dcc.assets.length;
  if (weights.length !== N) throw new RangeError(`${weights.length} weights for ${N} assets`);
  if (!(Number.isInteger(horizon) && horizon >= 1))
    throw new RangeError(`horizon must be an integer ≥ 1, got ${horizon}`);
  const mean = dcc.assets.reduce((m, f, i) => m + weights[i] * f.params.mu, 0);
  const closed = horizon === 1 && dcc.assets.every((f) => f.params.nu === undefined);
  if (request.method === 'normal' && !closed)
    throw new RangeError('the closed form needs one day and every asset normal');
  if (closed && request.method !== 'simulated') {
    const variance = portfolioVariance(dccForecast(dcc, 1).covariance[0], weights);
    return { ...tailRisk({ mean, variance, level }), method: 'normal' };
  }
  const paths = request.paths ?? 50_000;
  if (!(Number.isInteger(paths) && paths >= 2)) throw new RangeError(`paths must be an integer ≥ 2, got ${paths}`);
  if (!(level > 0 && level < 0.5)) throw new RangeError(`level must lie in (0, 0.5), got ${level}`);
  if (level * paths < 1) throw new RangeError('too few paths for this level');
  const seed = request.seed ?? 1;
  const draws = dcc.assets.map((f, i) =>
    innovations(f.params.nu, f.params.lambda, (seed + Math.imul(i, 0x9e3779b9)) >>> 0),
  );
  const first = dcc.assets.map((f) => forecastVariance(f, 1)[0]);
  const { a, b } = dcc;
  const c = 1 - a - b;
  const target = dcc.correlationTarget;
  const zT = dcc.lastStandardized;
  const Q1 = dcc.lastQ.map((row, i) => row.map((q, j) => c * target[i][j] + a * zT[i] * zT[j] + b * q));
  const sums = new Float64Array(paths);
  const s = new Float64Array(N);
  const eta = new Float64Array(N);
  const zt = new Float64Array(N);
  const R = new Float64Array(N * N);
  const L = new Float64Array(N * N);
  for (let path = 0; path < paths; path += 1) {
    s.set(first);
    const Q = Q1.map((row) => row.slice());
    let sum = 0;
    for (let t = 0; t < horizon; t += 1) {
      for (let i = 0; i < N; i += 1)
        for (let j = 0; j < N; j += 1) R[i * N + j] = i === j ? 1 : Q[i][j] / Math.sqrt(Q[i][i] * Q[j][j]);
      cholesky(R, L, N);
      for (let i = 0; i < N; i += 1) eta[i] = draws[i]();
      for (let i = 0; i < N; i += 1) {
        let v = 0;
        for (let k = 0; k <= i; k += 1) v += L[i * N + k] * eta[k];
        zt[i] = v;
      }
      for (let i = 0; i < N; i += 1) {
        const f = dcc.assets[i];
        const e = Math.sqrt(s[i]) * zt[i];
        sum += weights[i] * (f.params.mu + e);
        s[i] = nextVariance(f.model, f.params, s[i], e, zt[i]);
      }
      for (let i = 0; i < N; i += 1)
        for (let j = 0; j <= i; j += 1) Q[i][j] = Q[j][i] = c * target[i][j] + a * zt[i] * zt[j] + b * Q[i][j];
    }
    sums[path] = sum;
  }
  sums.sort();
  const k = Math.ceil(level * paths);
  let tail = 0;
  for (let i = 0; i < k; i += 1) tail += sums[i];
  return { valueAtRisk: -sums[k - 1], expectedShortfall: -tail / k, method: 'simulated' };
}
