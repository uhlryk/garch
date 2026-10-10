import { egarchLogLikelihood, egarchLogLikelihoodGradient } from './egarch.js';
import { robustStandardErrors } from './inference.js';
import {
  backcast,
  type Distribution,
  type GarchModel,
  type GarchParams,
  logLikelihood,
  logLikelihoodGradient,
  persistence,
} from './model.js';
import { bfgs } from './optimize.js';

export class GarchInputError extends Error {}

export type FitOptions = {
  readonly model?: GarchModel;
  /** The innovations: normal (default, `arch`'s) or Student-t with ν fitted. */
  readonly dist?: Distribution;
  /** The lowest ν a t fit may take (default `T_NU_MIN`); must exceed 2. */
  readonly nuMin?: number;
  /** A previous fit to start from as well — the walk-forward hands each origin the last one. */
  readonly warmStart?: GarchParams | null;
  /** Also compute robust (Bollerslev–Wooldridge) standard errors and two-sided p-values. */
  readonly inference?: boolean;
};

export type Inference = {
  readonly standardErrors: GarchParams;
  readonly pValues: GarchParams;
};

export type GarchFit = {
  readonly model: GarchModel;
  readonly converged: boolean;
  readonly params: GarchParams;
  readonly persistence: number;
  readonly logLikelihood: number;
  /** s_t for every observation, in the returns' units squared. */
  readonly conditionalVariance: Float64Array;
  /** e_T, the last residual. */
  readonly lastResidual: number;
  /** The pre-sample variance the recursion started from — its log for EGARCH. */
  readonly backcast: number;
  readonly inference: Inference | null;
  /**
   * For t innovations: the ν the likelihood wanted lies beyond a bound, and
   * the fit is the constrained optimum with ν held there. null when ν is
   * interior, and for normal innovations.
   */
  readonly nuBound: 'lower' | 'upper' | null;
  /**
   * Every distinct optimum the starts reached, best first — `modes[0]` is the
   * fit. More than one means the likelihood is multimodal on this series.
   */
  readonly modes: readonly GarchMode[];
};

/** One local maximum of the likelihood, reached from at least one start. */
export type GarchMode = {
  readonly params: GarchParams;
  readonly persistence: number;
  readonly logLikelihood: number;
  readonly converged: boolean;
  /** The previous origin's optimum reached it (a walk-forward's warm start). */
  readonly fromWarmStart: boolean;
  /** How many of the starts ended here. */
  readonly starts: number;
};

/** The persistence levels one start is taken at — the likelihood's modes sit at different ones. */
const START_PERSISTENCE = [0.5, 0.7, 0.9, 0.98, 0.995, 0.999];
const START_ALPHA = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4];
const START_GAMMA = [0, 0.05, 0.2];
/**
 * The default floor on ν. `arch` allows 2.05, but a t variance is
 * scale²·ν/(ν − 2): near 2 it is barely defined and its forecast explodes —
 * fitted freely, AAPL and IOVA came out at ν̂ ≈ 2.1–2.5, most of their early
 * origins not converging and the forecast worse than naive. At 4 the kurtosis is
 * still infinite but the variance forecast is stable.
 */
export const T_NU_MIN = 4;
const NU_MAX = 500;
const START_NU = 8;
/** How close to a bound ν must sit to be held there. */
const NU_AT_BOUND = 1e-3;

const logistic = (z: number) => 1 / (1 + Math.exp(-z));
const logit = (p: number) => Math.log(p / (1 - p));
const inside = (p: number) => Math.min(Math.max(p, 1e-9), 1 - 1e-9);

/**
 * Maximum likelihood by BFGS in an unconstrained space where every point is a
 * valid, covariance-stationary model:
 *
 *     ω = e^{x1}·var(r)   P = σ(x2) = α + γ/2 + β   β = P·σ(x3)
 *     q = P − β = α + γ/2   γ = 2q·tanh(x4)   α = q − γ/2
 *
 * so ω > 0, β ≥ 0, α ≥ 0, α + γ ≥ 0 and P < 1 hold by construction. One start
 * per persistence level (the best α / γ at that level), plus the warm start;
 * the highest likelihood wins. The GARCH(1,1) likelihood is multimodal on
 * short or jumpy series, and a single start stops in whichever mode is nearest.
 */
export function fitGarch(returns: ArrayLike<number>, options: FitOptions = {}): GarchFit {
  const model = options.model ?? 'garch';
  if (model === 'egarch') return fitEgarch(returns, options);
  const asymmetric = model === 'gjr';
  const studentT = options.dist === 't';
  const nuMin = options.nuMin ?? T_NU_MIN;
  if (studentT && !(nuMin > 2 && nuMin < NU_MAX))
    throw new RangeError(`nuMin must lie in (2, ${NU_MAX}), got ${nuMin}`);
  const nuSpan = NU_MAX - nuMin;
  const n = returns.length;
  const { mean, variance } = moments(returns);
  const bc = backcast(returns, mean);
  const sig2 = new Float64Array(n);

  // The unconstrained space; `heldNu` removes ν from it (a fit held on its bound).
  const spaceOf = (heldNu: number | null) => {
    const freeNu = studentT && heldNu === null;
    const toParams = (x: readonly number[]): GarchParams => {
      const omega = Math.exp(x[1]) * variance;
      const P = logistic(x[2]);
      const beta = P * logistic(x[3]);
      const q = P - beta;
      const gamma = asymmetric ? 2 * q * Math.tanh(x[4]) : 0;
      const p = { mu: x[0], omega, alpha: q - gamma / 2, gamma, beta };
      if (!studentT) return p;
      return { ...p, nu: freeNu ? nuMin + nuSpan * logistic(x[x.length - 1]) : heldNu! };
    };
    const toX = (p: GarchParams): number[] => {
      const P = inside(persistence(p));
      const x = [p.mu, Math.log(p.omega / variance), logit(P), logit(inside(p.beta / P))];
      if (asymmetric) {
        const q = P - p.beta;
        x.push(Math.atanh(Math.min(Math.max(p.gamma / (2 * q), -1 + 1e-9), 1 - 1e-9)));
      }
      if (freeNu) x.push(logit(inside(((p.nu ?? START_NU) - nuMin) / nuSpan)));
      return x;
    };
    // negative mean log-likelihood and its gradient in x (chain rule through toParams)
    const objective = (x: readonly number[]) => {
      const p = toParams(x);
      const { value, gradient: g } = logLikelihoodGradient(returns, p, bc);
      if (!Number.isFinite(value)) return { value: Infinity, gradient: x.map(() => NaN) };
      const P = logistic(x[2]);
      const s3 = logistic(x[3]);
      const dP2 = P * (1 - P);
      const dBeta2 = s3 * dP2;
      const dBeta3 = P * s3 * (1 - s3);
      const dq2 = dP2 - dBeta2;
      const dq3 = -dBeta3;
      const th = asymmetric ? Math.tanh(x[4]) : 0;
      const q = P - P * s3;
      // α = q(1 − th), γ = 2q·th
      const gx = [
        g[0],
        g[1] * p.omega,
        g[2] * (1 - th) * dq2 + g[3] * 2 * th * dq2 + g[4] * dBeta2,
        g[2] * (1 - th) * dq3 + g[3] * 2 * th * dq3 + g[4] * dBeta3,
      ];
      if (asymmetric) gx.push(-g[2] * q * (1 - th * th) + g[3] * 2 * q * (1 - th * th));
      if (freeNu) {
        const sn = logistic(x[x.length - 1]);
        gx.push(g[5] * nuSpan * sn * (1 - sn));
      }
      return { value: -value / n, gradient: gx.map((v) => -v / n) };
    };
    return { toParams, toX, objective };
  };
  const { toX, toParams, objective } = spaceOf(null);

  const starts: { x: number[]; warm: boolean }[] = [];
  for (const level of START_PERSISTENCE) {
    let best: { value: number; p: GarchParams } | null = null;
    for (const alpha of START_ALPHA)
      for (const gamma of asymmetric ? START_GAMMA : [0]) {
        const beta = level - alpha - gamma / 2;
        if (beta <= 0) continue;
        const p = {
          mu: mean,
          omega: variance * (1 - level),
          alpha,
          gamma,
          beta,
          ...(studentT ? { nu: START_NU } : {}),
        };
        const value = logLikelihood(returns, p, bc, sig2);
        if (Number.isFinite(value) && (!best || value > best.value)) best = { value, p };
      }
    if (best) starts.push({ x: toX(best.p), warm: false });
  }
  if (options.warmStart) {
    const w = options.warmStart;
    const nu = studentT ? Math.min(Math.max(w.nu ?? START_NU, nuMin), NU_MAX) : undefined;
    const warm = { ...(asymmetric ? w : { ...w, gamma: 0 }), nu };
    if (warm.omega > 0 && warm.alpha >= 0 && warm.beta >= 0 && warm.alpha + warm.gamma >= 0 && persistence(warm) < 1)
      starts.push({ x: toX(warm), warm: true });
  }

  const modes: { params: GarchParams; value: number; converged: boolean; fromWarmStart: boolean; starts: number }[] =
    [];
  for (const start of starts) {
    const run = bfgs(objective, start.x);
    if (!Number.isFinite(run.value)) continue;
    const params = toParams(run.x);
    const same = modes.find((m) => sameMode(m.params, m.value, params, run.value));
    if (!same) modes.push({ params, value: run.value, converged: run.converged, fromWarmStart: start.warm, starts: 1 });
    else {
      same.starts += 1;
      same.fromWarmStart ||= start.warm;
      if (run.value < same.value) Object.assign(same, { params, value: run.value, converged: run.converged });
    }
  }
  if (!modes.length) throw new GarchInputError('no starting point gave a finite likelihood');
  modes.sort((a, b) => a.value - b.value);
  // ν pushed against a bound: the optimum is the one with ν held there, where
  // the other parameters can converge (free, the logit of ν runs off to ±∞).
  let nuBound: 'lower' | 'upper' | null = null;
  if (studentT) {
    const nu = modes[0].params.nu!;
    nuBound = nu - nuMin < NU_AT_BOUND ? 'lower' : NU_MAX - nu < NU_AT_BOUND ? 'upper' : null;
    if (nuBound) {
      const held = spaceOf(nuBound === 'lower' ? nuMin : NU_MAX);
      const run = bfgs(held.objective, held.toX(modes[0].params));
      if (Number.isFinite(run.value))
        Object.assign(modes[0], { params: held.toParams(run.x), value: run.value, converged: run.converged });
    }
  }
  const best = modes[0];
  const params = best.params;
  const ll = logLikelihood(returns, params, bc, sig2);
  return {
    model,
    converged: best.converged,
    params,
    persistence: persistence(params),
    logLikelihood: ll,
    conditionalVariance: sig2,
    lastResidual: returns[n - 1] - params.mu,
    backcast: bc,
    inference: options.inference ? robustStandardErrors(returns, params, bc, asymmetric, studentT && !nuBound) : null,
    nuBound,
    modes: modes.map((m) => ({
      params: m.params,
      persistence: persistence(m.params),
      logLikelihood: -m.value * n,
      converged: m.converged,
      fromWarmStart: m.fromWarmStart,
      starts: m.starts,
    })),
  };
}

/** The returns' mean and variance, refusing what cannot be fitted. */
function moments(returns: ArrayLike<number>, minimum = 10): { mean: number; variance: number } {
  const n = returns.length;
  if (n < minimum) throw new GarchInputError(`a GARCH fit needs at least ${minimum} returns, got ${n}`);
  let mean = 0;
  for (let i = 0; i < n; i += 1) {
    if (!Number.isFinite(returns[i])) throw new GarchInputError(`return ${i} is not a finite number`);
    mean += returns[i];
  }
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i += 1) variance += (returns[i] - mean) ** 2;
  variance /= n;
  if (!(variance > 0)) throw new GarchInputError('the returns have no variance to model');
  return { mean, variance };
}

const START_EGARCH_ALPHA = [0.05, 0.1, 0.2, 0.4];
const START_EGARCH_GAMMA = [0, -0.05, -0.15];

/**
 * EGARCH(1,1,1) by maximum likelihood (`egarch.ts`), normal innovations only.
 * The unconstrained space:
 *
 *     β = σ(x2) ∈ (0, 1)   ω = (1 − β)·(ln var(r) + x1)   α = x3   γ = x4
 *
 * so x1 is the unconditional log-variance's distance from the sample's, and
 * the log-variance is stationary by construction (`arch` lets β run in
 * (−1, 1); a negative β — tomorrow's log-variance mirroring today's — is
 * left out). Starts as for GARCH: one per persistence level, plus the warm
 * start; every optimum reached is a mode. `persistence` is β.
 */
function fitEgarch(returns: ArrayLike<number>, options: FitOptions): GarchFit {
  if (options.dist === 't')
    throw new GarchInputError(
      'EGARCH takes normal innovations only: under Student-t the expected variance two steps ahead is infinite',
    );
  const n = returns.length;
  const { mean, variance } = moments(returns);
  const lnVar = Math.log(variance);
  const lnBc = Math.log(backcast(returns, mean));
  const sig2 = new Float64Array(n);
  const toParams = (x: readonly number[]): GarchParams => {
    const beta = logistic(x[2]);
    return { mu: x[0], omega: (1 - beta) * (lnVar + x[1]), alpha: x[3], gamma: x[4], beta };
  };
  const toX = (p: GarchParams): number[] => {
    const beta = inside(p.beta);
    return [p.mu, p.omega / (1 - beta) - lnVar, logit(beta), p.alpha, p.gamma];
  };
  const objective = (x: readonly number[]) => {
    const p = toParams(x);
    const { value, gradient: g } = egarchLogLikelihoodGradient(returns, p, lnBc);
    if (!Number.isFinite(value) || g.some((v) => !Number.isFinite(v)))
      return { value: Infinity, gradient: x.map(() => NaN) };
    const beta = p.beta;
    const level = lnVar + x[1];
    const gx = [g[0], g[1] * (1 - beta), (g[4] - g[1] * level) * beta * (1 - beta), g[2], g[3]];
    return { value: -value / n, gradient: gx.map((v) => -v / n) };
  };
  const starts: { x: number[]; warm: boolean }[] = [];
  for (const level of START_PERSISTENCE) {
    let best: { value: number; p: GarchParams } | null = null;
    for (const alpha of START_EGARCH_ALPHA)
      for (const gamma of START_EGARCH_GAMMA) {
        const p = { mu: mean, omega: (1 - level) * lnVar, alpha, gamma, beta: level };
        const value = egarchLogLikelihood(returns, p, lnBc, sig2);
        if (Number.isFinite(value) && (!best || value > best.value)) best = { value, p };
      }
    if (best) starts.push({ x: toX(best.p), warm: false });
  }
  const w = options.warmStart;
  if (w && w.beta > 0 && w.beta < 1 && [w.mu, w.omega, w.alpha, w.gamma].every(Number.isFinite))
    starts.push({ x: toX(w), warm: true });
  const modes: { params: GarchParams; value: number; converged: boolean; fromWarmStart: boolean; starts: number }[] =
    [];
  for (const start of starts) {
    const run = bfgs(objective, start.x);
    if (!Number.isFinite(run.value)) continue;
    const params = toParams(run.x);
    const same = modes.find(
      (m) =>
        Math.abs(m.value - run.value) < 1e-6 &&
        Math.abs(m.params.beta - params.beta) < 1e-3 &&
        Math.abs(m.params.alpha - params.alpha) < 1e-3 &&
        Math.abs(m.params.gamma - params.gamma) < 1e-3,
    );
    if (!same) modes.push({ params, value: run.value, converged: run.converged, fromWarmStart: start.warm, starts: 1 });
    else {
      same.starts += 1;
      same.fromWarmStart ||= start.warm;
      if (run.value < same.value) Object.assign(same, { params, value: run.value, converged: run.converged });
    }
  }
  if (!modes.length) throw new GarchInputError('no starting point gave a finite likelihood');
  modes.sort((a, b) => a.value - b.value);
  const params = modes[0].params;
  const ll = egarchLogLikelihood(returns, params, lnBc, sig2);
  return {
    model: 'egarch',
    converged: modes[0].converged,
    params,
    persistence: params.beta,
    logLikelihood: ll,
    conditionalVariance: sig2,
    lastResidual: returns[n - 1] - params.mu,
    backcast: lnBc,
    inference: options.inference
      ? robustStandardErrors(returns, params, lnBc, true, false, egarchLogLikelihoodGradient)
      : null,
    nuBound: null,
    modes: modes.map((m) => ({
      params: m.params,
      persistence: m.params.beta,
      logLikelihood: -m.value * n,
      converged: m.converged,
      fromWarmStart: m.fromWarmStart,
      starts: m.starts,
    })),
  };
}

/** Two runs stopped at the same optimum: the same likelihood and the same dynamics. */
function sameMode(a: GarchParams, aValue: number, b: GarchParams, bValue: number): boolean {
  return (
    Math.abs(aValue - bValue) < 1e-6 &&
    Math.abs(persistence(a) - persistence(b)) < 1e-3 &&
    Math.abs(a.alpha - b.alpha) < 1e-3 &&
    Math.abs(a.gamma - b.gamma) < 1e-3
  );
}
