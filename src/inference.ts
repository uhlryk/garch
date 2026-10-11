import type { Inference } from './fit.js';
import { type GarchParams, logLikelihoodGradient } from './model.js';

/** The log-likelihood and its gradient over (μ, ω, α, γ, β[, ν]), per-observation scores when asked. */
export type LikelihoodGradient = (
  returns: ArrayLike<number>,
  params: GarchParams,
  bc: number,
  scores?: Float64Array[],
) => { value: number; gradient: number[] };

const KEYS = ['mu', 'omega', 'alpha', 'gamma', 'beta', 'nu', 'lambda', 'delta'] as const;

/** erfc(x) for x ≥ 0 — Numerical Recipes' Chebyshev fit, relative error < 1.2e-7. */
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  // biome-ignore format: the Horner polynomial reads best on one line
  return (
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    )
  );
}

/** Φ(x), the standard normal CDF, each side computed from its own tail. */
export function normalCdf(x: number): number {
  return x < 0 ? 0.5 * erfc(-x / Math.SQRT2) : 1 - 0.5 * erfc(x / Math.SQRT2);
}

/** Two-sided normal p-value of a z statistic, computed in the tail (no 1 − Φ cancellation). */
export function twoSidedPValue(z: number): number {
  return erfc(Math.abs(z) / Math.SQRT2);
}

export function invert(matrix: readonly (readonly number[])[]): number[][] | null {
  const n = matrix.length;
  const a = matrix.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c += 1) {
    let p = c;
    for (let r = c + 1; r < n; r += 1) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    if (!(Math.abs(a[p][c]) > 1e-300)) return null;
    [a[c], a[p]] = [a[p], a[c]];
    const d = a[c][c];
    for (let j = 0; j < 2 * n; j += 1) a[c][j] /= d;
    for (let r = 0; r < n; r += 1)
      if (r !== c) {
        const f = a[r][c];
        if (f) for (let j = 0; j < 2 * n; j += 1) a[r][j] -= f * a[c][j];
      }
  }
  return a.map((row) => row.slice(n));
}

/** Which parameters were fitted: ω (unless targeted), γ (asymmetric models), ν (t or skewed t, interior), λ (skewed t), μ (a constant mean). */
export type FreeParameters = {
  /** False under variance targeting: ω is tied, not estimated. */
  readonly omega: boolean;
  readonly gamma: boolean;
  readonly nu: boolean;
  readonly lambda: boolean;
  readonly mu: boolean;
  /** GARCH-X's δ; absent, false. */
  readonly delta?: boolean;
};

/**
 * Bollerslev–Wooldridge robust covariance, `arch`'s default `cov_type="robust"`:
 * H⁻¹·(Σ s_t s_tᵀ)·H⁻¹, with the per-observation scores analytic and the
 * Hessian a central difference of the analytic gradient. A parameter on the
 * boundary of the space (α or β at 0) gets a standard error without meaning;
 * it is returned as computed, as `arch` does. With `freeNu`, ν is one of the
 * parameters (t innovations, ν interior); its p-value (against ν = 0) means
 * nothing and is NaN. A parameter not fitted (`free`) gets NaN.
 */
export function robustStandardErrors(
  returns: ArrayLike<number>,
  params: GarchParams,
  bc: number,
  free: FreeParameters,
  likelihood: LikelihoodGradient = logLikelihoodGradient,
): Inference | null {
  const n = returns.length;
  const keys = KEYS.filter(
    (k) =>
      (k !== 'omega' || free.omega) &&
      (k !== 'gamma' || free.gamma) &&
      (k !== 'nu' || free.nu) &&
      (k !== 'lambda' || free.lambda) &&
      (k !== 'mu' || free.mu) &&
      (k !== 'delta' || free.delta === true),
  );
  const index = keys.map((k) => KEYS.indexOf(k));
  const m = keys.length;
  const scores = KEYS.map(() => new Float64Array(n));
  likelihood(returns, params, bc, scores);
  const opg = index.map((i) =>
    index.map((j) => {
      let s = 0;
      for (let t = 0; t < n; t += 1) s += scores[i][t] * scores[j][t];
      return s;
    }),
  );
  const hessian = Array.from({ length: m }, () => new Array<number>(m).fill(0));
  for (let a = 0; a < m; a += 1) {
    const key = keys[a];
    const value = params[key]!;
    const h = Math.max(Math.abs(value) * 1e-5, 1e-8);
    const up = likelihood(returns, { ...params, [key]: value + h }, bc).gradient;
    const down = likelihood(returns, { ...params, [key]: value - h }, bc).gradient;
    for (let b = 0; b < m; b += 1) hessian[a][b] = -(up[index[b]] - down[index[b]]) / (2 * h);
  }
  // symmetrise the finite-difference information matrix
  for (let a = 0; a < m; a += 1)
    for (let b = a + 1; b < m; b += 1) hessian[a][b] = hessian[b][a] = (hessian[a][b] + hessian[b][a]) / 2;
  const inverse = invert(hessian);
  if (!inverse) return null;
  const se: Record<string, number> = { mu: NaN, omega: NaN, alpha: NaN, gamma: NaN, beta: NaN };
  if (params.nu !== undefined) se.nu = NaN;
  if (params.lambda !== undefined) se.lambda = NaN;
  if (params.delta !== undefined) se.delta = NaN;
  const pv: Record<string, number> = { ...se };
  for (let a = 0; a < m; a += 1) {
    let v = 0;
    for (let i = 0; i < m; i += 1) for (let j = 0; j < m; j += 1) v += inverse[a][i] * opg[i][j] * inverse[j][a];
    const key = keys[a];
    se[key] = v > 0 ? Math.sqrt(v) : NaN;
    pv[key] = v > 0 && key !== 'nu' ? twoSidedPValue(params[key]! / se[key]) : NaN;
  }
  return { standardErrors: se as GarchParams, pValues: pv as GarchParams };
}
