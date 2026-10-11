import { EGARCH_ABS_MEAN } from './egarch.js';
import { GarchInputError } from './fit.js';
import { type GarchModel, type GarchParams, persistence } from './model.js';
import { skewConstants, skewDraw } from './skewt.js';

/**
 * Paths of the model forward from given parameters: what a fan chart, a
 * quantile of the h-step return, or a check of a forecast against its own
 * Monte Carlo needs.
 */

export type SimulationRequest = {
  readonly model: GarchModel;
  /** With `nu`, the innovations are Student-t standardised to unit variance, with `lambda` too Hansen's skewed t (garch and gjr only). */
  readonly params: GarchParams;
  /**
   * Where the recursion stands: s and e of the last observed return. With
   * both, step 1's variance is the one-step forecast; without, the paths
   * start at the unconditional variance. A `FitAt` spreads into the request.
   */
  readonly lastVariance?: number;
  readonly lastResidual?: number;
  readonly steps: number;
  readonly paths: number;
  /** The same seed gives the same paths, on any platform. */
  readonly seed: number;
};

export type Simulation = {
  /** One per path: r_1 … r_steps. */
  readonly returns: Float64Array[];
  /** One per path: s_1 … s_steps, each the variance r_t was drawn with. */
  readonly variance: Float64Array[];
};

/** xoshiro128** seeded through splitmix32: a uniform stream on [0, 1). */
function uniforms(seed: number): () => number {
  let x = seed >>> 0;
  const split = () => {
    x = (x + 0x9e3779b9) >>> 0;
    let z = x;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
    return (z ^ (z >>> 16)) >>> 0;
  };
  let a = split();
  let b = split();
  let c = split();
  let d = split();
  return () => {
    const out = Math.imul(rotl(Math.imul(b, 5), 7), 9) >>> 0;
    const t = b << 9;
    c ^= a;
    d ^= b;
    b ^= c;
    a ^= d;
    c ^= t;
    d = rotl(d, 11);
    return out / 4294967296;
  };
}

const rotl = (v: number, k: number) => (v << k) | (v >>> (32 - k));

/** Standard normals by Box–Muller, both of each pair used. */
function normals(uniform: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const z = spare;
      spare = null;
      return z;
    }
    const radius = Math.sqrt(-2 * Math.log(1 - uniform()));
    const angle = 2 * Math.PI * uniform();
    spare = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  };
}

/** Gamma(shape, 1) by Marsaglia–Tsang; shape < 1 boosted by U^{1/shape}. */
function gammaVariate(shape: number, normal: () => number, uniform: () => number): number {
  if (shape < 1) return gammaVariate(shape + 1, normal, uniform) * (1 - uniform()) ** (1 / shape);
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    const x = normal();
    const v = (1 + c * x) ** 3;
    if (v <= 0) continue;
    const u = 1 - uniform();
    if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v;
  }
}

/** Unit-variance innovations: normal, Student-t with ν scaled by √((ν − 2)/ν), or Hansen's skewed t. */
function innovations(nu: number | undefined, lambda: number | undefined, seed: number): () => number {
  const uniform = uniforms(seed);
  const normal = normals(uniform);
  if (nu === undefined) return normal;
  const scale = Math.sqrt((nu - 2) / nu);
  const standardT = () => (normal() / Math.sqrt((2 * gammaVariate(nu / 2, normal, uniform)) / nu)) * scale;
  if (lambda === undefined) return standardT;
  const { a, b } = skewConstants(nu, lambda);
  return () => skewDraw(lambda, a, b, uniform(), standardT());
}

/**
 * `paths` paths of `steps` returns each, from one seeded stream. GARCH and
 * GJR run s_t = ω + α·e² + γ·e²·[e < 0] + β·s; EGARCH its log recursion. An
 * EGARCH with Student-t innovations is refused, as `fitGarch` refuses it: its
 * expected variance is infinite past one step.
 */
export function simulate(request: SimulationRequest): Simulation {
  const { model, params: p, steps, paths } = request;
  if (!(Number.isInteger(steps) && steps >= 1)) throw new RangeError(`steps must be a positive integer, got ${steps}`);
  if (!(Number.isInteger(paths) && paths >= 1)) throw new RangeError(`paths must be a positive integer, got ${paths}`);
  if (p.nu !== undefined && !(p.nu > 2)) throw new RangeError(`ν must exceed 2, got ${p.nu}`);
  if (p.lambda !== undefined && (p.nu === undefined || !(p.lambda > -1 && p.lambda < 1)))
    throw new RangeError(`λ needs ν and must lie in (−1, 1), got ${p.lambda}`);
  if (model === 'egarch' && p.nu !== undefined)
    throw new GarchInputError('EGARCH takes normal innovations only: its expected variance past one step is infinite');
  const known = request.lastVariance !== undefined;
  if (known !== (request.lastResidual !== undefined))
    throw new RangeError('lastVariance and lastResidual go together: give both or neither');
  if (known && !(request.lastVariance! > 0))
    throw new RangeError(`lastVariance must be positive, got ${request.lastVariance}`);
  const egarch = model === 'egarch';
  const gamma = model === 'garch' ? 0 : p.gamma;
  if (!egarch && !(persistence({ ...p, gamma }) < 1) && !known)
    throw new RangeError('a non-stationary model has no unconditional variance to start from');
  // s_1: the one-step forecast from where the recursion stands, or the unconditional variance
  let first: number;
  if (egarch) {
    if (known) {
      const z = request.lastResidual! / Math.sqrt(request.lastVariance!);
      first = Math.exp(
        p.omega + p.alpha * (Math.abs(z) - EGARCH_ABS_MEAN) + p.gamma * z + p.beta * Math.log(request.lastVariance!),
      );
    } else first = Math.exp(p.omega / (1 - p.beta));
  } else if (known) {
    const e = request.lastResidual!;
    first = p.omega + p.alpha * e * e + gamma * (e < 0 ? e * e : 0) + p.beta * request.lastVariance!;
  } else first = p.omega / (1 - persistence({ ...p, gamma }));

  const draw = innovations(p.nu, p.lambda, request.seed);
  const returns: Float64Array[] = [];
  const variance: Float64Array[] = [];
  for (let path = 0; path < paths; path += 1) {
    const r = new Float64Array(steps);
    const v = new Float64Array(steps);
    let s = first;
    for (let t = 0; t < steps; t += 1) {
      const z = draw();
      v[t] = s;
      r[t] = p.mu + Math.sqrt(s) * z;
      const e = r[t] - p.mu;
      s = egarch
        ? Math.exp(p.omega + p.alpha * (Math.abs(z) - EGARCH_ABS_MEAN) + p.gamma * z + p.beta * Math.log(s))
        : p.omega + p.alpha * e * e + gamma * (e < 0 ? e * e : 0) + p.beta * s;
    }
    returns.push(r);
    variance.push(v);
  }
  return { returns, variance };
}
