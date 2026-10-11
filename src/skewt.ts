import { digamma, logGamma, studentCdf, studentQuantile } from './special.js';

/**
 * Hansen's (1994) skewed Student-t, standardised to mean 0 and variance 1 —
 * `arch`'s `SkewStudent`, with η as `nu` and λ as `lambda`:
 *
 *     f(z) = b·c·(1 + ((b·z + a) / (1 − λ))² / (η − 2))^{−(η+1)/2}   for z < −a/b
 *     f(z) = b·c·(1 + ((b·z + a) / (1 + λ))² / (η − 2))^{−(η+1)/2}   for z ≥ −a/b
 *
 *     c = Γ((η+1)/2) / (√(π(η−2))·Γ(η/2)),  a = 4λc·(η−2)/(η−1),  b = √(1 + 3λ² − a²)
 *
 * η > 2, −1 < λ < 1; λ < 0 leans the mass left (a fatter loss tail), λ = 0 is
 * the standardised t. In w = (b·z + a)/(1 ∓ λ) each side is half a
 * standardised t, which every moment, quantile and draw below is built on.
 */

/** a, b, ln c and their derivatives in η and λ — what the log-likelihood and its gradient need. */
export type SkewConstants = {
  readonly a: number;
  readonly b: number;
  readonly c: number;
  readonly logC: number;
  readonly dLogCdNu: number;
  readonly dAdNu: number;
  readonly dAdLambda: number;
  readonly dBdNu: number;
  readonly dBdLambda: number;
};

export function skewConstants(nu: number, lambda: number): SkewConstants {
  const logC = logGamma((nu + 1) / 2) - logGamma(nu / 2) - 0.5 * Math.log(Math.PI * (nu - 2));
  const c = Math.exp(logC);
  const dLogCdNu = 0.5 * digamma((nu + 1) / 2) - 0.5 * digamma(nu / 2) - 0.5 / (nu - 2);
  const a = (4 * lambda * c * (nu - 2)) / (nu - 1);
  const b = Math.sqrt(1 + 3 * lambda * lambda - a * a);
  const dAdNu = 4 * lambda * ((c * dLogCdNu * (nu - 2)) / (nu - 1) + c / ((nu - 1) * (nu - 1)));
  const dAdLambda = (4 * c * (nu - 2)) / (nu - 1);
  return {
    a,
    b,
    c,
    logC,
    dLogCdNu,
    dAdNu,
    dAdLambda,
    dBdNu: (-a * dAdNu) / b,
    dBdLambda: (3 * lambda - a * dAdLambda) / b,
  };
}

/** P(W ≤ w) for a standardised t W (variance 1). */
const standardCdf = (w: number, nu: number) => studentCdf(w * Math.sqrt(nu / (nu - 2)), nu);

/** ∫_{−∞}^{x} wʲ·φ(w) dw for the standardised t density φ, j = 0, 1, 2 — closed forms, all 0 at −∞. */
function halfMoments(x: number, nu: number, c: number): [number, number, number] {
  if (x === -Infinity) return [0, 0, 0];
  if (x === Infinity) return [1, 0, 1];
  // w·φ(w) = d/dw h(w),  h(w) = −c·(η−2)/(η−1)·(1 + w²/(η−2))^{−(η−1)/2}
  const h = ((-c * (nu - 2)) / (nu - 1)) * (1 + (x * x) / (nu - 2)) ** (-(nu - 1) / 2);
  // ∫ w²φ = x·h(x) + F_{η−2}(x): by parts, the rest is the t kernel with η − 2 degrees of freedom
  return [standardCdf(x, nu), h, x * h + studentCdf(x, nu - 2)];
}

/**
 * E[zᵏ·1{z < x}] for k = 0, 1, 2: P(z < x), the partial mean, the partial
 * second moment. Each side is z = (m·w − a)/b with m = 1 ∓ λ and w half a
 * standardised t, so each is a sum of `halfMoments`.
 */
export function skewPartialMoments(x: number, nu: number, lambda: number): [number, number, number] {
  const { a, b, c } = skewConstants(nu, lambda);
  const out: [number, number, number] = [0, 0, 0];
  const add = (m: number, lo: number, hi: number) => {
    if (!(hi > lo)) return;
    const H = halfMoments(hi, nu, c);
    const L = halfMoments(lo, nu, c);
    const i0 = H[0] - L[0];
    const i1 = H[1] - L[1];
    const i2 = H[2] - L[2];
    out[0] += m * i0;
    out[1] += (m / b) * (m * i1 - a * i0);
    out[2] += (m / (b * b)) * (m * m * i2 - 2 * a * m * i1 + a * a * i0);
  };
  const edge = b * x + a; // < 0: x is on the left side
  add(1 - lambda, -Infinity, Math.min(0, edge / (1 - lambda)));
  if (edge > 0) add(1 + lambda, 0, edge / (1 + lambda));
  return out;
}

/** E[z²·1{z < 0}]: the share of the unit variance carried by down moves — ½ when λ = 0. */
export function skewDownShare(nu: number, lambda: number): number {
  return skewPartialMoments(0, nu, lambda)[2];
}

/** The p-quantile: the left side holds probability (1 − λ)/2. */
export function skewQuantile(p: number, nu: number, lambda: number): number {
  if (!(p > 0 && p < 1)) throw new RangeError(`p must lie in (0, 1), got ${p}`);
  const { a, b } = skewConstants(nu, lambda);
  const standard = (q: number) => studentQuantile(q, nu) * Math.sqrt((nu - 2) / nu);
  const left = (1 - lambda) / 2;
  const w =
    p < left ? (1 - lambda) * standard(p / (1 - lambda)) : (1 + lambda) * standard(0.5 + (p - left) / (1 + lambda));
  return (w - a) / b;
}

/** One draw from a uniform `side` and a standardised-t draw: the side by its mass (1 ∓ λ)/2, then half a t on it. */
export function skewDraw(lambda: number, a: number, b: number, side: number, standardT: number): number {
  const w = Math.abs(standardT);
  return side < (1 - lambda) / 2 ? (-(1 - lambda) * w - a) / b : ((1 + lambda) * w - a) / b;
}

/** ln f(z) and its derivatives in z, η and λ — one observation of the likelihood (without −½ ln s). */
export function skewLogDensity(
  z: number,
  nu: number,
  lambda: number,
  k: SkewConstants,
): { value: number; dz: number; dNu: number; dLambda: number } {
  const left = k.b * z + k.a < 0;
  const sign = left ? -1 : 1;
  const m = 1 + sign * lambda;
  const w = (k.b * z + k.a) / m;
  const q = (w * w) / (nu - 2);
  const value = Math.log(k.b) + k.logC - ((nu + 1) / 2) * Math.log1p(q);
  // ∂/∂w of −(η+1)/2·ln(1 + q)
  const dw = (-(nu + 1) * w) / ((nu - 2) * (1 + q));
  const dwNu = (z * k.dBdNu + k.dAdNu) / m;
  const dwLambda = (z * k.dBdLambda + k.dAdLambda) / m - (w * sign) / m;
  return {
    value,
    dz: (dw * k.b) / m,
    dNu: k.dBdNu / k.b + k.dLogCdNu - 0.5 * Math.log1p(q) + (((nu + 1) / 2) * (q / (nu - 2))) / (1 + q) + dw * dwNu,
    dLambda: k.dBdLambda / k.b + dw * dwLambda,
  };
}
