/** Special functions: Γ, ψ, the incomplete beta, and the normal and Student-t quantiles. */

const LANCZOS = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
];

/** ln Γ(x) for x > 0 — Lanczos (g = 7), relative error ~1e-15. */
export function logGamma(x: number): number {
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  const z = x - 1;
  let a = LANCZOS[0];
  const t = z + 7.5;
  for (let i = 1; i < 9; i += 1) a += LANCZOS[i] / (z + i);
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}

/** ψ(x) = d ln Γ(x) / dx for x > 0 — recurrence up to 6, then the asymptotic series. */
export function digamma(x: number): number {
  let r = 0;
  while (x < 6) {
    r -= 1 / x;
    x += 1;
  }
  const f = 1 / (x * x);
  return r + Math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))));
}

/** Φ⁻¹(p) — Acklam's rational approximation, relative error < 1.2e-9. */
export function normalQuantile(p: number): number {
  if (!(p > 0 && p < 1)) throw new RangeError(`p must lie in (0, 1), got ${p}`);
  const a = [
    -39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239,
  ];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [
    -0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968,
    2.938163982698783,
  ];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const low = 0.02425;
  if (p < low || p > 1 - low) {
    const q = Math.sqrt(-2 * Math.log(p < low ? p : 1 - p));
    const x =
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    return p < low ? x : -x;
  }
  const q = p - 0.5;
  const r = q * q;
  return (
    ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  );
}

/** The regularised incomplete beta I_x(a, b) — Lentz's continued fraction, on the side where it converges. */
export function incompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x));
  if (x > (a + 1) / (a + b + 2)) return 1 - incompleteBeta(1 - x, b, a);
  const tiny = 1e-300;
  let c = 1;
  let d = 1 - ((a + b) * x) / (a + 1);
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let h = d;
  for (let m = 1; m < 1000; m += 1) {
    const m2 = 2 * m;
    let aa = (m * (b - m) * x) / ((a + m2 - 1) * (a + m2));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    h *= d * c;
    aa = (-(a + m) * (a + b + m) * x) / ((a + m2) * (a + m2 + 1));
    d = 1 + aa * d;
    if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c;
    if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const delta = d * c;
    h *= delta;
    if (Math.abs(delta - 1) < 1e-16) break;
  }
  return (front * h) / a;
}

/** P(T ≤ t) for Student's t with ν degrees of freedom (not standardised). */
export function studentCdf(t: number, nu: number): number {
  const tail = 0.5 * incompleteBeta(nu / (nu + t * t), nu / 2, 0.5);
  return t < 0 ? tail : 1 - tail;
}

/** The p-quantile of Student's t with ν degrees of freedom: bisection on the CDF, to double precision. */
export function studentQuantile(p: number, nu: number): number {
  if (!(p > 0 && p < 1)) throw new RangeError(`p must lie in (0, 1), got ${p}`);
  if (!(nu > 0)) throw new RangeError(`ν must be positive, got ${nu}`);
  if (p === 0.5) return 0;
  if (p > 0.5) return -studentQuantile(1 - p, nu);
  let lo = -1;
  while (studentCdf(lo, nu) > p) lo *= 2;
  let hi = 0;
  for (let i = 0; i < 200 && hi - lo > 1e-15 * Math.max(1, Math.abs(lo)); i += 1) {
    const mid = (lo + hi) / 2;
    if (studentCdf(mid, nu) > p) hi = mid;
    else lo = mid;
  }
  return (lo + hi) / 2;
}
