/** Minimises `f` given its gradient: BFGS with Armijo backtracking. */

export type Objective = (x: readonly number[]) => { value: number; gradient: number[] };

export type Minimum = {
  readonly x: number[];
  readonly value: number;
  readonly converged: boolean;
};

const GRADIENT_TOL = 1e-7;

function maxAbs(v: readonly number[]): number {
  let m = 0;
  for (const x of v) m = Math.max(m, Math.abs(x));
  return m;
}

function identity(m: number): number[][] {
  return Array.from({ length: m }, (_, i) =>
    Array.from({ length: m }, (_, j) => (i === j ? 1 : 0)),
  );
}

export function bfgs(f: Objective, x0: readonly number[], maxIter = 300): Minimum {
  const m = x0.length;
  let x = x0.slice();
  let { value: fx, gradient: g } = f(x);
  if (!Number.isFinite(fx) || g.some((v) => !Number.isFinite(v)))
    return { x, value: fx, converged: false };
  let H = identity(m);
  for (let iter = 0; iter < maxIter; iter += 1) {
    if (maxAbs(g) < GRADIENT_TOL) return { x, value: fx, converged: true };
    let d = H.map((row) => -row.reduce((s, v, j) => s + v * g[j], 0));
    let slope = d.reduce((s, v, i) => s + v * g[i], 0);
    if (!(slope < 0)) {
      H = identity(m);
      d = g.map((v) => -v);
      slope = -g.reduce((s, v) => s + v * v, 0);
    }
    let step = 1;
    let xn: number[] = x;
    let next = { value: Infinity, gradient: g };
    for (let k = 0; k < 60; k += 1) {
      xn = x.map((v, i) => v + step * d[i]);
      next = f(xn);
      if (Number.isFinite(next.value) && next.value <= fx + 1e-4 * step * slope) break;
      step *= 0.5;
    }
    if (!(next.value < fx) || next.gradient.some((v) => !Number.isFinite(v)))
      return { x, value: fx, converged: maxAbs(g) < 1e-5 };
    const s = xn.map((v, i) => v - x[i]);
    const y = next.gradient.map((v, i) => v - g[i]);
    const sy = s.reduce((a, v, i) => a + v * y[i], 0);
    const improvement = fx - next.value;
    x = xn;
    fx = next.value;
    g = next.gradient;
    if (sy > 1e-14) {
      const Hy = H.map((row) => row.reduce((a, v, j) => a + v * y[j], 0));
      const yHy = y.reduce((a, v, i) => a + v * Hy[i], 0);
      H = H.map((row, i) =>
        row.map(
          (v, j) =>
            v +
            ((sy + yHy) * s[i] * s[j]) / (sy * sy) -
            (Hy[i] * s[j] + s[i] * Hy[j]) / sy,
        ),
      );
    }
    if (improvement < 1e-13 && maxAbs(s) < 1e-9) return { x, value: fx, converged: true };
  }
  return { x, value: fx, converged: maxAbs(g) < 1e-5 };
}
