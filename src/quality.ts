/**
 * Whether a return series is fine-grained enough for a likelihood.
 *
 * A price recorded on a coarse grid — cents on a stock that traded at $0.08
 * split-adjusted (AAPL until 2005), a penny stock's cents times a 1:100
 * reverse split (IOVA until 2014) — turns most days into an exact zero and
 * the rest into jumps of one tick. An exact zero lets the conditional
 * variance run to 0 with the likelihood rising without bound; under
 * Student-t innovations nothing stops it, and the fit degenerates (ω → 0,
 * μ → 0). The package does not decide what to do about it; it says where
 * such a stretch ends.
 */

export type QuantizationOptions = {
  /** Returns per window (default 250, a year of sessions). */
  readonly window?: number;
  /** A window with more than this share of exact-zero returns is quantized (default 0.2). */
  readonly maxZeroShare?: number;
};

/**
 * The index of the first return after the last quantized window — 0 when no
 * window is. `returns.slice(quantizedUntil(returns))` is the fine-grained
 * history.
 */
export function quantizedUntil(returns: ArrayLike<number>, options: QuantizationOptions = {}): number {
  const window = options.window ?? 250;
  const share = options.maxZeroShare ?? 0.2;
  if (!(Number.isInteger(window) && window >= 1))
    throw new RangeError(`window must be a positive integer, got ${window}`);
  if (!(share >= 0 && share < 1)) throw new RangeError(`maxZeroShare must lie in [0, 1), got ${share}`);
  const n = returns.length;
  if (n < window) return 0;
  let zeros = 0;
  for (let i = 0; i < window; i += 1) if (returns[i] === 0) zeros += 1;
  let until = zeros / window > share ? window : 0;
  for (let end = window + 1; end <= n; end += 1) {
    if (returns[end - 1] === 0) zeros += 1;
    if (returns[end - 1 - window] === 0) zeros -= 1;
    if (zeros / window > share) until = end;
  }
  return until;
}
