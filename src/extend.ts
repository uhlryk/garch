import { continueEgarchLikelihood } from './egarch.js';
import { type GarchFit, GarchInputError, informationCriteria } from './fit.js';
import { continueLikelihood } from './model.js';

/**
 * A fit carried over new returns without re-estimating it: what a caller who
 * refits weekly needs for today's forecast after each new return.
 */

/**
 * The fit with `newReturns` appended: the variance recursion runs on over
 * them at the fit's parameters, unchanged. The answer is a `GarchFit` —
 * `forecastVariance`, `simulate` and `tailRisk` take it as they take any fit —
 * whose variance path, `lastResidual`, log-likelihood, AIC and BIC cover the
 * whole series (k as estimated, n the whole length), and whose `extendedBy`
 * counts the returns added since the estimate. The parameters, `inference`,
 * `modes` and `converged` stay the estimate's.
 *
 * For garch and gjr the variance path and log-likelihood are those of
 * `logLikelihood` over the concatenated series at the same parameters, to
 * the bit. EGARCH restarts from ln of the last variance, which can differ from
 * the log-variance the recursion held by one rounding: the same to ~1e-15.
 */
export function extendFit(fit: GarchFit, newReturns: ArrayLike<number>, newExogenous?: ArrayLike<number>): GarchFit {
  const m = newReturns.length;
  for (let i = 0; i < m; i += 1)
    if (!Number.isFinite(newReturns[i])) throw new GarchInputError(`new return ${i} is not a finite number`);
  if ((fit.lastExogenous !== null) !== (newExogenous !== undefined))
    throw new GarchInputError(
      fit.lastExogenous === null
        ? 'the fit has no exogenous series to extend'
        : 'a GARCH-X fit needs the exogenous values of the new returns',
    );
  if (newExogenous) {
    if (newExogenous.length !== m)
      throw new GarchInputError(`${newExogenous.length} new exogenous values for ${m} new returns`);
    for (let i = 0; i < m; i += 1)
      if (!(Number.isFinite(newExogenous[i]) && newExogenous[i] >= 0))
        throw new GarchInputError(`new exogenous value ${i} must be a finite number ≥ 0`);
  }
  if (!m) return fit;
  const p = fit.params;
  const old = fit.conditionalVariance;
  const n = old.length;
  const s = old[n - 1];
  const e = fit.lastResidual;
  const variance = new Float64Array(n + m);
  variance.set(old);
  const ll =
    fit.model === 'egarch'
      ? continueEgarchLikelihood(
          newReturns,
          p,
          { L: Math.log(s), z: e / Math.sqrt(s) },
          fit.logLikelihood,
          variance.subarray(n),
        )
      : continueLikelihood(
          newReturns,
          p,
          { e2: e * e, neg: e < 0 ? e * e : 0, s, x: fit.lastExogenous ?? 0 },
          fit.logLikelihood,
          variance.subarray(n),
          undefined,
          newExogenous,
        );
  return {
    ...fit,
    logLikelihood: ll,
    ...informationCriteria(ll, fit.parameterCount, n + m),
    conditionalVariance: variance,
    lastResidual: newReturns[m - 1] - p.mu,
    extendedBy: fit.extendedBy + m,
    lastExogenous: newExogenous ? newExogenous[m - 1] : null,
  };
}
