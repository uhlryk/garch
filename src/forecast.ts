import { egarchForecast } from './egarch.js';
import type { GarchFit } from './fit.js';
import { checkCount, persistence } from './model.js';

/**
 * Per-step variance forecasts s_{T+1} … s_{T+h}, analytic: the first step
 * uses the observed last residual, later steps its expectation
 * (E[e²·[e<0]] = s/2), so s_{T+k} = ω + P·s_{T+k−1}. EGARCH's is
 * `egarchForecast`.
 *
 * GARCH-X adds δ·x_{T+k−1} to each step: x_T is the fit's last value;
 * `exogenous` gives x_{T+1} … x_{T+h−1} (a model's forecast of x, or a
 * scenario), and without it x is held at x_T.
 */
export function forecastVariance(fit: GarchFit, horizon: number, exogenous?: ArrayLike<number>): number[] {
  checkCount('horizon', horizon);
  const p = fit.params;
  const e = fit.lastResidual;
  const last = fit.conditionalVariance[fit.conditionalVariance.length - 1];
  if (fit.model === 'egarch') return egarchForecast(p, last, e, horizon);
  const delta = p.delta ?? 0;
  const xT = fit.lastExogenous ?? 0;
  if (exogenous !== undefined) {
    if (fit.lastExogenous === null) throw new RangeError('the fit has no exogenous series to forecast with');
    if (exogenous.length < horizon - 1)
      throw new RangeError(`${horizon - 1} future exogenous values are needed, got ${exogenous.length}`);
  }
  const P = persistence(p);
  const out: number[] = [];
  let v = p.omega + p.alpha * e * e + p.gamma * (e < 0 ? e * e : 0) + p.beta * last + delta * xT;
  out.push(v);
  for (let k = 1; k < horizon; k += 1) {
    v = p.omega + P * v + delta * (exogenous ? exogenous[k - 1] : xT);
    out.push(v);
  }
  return out;
}
