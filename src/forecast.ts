import { egarchForecast } from './egarch';
import type { GarchFit } from './fit';
import { persistence } from './model';

/**
 * Per-step variance forecasts s_{T+1} … s_{T+h}, analytic: the first step
 * uses the observed last residual, later steps its expectation
 * (E[e²·[e<0]] = s/2), so s_{T+k} = ω + P·s_{T+k−1}. EGARCH's is
 * `egarchForecast`.
 */
export function forecastVariance(fit: GarchFit, horizon: number): number[] {
  if (!(horizon >= 1)) throw new RangeError(`horizon must be ≥ 1, got ${horizon}`);
  const p = fit.params;
  const e = fit.lastResidual;
  const last = fit.conditionalVariance[fit.conditionalVariance.length - 1];
  if (fit.model === 'egarch') return egarchForecast(p, last, e, horizon);
  const P = persistence(p);
  const out: number[] = [];
  let v = p.omega + p.alpha * e * e + p.gamma * (e < 0 ? e * e : 0) + p.beta * last;
  out.push(v);
  for (let k = 1; k < horizon; k += 1) {
    v = p.omega + P * v;
    out.push(v);
  }
  return out;
}
