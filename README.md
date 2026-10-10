# @krzysztofsztompka/garch

GARCH(1,1), GJR-GARCH(1,1) and EGARCH(1,1,1) by maximum likelihood, with
walk-forward validation. Pure TypeScript, **no runtime dependencies**, CommonJS
and ES modules. Fits match Python's [`arch`](https://github.com/bashtage/arch)
where the likelihood has one optimum, and find a higher one where it has several.

```bash
npm install @krzysztofsztompka/garch
```

Node ≥ 20. The core (`@krzysztofsztompka/garch`) runs anywhere; the worker pool
(`@krzysztofsztompka/garch/node`) needs Node's `worker_threads`.

## Quick start

```ts
import { fitGarch, forecastVariance } from '@krzysztofsztompka/garch';

// returns in any units — e.g. ×100 log-returns; answers come in the same units squared
const fit = fitGarch(returns, { model: 'gjr', inference: true });
fit.params;                    // { mu, omega, alpha, gamma, beta }
fit.inference?.pValues.gamma;  // robust, two-sided
forecastVariance(fit, 20);     // s_{T+1} … s_{T+20}
```

A walk-forward refits at every origin, which takes seconds to a minute on one
core for a long history. The pool spreads it over every core but one, off the
caller's event loop, with the same answer to the bit:

```ts
import { volatilityForecastInWorkers, weightedForecastInWorkers } from '@krzysztofsztompka/garch/node';

await volatilityForecastInWorkers({ returns, model: 'gjr', horizon: 20 }, { timeoutMs: 120_000 });
await weightedForecastInWorkers({ returns, horizon: 20 });   // four models weighted by past loss
```

What it does not do, deliberately:

- **No prices and no percent.** It takes returns in whatever units it is given.
- **No interpretation.** When σ̂ is worth quoting, or what a regime means, is the caller's.
- **No trimming.** `quantizedUntil` says where a coarse-grid stretch of history ends; whether to fit from there is the caller's.
- **No EGARCH with Student-t innovations.** Its expected variance two steps ahead is infinite (below).

## The model

`arch`'s default specification: a constant mean and normal innovations.

    r_t = μ + e_t,   e_t ~ N(0, s_t)
    s_t = ω + α·e²_{t−1} + γ·e²_{t−1}·[e_{t−1} < 0] + β·s_{t−1}      (γ = 0 for garch)

- **Backcast as in `arch`.** The recursion starts from an exponentially weighted (0.94) mean of the first 75 squared residuals.
- **Analytic forecasts.** s_{T+k} = ω + (α + γ/2 + β)·s_{T+k−1}.
- **Robust standard errors.** Bollerslev–Wooldridge, `arch`'s `cov_type="robust"`: analytic scores and a Hessian taken by differencing the analytic gradient.

### Student-t innovations (`dist: 't'`)

    l_t = ln Γ((ν+1)/2) − ln Γ(ν/2) − ½ ln(π(ν−2)) − ½ ln s_t − (ν+1)/2 · ln(1 + e²_t / (s_t(ν−2)))

s_t is still the variance, so the forecasts are the same recursion. ν is a
parameter with its analytic gradient and robust standard error.

- **ν ≥ 4 by default** (`nuMin`; `arch` allows 2.05). A t variance is scale²·ν/(ν−2), and near 2 it is barely defined. Fitted freely on daily equity returns, ν̂ ≈ 2.1–2.5 gave a variance forecast worse than naive; at 4 the kurtosis is still infinite, but the forecast is stable.
- **A ν the data push past a bound is held there.** Free, the logit of ν runs off and the fit never converges. The fit is refitted with ν fixed on the bound (the constrained optimum) and reports `nuBound: 'lower' | 'upper'`. A held ν has no standard error.
- **Exact-zero returns break it.** Prices recorded on a coarse grid make most days an exact zero: cents on a stock worth $0.08 split-adjusted, or a penny stock's cents times a 1:100 reverse split. On an exact zero the variance can run to 0 while the likelihood rises without bound. The normal likelihood punishes that on the next non-zero day (e²/s); the t likelihood only logarithmically, so the fit degenerates to ω ≈ 0 and μ ≈ 0, **and can still report `converged`**. `quantizedUntil(returns)` returns the index after the last 250-return window with more than 20 % exact zeros; fit t from there.

### EGARCH (`model: 'egarch'`)

`arch`'s `vol="EGARCH", p=1, o=1, q=1` with normal innovations:

    ln s_t = ω + α·(|z_{t−1}| − √(2/π)) + γ·z_{t−1} + β·ln s_{t−1},   z = e / √s

- **The asymmetry is γ itself, and its sign is the other way round from GJR.** γ < 0 means a down move raises tomorrow's variance more than an up move of the same size. `persistence` is β.
- **The recursion starts from ln of the backcast**, with no shock term at t = 0, as in `arch`. `GarchFit.backcast` holds that log.
- **Parameter space.** β = σ(x2) ∈ (0, 1); ω = (1 − β)·(ln var(r) + x1); α and γ are free. `arch` lets β run in (−1, 1); a negative β is left out here. The gradient is analytic, through the recursion of d ln s.
- **The forecast is exact, not simulated.** E[s_{T+k}] = exp(m_k)·Π_{j<k−1} E[exp(βʲ·g(z))], with m_k = ω + β·m_{k−1}. Under the normal, E[exp(a|z| + bz)] = e^{(a+b)²/2}Φ(a+b) + e^{(a−b)²/2}Φ(a−b). Under Student-t that expectation is infinite for every a > 0, which is why a t EGARCH is refused (`GarchInputError`).
- **Against `arch` 8.0.0** (`test/fixtures/arch-egarch-2026-10-01.json`): the same optimum to ~1e-5 in α, γ, β, and an equal or slightly higher log-likelihood. γ's robust p-value agrees within 1 %. The closed-form forecast agrees with `arch`'s 200 000 simulated paths to 0.01 % and 0.24 % at 20 steps.
- **Its walk-forward baseline is symmetric GARCH,** like GJR's: is the asymmetry worth it?

## The optimiser, and why it has six starts

BFGS with an analytic gradient runs in an unconstrained space where every point
is a valid, stationary model: ω = e^{x1}·var(r), P = σ(x2) = α+γ/2+β,
β = P·σ(x3), γ = 2q·tanh(x4), α = q − γ/2.

It runs from **one start per persistence level** (0.5 … 0.999), plus the
previous origin's optimum during a walk-forward, and the highest likelihood
wins. Every distinct optimum the starts reached is returned as `modes`, best
first: on short or jumpy series (small-cap biotech, for one) the GARCH(1,1)
likelihood has **several modes**, and `arch`, with one start, stops in the
nearest one. There this package finds a higher likelihood than `arch` (+18 to
+56 log-likelihood on such series); where there is one mode, it lands on
`arch`'s numbers. `asOfFit` returns the rivals within a log-likelihood distance
you ask for, each with its own variance path and forecast.

What was tried and failed:

- **Nelder–Mead with a warm start** did not reach the optimum: warm and cold fits differed at most origins of a walk-forward.
- **A numerical gradient** reached the same optimum 4–8× slower than `arch`.
- **Fewer grid starts when a warm start is available.** A warm start alone misses the best optimum at ~1 % of origins on large caps and ETFs, but at 5–53 % on biotech. The grid runs at every origin.

## Walk-forward segments and the pool

A walk-forward is cut into **segments of 32 consecutive origins**
(`SEGMENT_ORIGINS`). Inside a segment each origin also starts from the previous
one's optimum; a segment's first origin starts from the grid alone. The cut
depends on the history only, so `walkForwardValidation` in-process and
`volatilityForecastInWorkers` on a pool of any size give the same answer, to
the bit. Each fit reads its train window as a `subarray` of one
`Float64Array` — no copy per origin.

`GarchPool` keeps up to *size* workers (default: every core but one), started
on demand and reused; an idle worker does not keep the process alive. A call's
tasks share one deadline: past it, or on one task's refusal, its queued tasks
are dropped and its running workers terminated, and the pool carries on. The
heaviest segments (the longest train windows) are dispatched first.

Measured on an i7-13700K (23 workers), one core → pool: S&P 500 (14 k returns)
GJR 93 s → 10 s, GARCH 46 s → 5 s; SPY GJR 36 s → 4 s.

## Several models, weighted by their past loss

`weightedForecast` / `weightedForecastInWorkers` fit {garch, gjr} × {normal, t}
(`WEIGHTED_MODELS`). Each model is walked forward, and the models are combined
at every origin by weights ∝ 1 / the model's mean QLIKE over the windows that
had **ended** by that origin. The weights are equal until each model has 20
such windows (`WEIGHT_BURN_IN`). The as-of forecast and variance path are
combined with the weights from every window.

The answer carries each model's as-of fit, its own validation and its weight,
next to the weighted forecast and its validation against the naive baseline —
and `walkForward`: at every origin, each model's fit (parameters, last
variance, last residual) and the weights in force there, so a simulation from
each origin needs no second walk-forward.

Why weights and not a pick: on 15 daily series, picking the model with the
lowest past loss beat the default on 8 and cost up to +47 % where the ranking
flipped; weighting averaged −5 % with a worst case of +4 %. Trim quantized
history (`quantizedUntil`) before putting t in the mix.

## Against the npm `garch` package

Judged on 49 daily series by four conclusions drawn from the output — is σ̂
worth quoting, the volatility regime, an implied-vs-realised variance verdict,
and the coverage of a forecast range — against what happened:

| class | npm `garch` 2.0.0 | this package |
|---|---|---|
| large cap (14) | fails three of four | passes |
| small/mid-cap biotech (18) | fails all four | fails on 4 names, all where `arch`'s lower-likelihood mode happened to read the future better |
| small-cap tools (4) | fails one | passes |
| ETF / index (6) | passes | passes |

npm `garch`'s variance forecast collapses to ≈ 0 at single origins (QLIKE 10¹⁰).

## API

```ts
import { egarchForecast, fitGarch, forecastVariance, quantizedUntil, walkForwardValidation, volatilityForecast, fitsAt } from '@krzysztofsztompka/garch';
import { fitsAtInWorkers, runInWorker, volatilityForecastInWorkers, weightedForecastInWorkers } from '@krzysztofsztompka/garch/node';

const fit = fitGarch(returns, { model: 'gjr', inference: true });
fit.modes;                     // every optimum reached, best first
fitGarch(returns, { model: 'egarch', inference: true });          // γ < 0: leverage; normal innovations only
fitGarch(returns.slice(quantizedUntil(returns)), { dist: 't' });   // Student-t on the fine-grained history
// → params.nu, nuBound: 'lower' | 'upper' | null

// Node only: the walk-forward's segments across a pool of workers — the in-process answer, faster
await volatilityForecastInWorkers({ returns, model: 'gjr', horizon: 20, rivalsWithin: 1.92 }, { timeoutMs: 120_000 });
await fitsAtInWorkers({ returns, ends: [999, 1499] });
await weightedForecastInWorkers({ returns, horizon: 20 });
// → forecast, conditionalVariance, models[i].{spec, asOf, validation, weight}, validation, walkForward[j].{origin, weights, fits}
await runInWorker('asOfFit', { returns, horizon: 20 });        // any one job on one worker
```

`fitGarch` throws `GarchInputError` on input it cannot fit. The pool's calls
reject with one of three errors, so a caller can word each:

- `GarchJobInputError`: the input cannot be fitted.
- `GarchJobTimeoutError`: the call ran past its deadline, and its workers are terminated.
- `GarchJobFailedError`: any other failure.

## Tests

`npm test` compiles `src/` and `test/` with tsc and runs them with `node --test`.
There is no test framework dependency.

- **The analytic gradient** matches its finite difference.
- **Simulated paths.** Parameters are recovered on simulated GARCH and GJR paths. A real leverage term is significant and an absent one is not.
- **Every fit is a valid, stationary model.** Inputs that cannot be fitted are refused.
- **Forecasts** decay to the unconditional variance.
- **Walk-forward windows** never leak from test into train.
- **Against `arch` (frozen 2026-09-25).** `test/fixtures/arch-2026-09-25.json` holds real IOVA and SPY closes and `arch`'s own answers. The package must match the parameters and σ̂ to 1e-3 / 1e-4, γ's p-value to 2 %, and IOVA's walk-forward QLIKE to 1e-3.
- **Worker round trip.** The same answer as in-process, an input refusal kept as an input refusal, and a timeout that terminates the worker.
- **The modes.** Best first, the fit being the first; the warm start's optimum marked; rivals only within the asked distance.
- **Student-t.** The gradient against finite differences, ν included. ν is recovered on a simulated t-GARCH path. A ν below the floor is held and converges, and its standard error is NaN. Robust SE come with a free ν.
- **Weighting.** Equal weights before the burn-in; the better model dominates after it; changing a forecast whose window ends after an origin does not move that origin's weights; the pool's answer equals the in-process one.
- **EGARCH.** The gradient against finite differences. A simulated path's parameters, leverage sign included. The closed-form forecast against its own Monte Carlo. `arch` 8.0.0's optimum, γ p-value and simulated forecast. A t EGARCH refused.
- **Quantized history.** On IOVA's frozen closes: the cut lands where the reverse-split cents end, t degenerates before it (ω ≈ 0 while `converged`), and fits cleanly after it.
- **The pool.** The walk-forward and `fitsAt` bit-identical to in-process on pools of 1 and 3 workers; one refused job fails its whole call; the pool keeps working after a call timed out.

## Build

```bash
npm install && npm run build     # → dist/cjs and dist/esm, each with .d.ts
```

## Releasing

Move `CHANGELOG.md`'s `Unreleased` entries under the new version, set
`package.json`'s version to match, commit, then push a tag `v<version>`. The
`release` workflow refuses a tag that differs from the version or has no
changelog section; otherwise it tests, publishes to npm with provenance and
makes the GitHub Release from that section. npm must trust the workflow: a
trusted publisher (`uhlryk/garch`, `release.yml`) set on npmjs.com, or an
`NPM_TOKEN` repository secret holding an automation token.

## Licence

MIT
