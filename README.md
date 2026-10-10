# garch/ — `@smi/garch`

GARCH(1,1), GJR-GARCH(1,1) and EGARCH(1,1,1) by maximum likelihood, with
walk-forward validation. Pure TypeScript with **no runtime dependencies**, so that it can
become a published npm library without changes. It is here instead of a Python
kernel since [#1004](https://github.com/uhlryk/stock-market-insights/issues/1004).

## Who uses it

| caller | what it asks for | why this way |
|---|---|---|
| `webapp/…/technical-analysis/volatility-fit.ts` | `quantizedUntil(returns)`, then `weightedForecastInWorkers(…)` on the returns after it: four models' as-of fits and walk-forwards, the weighted σ̂ and its QLIKE against the naive baseline, and every origin's fits with the weights in force there — what the GARCH-method price cone simulates from (#1113, #1114) | a walk-forward refits every model at every origin, seconds to a minute on one core for a long history. The pool spreads its segments over every core but one and the Nest event loop stays free; the cone draws on the same walk-forward instead of fitting again. The app turns closes into ×100 log-returns and maps the answer back onto bars; what the numbers MEAN (regime, leverage sentence, withholding, when the models disagree too much) is `garch.ts`'s |
| `garch/bench/` | everything, on the operator's real closes | the #1004 / #1113 judgement and the probes, kept so a change is judged on conclusions again |
| `garch/test/` | everything | see *Tests* |

Deliberate absences:

- **No prices, no bars, no percent.** The package takes returns in whatever units it is given and answers in the same units squared.
- **No labels and no notes.** A rule about when σ̂ may be quoted belongs to the consumer.
- **No EGARCH with Student-t innovations.** Its expected variance two steps ahead is infinite (below); `arch` simulates a finite, unstable number instead. EGARCH with normal innovations is here since [#1111](https://github.com/uhlryk/stock-market-insights/issues/1111). Student-t is on the route as two of the four weighted models since [#1113](https://github.com/uhlryk/stock-market-insights/issues/1113).
- **No trimming.** `quantizedUntil` says where a coarse-grid stretch of history ends; whether to fit from there is the consumer's (the app does, #1114).
- **No threshold on the modes.** The package lists every optimum; which are the data's "cannot tell apart" is the consumer's (`rivalsWithin`).

## The model

`arch`'s default specification: a constant mean and normal innovations.

    r_t = μ + e_t,   e_t ~ N(0, s_t)
    s_t = ω + α·e²_{t−1} + γ·e²_{t−1}·[e_{t−1} < 0] + β·s_{t−1}      (γ = 0 for garch)

- **Backcast as in `arch`.** The recursion starts from an exponentially weighted (0.94) mean of the first 75 squared residuals.
- **Analytic forecasts.** s_{T+k} = ω + (α + γ/2 + β)·s_{T+k−1}.
- **Robust standard errors.** Bollerslev–Wooldridge, `arch`'s `cov_type="robust"`: analytic scores and a Hessian taken by differencing the analytic gradient.

### Student-t innovations (`dist: 't'`, #1111)

    l_t = ln Γ((ν+1)/2) − ln Γ(ν/2) − ½ ln(π(ν−2)) − ½ ln s_t − (ν+1)/2 · ln(1 + e²_t / (s_t(ν−2)))

s_t is still the variance, so the forecasts are the same recursion. ν is a
parameter with its analytic gradient and robust standard error.

- **ν ≥ 4 by default** (`nuMin`; `arch` allows 2.05). A t variance is scale²·ν/(ν−2), and near 2 it is barely defined. The #1111 experiment fitted ν̂ ≈ 2.1–2.5 on AAPL and IOVA, and the variance forecast was worse than naive. At 4 the kurtosis is still infinite, but the variance forecast is stable.
- **A ν the data push past a bound is held there.** Free, the logit of ν runs off and the fit never converges. The fit is refitted with ν fixed on the bound (the constrained optimum) and reports `nuBound: 'lower' | 'upper'`. A held ν has no standard error. On biotech names ν sits on the floor at 4 almost always.
- **Exact-zero returns break it.** Prices recorded on a coarse grid make most days an exact zero: cents on a stock worth $0.08 split-adjusted (AAPL until 2005), or a penny stock's cents times a 1:100 reverse split (IOVA until 2014). On an exact zero the variance can run to 0 while the likelihood rises without bound. The normal likelihood punishes that on the next non-zero day (e²/s); the t likelihood only logarithmically, so the fit degenerates to ω ≈ 0 and μ ≈ 0, **and can still report `converged`** (IOVA's first 800 returns: ω = 1.7e-10). `quantizedUntil(returns)` returns the index after the last 250-return window with more than 20 % exact zeros. From there AAPL and IOVA fit t at every origin.

### EGARCH (`model: 'egarch'`, #1111)

`arch`'s `vol="EGARCH", p=1, o=1, q=1` with normal innovations:

    ln s_t = ω + α·(|z_{t−1}| − √(2/π)) + γ·z_{t−1} + β·ln s_{t−1},   z = e / √s

- **The asymmetry is γ itself, and its sign is the other way round from GJR.** γ < 0 means a down move raises tomorrow's variance more than an up move of the same size. `persistence` is β.
- **The recursion starts from ln of the backcast**, with no shock term at t = 0, as in `arch`. `GarchFit.backcast` holds that log.
- **Parameter space.** β = σ(x2) ∈ (0, 1); ω = (1 − β)·(ln var(r) + x1); α and γ are free. `arch` lets β run in (−1, 1); a negative β is left out here. The gradient is analytic, through the recursion of d ln s.
- **The forecast is exact, not simulated.** E[s_{T+k}] = exp(m_k)·Π_{j<k−1} E[exp(βʲ·g(z))], with m_k = ω + β·m_{k−1}. Under the normal, E[exp(a|z| + bz)] = e^{(a+b)²/2}Φ(a+b) + e^{(a−b)²/2}Φ(a−b). Under Student-t that expectation is infinite for every a > 0, which is why a t EGARCH is refused (`GarchInputError`).
- **Against `arch` 8.0.0** (`test/fixtures/arch-egarch-2026-10-01.json`, IOVA and SPY): the same optimum to ~1e-5 in α, γ, β, and an equal or slightly higher log-likelihood. γ's robust p-value agrees within 1 %. The closed-form forecast agrees with `arch`'s 200 000 simulated paths to 0.01 % (IOVA) and 0.24 % (SPY) at 20 steps.
- **Its walk-forward baseline is symmetric GARCH,** like GJR's: is the asymmetry worth it?

## The optimiser, and why it has six starts

BFGS with an analytic gradient runs in an unconstrained space where every point
is a valid, stationary model: ω = e^{x1}·var(r), P = σ(x2) = α+γ/2+β,
β = P·σ(x3), γ = 2q·tanh(x4), α = q − γ/2.

It runs from **one start per persistence level** (0.5 … 0.999), plus the
previous origin's optimum during a walk-forward, and the highest likelihood
wins. Every distinct optimum the starts reached is returned as `modes`, best
first (#1006): on short or jumpy series there are several, and a consumer can
ask `asOfFit` for the rivals within a log-likelihood distance, each with its
own variance path and forecast. The reason is #1004's main finding: on short or jumpy series (biotech)
the GARCH(1,1) likelihood has **several modes**. `arch`, with one start, stops
in the nearest one. There this package finds a higher likelihood than `arch`
(PRAX +27, REPL +18, VRTX +55, LAB +56 log-likelihood). Where there is one
mode, it lands on `arch`'s numbers. Two earlier attempts are recorded because
they failed:

- **Nelder–Mead with a warm start** did not reach the optimum. The warm fit and the cold fit differed at 556 of INSM's 610 origins.
- **A numerical gradient** reached the same optimum 4–8× slower than `arch`.
- **Fewer grid starts when a warm start is available** (#1006, measured, not built). A warm start alone misses the best optimum at 0–14 of ~1 100 origins on large caps and ETFs, but at 5–53 % of them on biotech (PRAX gjr 51 / 96). The grid runs at every origin.

## Walk-forward segments and the pool (#1006)

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

Measured 2026-09-30 (i7-13700K: 8 performance + 8 efficiency cores, 23
workers), one core → pool: GSPC (14 k returns) gjr 93 s → 10 s, garch 46 s →
5 s; SPY gjr 36 s → 4 s; IOVA gjr 7.5 s → 1.1 s. Against the unsegmented
walk-forward of #1004 on 15 real series × 2 models, QLIKE moved by at most
1.2e−8 relative and no verdict changed.

## Several models, weighted by their past loss (#1113)

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
Why weights and not a pick: #1111 measured both on 15 series. Picking the
model with the lowest past loss beat the default on 8 and cost up to +47 %
where the ranking flipped. Weighting averaged −5 % with a worst case of +4 %.
The package does not trim quantized history; a consumer that wants t in the
mix should (`quantizedUntil`).

## Why this and not the npm `garch` package or `arch` (#1004)

The operator's ruling of 2026-09-24: a replacement passes if the **conclusions**
drawn from its output are right, per instrument class, not if it reproduces the
old numbers. Four conclusions were judged against what happened, on 49 series,
with the criteria fixed before any candidate was looked at:

1. Is σ̂ worth quoting?
2. The volatility regime.
3. The variance premium's rich / cheap verdict.
4. The GARCH range's coverage.

| class | npm `garch` 2.0.0 | this package |
|---|---|---|
| large cap (14) | fails C1, C2, C4 | passes |
| small/mid-cap biotech (18) | fails all four | fails on 4 names: PRAX, SMMT (C1), GPCR, REPL (C2), GPCR (C3) |
| small-cap tools (4) | fails C1 | passes |
| ETF / index (6) | passes | passes |
| commodity ETF, fx, reference | too few | too few |

npm's variance forecast collapses to ≈ 0 at single origins (QLIKE 10¹⁰). The
four biotech failures here are all series where `arch`'s lower-likelihood mode
happened to read the future better. The operator chose this package on
2026-09-25. The full per-instrument record is the result comment on #1004.

## API

```ts
import { egarchForecast, fitGarch, forecastVariance, quantizedUntil, walkForwardValidation, volatilityForecast, fitsAt } from '@smi/garch';
import { fitsAtInWorkers, runInWorker, volatilityForecastInWorkers, weightedForecastInWorkers } from '@smi/garch/node';

const fit = fitGarch(returns, { model: 'gjr', inference: true });
fit.params;                    // { mu, omega, alpha, gamma, beta }
fit.inference?.pValues.gamma;  // robust, two-sided
forecastVariance(fit, 20);     // s_{T+1} … s_{T+20}

fit.modes;                     // every optimum reached, best first
fitGarch(returns, { model: 'egarch', inference: true });          // γ < 0: leverage; normal innovations only
fitGarch(returns.slice(quantizedUntil(returns)), { dist: 't' });   // Student-t on the fine-grained history
// → params.nu, nuBound: 'lower' | 'upper' | null

// Node only: the walk-forward's segments across a pool of workers — the in-process answer, faster
await volatilityForecastInWorkers({ returns, model: 'gjr', horizon: 20, rivalsWithin: 1.92 }, { timeoutMs: 120_000 });
await fitsAtInWorkers({ returns, ends: [999, 1499] });
await weightedForecastInWorkers({ returns, horizon: 20 });     // four models weighted by past loss
// → forecast, conditionalVariance, models[i].{spec, asOf, validation, weight}, validation, walkForward[j].{origin, weights, fits}
await runInWorker('asOfFit', { returns, horizon: 20 });        // any one job on one worker
```

Each rejects with one of three errors, so the consumer can word each:

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
- **Weighting.** Equal weights before the burn-in; the better model dominates after it; changing a forecast whose window ends after an origin does not move that origin's weights; the pool's answer equals the in-process one; every origin's fits are `fitsAt`'s on the same ends, its weights summing to one.
- **EGARCH.** The gradient against finite differences. A simulated path's parameters, leverage sign included. The closed-form forecast against its own Monte Carlo. `arch` 8.0.0's optimum, γ p-value and simulated forecast on IOVA and SPY. A t EGARCH refused. The walk-forward the same in-process and on the pool.
- **Quantized history.** On IOVA's frozen closes: the cut lands where the reverse-split cents end, t degenerates before it (ω ≈ 0 while `converged`), and fits cleanly after it.
- **The pool.** The walk-forward and `fitsAt` bit-identical to in-process on pools of 1 and 3 workers; one refused job fails its whole call; the pool keeps working after a call timed out.

The webapp runs these through `npm run test:garch`. `npm test` there builds the
package first, because the backend requires `dist/`.

## Build

```bash
cd garch && npm install && npm run build     # → dist/ (CommonJS + .d.ts)
```

`cd webapp && npm run build` (and `npm start`) builds it first.
