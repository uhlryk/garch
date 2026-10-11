# @krzysztofsztompka/garch

GARCH(1,1), GJR-GARCH(1,1) and EGARCH(1,1,1) by maximum likelihood — normal,
Student-t or skewed-t innovations, a constant or zero mean — with variance
forecasts, walk-forward validation, simulation, value at risk with backtests,
and residual diagnostics; DCC-GARCH for the correlations of several assets. Pure TypeScript, **no runtime dependencies**,
CommonJS and ES modules, worker pools for Node and the browser. Fits match
Python's [`arch`](https://github.com/bashtage/arch) where the likelihood has
one optimum, and find a higher one where it has several.

```bash
npm install @krzysztofsztompka/garch
```

**API documentation:** <https://uhlryk.github.io/garch/> — every export, option
type and result field, built by TypeDoc from the declarations' comments.

Node ≥ 20. The core (`@krzysztofsztompka/garch`) runs anywhere; the worker pool
comes as `@krzysztofsztompka/garch/node` (Node's `worker_threads`) and
`@krzysztofsztompka/garch/web` (Web Workers, ES modules only).

## Quick start

```ts
import { fitGarch, forecastVariance } from '@krzysztofsztompka/garch';

// returns in any units — e.g. ×100 log-returns; answers come in the same units squared
const fit = fitGarch(returns, { model: 'gjr', inference: true });
fit.params;                    // { mu, omega, alpha, gamma, beta }
fit.inference?.pValues.gamma;  // robust, two-sided
forecastVariance(fit, 20);     // s_{T+1} … s_{T+20}
```

A complete program: [`examples/forecast.mjs`](examples/forecast.mjs) fits
gjr-t on SPY, forecasts 20 days and grades the model walked forward.

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
- **No EGARCH with Student-t or skewed-t innovations.** Its expected variance two steps ahead is infinite (below).
- **No AR or ARMA mean.** The mean is a constant or zero. Daily returns have next to no autocorrelation to model; a caller who needs it can filter the returns first and fit with `mean: 'zero'`.
- **No GED innovations.** Normal, t and skewed t cover thin tails, fat tails and a skew; GED adds a tail shape between normal and t that daily equity returns rarely ask for.
- **No GARCH(p, q) beyond (1, 1).** Higher orders rarely beat (1, 1) out of sample, and they need a different stationarity parameterisation; every model here is (1, 1).

## The model

`arch`'s default specification: a constant mean and normal innovations.

    r_t = μ + e_t,   e_t ~ N(0, s_t)
    s_t = ω + α·e²_{t−1} + γ·e²_{t−1}·[e_{t−1} < 0] + β·s_{t−1}      (γ = 0 for garch)

- **A zero mean** (`mean: 'zero'`, `arch`'s `mean='Zero'`) holds μ at 0 for returns already demeaned; every model takes it, and so do the walk-forward, the pool and the weighted forecast. It matches `arch` on the fixture series.
- **Backcast as in `arch`.** The recursion starts from an exponentially weighted (0.94) mean of the first 75 squared residuals.
- **Analytic forecasts.** s_{T+k} = ω + (α + κγ + β)·s_{T+k−1}, with κ = E[z²·1{z < 0}] the down moves' share of a unit shock (`downShare`): ½ for normal and t innovations, computed exactly for a skewed t (below). `persistence` is α + κγ + β.
- **AIC and BIC** as `arch` computes them, −2·ln L + 2k and −2·ln L + k·ln n, with k the free parameters (`parameterCount`): ω (unless targeted), α, β, plus μ unless the mean is zero, γ for gjr and egarch, ν for t and skewed t unless it is held on a bound, and λ for skewed t.
- **Robust standard errors.** Bollerslev–Wooldridge, `arch`'s `cov_type="robust"`: analytic scores and a Hessian taken by differencing the analytic gradient. A parameter that was not fitted (μ under a zero mean, a ν held on its bound, a targeted ω) gets NaN.
- **Refusals.** Input that cannot be fitted — fewer than 10 returns, a non-finite one, no variance — throws `GarchInputError`; a count that is not a whole number (`horizon`, `minTrain`, `step`, `burnIn`) throws `RangeError`.

### Variance targeting (`targetVariance: true`)

ω is not estimated: it is tied to the returns' variance V so that the model's
unconditional variance is the sample's, ω = V·(1 − persistence), V about the
sample mean (about 0 under a zero mean). One parameter fewer: fits are steadier
on short series and faster in a walk-forward.

- **garch and gjr, every distribution and mean.** Under a skewed GJR, persistence carries the exact down share κ(ν, λ), so ω moves with ν and λ as well; the gradient follows it (κ's derivatives by central differences). The fit reports V as `varianceTarget` (null when ω was estimated), and ω in `params` as always.
- **Out of the count.** ω leaves the search space, `parameterCount` (so AIC and BIC) and the robust covariance: its standard error is NaN, and the others' carry its dependence on them.
- **Threaded through** the walk-forward, `varBacktest`, `fitsAt`, the pool and the weighted forecast, like `mean`. In a weighted set, garch and gjr models are targeted and an egarch one is fitted with ω free.
- **Not for EGARCH** (`GarchInputError`): its unconditional log-variance is not ln V (Jensen's inequality and the shock term's mean move it), so there is no exact target. Its parameter space already centres ω on ln var(r).
- **Against a reference.** `arch` 8 has no variance targeting, so `scripts/arch-fixtures/generate.py` writes `arch`'s likelihood out in numpy (checked against `arch`'s own at its optimum), ties ω the same way and maximises it with scipy from several starts. On SPY and IOVA, garch and gjr, the fit lands on scipy's optimum to 1e-4 with an equal or higher likelihood.

### GARCH-X: an exogenous variance driver (`exogenous`)

Daily returns are a noisy view of volatility; an implied variance (VIX²) or an
intraday realised measure carries more. GARCH-X adds such a series x ≥ 0,
aligned with the returns, to the variance equation:

    s_t = ω + α·e²_{t−1} + γ·e²_{t−1}·[e_{t−1} < 0] + β·s_{t−1} + δ·x_{t−1},   δ ≥ 0

- **Fitting.** δ is a parameter like the others, with its analytic gradient (d s_t/dδ = x_{t−1} + β·d s_{t−1}/dδ), a robust standard error and p-value, and one more in `parameterCount`. The recursion starts from x_{−1} = the 0.94-weighted mean of the first 75 x (`exogenousBackcast`), as the variance starts from `arch`'s backcast. garch and gjr, every distribution and mean, and variance targeting (ω = V·(1 − P) − δ·mean(x), so the unconditional variance stays V). EGARCH refuses it.
- **Forecasts need x ahead.** s_{T+1} takes x_T, which the fit keeps (`lastExogenous`). Beyond one step x is not known: `forecastVariance(fit, h, xPath)` takes x_{T+1} … x_{T+h−1} (a forecast of x, or a scenario); without one, x is held at x_T. `simulate` takes the x of every step (`exogenous`), and `extendFit(fit, r, x)` the new x with the new returns.
- **In a walk-forward** each origin's fit sees x up to the origin and its forecast holds x at x_origin, so nothing from after the origin leaks in. `exogenous` goes through `walkForwardValidation`, `volatilityForecast`, `fitsAt`, `varBacktest`, the pool and the weighted forecast (whose egarch models are fitted without it). Whether x earns its keep is `compareValidations` of the runs with and without it.
- **Against a reference.** `arch` takes exogenous regressors in the mean only, so `scripts/arch-fixtures/generate.py` writes the GARCH-X likelihood out in numpy and maximises it with scipy, on SPY with x the trailing five-day mean of r². The fit lands on scipy's optimum to 1e-5 in every parameter, garch and gjr, with an equal likelihood.
- **Not Realized GARCH.** Hansen, Huang and Shek's model also has a measurement equation for x and a joint likelihood; it is not here (issue #26).

### Student-t innovations (`dist: 't'`)

    l_t = ln Γ((ν+1)/2) − ln Γ(ν/2) − ½ ln(π(ν−2)) − ½ ln s_t − (ν+1)/2 · ln(1 + e²_t / (s_t(ν−2)))

s_t is still the variance, so the forecasts are the same recursion. ν is a
parameter with its analytic gradient and robust standard error.

- **ν ≥ 4 by default** (`nuMin`; `arch` allows 2.05). A t variance is scale²·ν/(ν−2), and near 2 it is barely defined. Fitted freely on daily equity returns, ν̂ ≈ 2.1–2.5 gave a variance forecast worse than naive; at 4 the kurtosis is still infinite, but the forecast is stable.
- **A ν the data push past a bound is held there.** Free, the logit of ν runs off and the fit never converges. The fit is refitted with ν fixed on the bound (the constrained optimum) and reports `nuBound: 'lower' | 'upper'`. A held ν has no standard error.
- **Exact-zero returns break it.** Prices recorded on a coarse grid make most days an exact zero: cents on a stock worth $0.08 split-adjusted, or a penny stock's cents times a 1:100 reverse split. On an exact zero the variance can run to 0 while the likelihood rises without bound. The normal likelihood punishes that on the next non-zero day (e²/s); the t likelihood only logarithmically, so the fit degenerates to ω ≈ 0 and μ ≈ 0, **and can still report `converged`**. `quantizedUntil(returns)` returns the index after the last 250-return window with more than 20 % exact zeros; fit t from there.

### Skewed-t innovations (`dist: 'skewt'`)

Hansen's skewed t, `arch`'s `SkewStudent`: standardised to mean 0 and variance
1, with ν (`arch`'s η) and λ ∈ (−1, 1); λ < 0 leans the mass left, a fatter
loss tail. Each side of the density is half a standardised t, so its CDF,
quantile and partial moments E[zᵏ·1{z < x}] are closed forms over the
incomplete beta; they agree with `arch`'s to 1e-14. The gradient is analytic
in λ and ν as well, λ has a robust standard error and p-value, and ν has the
same floor and bound handling as under t.

- **GJR's asymmetry under a skew.** A down move's share of a unit shock's variance, κ = E[z²·1{z < 0}], is ½ only for a symmetric innovation; under a skewed t it is computed exactly (`downShare`, `skewDownShare`). `persistence` and the forecast use κ: s_{T+k} = ω + (α + κγ + β)·s_{T+k−1}. `arch` uses ½ there whatever the distribution, so for a skewed GJR its analytic forecast and this one differ by that much; the simulated paths here agree with this one. The search space keeps `arch`'s constraint α + γ/2 + β < 1.
- **Against `arch`** on SPY: garch and gjr fits to ~1e-5, with an equal or higher likelihood, and λ's and γ's p-values within 2 %.
- **VaR and ES** take λ (`tailRisk({ ..., nu, lambda })`), and `simulate` draws skewed-t innovations when the parameters carry λ.

### EGARCH (`model: 'egarch'`)

`arch`'s `vol="EGARCH", p=1, o=1, q=1` with normal innovations:

    ln s_t = ω + α·(|z_{t−1}| − √(2/π)) + γ·z_{t−1} + β·ln s_{t−1},   z = e / √s

- **The asymmetry is γ itself, and its sign is the other way round from GJR.** γ < 0 means a down move raises tomorrow's variance more than an up move of the same size. `persistence` is β.
- **The recursion starts from ln of the backcast**, with no shock term at t = 0, as in `arch`. `GarchFit.backcast` holds that log.
- **Parameter space.** β = σ(x2) ∈ (0, 1); ω = (1 − β)·(ln var(r) + x1); α and γ are free. `arch` lets β run in (−1, 1); a negative β is left out here. The gradient is analytic, through the recursion of d ln s.
- **The forecast is exact, not simulated.** E[s_{T+k}] = exp(m_k)·Π_{j<k−1} E[exp(βʲ·g(z))], with m_k = ω + β·m_{k−1}. Under the normal, E[exp(a|z| + bz)] = e^{(a+b)²/2}Φ(a+b) + e^{(a−b)²/2}Φ(a−b). Under Student-t (skewed or not) that expectation is infinite for every a > 0, which is why a t or skewed-t EGARCH is refused (`GarchInputError`).
- **Against `arch` 8.0.0** (`test/fixtures/arch-egarch-2026-10-01.json`): the same optimum to ~1e-5 in α, γ, β, and an equal or slightly higher log-likelihood. γ's robust p-value agrees within 1 %. The closed-form forecast agrees with `arch`'s 200 000 simulated paths (seeded) to 0.02 % and 0.08 % at 20 steps.
- **Its walk-forward baseline is symmetric GARCH,** like GJR's: is the asymmetry worth it?

### Simulation

`simulate` runs the model's own recursion forward: `paths` paths of `steps`
returns, from where a fit stands (its last variance and residual, so step 1's
variance is the one-step forecast) or from the unconditional variance. The
innovations are normal, Student-t scaled to unit variance when the parameters
carry ν, Hansen's skewed t when they also carry λ, or the fit's own residuals
resampled (`innovations: { bootstrap: z }`, below); an EGARCH with t is
refused, as in fitting. One seeded
stream (xoshiro128**) feeds every path, so a seed gives the same paths on any
platform. The paths' mean variance converges to `forecastVariance`.
[`examples/simulate-fan.mjs`](examples/simulate-fan.mjs) reads the percentiles
of the 10-day return, a fan chart's edges, off 20 000 paths.

### Extending a fit without refitting

`extendFit(fit, newReturns)` runs the variance recursion on over returns that
came after the estimate, at the fit's parameters, unchanged — what a caller who
refits weekly needs for today's forecast after each new return: one pass over the
new returns instead of a multi-start fit. The answer is a `GarchFit`:
`forecastVariance`, `simulate` and `tailRisk` take it as any fit. Its variance
path, `lastResidual`, log-likelihood, AIC and BIC cover the whole series; its
parameters, `inference`, `modes` and `converged` stay the estimate's, and
`extendedBy` counts the returns added since it (0 for a fit from `fitGarch`).

- **The same numbers as a run over the whole series.** For garch and gjr the variance path and log-likelihood are those of `logLikelihood` over the concatenated returns at the same parameters, to the bit; EGARCH restarts from ln of the last variance, one rounding away (~1e-15).
- **Against `arch`.** It is `arch`'s `fix(params)` on the longer series (statsmodels' state-space results call the same step `append(refit=False)`): on SPY, a gjr fit on all but the last 250 returns, carried over them, reaches `fix`'s last variance and one-step forecast to 1e-10. The log-likelihoods differ by ~1e-4 because `fix` backcasts at the fixed μ rather than the sample mean.
- **When to refit is the caller's.** The parameters drift slowly, so a weekly or monthly refit with daily extensions in between is the usual pattern; a large shock or a regime change is the time to refit sooner. `extendedBy` says how stale the estimate is.

### Residual diagnostics

`standardizedResiduals(fit, returns)` gives z_t = (r_t − μ)/√s_t. `ljungBox`
(on z for the mean, on z² for the variance) and `archLm` (Engle's LM test on
z) say whether the model left anything behind; each returns the statistic, its
χ² degrees of freedom and the p-value. They follow statsmodels'
`acorr_ljungbox` and `het_arch` and agree with them on the fixture series to
1e-3; `chiSquareSurvival` agrees with scipy to 1e-12.

### Value at risk

VaR and ES are positive losses in the returns' units at a tail probability
`level` (0.01 for 99 %). `tailRisk` gives the one-step answer in closed form
from μ, s_{T+1} and the innovations (normal, standardised t with ν, or skewed
t with ν and λ); its quantiles and tail means agree with scipy and `arch`. Over several steps the cumulative
return under GARCH is not normal, so `simulatedTailRisk` reads VaR and ES off
`simulate`'s paths. `varBacktest` walks a one-step VaR forward and
`coverageTest` grades the violations: Kupiec's unconditional coverage,
Christoffersen's independence, and both together.
[`examples/var-backtest.mjs`](examples/var-backtest.mjs) walks a 99 % VaR
forward day by day on SPY, parametric and FHS.

### Filtered historical simulation

`tailRisk` and `simulate` assume the innovations' family. Filtered historical
simulation (FHS) takes the tail from the data instead: the fit's own
standardised residuals z_t, pushed through the variance recursion.

- **One step:** `filteredTailRisk({ mean, variance, residuals, level })` is μ + √s_{T+1}·q̂, q̂ the ⌈level·n⌉-th smallest residual, and the ES the mean of the residuals at or below it — no paths needed.
- **Several steps:** `simulate({ ..., innovations: { bootstrap: z } })` draws the innovations with replacement from z (seeded, as any simulation) and runs the same recursion; `simulatedTailRisk` reads VaR and ES off the paths. This is `arch`'s `forecast(method='bootstrap')`; on SPY's gjr fit the paths' mean variance agrees with `arch`'s over 10 steps to 2 % (Monte Carlo on both sides).
- **Walked forward:** `varBacktest(returns, { ..., method: 'fhs' })` reads each origin's VaR off the residuals of that origin's fit over its train window, so it can be graded against the parametric VaR by the same coverage tests.
- **Limits.** The residuals are resampled as they are, not rescaled to unit variance, and a level below 1/n has no residual in its tail and is refused. A GJR's unconditional start (no `lastVariance`) still uses the parameters' down share, not the residuals'.

### Is the difference more than noise? (Diebold–Mariano)

A walk-forward says the model's mean QLIKE is below the baseline's; the
Diebold–Mariano test says whether that is more than noise. Every `Validation`
keeps each scored origin's QLIKE term and the baseline's (`scored`), and
carries `dieboldMariano`: the mean difference (model − baseline, below 0 the
model lost less), its HAC standard error, the statistic with the
Harvey–Leybourne–Newbold small-sample correction, and a two-sided p-value from
t(n − 1). For gjr and egarch the baseline is symmetric garch, so the test asks
whether the asymmetry earns its keep; for garch, whether it beats the naive
trailing variance.

- **Overlapping windows.** Windows `horizon` long and `step` apart share test days with the next ⌈horizon/step⌉ − 1 windows, so their loss differences are autocorrelated that far; those are the HAC lags (the test's "h" is ⌈horizon/step⌉). At the default step 10 and horizon 20, one lag.
- **The HAC weights** are Newey–West's Bartlett 1 − k/h by default, never negative; `kernel: 'uniform'` gives Diebold and Mariano's original, R's `forecast::dm.test` default, which can come out non-positive (then refused). Neither `arch` nor statsmodels has the test; `scripts/arch-fixtures/generate.py` computes it by hand in numpy, and the statistic and p-value here agree to 1e-12.
- **Model against model.** `compareValidations(a, b, overlap)` tests two validations' terms on the origins both scored — the weighted forecast's models against one another, say.
- A skipped origin (a fit that did not converge) leaves a gap the HAC lags do not see; with few skips the effect is small.

## Several assets: DCC-GARCH

`fitDcc(series, { models })` fits Engle's (2002) dynamic conditional
correlation in two stages. Stage 1 fits each asset's own model with `fitGarch`
(any model, distribution and mean; one specification for all or one per
asset) and standardises its residuals. Stage 2 fits DCC(1,1) on them with
correlation targeting:

    Q_t = (1 − a − b)·Q̄ + a·z_{t−1}z_{t−1}ᵀ + b·Q_{t−1},   Q̄ = (1/T)·Σ z_t z_tᵀ
    R_t = diag(Q_t)^{−½}·Q_t·diag(Q_t)^{−½},   H_t = D_t·R_t·D_t

with (a, b) maximising the correlation part of the Gaussian likelihood, a, b ≥
0 and a + b < 1 by construction (BFGS from four starts). The fit keeps every
asset's `GarchFit`, a, b, Q̄, R_t for every day, and both log-likelihoods: stage
2's and the total.

- **Forecasts.** `dccForecast(fit, h)` gives H_{T+1} … H_{T+h}, R and each asset's variance. R_{T+1} is exact; later steps use Engle and Sheppard's approximation, as R's `rmgarch` does: R_{T+k} = (1 − (a+b)^{k−1})·R̄ + (a+b)^{k−1}·R_{T+1}.
- **Portfolios.** `portfolioVariance(H, w)` is wᵀHw. `portfolioTailRisk({ dcc, weights, horizon, level })` is closed form for one day with every asset normal, and simulated otherwise (`paths`, `seed`): each day draws every asset's own unit-variance innovations, correlates them through the Cholesky factor of R_t and runs every recursion forward. Under t or skewed-t assets the correlated draws keep the correlation but not exactly each marginal.
- **The pool.** `fitDccInWorkers(series, options)` (`/node`, `/web`) runs stage 1 one asset per worker and stage 2 in the caller, for the in-process answer.
- **Against a reference.** `arch` has no DCC, so `scripts/arch-fixtures/generate.py` writes stage 2 out in numpy over `arch`'s stage-1 residuals and maximises it with scipy, on IOVA and SPY over IOVA's history (aligned by position from their common last date). Stage 2 on those residuals lands on scipy's (a, b) to 1e-4 with an equal or higher likelihood; the whole fit, forecast covariances and portfolio VaR agree to 1 %.
- **Limits.** DCC(1,1) only, the series aligned by the caller (equal lengths, no gaps), and no standard errors for a and b: a two-stage estimate's covariance needs the correction for stage 1's estimation error, which is not computed. Stage 2 is Gaussian quasi-likelihood whatever the assets' innovations. The per-day correlations take T·N² numbers; meant for a handful of assets, not hundreds.

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

`npm run bench` measures it on SPY's closes from the `arch` fixture (8 469
returns, horizon 20) and asserts the pool's answer equals the in-process one.
On an i7-13700K, Node 24, 23 workers, one core → pool:

| job | one core | pool |
|---|---|---|
| garch | 15.8 s | 2.3 s |
| gjr | 42.4 s | 4.0 s |
| egarch | 51.7 s | 3.9 s |
| weighted (`-- --weighted`) | 122 s | 11.2 s |

### In the browser

`@krzysztofsztompka/garch/web` is the same pool on Web Workers: `GarchWebPool`
(default: every logical core but one) and the same calls —
`volatilityForecastInWorkers`, `fitsAtInWorkers`, `weightedForecastInWorkers`,
`runInWorker` — with the in-process answer. The worker is started as
`new Worker(new URL('./worker-web.js', import.meta.url), { type: 'module' })`,
written out literally, so Vite and webpack bundle it without configuration;
both were checked in Chrome. A bundler that needs the worker its own way passes
`new GarchWebPool(size, () => new Worker(...))`. Node's pool and this one share
one implementation of the queue, deadline and cancellation (`WorkerPool`).
[`examples/browser/`](https://github.com/uhlryk/garch/tree/main/examples/browser) is a Vite page running a walk-forward
on the web pool (`npm install && npm run dev` there).

## Several models, weighted by their past loss

`weightedForecast` / `weightedForecastInWorkers` fit {garch, gjr} × {normal, t}
(`WEIGHTED_MODELS`) by default; `models` takes any other set of
`{ model, dist }` (egarch and skewed t included), and `mean: 'zero'` applies to
all of them. Each model is walked forward, and the models are combined
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

## Examples

Complete programs in [`examples/`](https://github.com/uhlryk/garch/tree/main/examples), each importing the package by
name; CI runs the Node ones against the packed tarball. In a checkout,
`npm run build` first (the name resolves to `dist/`).

| example | what it shows |
|---|---|
| [`forecast.mjs`](examples/forecast.mjs) | gjr-t on SPY on the worker pool: parameters, σ̂ tomorrow and over 20 days, the walk-forward verdict and its Diebold–Mariano p-value. |
| [`var-backtest.mjs`](examples/var-backtest.mjs) | A one-step 99 % VaR walked forward day by day, parametric and FHS, with Kupiec and Christoffersen. |
| [`simulate-fan.mjs`](examples/simulate-fan.mjs) | Paths from the fit: percentiles of the 10-day return, and the 10-day VaR and ES. |
| [`browser/`](https://github.com/uhlryk/garch/tree/main/examples/browser) | A Vite page running the walk-forward on Web Workers. |

`examples/spy.json` holds the last 2 501 SPY closes of the test fixture.

## Usage

```ts
import { archLm, compareValidations, dccForecast, dieboldMariano, egarchForecast, extendFit, filteredTailRisk, fitDcc, fitGarch, forecastVariance, ljungBox, portfolioTailRisk, portfolioVariance, quantizedUntil, simulate, simulatedTailRisk, standardizedResiduals, tailRisk, varBacktest, walkForwardValidation, volatilityForecast, fitsAt } from '@krzysztofsztompka/garch';
import { fitDccInWorkers, fitsAtInWorkers, runInWorker, volatilityForecastInWorkers, weightedForecastInWorkers } from '@krzysztofsztompka/garch/node'; // or '/web'

const fit = fitGarch(returns, { model: 'gjr', inference: true });
fit.modes;                     // every optimum reached, best first
fit.aic; fit.bic;              // as arch reports them; a ν held on a bound is not counted
fitGarch(returns, { model: 'egarch', inference: true });          // γ < 0: leverage; normal innovations only
fitGarch(returns, { model: 'gjr', mean: 'zero' });                // μ held at 0
fitGarch(returns, { model: 'gjr', targetVariance: true });        // ω = var(r)·(1 − persistence), → varianceTarget
const gx = fitGarch(returns, { model: 'gjr', exogenous: vix2 });  // GARCH-X: + δ·x_{t−1}, → params.delta
forecastVariance(gx, 20, vix2Path);                               // x_{T+1} … x_{T+19}; held at x_T without
fitGarch(returns, { model: 'gjr', dist: 'skewt', inference: true }); // → params.nu, params.lambda; λ < 0: a fatter loss tail
fitGarch(returns.slice(quantizedUntil(returns)), { dist: 't' });   // Student-t on the fine-grained history
// → params.nu, nuBound: 'lower' | 'upper' | null

// paths forward from where the fit stands (seeded); a FitAt from fitsAt spreads in as it is
simulate({ model: 'gjr', params: fit.params, lastVariance: fit.conditionalVariance.at(-1), lastResidual: fit.lastResidual,
           steps: 20, paths: 10_000, seed: 1 });   // → { returns, variance }: Float64Array per path

const today = extendFit(fit, newReturns);   // the recursion carried over new returns, parameters unchanged
forecastVariance(today, 20);                 // today.extendedBy: returns since the estimate

const z = standardizedResiduals(fit, returns);
ljungBox(z.map((v) => v * v), 10);   // → { statistic, df, pValue }: ARCH left in the residuals?
archLm(z, 10);

tailRisk({ mean: fit.params.mu, variance: forecastVariance(fit, 1)[0], nu: fit.params.nu, level: 0.01 });
simulatedTailRisk(simulate({ ... , steps: 10, paths: 100_000, seed: 1 }), 0.01);   // 10-day VaR, ES
filteredTailRisk({ mean: fit.params.mu, variance: forecastVariance(fit, 1)[0], residuals: z, level: 0.01 });   // FHS, one step
simulate({ ..., innovations: { bootstrap: z } });                                   // FHS paths: residuals resampled
varBacktest(returns, { model: 'gjr', level: 0.01, method: 'fhs' });
varBacktest(returns, { model: 'gjr', dist: 't', level: 0.01, step: 1 });          // → violations, rate, Kupiec, Christoffersen

const { validation } = volatilityForecast({ returns, model: 'gjr', horizon: 20 });
validation.dieboldMariano;           // → { meanDifference, statistic, pValue, … }: gjr vs garch, more than noise?
dieboldMariano(lossA, lossB, 2);     // any two loss series; h = ⌈horizon/step⌉
compareValidations(models[1].validation, models[0].validation, 2);   // model against model

// on a pool (Node: /node, browser: /web): the walk-forward's segments across workers — the in-process answer, faster
await volatilityForecastInWorkers({ returns, model: 'gjr', horizon: 20, rivalsWithin: 1.92 }, { timeoutMs: 120_000 });
await fitsAtInWorkers({ returns, ends: [999, 1499] });
await weightedForecastInWorkers({ returns, horizon: 20 });
// → forecast, conditionalVariance, models[i].{spec, asOf, validation, weight}, validation, walkForward[j].{origin, weights, fits}
await runInWorker('asOfFit', { returns, horizon: 20 });        // any one job on one worker

// several assets: DCC-GARCH; one array of returns per asset, aligned by day
const dcc = fitDcc([spy, tlt, gld], { models: { model: 'gjr', dist: 't' } });   // → assets, a, b, correlation
const ahead = dccForecast(dcc, 10);                       // → covariance[k], correlation[k]: H_{T+k}, R_{T+k}
portfolioVariance(ahead.covariance[0], [0.6, 0.3, 0.1]);
portfolioTailRisk({ dcc, weights: [0.6, 0.3, 0.1], horizon: 10, level: 0.01, paths: 50_000, seed: 1 });
await fitDccInWorkers([spy, tlt, gld], { models: { model: 'gjr' } });   // stage 1 across the pool
```

`fitGarch` throws `GarchInputError` on input it cannot fit. The pool's calls
reject with one of three errors, so a caller can word each:

- `GarchJobInputError`: the input cannot be fitted.
- `GarchJobTimeoutError`: the call ran past its deadline, and its workers are terminated.
- `GarchJobFailedError`: any other failure.

## API reference

Every value each entry point exports, one line each. The types (`GarchFit`,
`FitOptions`, `VolatilityForecast`, …) come with the declarations, and each
field is commented there; the [API documentation](https://uhlryk.github.io/garch/)
renders them. `npm run docs:check` fails when an export is missing from this
section, and `npm run docs` (TypeDoc, into `docs-site/`) when an export has no
doc comment; CI runs both, and the `pages` workflow publishes the site on every
push to `main`.

### `@krzysztofsztompka/garch`

**Fitting and forecasting**

| export | what it does |
|---|---|
| `fitGarch(returns, options?)` | Maximum likelihood fit: `model` (`'garch'` \| `'gjr'` \| `'egarch'`), `dist` (`'normal'` \| `'t'` \| `'skewt'`), `mean` (`'constant'` \| `'zero'`), `targetVariance`, `exogenous`, `nuMin`, `warmStart`, `inference`. Returns `GarchFit`: params, persistence, log-likelihood, `aic`, `bic`, variance path, `modes`, `nuBound`, `inference`, `extendedBy`, `varianceTarget`, `lastExogenous`. |
| `extendFit(fit, newReturns, newExogenous?)` | The fit carried over new returns at its parameters, without re-estimating: variance path, `lastResidual`, log-likelihood and criteria over the whole series; `extendedBy` counts the returns added. |
| `forecastVariance(fit, horizon, exogenous?)` | s_{T+1} … s_{T+h}, analytic (EGARCH's exact); GARCH-X takes x_{T+1} … x_{T+h−1} or holds x_T. |
| `egarchForecast(params, lastVariance, lastResidual, horizon)` | EGARCH's closed-form expected variance path, from a state. |
| `persistence(params)` | α + κγ + β for garch/gjr. |
| `downShare(params)` | κ = E[z²·1{z < 0}] of the params' innovations: ½, or the skewed t's. |
| `GarchInputError` | Thrown on input that cannot be fitted. |
| `T_NU_MIN` | The default floor on ν (4). |

**Whole questions**

| export | what it does |
|---|---|
| `volatilityForecast(request)` | Fit on everything, forecast `horizon` steps, walk the model forward against its baseline (naive for garch, symmetric garch for gjr and egarch). |
| `asOfFit(request)` | The as-of half of it: the fit, its forecast, γ's p-value, and `rivalsWithin` rival optima. |
| `fitsAt(request)` | One fit per train end — the state a simulation from each window needs. |
| `weightedForecast(request)` | Several models weighted by 1 / their past QLIKE (below). |
| `WEIGHTED_MODELS`, `WEIGHT_BURN_IN` | The default model set and burn-in of the weighting. |
| `weighForecasts(returns, windows, perModel, burnIn?)` | The weighting itself, on forecasts already made. |

**Walk-forward validation**

| export | what it does |
|---|---|
| `walkForwardValidation(returns, options)` | QLIKE of the model's walked-forward forecasts against a `'naive'` or `'garch'` baseline, each origin's terms (`scored`) and their Diebold–Mariano test. |
| `dieboldMariano(lossA, lossB, horizon?, options?)` | Diebold–Mariano with a HAC variance (Bartlett or uniform, horizon − 1 lags) and the HLN correction: is A's loss below B's by more than noise? |
| `compareValidations(a, b, overlap?, options?)` | Diebold–Mariano of two validations' QLIKE terms on the origins both scored. |
| `overlapOf(windows, horizon)` | ⌈horizon/step⌉: how many consecutive windows share a test day — the test's h. |
| `walkForwardOrigins(n, minTrain, horizon, step?)` | The expanding windows; no test index ever precedes a train index. |
| `validationWindows(n, options)` | The windows a validation scores (minTrain 500, step 10 by default). |
| `walkForwardForecasts(request)`, `walkForwardSteps(request)` | One segment's forecasts, and with each origin's fit kept. |
| `scoreWalkForward(returns, windows, horizon, forecasts, baseline)` | QLIKE of given forecasts and their baseline. |
| `segments(items, size?)`, `SEGMENT_ORIGINS` | The cut into 32-origin segments that keeps the pool's answer identical. |
| `qlike(realized, forecast)` | Patton's QLIKE loss over variances. |
| `realizedVariance(returns, from?, to?)` | The mean squared return over a window. |

**Simulation, risk and diagnostics**

| export | what it does |
|---|---|
| `simulate(request)` | Seeded paths of returns and variances from a fit's state or the unconditional variance; `innovations: { bootstrap: z }` resamples residuals (FHS). |
| `tailRisk({ mean, variance, nu?, lambda?, level })` | One-step VaR and ES, closed form. |
| `filteredTailRisk({ mean, variance, residuals, level })` | One-step VaR and ES by filtered historical simulation: the empirical quantile of the standardised residuals. |
| `simulatedTailRisk(simulation, level, steps?)` | VaR and ES of the cumulative return over simulated paths. |
| `varBacktest(returns, options)` | A one-step VaR walked forward, its violations graded; `method: 'fhs'` reads it off each fit's residuals. |
| `coverageTest(violations, level)` | Kupiec, Christoffersen and conditional coverage tests. |
| `standardizedResiduals(fit, returns)` | z_t = (r_t − μ)/√s_t. |
| `ljungBox(series, lags, modelDf?)`, `archLm(series, lags)` | Serial correlation and ARCH left in the residuals. |
| `quantizedUntil(returns, options?)` | Where coarse-grid history ends. |

**Several assets**

| export | what it does |
|---|---|
| `fitDcc(series, options?)` | DCC-GARCH: each asset by `fitGarch` (`models`: one spec or one per asset), then DCC(1,1) (a, b) with correlation targeting. Returns `DccFit`: `assets`, `a`, `b`, `correlationTarget`, `correlation` per day, both log-likelihoods. |
| `dccForecast(fit, horizon)` | H_{T+k}, R_{T+k} and each asset's variance, k = 1 … h. |
| `portfolioVariance(covariance, weights)` | wᵀHw. |
| `portfolioTailRisk(request)` | Portfolio VaR and ES over `horizon` days: closed form (one day, normal assets) or simulated paths. |

**Distributions and special functions**

| export | what it does |
|---|---|
| `normalCdf(x)`, `normalQuantile(p)`, `twoSidedPValue(z)` | The normal CDF, its inverse, and a two-sided p-value. |
| `studentCdf(t, nu)`, `studentQuantile(p, nu)` | Student's t (not standardised). |
| `skewQuantile(p, nu, lambda)`, `skewPartialMoments(x, nu, lambda)`, `skewDownShare(nu, lambda)` | Hansen's skewed t: quantile, E[zᵏ·1{z < x}] for k = 0, 1, 2, and κ. |
| `chiSquareSurvival(x, df)` | P(χ²_df > x). |
| `logGamma(x)`, `digamma(x)` | ln Γ and ψ. |

**Likelihoods** — for checking a fit or building on one

| export | what it does |
|---|---|
| `logLikelihood(returns, params, backcast, variance, contributions?, exogenous?)` | The garch/gjr log-likelihood, filling the variance path; GARCH-X with `{ values, backcast }`. |
| `logLikelihoodGradient(returns, params, backcast, scores?, exogenous?)` | It and its analytic gradient (and per-observation scores); δ's is the eighth entry. |
| `egarchLogLikelihood(…)`, `egarchLogLikelihoodGradient(…)` | The same for EGARCH, from the log of the backcast. |
| `backcast(returns, mean?)` | `arch`'s pre-sample variance. |
| `exogenousBackcast(x)` | GARCH-X's x_{−1}: the 0.94-weighted mean of the first 75 x. |
| `EGARCH_ABS_MEAN` | E\|z\| = √(2/π). |

### `@krzysztofsztompka/garch/node` and `@krzysztofsztompka/garch/web`

Both export the same names; `/node` runs worker threads, `/web` Web Workers.

| export | what it does |
|---|---|
| `volatilityForecastInWorkers(request, options?)` | `volatilityForecast` across the pool. |
| `fitsAtInWorkers(request, options?)` | `fitsAt` across the pool. |
| `weightedForecastInWorkers(request, options?)` | `weightedForecast` across the pool. |
| `fitDccInWorkers(series, options?, runOptions?)` | `fitDcc` with stage 1 across the pool. |
| `runInWorker(job, request, options?)` | Any one job on one worker. |
| `GarchPool` (`/node`), `GarchWebPool` (`/web`) | A pool of `size` workers; `close()` terminates them. |
| `defaultPool()` | The shared pool the calls use when `options.pool` is absent. |
| `WorkerPool` | The transport-independent pool both extend. |
| `GarchJobInputError`, `GarchJobTimeoutError`, `GarchJobFailedError` | How a pool call rejects. |

## Tests

`npm test` compiles `src/` and `test/` with tsc and runs them with `node --test`.
There is no test framework dependency. One file per area, run in parallel,
with what they share (the fixtures, a seeded path, the Web Worker shim) in
`test/helpers.ts`:

| file | area |
|---|---|
| [`fit.test.ts`](test/fit.test.ts) | The likelihood and its gradient, `fitGarch`, variance targeting, the modes, Student-t, `quantizedUntil`. |
| [`egarch.test.ts`](test/egarch.test.ts) | EGARCH. |
| [`skewt.test.ts`](test/skewt.test.ts) | Skewed-t innovations. |
| [`garch-x.test.ts`](test/garch-x.test.ts) | GARCH-X. |
| [`forecast.test.ts`](test/forecast.test.ts) | `forecastVariance`, `extendFit`, `simulate`. |
| [`validation.test.ts`](test/validation.test.ts) | Walk-forward windows, QLIKE, Diebold–Mariano, `fitsAt`, the weighted forecast. |
| [`risk.test.ts`](test/risk.test.ts) | Value at risk, backtests, filtered historical simulation. |
| [`diagnostics.test.ts`](test/diagnostics.test.ts) | Ljung–Box, ARCH-LM, standardised residuals. |
| [`dcc.test.ts`](test/dcc.test.ts) | DCC-GARCH. |
| [`pool.test.ts`](test/pool.test.ts) | `runInWorker`, the Node pool and the web pool. |
| [`against-arch.test.ts`](test/against-arch.test.ts) | `arch`'s answers on the fixture series. |

`node --test dist-test/test/fit.test.js` (after `npm test` has compiled) runs one area.
`npm run coverage` (Node ≥ 22.8) adds Node's own coverage of the compiled
code and fails under 95 % of lines, 85 % of branches or 80 % of functions; CI
runs it on Node 22 and 24. Code that runs only inside a worker is counted where
the tests also run it in-process.

- **The analytic gradient** matches its finite difference.
- **Simulated paths.** Parameters are recovered on simulated GARCH and GJR paths. A real leverage term is significant and an absent one is not.
- **Every fit is a valid, stationary model.** Inputs that cannot be fitted are refused.
- **Forecasts** decay to the unconditional variance.
- **Walk-forward windows** never leak from test into train.
- **DCC-GARCH.** a, b and the correlation target recovered from a simulated three-asset path; stage 2 on `arch`'s IOVA and SPY residuals at scipy's optimum to 1e-4, the likelihood equal at scipy's (a, b) to 1e-8; the whole fit, forecast covariances, portfolio variance and VaR against the fixture to 1 %; R_{T+1} exact and the forecast reverting to R̄; the simulated one-day portfolio VaR and ES against the closed form, a seed reproducing them; stage 1 on the pool equal to in-process.
- **Diebold–Mariano.** Statistic and p-value against a numpy hand computation to 1e-12 / 1e-9, for h = 1, 3, 5 and both kernels; IOVA's garch-vs-naive test against the same computation on `arch`'s walk-forward to 1 %; the per-origin terms average to the validation's loss; two validations compared on the origins both scored.
- **Extending a fit.** For garch, gjr (normal, t, skewed t, zero mean) the extended fit's variance path and log-likelihood equal `logLikelihood` over the whole series to the bit, EGARCH's to 1e-13; extending in steps equals extending at once; the forecast starts from the new last state; on SPY it reaches `arch`'s `fix()` variance and one-step forecast to 1e-10.
- **Simulation.** A seed gives the same paths; the paths' mean variance matches the analytic forecast for GJR (normal, t and skewed t) and EGARCH, and their innovations have unit variance.
- **GARCH-X.** The gradient against finite differences, δ included; δ recovered and significant on a simulated path with a persistent driver; SPY's garch-X and gjr-X against scipy's optimum to 1e-5; the forecast with x held and along a path; `extendFit` with new x equal to the whole-series likelihood to the bit; the walk-forward on the pool equal to in-process; variance targeting with x; refusals.
- **Variance targeting.** On SPY and IOVA, garch and gjr against scipy's optimum of the same likelihood with ω tied, to 1e-4 and an equal or higher likelihood; ω = V·(1 − persistence) to the bit; one parameter fewer in the count and none in the covariance; t, skewed-t and zero-mean fits keep the unconditional variance at V; EGARCH refused; carried through the pool's walk-forward and weighted forecast as in-process.
- **A zero mean and the criteria.** garch, gjr and egarch with `mean: 'zero'` against `arch`'s `mean='Zero'`; AIC and BIC against `arch`'s; the zero mean carried through the pool.
- **Refusals.** Fractional counts are refused; a pool timeout names the jobs it cut short.
- **Value at risk.** Normal and t quantiles, VaR and ES against scipy; the one-step VaR read off simulated paths; Kupiec and Christoffersen statistics against a hand computation; a walked-forward VaR covers at its level when the model is the truth.
- **Filtered historical simulation.** On SPY, the one-step VaR and ES against the same empirical quantile of `arch`'s residuals to 1e-9, and bootstrapped paths' mean variance against `arch`'s `method='bootstrap'` forecast to 2 %; on t draws the FHS VaR and ES converge to the t's; a seed gives the same resampled paths, drawn only from the residuals given; a walked-forward FHS VaR is each origin's fit's and covers at its level on a t path.
- **Residual diagnostics.** Ljung–Box and ARCH-LM against statsmodels on the fixture series; the χ² tail against scipy; ARCH found in a GARCH path and none left in its standardised residuals.
- **Against `arch`.** `test/fixtures/arch-2026-09-25.json` holds real IOVA and SPY closes (to 2026-09-23) and `arch`'s own answers. The package must match the parameters and σ̂ to 1e-3 / 1e-4, γ's p-value to 2 %, and IOVA's walk-forward QLIKE to 1e-3.
- **The fixtures are regenerated, not hand-written.** `scripts/arch-fixtures/generate.py` reads the closes from the fixture and rewrites `arch`'s answers in both files, with the versions in its `requirements.txt` (`arch` 8.0.0) and a seeded simulation, recording both in each file's `source`:

  ```bash
  python3 -m venv .venv && .venv/bin/pip install -r scripts/arch-fixtures/requirements.txt
  .venv/bin/python scripts/arch-fixtures/generate.py
  ```
- **The examples** run in CI on every Node version against the packed tarball, so the published entry points work as a caller imports them.
- **Worker round trip.** The same answer as in-process, an input refusal kept as an input refusal, and a timeout that terminates the worker.
- **The modes.** Best first, the fit being the first; the warm start's optimum marked; rivals only within the asked distance.
- **Student-t.** The gradient against finite differences, ν included. ν is recovered on a simulated t-GARCH path. A ν below the floor is held and converges, and its standard error is NaN. Robust SE come with a free ν.
- **Skewed t.** `arch`'s `SkewStudent` quantiles and partial moments to 1e-12; the gradient against finite differences, ν and λ included; ν and λ recovered on a simulated path; SPY's garch and gjr fits against `arch`'s; the skewed VaR and ES against the model's own paths; the simulated variance against the forecast with the exact down share.
- **Weighting.** Equal weights before the burn-in; the better model dominates after it; changing a forecast whose window ends after an origin does not move that origin's weights; the pool's answer equals the in-process one.
- **EGARCH.** The gradient against finite differences. A simulated path's parameters, leverage sign included. The closed-form forecast against its own Monte Carlo. `arch` 8.0.0's optimum, γ p-value and simulated forecast. A t EGARCH refused.
- **Quantized history.** On IOVA's frozen closes: the cut lands where the reverse-split cents end, t degenerates before it (ω ≈ 0 while `converged`), and fits cleanly after it.
- **The web pool.** Through a Web Worker made of a worker thread: the walk-forward identical to in-process, an input refusal and a deadline carried as on Node.
- **The pool.** The walk-forward and `fitsAt` bit-identical to in-process on pools of 1 and 3 workers; one refused job fails its whole call; the pool keeps working after a call timed out.

## Build

```bash
npm install && npm run build     # → dist/cjs and dist/esm, each with .d.ts
```

## Contributing

A change to the public API or its behaviour lands with its documentation: the
feature's section in this README, a Usage line, an **API reference** row for
each new export, a Tests bullet, and a CHANGELOG `Unreleased` entry
(`CLAUDE.md` and the issue template list it). CI runs `npm run docs:check`,
which fails when an export has no API reference row.

`main` is protected by a ruleset: no deletion, no force push, and a change
from anyone but the owner lands only with CI green on Node 20, 22 and 24.

Every action in the workflows is pinned to a full commit SHA (the version in a
comment), so a moved tag cannot change what runs with publish rights; CI runs
with `contents: read` only, and the release job alone gets `contents: write`
and `id-token: write`. Dependabot (`.github/dependabot.yml`) opens weekly PRs
for the actions and, grouped, the npm devDependencies; Dependabot alerts and
security updates are on. A Dependabot PR is read like any other — the
changelog of the bumped action or tool, and for an action the new SHA against
its release tag — and merged only with CI green on all three Node versions.

## Releasing

Move `CHANGELOG.md`'s `Unreleased` entries under the new version, set
`package.json`'s version to match, commit, then push a tag `v<version>`. The
`release` workflow refuses a tag that differs from the version or has no
changelog section; otherwise it tests, publishes to npm with provenance and
makes the GitHub Release from that section. npm trusts the workflow as a
trusted publisher (`uhlryk/garch`, `release.yml`), so no token is stored.

## Licence

MIT
