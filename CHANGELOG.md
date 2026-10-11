# Changelog

All notable changes to `@krzysztofsztompka/garch`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `extendFit(fit, newReturns)`: a fit carried over new returns at its
  parameters, without re-estimating — the variance path, last residual,
  log-likelihood and criteria over the whole series, for every model,
  distribution and mean. `GarchFit.extendedBy` counts the returns added since
  the estimate (0 for a fit from `fitGarch`).
- `dieboldMariano(lossA, lossB, horizon, { kernel })`: the Diebold–Mariano
  test with a HAC (Newey–West or uniform) variance and the
  Harvey–Leybourne–Newbold correction. Every `Validation` now carries each
  scored origin's QLIKE terms (`scored`) and `dieboldMariano` against its
  baseline, with HAC lags for overlapping windows; `compareValidations` tests
  two validations against each other; `overlapOf` gives ⌈horizon/step⌉.
- Filtered historical simulation: `simulate({ innovations: { bootstrap: z } })`
  resamples standardised residuals through the recursion (as `arch`'s
  `method='bootstrap'`), `filteredTailRisk` gives the one-step VaR and ES from
  their empirical quantile, and `varBacktest({ method: 'fhs' })` walks it
  forward.
- Variance targeting, `targetVariance: true`, for garch and gjr: ω is tied to
  the returns' variance, ω = var(r)·(1 − persistence), and leaves the search
  space, the parameter count and the robust covariance; `GarchFit.varianceTarget`
  reports the variance targeted. Threaded through the walk-forward,
  `varBacktest`, `fitsAt`, the pool and the weighted forecast. EGARCH refuses it.
- DCC-GARCH for several assets: `fitDcc` (stage 1 any model per asset, stage 2
  DCC(1,1) with correlation targeting), `dccForecast` (covariance and
  correlation h steps ahead), `portfolioVariance`, `portfolioTailRisk` (closed
  form or simulated), and `fitDccInWorkers` on `/node` and `/web` with stage 1
  on the pool. `runInWorker` takes a `fitGarch` job.
- GARCH-X, `exogenous: x`, for garch and gjr: δ·x_{t−1} in the variance
  equation, δ ≥ 0 fitted with an analytic gradient and robust inference
  (`params.delta`, `GarchFit.lastExogenous`, `exogenousBackcast`).
  `forecastVariance` takes a future x path (x held at its last value without),
  `simulate` and `extendFit` take x, and the walk-forward, `varBacktest`,
  `fitsAt`, the pool and the weighted forecast thread it through.

- API documentation built by TypeDoc (`npm run docs`, which fails on an
  export without a doc comment) and published on GitHub Pages; the package's
  `homepage` points there. `/node` and `/web` export the types `PoolWorker`,
  `WorkerEvents` and `Answer` that `WorkerPool`'s constructor refers to.

### Changed

- `logLikelihood` and `logLikelihoodGradient` take an optional exogenous
  series; with one, the gradient has eight entries (δ's last).

## [0.2.0] - 2026-10-11

### Added

- `GarchFit.aic`, `.bic` and `.parameterCount`; `aic` and `bic` on `asOfFit` /
  `volatilityForecast` answers. They match `arch` on the fixture series.
- `simulate`: seeded paths of returns and variances for garch, gjr (normal or
  Student-t) and egarch, from a fit's last state or the unconditional variance.
- Residual diagnostics: `standardizedResiduals`, `ljungBox`, `archLm` and
  `chiSquareSurvival`, matching statsmodels and scipy.
- Value at risk: `tailRisk` (one step, normal or t), `simulatedTailRisk`
  (multi-step, from paths), `varBacktest` and `coverageTest` (Kupiec,
  Christoffersen); `normalQuantile`, `studentCdf`, `studentQuantile`.
- A zero mean, `mean: 'zero'`, for every model and through the walk-forward,
  the pool and the weighted forecast; `GarchFit.mean` says which was fitted.
- `@krzysztofsztompka/garch/web`: the worker pool on Web Workers
  (`GarchWebPool`, ES modules only), bundled by Vite and webpack without
  configuration. `WorkerPool` is the transport-independent pool both share.
- Hansen's skewed-t innovations, `dist: 'skewt'` (`params.lambda`), for garch
  and gjr: fitting with an analytic gradient, robust inference, VaR and ES,
  simulation; `skewQuantile`, `skewPartialMoments`, `skewDownShare`,
  `downShare`. They match `arch`'s `SkewStudent`.

### Changed

- `persistence` and the GJR forecast use the down share E[z²·1{z < 0}]
  (½ for normal and t innovations, as before; exact for a skewed t).
- `logGamma`, `digamma`, `normalQuantile`, `studentCdf` and `studentQuantile`
  live in one module; `studentQuantile` takes any p in (0, 1).
- Robust standard errors take the set of fitted parameters, so a parameter
  held fixed (μ under a zero mean, a ν on its bound) gets NaN.

### Fixed

- `horizon`, `minTrain`, `step` and `burnIn` must be whole numbers; a fraction
  is refused with a `RangeError` instead of giving odd windows or an extra step.
- A pool timeout names the jobs that had not finished, with how many of each,
  instead of the call's first job.

## [0.1.0] - 2026-10-10

The first public release.

- GARCH(1,1), GJR-GARCH(1,1) and EGARCH(1,1,1) by maximum likelihood: BFGS
  with an analytic gradient from six persistence-level starts plus an optional
  warm start; every optimum reached is returned (`modes`, `rivals`).
- Normal and Student-t innovations (ν fitted, floor `nuMin`, a ν held on a
  bound reported by `nuBound`); EGARCH takes normal innovations only.
- Robust (Bollerslev–Wooldridge) standard errors and p-values.
- Analytic variance forecasts; EGARCH's in closed form.
- Walk-forward validation by QLIKE against a naive or symmetric-GARCH
  baseline; `fitsAt` for fits at given train ends.
- `weightedForecast`: {garch, gjr} × {normal, t} weighted by past loss.
- `quantizedUntil`: where coarse-grid history ends.
- `@krzysztofsztompka/garch/node`: a worker-thread pool with the in-process
  answer to the bit, one deadline per call and cancellation.
- ES modules and CommonJS, with type declarations; Node ≥ 20.

[Unreleased]: https://github.com/uhlryk/garch/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/uhlryk/garch/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/uhlryk/garch/releases/tag/v0.1.0
