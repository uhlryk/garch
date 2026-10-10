# Changelog

All notable changes to `@krzysztofsztompka/garch`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `GarchFit.aic`, `.bic` and `.parameterCount`; `aic` and `bic` on `asOfFit` /
  `volatilityForecast` answers. They match `arch` on the fixture series.
- `simulate`: seeded paths of returns and variances for garch, gjr (normal or
  Student-t) and egarch, from a fit's last state or the unconditional variance.
- Residual diagnostics: `standardizedResiduals`, `ljungBox`, `archLm` and
  `chiSquareSurvival`, matching statsmodels and scipy.

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

[Unreleased]: https://github.com/uhlryk/garch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/uhlryk/garch/releases/tag/v0.1.0
