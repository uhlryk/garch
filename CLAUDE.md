# @krzysztofsztompka/garch

Pure-TypeScript GARCH library on npm. `src/` is the package, `test/garch.test.ts`
the suite (node --test, no framework), `scripts/` dev tools, `build/` the post-tsc step.

## Every change that touches the public API or behaviour

Done means all of these, in the same commit as the code:

- **README** — the prose section for the feature (what it does, how it compares to
  `arch`/statsmodels/scipy, its limits), a line in **Usage** if a caller would type it,
  a row in **API reference** for every new export, a bullet under **Tests** for what
  the new tests prove. Fix any passage the change makes stale (formulas, lists of
  distributions/models, "what it does not do").
- **CHANGELOG.md** — under `## [Unreleased]`: Added / Changed / Fixed.
- **package.json** `description`/`keywords` when the feature changes what the package is.
- `npm run docs:check` passes (CI runs it): it fails on any export without an API
  reference row. It cannot check prose — that part is on the change's author.

## Checks before committing

`npm run lint`, `npm run typecheck`, `npm test` (or `npm run coverage` on Node ≥ 22.8),
`npm run build && npm run docs:check`. Numbers compared against Python come from
`scripts/arch-fixtures/generate.py` (pinned versions), never typed in by hand.

## Workflow

Commit straight to `main`, one commit per issue, `Closes #N`. Releases: move
`Unreleased` in CHANGELOG under the new version, bump `package.json`, push tag
`v<version>` — `release.yml` publishes via npm trusted publishing (no token).
