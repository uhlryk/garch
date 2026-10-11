# @krzysztofsztompka/garch

Pure-TypeScript GARCH library on npm. `src/` is the package, `test/*.test.ts`
the suite, one file per area (node --test, no framework; shared helpers in `test/helpers.ts`), `scripts/` dev tools, `build/` the post-tsc step.

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
- `npm run docs` passes (CI runs it): TypeDoc fails on an exported value or type
  without a doc comment.

## Checks before committing

`npm run lint`, `npm run typecheck`, `npm test` (or `npm run coverage` on Node ≥ 22.8),
`npm run build && npm run docs:check`, `npm run docs`. Numbers compared against Python come from
`scripts/arch-fixtures/generate.py` (pinned versions), never typed in by hand.

## Workflow

Commit straight to `main`, one commit per issue, `Closes #N` — the owner bypasses
the `main` ruleset (no deletion, no force push, `test (20|22|24)` required for
everyone else, Dependabot PRs included); never force-push. Releases: move
`Unreleased` in CHANGELOG under the new version, bump `package.json`, push tag
`v<version>` — `release.yml` publishes via npm trusted publishing (no token).

Workflow actions are pinned by full SHA with the version in a comment
(`actions/checkout@<sha> # v4.4.0`); a new `uses:` follows that form. Dependabot
PRs (weekly: actions, grouped npm devDependencies) need CI green on 20/22/24;
check the bumped release notes (and an action's SHA against its tag) before merging.
