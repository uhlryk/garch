"""The answers `arch` gives on the frozen closes, which test/garch.test.ts holds this package to.

    python -m venv .venv && .venv/bin/pip install -r scripts/arch-fixtures/requirements.txt
    .venv/bin/python scripts/arch-fixtures/generate.py

Reads the closes already in test/fixtures/arch-2026-09-25.json (nothing is fetched) and
rewrites both fixtures in place: the closes and their dates are kept, every `arch` answer
and the `recorded` date are new. The file names date the closes, `recorded` the answers.

Returns are ×100 log returns, as the tests compute them. Every fit is `arch`'s default
specification: a constant mean, normal innovations, `arch`'s own single start.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import arch
import numpy as np
from arch import arch_model

FIXTURES = Path(__file__).resolve().parents[2] / 'test' / 'fixtures'
GARCH_FIXTURE = FIXTURES / 'arch-2026-09-25.json'
EGARCH_FIXTURE = FIXTURES / 'arch-egarch-2026-10-01.json'

HORIZON = 20
# the walk-forward's windows: validationWindows' defaults (minTrain 500, step 10)
MIN_TRAIN = 500
STEP = 10
SIMULATIONS = 200_000
SEED = 20261001


def returns_of(closes: list[float]) -> np.ndarray:
    c = np.asarray(closes, dtype=float)
    return np.log(c[1:] / c[:-1]) * 100


def fit(r: np.ndarray, vol: str, o: int):
    return arch_model(r, mean='Constant', vol=vol, p=1, o=o, q=1, dist='normal').fit(disp='off')


def garch_answers(r: np.ndarray, o: int) -> dict:
    res = fit(r, 'GARCH', o)
    p = res.params
    variance = res.forecast(horizon=HORIZON, reindex=False).variance.values[-1]
    gamma = float(p['gamma[1]']) if o else None
    return {
        'omega': float(p['omega']),
        'alpha': float(p['alpha[1]']),
        'beta': float(p['beta[1]']),
        'gamma': gamma,
        'gamma_pvalue': float(res.pvalues['gamma[1]']) if o else None,
        'persistence': float(p['alpha[1]'] + (gamma or 0) / 2 + p['beta[1]']),
        'sigma_daily': float(np.sqrt(variance[0]) / 100),
        'sigma_horizon': float(np.sqrt(variance.sum()) / 100),
    }


def qlike(realized: np.ndarray, forecast: np.ndarray) -> float:
    ratio = np.maximum(realized / forecast, 1e-300)
    return float(np.mean(ratio - np.log(ratio) - 1))


def walk_forward(r: np.ndarray) -> dict:
    """At every origin: fit on r[0 … origin], the mean variance of the next HORIZON returns,
    against the mean squared return they realised; naive is the trailing HORIZON's. An origin
    whose fit did not converge, or whose forecast or naive variance is not positive, is
    skipped — as scoreWalkForward does."""
    realized, forecast, naive = [], [], []
    for origin in range(MIN_TRAIN - 1, len(r) - HORIZON, STEP):
        res = fit(r[: origin + 1], 'GARCH', 0)
        if res.convergence_flag != 0:
            continue
        f = res.forecast(horizon=HORIZON, reindex=False).variance.values[-1].mean()
        b = np.mean(r[origin + 1 - HORIZON : origin + 1] ** 2)
        if not (np.isfinite(f) and f > 0 and b > 0):
            continue
        realized.append(np.mean(r[origin + 1 : origin + 1 + HORIZON] ** 2))
        forecast.append(f)
        naive.append(b)
    realized, forecast, naive = map(np.asarray, (realized, forecast, naive))
    return {'qlike': qlike(realized, forecast), 'naive_qlike': qlike(realized, naive), 'origins': len(realized)}


def egarch_answers(r: np.ndarray) -> dict:
    res = fit(r, 'EGARCH', 1)
    p = res.params
    simulated = res.forecast(
        horizon=HORIZON,
        method='simulation',
        simulations=SIMULATIONS,
        reindex=False,
        rng=np.random.default_rng(SEED).standard_normal,
    ).variance.values[-1]
    return {
        'mu': float(p['mu']),
        'omega': float(p['omega']),
        'alpha': float(p['alpha[1]']),
        'gamma': float(p['gamma[1]']),
        'beta': float(p['beta[1]']),
        'loglik': float(res.loglikelihood),
        'gamma_pvalue': float(res.pvalues['gamma[1]']),
        'converged': res.convergence_flag == 0,
        'variance_1': float(simulated[0]),
        'variance_20_simulated': [float(v) for v in simulated],
    }


def main() -> None:
    today = datetime.date.today().isoformat()
    versions = f'arch {arch.__version__}, numpy {np.__version__}, python {sys.version.split()[0]}'
    garch = json.loads(GARCH_FIXTURE.read_text())
    egarch = {
        'recorded': today,
        'source': (
            f'the closes of {GARCH_FIXTURE.name}; {versions}; '
            "arch_model(mean='Constant', vol='EGARCH', p=1, o=1, q=1, dist='normal').fit(), "
            f"forecast(horizon={HORIZON}, method='simulation', simulations={SIMULATIONS}, "
            f'rng=default_rng({SEED})); scripts/arch-fixtures/generate.py'
        ),
        'series': {},
    }
    for name, s in garch['series'].items():
        print(f'{name}: fitting', file=sys.stderr)
        r = returns_of(s['closes'])
        s['arch'] = {'garch': {**garch_answers(r, 0), **walk_forward(r)}, 'gjr': garch_answers(r, 1)}
        egarch['series'][name] = egarch_answers(r)
    garch['recorded'] = today
    garch['source'] = (
        f'daily closes as recorded; {versions}; '
        "arch_model(mean='Constant', vol='GARCH', p=1, o=0|1, q=1, dist='normal').fit() on ×100 log returns, "
        f'walk-forward minTrain {MIN_TRAIN}, step {STEP}, horizon {HORIZON}; scripts/arch-fixtures/generate.py'
    )
    GARCH_FIXTURE.write_text(json.dumps(garch, ensure_ascii=False))
    EGARCH_FIXTURE.write_text(json.dumps(egarch, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
