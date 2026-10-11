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
import warnings
from pathlib import Path

import arch
import numpy as np
from arch import arch_model
from arch.univariate import SkewStudent
from scipy import optimize, stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

FIXTURES = Path(__file__).resolve().parents[2] / 'test' / 'fixtures'
GARCH_FIXTURE = FIXTURES / 'arch-2026-09-25.json'
EGARCH_FIXTURE = FIXTURES / 'arch-egarch-2026-10-01.json'

HORIZON = 20
# the walk-forward's windows: validationWindows' defaults (minTrain 500, step 10)
MIN_TRAIN = 500
STEP = 10
SIMULATIONS = 200_000
SEED = 20261001
DIAGNOSTIC_LAGS = 10


def returns_of(closes: list[float]) -> np.ndarray:
    c = np.asarray(closes, dtype=float)
    return np.log(c[1:] / c[:-1]) * 100


def fit(r: np.ndarray, vol: str, o: int, mean: str = 'Constant', dist: str = 'normal'):
    return arch_model(r, mean=mean, vol=vol, p=1, o=o, q=1, dist=dist).fit(disp='off')


def criteria(res) -> dict:
    return {'loglik': float(res.loglikelihood), 'aic': float(res.aic), 'bic': float(res.bic)}


def diagnostics(res) -> dict:
    """Ljung–Box on z and z², ARCH-LM on z, over the fit's standardised residuals."""
    z = np.asarray(res.std_resid)
    out = {'lags': DIAGNOSTIC_LAGS}
    for key, series in (('ljung_box_z', z), ('ljung_box_z2', z**2)):
        lb = acorr_ljungbox(series, lags=[DIAGNOSTIC_LAGS])
        out[key] = {'statistic': float(lb['lb_stat'].iloc[0]), 'p_value': float(lb['lb_pvalue'].iloc[0])}
    with warnings.catch_warnings():  # statsmodels 0.15 warns that het_arch's tuple will become a NamedTuple
        warnings.simplefilter('ignore', FutureWarning)
        lm, p, _, _ = het_arch(z, nlags=DIAGNOSTIC_LAGS)
    out['arch_lm'] = {'statistic': float(lm), 'p_value': float(p)}
    return out


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
        **criteria(res),
        'diagnostics': diagnostics(res),
    }


def zero_mean_answers(r: np.ndarray) -> dict:
    """Each model with mean='Zero': its parameters, likelihood, criteria and γ's robust p-value."""
    out = {}
    for model, vol, o in (('garch', 'GARCH', 0), ('gjr', 'GARCH', 1), ('egarch', 'EGARCH', 1)):
        res = fit(r, vol, o, 'Zero')
        p = res.params
        out[model] = {
            'omega': float(p['omega']),
            'alpha': float(p['alpha[1]']),
            'gamma': float(p['gamma[1]']) if o else None,
            'beta': float(p['beta[1]']),
            'gamma_pvalue': float(res.pvalues['gamma[1]']) if o else None,
            **criteria(res),
        }
    return out


def skewt_answers(r: np.ndarray) -> dict:
    """garch and gjr with Hansen's skewed t: parameters (η as nu, λ as lambda), likelihood, criteria, p-values."""
    out = {}
    for model, o in (('garch', 0), ('gjr', 1)):
        res = fit(r, 'GARCH', o, dist='skewt')
        p = res.params
        out[model] = {
            'mu': float(p['mu']),
            'omega': float(p['omega']),
            'alpha': float(p['alpha[1]']),
            'gamma': float(p['gamma[1]']) if o else None,
            'beta': float(p['beta[1]']),
            'nu': float(p['eta']),
            'lambda': float(p['lambda']),
            'gamma_pvalue': float(res.pvalues['gamma[1]']) if o else None,
            'lambda_pvalue': float(res.pvalues['lambda']),
            'converged': res.convergence_flag == 0,
            **criteria(res),
        }
    return out


def skewt_distribution() -> list:
    """arch's SkewStudent itself: quantiles and partial moments E[zᵏ·1{z < x}] at a few (η, λ)."""
    d = SkewStudent()
    out = []
    for nu, lam in ((5.0, 0.0), (5.0, -0.3), (8.0, 0.4), (4.5, -0.7)):
        params = np.array([nu, lam])
        out.append({
            'nu': nu,
            'lambda': lam,
            'quantiles': {str(q): float(d.ppf(q, params)) for q in (0.001, 0.01, 0.05, 0.5, 0.9)},
            'partial_moments': [
                {'x': x, 'k': k, 'value': float(d.partial_moment(k, x, params))}
                for x in (-2.0, 0.0, 1.0)
                for k in (0, 1, 2)
            ],
        })
    return out


EXTEND_BY = 250


def extend_answers(r: np.ndarray) -> dict:
    """gjr fitted on all but the last EXTEND_BY returns, then those parameters held fixed over
    the whole series (`fix`): the variance the recursion reaches, the one-step forecast, and
    the log-likelihood — what extendFit carries a fit to without re-estimating it."""
    res = fit(r[:-EXTEND_BY], 'GARCH', 1)
    fixed = arch_model(r, mean='Constant', vol='GARCH', p=1, o=1, q=1, dist='normal').fix(res.params)
    p = res.params
    return {
        'added': EXTEND_BY,
        'mu': float(p['mu']),
        'omega': float(p['omega']),
        'alpha': float(p['alpha[1]']),
        'gamma': float(p['gamma[1]']),
        'beta': float(p['beta[1]']),
        'last_variance': float(fixed.conditional_volatility[-1] ** 2),
        'forecast_1': float(fixed.forecast(horizon=1, reindex=False).variance.values[-1][0]),
        'loglik': float(fixed.loglikelihood),
    }


def gjr_loglik(r: np.ndarray, mu: float, omega: float, alpha: float, gamma: float, beta: float) -> float:
    """arch's GJR-GARCH(1,1) normal log-likelihood, written out: the recursion from arch's
    backcast (0.94-weighted mean of the first 75 squared residuals about the sample mean)."""
    e = r - mu
    tau = min(75, len(r))
    w = 0.94 ** np.arange(tau)
    bc = float(w @ (r[:tau] - r.mean()) ** 2 / w.sum())
    s = np.empty(len(r))
    prev_e2, prev_neg, prev_s = bc, 0.5 * bc, bc
    for t in range(len(r)):
        s[t] = omega + alpha * prev_e2 + gamma * prev_neg + beta * prev_s
        prev_e2 = e[t] ** 2
        prev_neg = prev_e2 if e[t] < 0 else 0.0
        prev_s = s[t]
    return float(-0.5 * np.sum(np.log(2 * np.pi) + np.log(s) + e**2 / s))


def targeting_answers(r: np.ndarray) -> dict:
    """arch 8 has no variance targeting: the same likelihood (checked against arch's own at its
    optimum) maximised by scipy with ω = var(r)·(1 − α − γ/2 − β), var about the sample mean,
    from several starts."""
    target = float(np.var(r))
    out = {'variance': target}
    for model, o in (('garch', 0), ('gjr', 1)):
        res = fit(r, 'GARCH', o)
        p = res.params
        free = gjr_loglik(r, p['mu'], p['omega'], p['alpha[1]'], p['gamma[1]'] if o else 0.0, p['beta[1]'])
        assert abs(free - res.loglikelihood) < 1e-6, (model, free, res.loglikelihood)

        def unpack(x):
            mu, alpha, beta = x[0], x[1], x[2]
            gamma = x[3] if o else 0.0
            return mu, target * (1 - alpha - gamma / 2 - beta), alpha, gamma, beta

        def negative(x):
            with np.errstate(invalid='ignore', divide='ignore'):  # a trial step off the valid region
                value = -gjr_loglik(r, *unpack(x))
            return value if np.isfinite(value) else 1e300

        constraints = [
            {'type': 'ineq', 'fun': lambda x: 1 - 1e-6 - (x[1] + (x[3] / 2 if o else 0) + x[2])},
        ]
        if o:
            constraints.append({'type': 'ineq', 'fun': lambda x: x[1] + x[3]})
        bounds = [(None, None), (0, 1), (0, 1)] + ([(-1, 2)] if o else [])
        best = None
        for alpha, beta in ((p['alpha[1]'], p['beta[1]']), (0.05, 0.9), (0.1, 0.8), (0.02, 0.97)):
            x0 = [p['mu'], alpha, beta] + ([max(p['gamma[1]'], 0.0)] if o else [])
            run = optimize.minimize(
                negative, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                options={'ftol': 1e-14, 'maxiter': 1000},
            )
            if best is None or run.fun < best.fun:
                best = run
        mu, omega, alpha, gamma, beta = unpack(best.x)
        out[model] = {
            'mu': float(mu),
            'omega': float(omega),
            'alpha': float(alpha),
            'gamma': float(gamma) if o else None,
            'beta': float(beta),
            'loglik': float(-best.fun),
        }
    return out


DCC_HORIZON = 5
DCC_WEIGHTS = (0.3, 0.7)


def dcc_correlation(z: np.ndarray, a: float, b: float) -> tuple[float, np.ndarray, list]:
    """DCC(1,1) with correlation targeting over z (T×N): the correlation part of the Gaussian
    log-likelihood, Q_T and every R_t. Q_1 = Q̄ = zᵀz/T."""
    T, N = z.shape
    qbar = z.T @ z / T
    q = qbar.copy()
    total = 0.0
    rs = []
    for t in range(T):
        if t > 0:
            q = (1 - a - b) * qbar + a * np.outer(z[t - 1], z[t - 1]) + b * q
        d = 1 / np.sqrt(np.diag(q))
        r = q * np.outer(d, d)
        rs.append(r)
        sign, logdet = np.linalg.slogdet(r)
        if sign <= 0:
            return -np.inf, q, rs
        total += -0.5 * (logdet + z[t] @ np.linalg.solve(r, z[t]) - z[t] @ z[t])
    return total, q, rs


def dcc_answers(series: dict) -> dict:
    """arch has no DCC: stage 1 is arch's GARCH(1,1) on each series, stage 2 written out here and
    maximised by scipy (a, b ≥ 0, a + b < 1). IOVA and SPY over IOVA's whole history, aligned by
    position from their common last date. The forecast is rmgarch's: R_{T+1} exact, then
    R_{T+k} = (1 − (a+b)^{k−1})·R̄ + (a+b)^{k−1}·R_{T+1}."""
    rs = [returns_of(series[name]['closes']) for name in ('iova', 'spy')]
    n = min(len(r) for r in rs)
    rs = [r[-n:] for r in rs]
    stage1 = [fit(r, 'GARCH', 0) for r in rs]
    z = np.column_stack([np.asarray(res.std_resid) for res in stage1])

    def negative(x):
        value, _, _ = dcc_correlation(z, x[0], x[1])
        return -value if np.isfinite(value) else 1e300

    best = None
    for x0 in ((0.01, 0.97), (0.03, 0.95), (0.05, 0.9)):
        run = optimize.minimize(
            negative, x0, method='SLSQP', bounds=[(0, 1), (0, 1)],
            constraints=[{'type': 'ineq', 'fun': lambda x: 1 - 1e-6 - x[0] - x[1]}],
            options={'ftol': 1e-14, 'maxiter': 1000},
        )
        if best is None or run.fun < best.fun:
            best = run
    a, b = (float(v) for v in best.x)
    loglik, q_last, r_path = dcc_correlation(z, a, b)
    qbar = z.T @ z / len(z)
    q1 = (1 - a - b) * qbar + a * np.outer(z[-1], z[-1]) + b * q_last
    normalise = lambda m: m / np.sqrt(np.outer(np.diag(m), np.diag(m)))
    r1, rbar = normalise(q1), normalise(qbar)
    variance = np.column_stack(
        [res.forecast(horizon=DCC_HORIZON, reindex=False).variance.values[-1] for res in stage1]
    )
    correlation, covariance = [], []
    for k in range(DCC_HORIZON):
        r = r1 if k == 0 else (1 - (a + b) ** k) * rbar + (a + b) ** k * r1
        sd = np.sqrt(variance[k])
        correlation.append(r.tolist())
        covariance.append((r * np.outer(sd, sd)).tolist())
    w = np.asarray(DCC_WEIGHTS)
    mean = float(w @ [res.params['mu'] for res in stage1])
    pv = float(w @ np.asarray(covariance[0]) @ w)
    level = 0.01
    return {
        'returns': n,
        'stage1': [
            {k: float(res.params[key]) for k, key in (('mu', 'mu'), ('omega', 'omega'), ('alpha', 'alpha[1]'), ('beta', 'beta[1]'))}
            for res in stage1
        ],
        'a': a,
        'b': b,
        'correlation_loglik': float(loglik),
        'loglik': float(loglik + sum(res.loglikelihood for res in stage1)),
        'last_correlation': r_path[-1].tolist(),
        'correlation': correlation,
        'covariance': covariance,
        'weights': list(DCC_WEIGHTS),
        'portfolio_variance': pv,
        'var_1pct': float(-(mean + np.sqrt(pv) * stats.norm.ppf(level))),
    }


FHS_HORIZON = 10
FHS_SIMULATIONS = 20_000


def fhs_answers(r: np.ndarray) -> dict:
    """Filtered historical simulation on a gjr fit: the one-step VaR and ES from the empirical
    quantile of arch's standardised residuals (the ⌈level·n⌉-th smallest, the mean of those at or
    below it — computed here, arch has no such function), and arch's own bootstrap forecast
    (`method='bootstrap'`), which resamples the same residuals through the recursion."""
    res = fit(r, 'GARCH', 1)
    p = res.params
    z = np.sort(np.asarray(res.std_resid))
    variance_1 = float(res.forecast(horizon=1, reindex=False).variance.values[-1][0])
    tail = {}
    for level in (0.01, 0.05):
        k = int(np.ceil(level * len(z)))
        tail[str(level)] = {
            'var': float(-(p['mu'] + np.sqrt(variance_1) * z[k - 1])),
            'es': float(-(p['mu'] + np.sqrt(variance_1) * z[:k].mean())),
        }
    bootstrap = res.forecast(
        horizon=FHS_HORIZON,
        method='bootstrap',
        simulations=FHS_SIMULATIONS,
        random_state=np.random.RandomState(SEED),
        reindex=False,
    ).variance.values[-1]
    return {
        'mu': float(p['mu']),
        'omega': float(p['omega']),
        'alpha': float(p['alpha[1]']),
        'gamma': float(p['gamma[1]']),
        'beta': float(p['beta[1]']),
        'variance_1': variance_1,
        'tail': tail,
        'bootstrap_variance': [float(v) for v in bootstrap],
        'bootstrap_simulations': FHS_SIMULATIONS,
    }


def qlike_terms(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    ratio = np.maximum(realized / forecast, 1e-300)
    return ratio - np.log(ratio) - 1


def qlike(realized: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(qlike_terms(realized, forecast)))


def diebold_mariano(a: np.ndarray, b: np.ndarray, h: int, kernel: str = 'bartlett') -> dict:
    """Diebold–Mariano by hand (no reference library has it): d = a − b, a HAC variance of its
    mean with h − 1 lags (Bartlett 1 − k/h, or uniform), the Harvey–Leybourne–Newbold
    correction, a two-sided p-value from t(n − 1)."""
    d = a - b
    n = len(d)
    dc = d - d.mean()
    gamma = [float(dc[k:] @ dc[: n - k]) / n for k in range(h)]
    weights = [1.0] + [2 * (1 - k / h if kernel == 'bartlett' else 1) for k in range(1, h)]
    long_run = sum(w * g for w, g in zip(weights, gamma))
    dm = d.mean() / np.sqrt(long_run / n)
    hln = dm * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {'statistic': float(hln), 'p_value': float(2 * stats.t.sf(abs(hln), n - 1)), 'mean': float(d.mean())}


def diebold_mariano_cases() -> list:
    """Two seeded, autocorrelated loss series, every (h, kernel) the tests check."""
    rng = np.random.default_rng(SEED)
    shocks = rng.standard_normal(81)
    a = np.exp(0.3 * (shocks[1:] + 0.6 * shocks[:-1]))
    b = np.exp(0.3 * rng.standard_normal(80)) + 0.05
    return [
        {'h': h, 'kernel': kernel, 'a': a.tolist(), 'b': b.tolist(), **diebold_mariano(a, b, h, kernel)}
        for h in (1, 3, 5)
        for kernel in ('bartlett', 'uniform')
    ]


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
    return {
        'qlike': qlike(realized, forecast),
        'naive_qlike': qlike(realized, naive),
        'origins': len(realized),
        # windows STEP apart, HORIZON long: ⌈HORIZON/STEP⌉ consecutive ones overlap
        'diebold_mariano': diebold_mariano(
            qlike_terms(realized, forecast), qlike_terms(realized, naive), -(-HORIZON // STEP)
        ),
    }


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
        **criteria(res),
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
        s['arch'] = {
            'garch': {**garch_answers(r, 0), **walk_forward(r)},
            'gjr': garch_answers(r, 1),
            'zero_mean': zero_mean_answers(r),
            'targeting': targeting_answers(r),
        }
        if name == 'spy':  # IOVA's quantized history makes a t fit degenerate (see quantizedUntil)
            s['arch']['skewt'] = skewt_answers(r)
            s['arch']['extend'] = extend_answers(r)
            s['arch']['fhs'] = fhs_answers(r)
        egarch['series'][name] = egarch_answers(r)
    garch['skewt_distribution'] = skewt_distribution()
    garch['diebold_mariano'] = diebold_mariano_cases()
    garch['dcc'] = dcc_answers(garch['series'])
    garch['recorded'] = today
    garch['source'] = (
        f'daily closes as recorded; {versions}; '
        "arch_model(mean='Constant' (zero_mean: 'Zero'), vol='GARCH'|'EGARCH', p=1, o=0|1, q=1, dist='normal').fit() "
        'on ×100 log returns, '
        f'walk-forward minTrain {MIN_TRAIN}, step {STEP}, horizon {HORIZON}; scripts/arch-fixtures/generate.py'
    )
    GARCH_FIXTURE.write_text(json.dumps(garch, ensure_ascii=False))
    EGARCH_FIXTURE.write_text(json.dumps(egarch, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
