# benchmarks/bench_ols.py
"""OLS benchmark: quarreLM (GE vs Cholesky) against sklearn / statsmodels / numpy.

All contenders solve the same no-intercept least-squares problem.
Coefficients are cross-checked against numpy lstsq before anything is timed.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy.linalg import solve as scipy_solve

from quarrelm.api import ols  # noqa: E402
from quarrelm._core import get_version


def generate_data(n, p, seed=42):
    """Test data as (polars df, numpy X, numpy y)."""
    rng = np.random.default_rng(seed)
    true_coefs = rng.standard_normal(p)
    X = rng.standard_normal((n, p))
    y = X @ true_coefs + rng.standard_normal(n) * 0.1

    data = {"y": y}
    for i in range(p):
        data[f"x{i}"] = X[:, i]
    return pl.DataFrame(data), pd.DataFrame(data), X, y


def bench(fn, *args, warmup=3, runs=20, label=""):
    """Median-of-runs wall time in ms."""
    for _ in range(warmup):
        fn(*args)

    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        fn(*args)
        times.append((time.perf_counter_ns() - start) / 1e6)

    times.sort()
    median = times[len(times) // 2]
    print(
        f"  {label:32s}  median={median:9.3f}ms"
        f"  best={times[0]:9.3f}ms  worst={times[-1]:9.3f}ms"
    )
    return median


# --- contenders (each: data in -> coefficient vector out) ---


def fit_numpy(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def fit_normal_eq(X, y):
    """Normal equations via LAPACK dposv — numpy/scipy doing quarreLM's algorithm."""
    return scipy_solve(X.T @ X, X.T @ y, assume_a="pos")


def fit_sklearn(X, y):
    return LinearRegression(fit_intercept=False).fit(X, y).coef_


def fit_statsmodels(X, y):
    return sm.OLS(y, X).fit().params


def fit_quarrelm_ge(df):
    return ols(df, target="y", method="ge").coef_array


def fit_quarrelm_cholesky(df):
    return ols(df, target="y", method="cholesky").coef_array


def check_correctness(df, X, y):
    """Every contender must agree with numpy lstsq before we time anything."""
    ref = fit_numpy(X, y)
    results = {
        "normal eq (scipy pos)": fit_normal_eq(X, y),
        "sklearn": fit_sklearn(X, y),
        "statsmodels": np.asarray(fit_statsmodels(X, y)),
        "quarreLM GE": fit_quarrelm_ge(df),
        "quarreLM cholesky": fit_quarrelm_cholesky(df),
    }
    ok = True
    for label, coefs in results.items():
        max_diff = float(np.max(np.abs(coefs - ref)))
        flag = ""
        if not np.allclose(coefs, ref, rtol=1e-6, atol=1e-8):
            flag = "  <-- DISAGREES, timings below are meaningless"
            ok = False
        print(f"  {label:32s}  max|diff vs lstsq| = {max_diff:.2e}{flag}")
    return ok


def run_suite(n, p, runs=20):
    print(f"\n{'=' * 74}")
    print(f"  n={n:,}  p={p}  (X: {n * p * 8 / 1e6:.1f} MB)")
    print(f"{'=' * 74}")

    df_pl, df_pd, X, y = generate_data(n, p)

    print("\n  Correctness (vs numpy lstsq):")
    check_correctness(df_pl, X, y)

    print("\n  End-to-end (data in -> coefficients out):")
    t_np = bench(fit_numpy, X, y, runs=runs, label="numpy lstsq (numpy)")
    t_ne = bench(fit_normal_eq, X, y, runs=runs, label="normal eq dposv (numpy)")
    t_sk = bench(fit_sklearn, X, y, runs=runs, label="sklearn (numpy)")
    t_sm = bench(fit_statsmodels, X, y, runs=runs, label="statsmodels (numpy)")
    t_ge = bench(fit_quarrelm_ge, df_pl, runs=runs, label="quarreLM GE (polars)")
    t_ch = bench(
        fit_quarrelm_cholesky, df_pl, runs=runs, label="quarreLM cholesky (polars)"
    )
    t_cp = bench(
        fit_quarrelm_cholesky, df_pd, runs=runs, label="quarreLM cholesky (pandas)"
    )

    print("\n  Speedups:")
    print(f"    cholesky vs GE:          {t_ge / t_ch:6.2f}x")
    print(f"    cholesky vs sklearn:     {t_sk / t_ch:6.2f}x")
    print(f"    cholesky vs statsmodels: {t_sm / t_ch:6.2f}x")
    print(f"    cholesky vs numpy lstsq: {t_np / t_ch:6.2f}x")
    print(f"    cholesky vs normal eq:   {t_ne / t_ch:6.2f}x")
    print(f"    choleskypd vs normal eq: {t_cp / t_ch:6.2f}x")


if __name__ == "__main__":
    import platform, sklearn, scipy

    print(f"quarreLM OLS benchmark — {time.strftime('%Y-%m-%d')}")
    print(f"quarreLM build: {get_version()}")
    print(
        f"python {platform.python_version()}  numpy {np.__version__}  "
        f"scipy {scipy.__version__}  sklearn {sklearn.__version__}"
    )
    print(f"machine: {platform.processor() or platform.machine()}")

    # Small: overhead-dominated (measures the boundary, not the solver)
    run_suite(n=100, p=5)

    # Typical DS regression shapes
    run_suite(n=10_000, p=50)
    run_suite(n=100_000, p=100)

    # Tall-skinny: the bread-and-butter case, gram formation dominates
    run_suite(n=1_000_000, p=20, runs=10)

    # Wide: p pressure — where O(p^3) methods separate
    run_suite(n=1_000, p=500, runs=10)

    # Tall-Wide: p pressure — where O(p^3) methods separate
    run_suite(n=100_000, p=500, runs=10)
