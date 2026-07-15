import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import lasso_path, enet_path as sklearn_enet_path

from quarrelm.api import enet_path  # noqa: E402
from quarrelm._core import get_version  # noqa: E402


def generate_data(n, p, sparsity=0.5, seed=42):
    """Returns (polars df, numpy X, numpy y, true_coefs)."""
    rng = np.random.default_rng(seed)
    true_coefs = rng.standard_normal(p)
    n_zero = int(p * sparsity)
    if n_zero:
        true_coefs[rng.choice(p, n_zero, replace=False)] = 0.0

    X = rng.standard_normal((n, p))
    y = X @ true_coefs + rng.standard_normal(n) * 0.1

    data = {"y": y}
    for i in range(p):
        data[f"x{i}"] = X[:, i]
    return pl.DataFrame(data), X, y, true_coefs


def bench(fn, *args, warmup=3, runs=10, label=""):
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
        f"  {label:40s}  median={median:9.3f}ms"
        f"  best={times[0]:9.3f}ms  worst={times[-1]:9.3f}ms"
    )
    return median


def _sklearn_alpha_grid(X, y, n_lambda, l1_ratio):
    """A \u03bb grid shaped like our \u03bb_max/\u03bb_min convention (l1_ratio = our alpha)."""
    lam_max = np.max(np.abs(X.T @ y)) / (len(y) * l1_ratio)
    return np.logspace(np.log10(lam_max), np.log10(lam_max * 1e-4), n_lambda)


# --- contenders (each: data in -> coefficient path out) ---


def fit_sklearn_path(X, y, n_lambda, alpha):
    if alpha == 1.0:
        alphas = _sklearn_alpha_grid(X, y, n_lambda, 1.0)
        _, coef_path, _ = lasso_path(X, y, alphas=alphas, max_iter=100000, tol=1e-7)
    else:
        alphas = _sklearn_alpha_grid(X, y, n_lambda, alpha)
        _, coef_path, _ = sklearn_enet_path(
            X, y, alphas=alphas, l1_ratio=alpha, max_iter=100000, tol=1e-7
        )
    return coef_path


def fit_quarrelm_path(df, alpha, n_lambda):
    return enet_path(df, target="y", alpha=alpha, n_lambda=n_lambda)


def run_suite(n, p, alpha, n_lambda=100, sparsity=0.5, runs=10):
    penalty_name = "lasso" if alpha == 1.0 else f"enet(\u03b1={alpha})"
    print(f"\n{'=' * 88}")
    print(
        f"  n={n:,}  p={p}  {penalty_name}  {n_lambda} lambdas  (X: {n * p * 8 / 1e6:.1f} MB)"
    )
    print(f"{'=' * 88}")

    df, X, y, true_coefs = generate_data(n, p, sparsity=sparsity)
    print(f"  True nonzero coefficients: {int(np.sum(true_coefs != 0))}/{p}")

    print("\n  End-to-end (data in -> coefficient path out):")
    sk_label = (
        f"sklearn lasso_path ({n_lambda}\u03bb, numpy)"
        if alpha == 1.0
        else f"sklearn enet_path ({n_lambda}\u03bb, numpy)"
    )
    t_sk = bench(fit_sklearn_path, X, y, n_lambda, alpha, runs=runs, label=sk_label)
    t_q = bench(
        fit_quarrelm_path,
        df,
        alpha,
        n_lambda,
        runs=runs,
        label=f"quarreLM enet_path ({n_lambda}\u03bb, polars)",
    )

    # TODO(zig-bench): the "Zig solver only" path timing (raw Arrow export +
    # quarrel_enet_path via the C ABI) was removed with the _core/ctypes plumbing.
    # It returns as a native `zig build bench` step.

    # quarreLM path iteration stats (its own \u03bb grid, warm-started across the path)
    result = fit_quarrelm_path(df, alpha, n_lambda)
    n_iter = result["n_iter"]
    print("\n  Speedups:")
    print(f"    quarreLM vs sklearn:  {t_sk / t_q:6.2f}x")
    print(f"  Path stats: {n_iter} total iters, {n_iter / n_lambda:.1f} avg per \u03bb")


if __name__ == "__main__":
    import platform
    import sklearn

    print(f"quarreLM elastic-net PATH benchmark \u2014 {time.strftime('%Y-%m-%d')}")
    print(f"quarreLM build: {get_version()}")
    print(
        f"python {platform.python_version()}  numpy {np.__version__}  "
        f"sklearn {sklearn.__version__}"
    )
    print(f"machine: {platform.processor() or platform.machine()}")

    print("\n" + "=" * 88)
    print("  LASSO PATH BENCHMARKS (alpha=1.0)")
    print("=" * 88)
    run_suite(n=1_000, p=10, alpha=1.0, n_lambda=100)
    run_suite(n=10_000, p=50, alpha=1.0, n_lambda=100)
    run_suite(n=100_000, p=100, alpha=1.0, n_lambda=100)
    run_suite(n=1_000, p=500, alpha=1.0, n_lambda=100)
    run_suite(n=100_000, p=500, alpha=1.0, n_lambda=100)

    print("\n" + "=" * 88)
    print("  ELASTIC NET PATH BENCHMARKS (alpha=0.5)")
    print("=" * 88)
    run_suite(n=1_000, p=10, alpha=0.5, n_lambda=100)
    run_suite(n=10_000, p=50, alpha=0.5, n_lambda=100)
    run_suite(n=100_000, p=100, alpha=0.5, n_lambda=100)
    run_suite(n=1_000, p=500, alpha=0.5, n_lambda=100)
    run_suite(n=100_000, p=500, alpha=0.5, n_lambda=100)
