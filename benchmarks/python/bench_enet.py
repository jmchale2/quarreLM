import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNet, Lasso

from quarrelm.api import enet  # noqa: E402
from quarrelm._core import get_version  # noqa: E402


def generate_data(n, p, sparsity=0.5, seed=42):
    """Test data with a fraction of true-zero coefficients.

    Returns (polars df, numpy X, numpy y, true_coefs).
    """
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
        f"  {label:36s}  median={median:9.3f}ms"
        f"  best={times[0]:9.3f}ms  worst={times[-1]:9.3f}ms"
    )
    return median


# --- contenders (each: data in -> coefficient vector out) ---


def fit_sklearn(X, y, alpha, lambda_, tol=1e-7):
    """Lasso when alpha==1, else ElasticNet (sklearn alpha=our lambda, l1_ratio=our alpha)."""
    if alpha == 1.0:
        model = Lasso(alpha=lambda_, fit_intercept=False, max_iter=100000, tol=tol)
    else:
        model = ElasticNet(
            alpha=lambda_, l1_ratio=alpha, fit_intercept=False, max_iter=100000, tol=tol
        )
    return model.fit(X, y).coef_


def fit_quarrelm(df, alpha, lambda_, tol=1e-7):
    return enet(df, target="y", alpha=alpha, lambda_=lambda_, max_iter=100000, tol=tol)


def check_correctness(df, X, y, alpha, lambda_):
    """quarreLM must agree with sklearn (tight tol) before we time anything."""
    sk_coefs = fit_sklearn(X, y, alpha, lambda_, tol=1e-10)
    result = fit_quarrelm(df, alpha, lambda_, tol=1e-10)
    zig_coefs = result.coef_array

    max_diff = float(np.max(np.abs(zig_coefs - sk_coefs)))
    n_nonzero = int(np.sum(np.abs(zig_coefs) > 1e-10))
    flag = "  <-- DISAGREES, timings below are meaningless" if max_diff > 1e-3 else ""
    print(f"  quarreLM vs sklearn   max|diff| = {max_diff:.2e}{flag}")
    print(f"  sparsity: {n_nonzero}/{len(zig_coefs)} nonzero ({result.n_iter} iters)")
    return max_diff <= 1e-3


def run_suite(n, p, alpha, lambda_, sparsity=0.5, runs=20):
    penalty_name = "lasso" if alpha == 1.0 else f"enet(\u03b1={alpha})"
    print(f"\n{'=' * 78}")
    print(
        f"  n={n:,}  p={p}  \u03bb={lambda_}  {penalty_name}  (X: {n * p * 8 / 1e6:.1f} MB)"
    )
    print(f"{'=' * 78}")

    df, X, y, true_coefs = generate_data(n, p, sparsity=sparsity)
    print(f"  True nonzero coefficients: {int(np.sum(true_coefs != 0))}/{p}")

    print("\n  Correctness (vs sklearn):")
    check_correctness(df, X, y, alpha, lambda_)

    print("\n  End-to-end (data in -> coefficients out):")
    sk_label = "sklearn Lasso (numpy)" if alpha == 1.0 else "sklearn ElasticNet (numpy)"
    t_sk = bench(fit_sklearn, X, y, alpha, lambda_, runs=runs, label=sk_label)
    t_q = bench(
        fit_quarrelm, df, alpha, lambda_, runs=runs, label="quarreLM enet (polars)"
    )

    # TODO(zig-bench): the "Zig solver only" timing (raw Arrow export + solve via
    # the C ABI, pre-extracted pointers) was removed with the _core/ctypes plumbing.
    # It returns as a native `zig build bench` step so the solver is timed without
    # the Python boundary in the loop.

    print("\n  Speedups:")
    print(f"    quarreLM vs sklearn:  {t_sk / t_q:6.2f}x")


if __name__ == "__main__":
    import platform
    import sklearn

    print(f"quarreLM elastic-net benchmark \u2014 {time.strftime('%Y-%m-%d')}")
    print(f"quarreLM build: {get_version()}")
    print(
        f"python {platform.python_version()}  numpy {np.__version__}  "
        f"sklearn {sklearn.__version__}"
    )
    print(f"machine: {platform.processor() or platform.machine()}")

    # --- Lasso (alpha=1.0) ---
    print("\n" + "=" * 78)
    print("  LASSO BENCHMARKS (alpha=1.0)")
    print("=" * 78)
    run_suite(n=100, p=5, alpha=1.0, lambda_=0.05)
    run_suite(n=10_000, p=50, alpha=1.0, lambda_=0.05)
    run_suite(n=100_000, p=100, alpha=1.0, lambda_=0.05)
    run_suite(n=1_000, p=500, alpha=1.0, lambda_=0.1, runs=10)

    # --- Elastic Net (alpha=0.5) ---
    print("\n" + "=" * 78)
    print("  ELASTIC NET BENCHMARKS (alpha=0.5)")
    print("=" * 78)
    run_suite(n=100, p=5, alpha=0.5, lambda_=0.05)
    run_suite(n=10_000, p=50, alpha=0.5, lambda_=0.05)
    run_suite(n=100_000, p=100, alpha=0.5, lambda_=0.05)
    run_suite(n=100_000, p=500, alpha=0.5, lambda_=0.05, runs=10)
    run_suite(n=1_000, p=500, alpha=0.5, lambda_=0.1, runs=10)
