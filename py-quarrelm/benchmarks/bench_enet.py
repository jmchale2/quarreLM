# benchmarks/bench_enet.py
import time
import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNet, Lasso
import pyarrow as pa
import ctypes

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from quarrelm._core import (
    enet,
    _lib,
    _extract_stream_pointer,
    _extract_array_pointers,
)


def generate_data(n, p, sparsity=0.5, seed=42):
    """Generate test data with some true zeros."""
    np.random.seed(seed)
    true_coefs = np.random.randn(p)
    # Zero out a fraction of coefficients
    n_zero = int(p * sparsity)
    true_coefs[np.random.choice(p, n_zero, replace=False)] = 0.0

    X = np.random.randn(n, p)
    y = X @ true_coefs + np.random.randn(n) * 0.1

    data = {"y": y.tolist()}
    for i in range(p):
        data[f"x{i}"] = X[:, i].tolist()

    return pl.DataFrame(data), X, y, true_coefs


def bench(fn, *args, warmup=3, runs=20, label=""):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn(*args)

    times = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        fn(*args)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        times.append(elapsed)

    times.sort()
    median = times[len(times) // 2]
    best = times[0]
    worst = times[-1]
    print(
        f"  {label:40s}  median={median:8.3f}ms  best={best:8.3f}ms  worst={worst:8.3f}ms"
    )
    return median


def bench_sklearn_lasso(X, y, lambda_):
    sk = Lasso(alpha=lambda_, fit_intercept=False, max_iter=100000, tol=1e-7)
    sk.fit(X, y)
    return sk.coef_


def bench_sklearn_enet(X, y, lambda_, alpha):
    sk = ElasticNet(
        alpha=lambda_, l1_ratio=alpha, fit_intercept=False, max_iter=100000, tol=1e-7
    )
    sk.fit(X, y)
    return sk.coef_


def bench_quarrelm_enet(df, alpha, lambda_):
    return enet(df, target="y", alpha=alpha, lambda_=lambda_, max_iter=100000, tol=1e-7)


def bench_zig_only_enet(X, y_np, alpha, lambda_):
    p = X.shape[1]
    table = pa.table({f"x{i}": X[:, i] for i in range(p)})
    y_arr = pa.array(y_np, type=pa.float64())
    stream_ptr, _sc = _extract_stream_pointer(table)
    y_ptr, y_sch, _a, _b = _extract_array_pointers(y_arr)
    out = np.zeros(p, dtype=np.float64)
    out_ptr = out.ctypes.data_as(ctypes.c_void_p)

    pf = np.ones(p, dtype=np.float64)
    lb = np.full(p, -np.inf, dtype=np.float64)
    ub = np.full(p, np.inf, dtype=np.float64)
    pf_ptr = pf.ctypes.data_as(ctypes.c_void_p)
    lb_ptr = lb.ctypes.data_as(ctypes.c_void_p)
    ub_ptr = ub.ctypes.data_as(ctypes.c_void_p)

    _lib.quarrel_enet_fit(
        stream_ptr,
        y_ptr,
        y_sch,
        pf_ptr,
        lb_ptr,
        ub_ptr,
        out_ptr,
        ctypes.c_int(p),
        ctypes.c_double(lambda_),
        ctypes.c_double(alpha),
        ctypes.c_double(1e-7),
        ctypes.c_int(100000),
    )
    return out


def run_suite(n, p, alpha, lambda_, sparsity=0.5):
    penalty_name = "lasso" if alpha == 1.0 else f"enet(α={alpha})"
    print(f"\n{'=' * 80}")
    print(
        f"  n={n:,}  p={p}  λ={lambda_}  {penalty_name}  (matrix: {n * p * 8 / 1e6:.1f} MB)"
    )
    print(f"{'=' * 80}")

    df, X, y, true_coefs = generate_data(n, p, sparsity=sparsity)
    n_true_nonzero = np.sum(true_coefs != 0)
    print(f"  True nonzero coefficients: {n_true_nonzero}/{p}")

    # End-to-end comparisons
    print(f"\n  End-to-end (data in → coefficients out):")

    if alpha == 1.0:
        t_sk = bench(bench_sklearn_lasso, X, y, lambda_, label="sklearn Lasso (numpy)")
    else:
        t_sk = bench(
            bench_sklearn_enet,
            X,
            y,
            lambda_,
            alpha,
            label=f"sklearn ElasticNet (numpy)",
        )

    t_enet = bench(
        bench_quarrelm_enet, df, alpha, lambda_, label="quarreLM enet (polars)"
    )

    # Zig-only
    print(f"\n  Zig solver only (arrow export + solve):")

    def zig_fresh():
        return bench_zig_only_enet(X, y, alpha, lambda_)

    t_zig = bench(zig_fresh, label="Zig enet (arrow export + solve)")

    # Quick correctness check
    if alpha == 1.0:
        sk_coefs = (
            Lasso(alpha=lambda_, fit_intercept=False, max_iter=100000, tol=1e-10)
            .fit(X, y)
            .coef_
        )
    else:
        sk_coefs = (
            ElasticNet(
                alpha=lambda_,
                l1_ratio=alpha,
                fit_intercept=False,
                max_iter=100000,
                tol=1e-10,
            )
            .fit(X, y)
            .coef_
        )

    zig_result = enet(
        df, target="y", alpha=alpha, lambda_=lambda_, max_iter=100000, tol=1e-10
    )
    zig_coefs = zig_result.coef_array
    max_diff = np.max(np.abs(zig_coefs - sk_coefs))
    n_zig_nonzero = np.sum(np.abs(zig_coefs) > 1e-10)

    # Summary
    print(f"\n  Speedups vs sklearn:")
    print(f"    quarreLM enet e2e:    {t_sk / t_enet:.2f}x")
    print(f"    Zig enet solver only: {t_sk / t_zig:.2f}x")
    print(f"  Correctness: max |zig - sklearn| = {max_diff:.2e}")
    print(f"  Sparsity: {n_zig_nonzero}/{p} nonzero ({zig_result.n_iter} iterations)")


if __name__ == "__main__":
    # --- Lasso (alpha=1.0) ---
    print("\n" + "=" * 80)
    print("  LASSO BENCHMARKS (alpha=1.0)")
    print("=" * 80)

    run_suite(n=100, p=5, alpha=1.0, lambda_=0.05)
    run_suite(n=10_000, p=50, alpha=1.0, lambda_=0.05)
    run_suite(n=100_000, p=100, alpha=1.0, lambda_=0.05)
    run_suite(n=1_000, p=500, alpha=1.0, lambda_=0.1)

    # --- Elastic Net (alpha=0.5) ---
    print("\n" + "=" * 80)
    print("  ELASTIC NET BENCHMARKS (alpha=0.5)")
    print("=" * 80)

    run_suite(n=100, p=5, alpha=0.5, lambda_=0.05)
    run_suite(n=10_000, p=50, alpha=0.5, lambda_=0.05)
    run_suite(n=100_000, p=100, alpha=0.5, lambda_=0.05)
    run_suite(n=100_000, p=500, alpha=0.5, lambda_=0.05)
    run_suite(n=1_000, p=500, alpha=0.5, lambda_=0.1)
