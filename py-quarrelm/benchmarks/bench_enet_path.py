# benchmarks/bench_enet_path.py
import time
import numpy as np
import polars as pl
from sklearn.linear_model import lasso_path, enet_path as sklearn_enet_path
import pyarrow as pa
import ctypes

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from quarrelm._core import (
    enet_path,
    _lib,
    _extract_stream_pointer,
    _extract_array_pointers,
)


def generate_data(n, p, sparsity=0.5, seed=42):
    np.random.seed(seed)
    true_coefs = np.random.randn(p)
    n_zero = int(p * sparsity)
    true_coefs[np.random.choice(p, n_zero, replace=False)] = 0.0

    X = np.random.randn(n, p)
    y = X @ true_coefs + np.random.randn(n) * 0.1

    data = {"y": y.tolist()}
    for i in range(p):
        data[f"x{i}"] = X[:, i].tolist()

    return pl.DataFrame(data), X, y, true_coefs


def bench(fn, *args, warmup=3, runs=10, label=""):
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
        f"  {label:45s}  median={median:8.3f}ms  best={best:8.3f}ms  worst={worst:8.3f}ms"
    )
    return median


def bench_sklearn_lasso_path(X, y, n_lambda):
    alphas = np.logspace(
        np.log10(np.max(np.abs(X.T @ y)) / len(y)),
        np.log10(np.max(np.abs(X.T @ y)) / len(y) * 1e-4),
        n_lambda,
    )
    _, coef_path, _ = lasso_path(X, y, alphas=alphas, max_iter=100000, tol=1e-7)
    return coef_path


def bench_sklearn_enet_path(X, y, n_lambda, l1_ratio):
    alphas = np.logspace(
        np.log10(np.max(np.abs(X.T @ y)) / (len(y) * l1_ratio)),
        np.log10(np.max(np.abs(X.T @ y)) / (len(y) * l1_ratio) * 1e-4),
        n_lambda,
    )
    _, coef_path, _ = sklearn_enet_path(
        X, y, alphas=alphas, l1_ratio=l1_ratio, max_iter=100000, tol=1e-7
    )
    return coef_path


def bench_quarrelm_path(df, alpha, n_lambda):
    return enet_path(df, target="y", alpha=alpha, n_lambda=n_lambda)


def bench_zig_only_path(X, y_np, alpha, n_lambda):
    p = X.shape[1]
    table = pa.table({f"x{i}": X[:, i] for i in range(p)})
    y_arr = pa.array(y_np, type=pa.float64())
    stream_ptr, _sc = _extract_stream_pointer(table)
    y_ptr, y_sch, _a, _b = _extract_array_pointers(y_arr)

    out_coef_matrix = np.zeros(p * n_lambda, dtype=np.float64)
    out_lambdas = np.zeros(n_lambda, dtype=np.float64)
    pf = np.ones(p, dtype=np.float64)
    lb = np.full(p, -np.inf, dtype=np.float64)
    ub = np.full(p, np.inf, dtype=np.float64)

    _lib.quarrel_enet_path(
        stream_ptr,
        y_ptr,
        y_sch,
        pf.ctypes.data_as(ctypes.c_void_p),
        lb.ctypes.data_as(ctypes.c_void_p),
        ub.ctypes.data_as(ctypes.c_void_p),
        out_coef_matrix.ctypes.data_as(ctypes.c_void_p),
        out_lambdas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(p),
        ctypes.c_int(n_lambda),
        ctypes.c_double(alpha),
        ctypes.c_double(1e-4),
        ctypes.c_double(1e-7),
        ctypes.c_int(10000),
    )
    return out_coef_matrix


def run_suite(n, p, alpha, n_lambda=100, sparsity=0.5):
    penalty_name = "lasso" if alpha == 1.0 else f"enet(α={alpha})"
    print(f"\n{'=' * 90}")
    print(
        f"  n={n:,}  p={p}  {penalty_name}  {n_lambda} lambdas  (matrix: {n * p * 8 / 1e6:.1f} MB)"
    )
    print(f"{'=' * 90}")

    df, X, y, true_coefs = generate_data(n, p, sparsity=sparsity)
    n_true_nonzero = np.sum(true_coefs != 0)
    print(f"  True nonzero coefficients: {n_true_nonzero}/{p}")

    print(f"\n  End-to-end (data in → coefficient path out):")

    if alpha == 1.0:
        t_sk = bench(
            bench_sklearn_lasso_path,
            X,
            y,
            n_lambda,
            label=f"sklearn lasso_path ({n_lambda}λ, numpy)",
        )
    else:
        t_sk = bench(
            bench_sklearn_enet_path,
            X,
            y,
            n_lambda,
            alpha,
            label=f"sklearn enet_path ({n_lambda}λ, numpy)",
        )

    t_qr = bench(
        bench_quarrelm_path,
        df,
        alpha,
        n_lambda,
        label=f"quarreLM enet_path ({n_lambda}λ, polars)",
    )

    print(f"\n  Zig solver only (arrow export + path):")

    def zig_fresh():
        return bench_zig_only_path(X, y, alpha, n_lambda)

    t_zig = bench(zig_fresh, label=f"Zig enet_path ({n_lambda}λ, arrow export)")

    # Quick correctness check
    result = enet_path(df, target="y", alpha=alpha, n_lambda=n_lambda)
    n_iter = result["n_iter"]
    avg_iter = n_iter / n_lambda

    print(f"\n  Speedups vs sklearn:")
    print(f"    quarreLM path e2e:    {t_sk / t_qr:.2f}x")
    print(f"    Zig path solver only: {t_sk / t_zig:.2f}x")
    print(f"  Path stats: {n_iter} total iterations, {avg_iter:.1f} avg per lambda")


if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("  LASSO PATH BENCHMARKS (alpha=1.0)")
    print("=" * 90)

    run_suite(n=1_000, p=10, alpha=1.0, n_lambda=100)
    run_suite(n=10_000, p=50, alpha=1.0, n_lambda=100)
    run_suite(n=100_000, p=100, alpha=1.0, n_lambda=100)
    run_suite(n=1_000, p=500, alpha=1.0, n_lambda=100)
    run_suite(n=100_000, p=500, alpha=1.0, n_lambda=100)

    print("\n" + "=" * 90)
    print("  ELASTIC NET PATH BENCHMARKS (alpha=0.5)")
    print("=" * 90)

    run_suite(n=1_000, p=10, alpha=0.5, n_lambda=100)
    run_suite(n=10_000, p=50, alpha=0.5, n_lambda=100)
    run_suite(n=100_000, p=100, alpha=0.5, n_lambda=100)
    run_suite(n=1_000, p=500, alpha=0.5, n_lambda=100)
    run_suite(n=100_000, p=500, alpha=0.5, n_lambda=100)
