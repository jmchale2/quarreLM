# benchmarks/bench_ols.py
import time
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from quarrelm._core import (
    ols,
    ols_simd,
    _lib,
    _extract_stream_pointer,
    _extract_array_pointers,
)
import ctypes
import pyarrow as pa


def generate_data(n, p, seed=42):
    """Generate test data, return as Polars DataFrame."""
    np.random.seed(seed)
    true_coefs = np.random.randn(p)
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
        elapsed = (time.perf_counter_ns() - start) / 1e6  # ms
        times.append(elapsed)

    times.sort()
    median = times[len(times) // 2]
    best = times[0]
    worst = times[-1]
    print(
        f"  {label:30s}  median={median:8.3f}ms  best={best:8.3f}ms  worst={worst:8.3f}ms"
    )
    return median


def bench_sklearn(X, y):
    """Sklearn end-to-end (numpy in, coefficients out)."""
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    return model.coef_


def bench_statsmodels(X, y):
    """Statsmodels end-to-end."""
    model = sm.OLS(y, X)
    result = model.fit()
    return result.params


def bench_quarrelm(df):
    """quarreLM scalar, end-to-end (Polars in, result out)."""
    return ols(df, target="y")


def bench_quarrelm_simd(df):
    """quarreLM SIMD, end-to-end (Polars in, result out)."""
    return ols_simd(df, target="y")


def bench_zig_only(X, y_np):
    """Zig SIMD, minimal overhead: pre-extracted Arrow pointers."""
    # Pre-convert to Arrow outside the timer
    p = X.shape[1]
    table = pa.table({f"x{i}": X[:, i] for i in range(p)})
    y_arr = pa.array(y_np, type=pa.float64())

    stream_ptr, _sc = _extract_stream_pointer(table)
    y_array_ptr, y_schema_ptr, _yac, _ysc = _extract_array_pointers(y_arr)
    out = np.zeros(p, dtype=np.float64)
    out_ptr = out.ctypes.data_as(ctypes.c_void_p)

    # This measures ONLY the Zig computation
    _lib.quarrel_ols_fit_simd(
        stream_ptr, y_array_ptr, y_schema_ptr, out_ptr, ctypes.c_int(p)
    )
    return out


def run_suite(n, p):
    print(f"\n{'=' * 70}")
    print(f"  n={n:,}  p={p}  (matrix: {n * p * 8 / 1e6:.1f} MB)")
    print(f"{'=' * 70}")

    df, X, y, true_coefs = generate_data(n, p)

    # End-to-end comparisons
    print("\n  End-to-end (data in → coefficients out):")
    t_sk = bench(bench_sklearn, X, y, label="sklearn (numpy)")
    t_sm = bench(bench_statsmodels, X, y, label="statsmodels (numpy)")
    t_scalar = bench(bench_quarrelm, df, label="quarreLM scalar (polars)")
    t_simd = bench(bench_quarrelm_simd, df, label="quarreLM SIMD (polars)")

    # Zig-only (no Python overhead)
    # Need fresh Arrow export each run since PyCapsules are single-use
    print("\n  Zig solver only (pre-extracted Arrow pointers):")

    def zig_only_fresh():
        """Fresh Arrow export + Zig solve each call."""
        table = pa.table({f"x{i}": X[:, i] for i in range(p)})
        y_arr = pa.array(y, type=pa.float64())
        stream_ptr, _sc = _extract_stream_pointer(table)
        y_ptr, y_sch, _a, _b = _extract_array_pointers(y_arr)
        out = np.zeros(p, dtype=np.float64)
        out_ptr = out.ctypes.data_as(ctypes.c_void_p)
        _lib.quarrel_ols_fit_simd(stream_ptr, y_ptr, y_sch, out_ptr, ctypes.c_int(p))
        return out

    t_zig = bench(zig_only_fresh, label="Zig SIMD (arrow export + solve)")

    # Summary
    print(f"\n  Speedups vs sklearn:")
    print(f"    quarreLM scalar e2e:  {t_sk / t_scalar:.2f}x")
    print(f"    quarreLM SIMD e2e:    {t_sk / t_simd:.2f}x")
    print(f"    Zig SIMD solver only: {t_sk / t_zig:.2f}x")
    print(f"  SIMD vs scalar:         {t_scalar / t_simd:.2f}x")


if __name__ == "__main__":
    # Small (sanity check)
    run_suite(n=100, p=5)

    # Medium (typical use case)
    run_suite(n=10_000, p=50)

    # Large (where SIMD should shine)
    run_suite(n=100_000, p=100)

    # Wide (p approaching n)
    run_suite(n=1_000, p=500)
