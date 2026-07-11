# tests/test_core.py — FFI boundary tests: do parameters and data actually
# cross the ctypes/C-ABI boundary intact? Math correctness lives in test_enet.py.
import ctypes
import pytest
import numpy as np
import polars as pl

import sys


sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
from quarrelm import errors
from quarrelm._core import (
    _lib,
    enet,
    quarrel_fit,
    quarrel_fit_path,
    SOLVER,
    quarrel_error_name,
    quarrel_last_error,
)
from quarrelm._params import FitOptions
from quarrelm.errors import ErrorCode


def _small_df(n: int = 200, p: int = 3, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    coefs = np.arange(1, p + 1, dtype=np.float64)  # [1, 2, 3]
    y = X @ coefs + rng.standard_normal(n) * 0.1
    data = {"y": y.tolist()}
    for i in range(p):
        data[f"x{i}"] = X[:, i].tolist()
    return pl.DataFrame(data)


def test_abi_probe():
    """Known values must cross the C ABI intact (guards against argtypes
    drift and backend miscompilation — see https://codeberg.org/ziglang/zig/issues/36038)."""
    _lib.quarrel_abi_probe.restype = ctypes.c_int
    _lib.quarrel_abi_probe.argtypes = (
        [ctypes.c_void_p] * 7 + [ctypes.c_int] + [ctypes.c_double] * 3 + [ctypes.c_int]
    )
    buf = (ctypes.c_double * 1)()
    p = ctypes.cast(buf, ctypes.c_void_p)
    ok = _lib.quarrel_abi_probe(p, p, p, p, p, p, p, 42, 1.5, 2.5, 3.5, 7)
    assert ok == 5, f"only {ok}/5 args crossed the ABI intact"


def test_lambda_reaches_solver():
    """Coefficients must respond to lambda; invariance means the parameter
    never reached the solver."""
    df = _small_df()
    weak = enet(df, target="y", alpha=1.0, lambda_=0.001, max_iter=10_000, tol=1e-8)
    strong = enet(df, target="y", alpha=1.0, lambda_=1.0, max_iter=10_000, tol=1e-8)
    assert not np.allclose(weak.coef_array, strong.coef_array), (
        "coefficients invariant to lambda — parameters are not reaching the solver"
    )


def test_alpha_reaches_solver():
    """Same guard for alpha: lasso and ridge-ish fits must differ."""
    df = _small_df()
    lasso = enet(df, target="y", alpha=1.0, lambda_=0.1, max_iter=10_000, tol=1e-8)
    ridge = enet(df, target="y", alpha=0.01, lambda_=0.1, max_iter=10_000, tol=1e-8)
    assert not np.allclose(lasso.coef_array, ridge.coef_array), (
        "coefficients invariant to alpha — parameters are not reaching the solver"
    )


def test_quarrel_fit_smoke():
    df = _small_df()
    fitopts = FitOptions(
        lambda_=0.01,
        alpha=0.5,
        n_lambda=100,
        lambda_min_ratio=0.1,
        penalty_factors=None,
        lower_bounds=None,
        upper_bounds=None,
        warm_start=None,
    )
    rc = quarrel_fit(df, "y", SOLVER.ENET, fitopts)


def test_quarrel_fit_path_smoke():
    df = _small_df()

    fitopts = FitOptions(
        lambda_=0.01,
        alpha=0.5,
        n_lambda=100,
        lambda_min_ratio=0.1,
        penalty_factors=None,
        lower_bounds=None,
        upper_bounds=None,
        warm_start=None,
    )
    rc = quarrel_fit_path(df, "y", SOLVER.ENET_PATH, 100, fitopts)


def test_quarrel_fit_bad_solver():
    df = _small_df()

    fitopts = FitOptions(
        lambda_=0.01,
        alpha=0.5,
        n_lambda=100,
        lambda_min_ratio=0.1,
        penalty_factors=None,
        lower_bounds=None,
        upper_bounds=None,
        warm_start=None,
    )

    with pytest.raises(errors.QuarrelError) as exc_info:
        rc = quarrel_fit(df, "y", 99, fitopts)  # type: ignore


def test_error_codes_match_zig():
    for code in ErrorCode:
        zig_name = quarrel_error_name(code)
        assert zig_name == code.name, f"{code.value}: python={code.name} zig={zig_name}"
