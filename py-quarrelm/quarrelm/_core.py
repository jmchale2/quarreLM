from quarrelm.libpath import _lib, _verify_lib  # isort: skip

_verify_lib(_lib)

from pathlib import Path
import ctypes
from quarrelm.errors import raise_for_code

import os
import sys
from dataclasses import dataclass, asdict
from typing import Any
from enum import IntEnum

import narwhals as nw
import numpy as np
from numpy.typing import NDArray
import pyarrow as pa

from quarrelm._params import (
    OLSResult,
    ElasticNetResult,
    ElasticNetPathResult,
    FitOptions,
)
from quarrelm import errors


_lib.quarrel_version.restype = ctypes.c_char_p
_lib.quarrel_version.argtypes = []


def get_version():
    return _lib.quarrel_version()


_lib.quarrel_ping.restype = ctypes.c_int
_lib.quarrel_ping.argtypes = []


def ping():
    return _lib.quarrel_ping()


def _extract_stream_pointer(obj) -> tuple[int, Any]:
    capsule = obj.__arrow_c_stream__()
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, b"arrow_array_stream")
    return ptr, capsule


def _extract_array_pointers(arr: pa.Array) -> tuple[int, int, Any, Any]:
    """
    Extract ArrowArray* and ArrowSchema* from a PyArrow Array.
    Returns (array_ptr, schema_ptr) as integers suitable for ctypes.
    """
    schema_capsule, array_capsule = arr.__arrow_c_array__()

    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

    schema_ptr = ctypes.pythonapi.PyCapsule_GetPointer(schema_capsule, b"arrow_schema")
    array_ptr = ctypes.pythonapi.PyCapsule_GetPointer(array_capsule, b"arrow_array")

    return array_ptr, schema_ptr, schema_capsule, array_capsule


_lib.quarrel_array_len.restype = ctypes.c_int
_lib.quarrel_array_len.argtypes = [
    ctypes.c_void_p,  # ArrowArray* (y)
    ctypes.c_void_p,  # ArrowSchema* (y)
]


def _get_array_len(arr):
    array_ptr, schema_ptr, _sc, _ac = _extract_array_pointers(arr)
    len = _lib.quarrel_array_len(array_ptr, schema_ptr)

    return len


_lib.quarrel_error_name.restype = ctypes.c_char_p
_lib.quarrel_error_name.argtypes = [ctypes.c_int]


def quarrel_error_name(code: int):
    return _lib.quarrel_error_name(code).decode()


_lib.quarrel_last_error.restype = ctypes.c_char_p


def quarrel_last_error():
    return _lib.quarrel_last_error().decode()


_lib.quarrel_last_error_context.restype = ctypes.c_char_p


def quarrel_last_error_context():
    return _lib.quarrel_last_error_context().decode()


class _ArrowData:
    """Marshalled feature stream + y array for one native call.

    The pointers are only valid while this object is alive."""

    feature_names: list[str]
    n_features: int
    stream_ptr: int
    y_array_ptr: int
    y_schema_ptr: int
    _stream_capsule: object
    _y_pa: object
    _y_schema_capsule: object
    _y_array_capsule: object

    @classmethod
    def from_frame(cls, df, target: str):
        self = cls.__new__(cls)
        nw_df = nw.from_native(df)
        self.feature_names = [c for c in nw_df.columns if c != target]
        self.n_features = len(self.feature_names)
        if self.n_features == 0:
            raise ValueError("No feature columns found")

        # --- features -> Arrow C stream ---
        features_df = nw_df.select(
            *[nw.col(f).cast(nw.Float64) for f in self.feature_names]
        )
        features_arrow = features_df.to_native()
        if hasattr(features_arrow, "__arrow_c_stream__"):
            features_pa = features_arrow
        else:
            features_pa = (
                pa.Table.from_pandas(features_arrow)
                if hasattr(features_arrow, "dtypes")
                else pa.table(features_arrow)
            )
        self.stream_ptr, self._stream_capsule = _extract_stream_pointer(features_pa)

        # --- y -> Arrow C array ---
        y_series = nw_df.get_column(target).cast(nw.Float64)
        y_native = y_series.to_native()
        if hasattr(y_native, "to_arrow"):
            y_pa = y_native.to_arrow()
            if isinstance(y_pa, pa.ChunkedArray):
                y_pa = y_pa.combine_chunks()
        elif isinstance(y_native, pa.Array):
            y_pa = y_native
        elif isinstance(y_native, pa.ChunkedArray):
            y_pa = y_native.combine_chunks()
        else:
            y_pa = pa.array(y_native, type=pa.float64())
        self._y_pa = y_pa  # free insurance alongside the capsules
        (
            self.y_array_ptr,
            self.y_schema_ptr,
            self._y_schema_capsule,
            self._y_array_capsule,
        ) = _extract_array_pointers(y_pa)
        return self

    # @classmethod
    # def from_xy(cls, X, y):
    #     pass


class SOLVER(IntEnum):
    OLS = 0
    ENET = 1
    ENET_PATH = 2


class OLSMETHOD(IntEnum):
    AUTO = 0
    CHOLESKY = 1
    GAUSSIAN_ELIM = 2


class _CFitOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("lambda_", ctypes.c_double),
        ("alpha", ctypes.c_double),
        ("tol", ctypes.c_double),
        ("max_iter", ctypes.c_uint64),
        ("n_lambda", ctypes.c_uint64),
        ("lambda_min_ratio", ctypes.c_double),
        ("ols_method", ctypes.c_uint64),
        ("penalty_factors", ctypes.POINTER(ctypes.c_double)),
        ("lower_bounds", ctypes.POINTER(ctypes.c_double)),
        ("upper_bounds", ctypes.POINTER(ctypes.c_double)),
        ("warm_start", ctypes.POINTER(ctypes.c_double)),
    ]


# Asserts that fitoptions sizes agree at import time.
if ctypes.sizeof(_CFitOptions) != _lib.quarrel_sizeof_fit_options():
    raise ImportError(
        f"_CFitOptions is {ctypes.sizeof(_CFitOptions)} bytes but libquarrelm expects "
        f"{_lib.quarrel_sizeof_fit_options()} — the Python mirror is out of sync with the ABI"
    )


class _CFitResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("n_iter", ctypes.c_uint64),
        ("out_coefs", ctypes.POINTER(ctypes.c_double)),
    ]


# Asserts that fitresult sizes agree at import time.
if ctypes.sizeof(_CFitResult) != _lib.quarrel_sizeof_fit_result():
    raise ImportError(
        f"_CFitResult is {ctypes.sizeof(_CFitResult)} bytes but libquarrelm expects "
        f"{_lib.quarrel_sizeof_fit_result()} — the Python mirror is out of sync with the ABI"
    )


class _CPathResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("n_iters", ctypes.POINTER(ctypes.c_uint64)),
        ("lambda_paths", ctypes.POINTER(ctypes.c_double)),
        ("out_coefs_matrix", ctypes.POINTER(ctypes.c_double)),
    ]


# Asserts that pathresult sizes agree at import time.
if ctypes.sizeof(_CPathResult) != _lib.quarrel_sizeof_path_result():
    raise ImportError(
        f"_CPathResult is {ctypes.sizeof(_CPathResult)} bytes but libquarrelm expects "
        f"{_lib.quarrel_sizeof_path_result()} — the Python mirror is out of sync with the ABI"
    )


def _ptr(
    arr,
    keepalive: list,
    expected_len: int,
    name: str,
    dtype=np.float64,
    ctype=ctypes.c_double,
):
    a = np.ascontiguousarray(arr, dtype=dtype)

    if a.ndim != 1 or a.shape[0] != expected_len:
        raise errors.DimensionError(
            errors.ErrorCode.DimensionMismatch,
            f"{name} has shape {a.shape}, expected ({expected_len},)",
        )
    keepalive.append(a)  # numpy array must outlive the call
    return a.ctypes.data_as(ctypes.POINTER(ctype))


def _build_opts(
    *,
    lambda_=0,
    alpha,
    tol,
    max_iter,
    n_lambda=0,
    lambda_min_ratio=-1,  # -1 is treated as None in capi.zig
    ols_method=0,
    penalty_factors=None,
    lower_bounds=None,
    upper_bounds=None,
    warm_start=None,
    n_features: int | None = None,
):
    opts = _CFitOptions()  # zero-initialized: all pointers start NULL
    opts.struct_size = ctypes.sizeof(_CFitOptions)
    opts.lambda_ = lambda_
    opts.alpha = alpha
    opts.tol = tol
    opts.max_iter = max_iter
    opts.n_lambda = n_lambda
    opts.lambda_min_ratio = lambda_min_ratio
    opts.ols_method = ols_method

    keepalive = []

    if penalty_factors is not None:
        opts.penalty_factors = _ptr(
            penalty_factors, keepalive, expected_len=n_features, name="penalty_factors"
        )
    if lower_bounds is not None:
        opts.lower_bounds = _ptr(
            lower_bounds,
            keepalive,
            expected_len=n_features,
            name="lower_bounds",
        )
    if upper_bounds is not None:
        opts.upper_bounds = _ptr(
            upper_bounds,
            keepalive,
            expected_len=n_features,
            name="upper_bounds",
        )
    if warm_start is not None:
        opts.warm_start = _ptr(
            warm_start,
            keepalive,
            expected_len=n_features,
            name="warm_start",
        )
    return opts, keepalive


def _build_fit_result(n_features: int):
    result = _CFitResult()
    result.struct_size = ctypes.sizeof(_CFitResult)
    out_coefs = np.zeros(n_features, dtype=np.float64)

    keepalive = []

    result.out_coefs = _ptr(
        out_coefs, keepalive, expected_len=n_features, name="out_coefs"
    )

    return result, out_coefs


def _build_path_result(n_features: int, n_lambdas: int):
    result = _CPathResult()
    result.struct_size = ctypes.sizeof(_CPathResult)
    out_coefs = np.zeros(n_features * n_lambdas, dtype=np.float64)
    lambdas = np.zeros(n_lambdas, dtype=np.float64)
    n_iters = np.zeros(n_lambdas, dtype=np.uint64)

    keepalive = []

    result.out_coefs_matrix = _ptr(
        out_coefs,
        keepalive,
        expected_len=n_features * n_lambdas,
        name="out_coefs",
    )
    result.lambda_paths = _ptr(
        lambdas,
        keepalive,
        expected_len=n_lambdas,
        name="lambda_paths",
    )
    result.n_iters = _ptr(
        n_iters,
        keepalive,
        dtype=np.uint64,
        ctype=ctypes.c_uint64,
        expected_len=n_lambdas,
        name="n_iters",
    )

    return result, out_coefs, lambdas, n_iters


_lib.quarrel_fit.restype = ctypes.c_int
_lib.quarrel_fit.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,  # stream, y_array, y_schema
    ctypes.c_int,  # n_features
    ctypes.c_int,  # solver
    ctypes.POINTER(_CFitOptions),
    ctypes.POINTER(_CFitResult),
]


def quarrel_fit(df, target: str, solver: SOLVER, fitopts: FitOptions):
    data = _ArrowData.from_frame(df, target)
    opts, _keep_o = _build_opts(
        lambda_=fitopts.lambda_,
        alpha=fitopts.alpha,
        tol=fitopts.tol,
        max_iter=fitopts.max_iter,
        penalty_factors=fitopts.penalty_factors,
        lower_bounds=fitopts.lower_bounds,
        upper_bounds=fitopts.upper_bounds,
        warm_start=fitopts.warm_start,
        n_features=data.n_features,
    )

    result, out_coefs = _build_fit_result(data.n_features)
    rc = _lib.quarrel_fit(
        data.stream_ptr,
        data.y_array_ptr,
        data.y_schema_ptr,
        data.n_features,
        ctypes.c_int(solver),
        ctypes.byref(opts),
        ctypes.byref(result),
    )

    raise_for_code(rc, quarrel_last_error(), quarrel_last_error_context())
    match solver:
        case SOLVER.OLS:
            result = OLSResult(
                coefficients={
                    f: c for f, c in zip(data.feature_names, out_coefs.ravel())
                },
                feature_names=data.feature_names,
                coef_array=out_coefs,
            )
        case SOLVER.ENET:
            result = ElasticNetResult(
                coefficients={
                    f: c for f, c in zip(data.feature_names, out_coefs.ravel())
                },
                feature_names=data.feature_names,
                coef_array=out_coefs,
                penalty_factors=fitopts.penalty_factors,
                lower_bounds=fitopts.lower_bounds,
                upper_bounds=fitopts.upper_bounds,
                alpha=fitopts.alpha,
                lambda_=fitopts.lambda_,
                n_iter=rc,
            )

    return result


_lib.quarrel_fit_path.restype = ctypes.c_int
_lib.quarrel_fit_path.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,  # stream, y_array, y_schema
    ctypes.c_int,  # n_features
    ctypes.c_int,  # solver
    ctypes.POINTER(_CFitOptions),
    ctypes.POINTER(_CPathResult),
]


def quarrel_fit_path(df, target: str, solver: SOLVER, fitopts: FitOptions):
    data = _ArrowData.from_frame(df, target)

    opts, _keep_o = _build_opts(
        lambda_=fitopts.lambda_,
        alpha=fitopts.alpha,
        tol=fitopts.tol,
        max_iter=fitopts.max_iter,
        penalty_factors=fitopts.penalty_factors,
        lower_bounds=fitopts.lower_bounds,
        upper_bounds=fitopts.upper_bounds,
        warm_start=fitopts.warm_start,
        n_lambda=fitopts.n_lambda,
        lambda_min_ratio=fitopts.lambda_min_ratio,
        n_features=data.n_features,
    )
    result, out_coefs_arr, lambdas, n_iters = _build_path_result(
        data.n_features, fitopts.n_lambda
    )

    rc = _lib.quarrel_fit_path(
        data.stream_ptr,
        data.y_array_ptr,
        data.y_schema_ptr,
        data.n_features,
        ctypes.c_int(solver),
        ctypes.byref(opts),
        ctypes.byref(result),
    )

    raise_for_code(rc, quarrel_last_error(), quarrel_last_error_context())

    coef_matrix = out_coefs_arr.reshape(data.n_features, fitopts.n_lambda)
    coefs = {}
    for feature in range(len(data.feature_names)):
        coefs[data.feature_names[feature]] = coef_matrix[feature, :]

    result = ElasticNetPathResult(
        coefficients=coefs,
        feature_names=data.feature_names,
        penalty_factors=fitopts.penalty_factors,
        lower_bounds=fitopts.lower_bounds,
        upper_bounds=fitopts.upper_bounds,
        coef_matrix=coef_matrix,
        lambda_=lambdas,
        alpha=fitopts.alpha,
        n_iters=n_iters,
        total_iters=rc,
    )
    return result
