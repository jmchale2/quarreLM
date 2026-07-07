from pathlib import Path
import ctypes
from quarrelm.errors import raise_for_code

import os
import sys
from dataclasses import dataclass
from typing import Any
from enum import IntEnum

import narwhals as nw
import numpy as np
from numpy.typing import NDArray
import pyarrow as pa


def _lib_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    elif sys.platform == "win32":
        return ".dll"
    else:
        return ".so"


def _find_lib() -> ctypes.CDLL:
    lib_suffix = _lib_suffix()
    libpath = Path(__file__).parents[2].resolve()

    candidates = [
        (libpath / f"zig-out/lib/libquarrelm{lib_suffix}"),
        (libpath / f"libquarrelm{lib_suffix}"),
    ]

    for path in candidates:
        if path.exists():
            return ctypes.CDLL(str(path))

    raise FileNotFoundError(
        f"Could not find libquarreLM.\nSearched: {[str(p) for p in candidates]}"
    )


_lib = _find_lib()


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


# Conversion to fit api, and not per solver calls


class SOLVER(IntEnum):
    OLS = 0
    ENET = 1
    ENET_PATH = 2


class _CFitOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("lambda_", ctypes.c_double),
        ("alpha", ctypes.c_double),
        ("tol", ctypes.c_double),
        ("max_iter", ctypes.c_uint64),
        ("n_lambda", ctypes.c_uint64),
        ("lambda_min_ratio", ctypes.c_double),
        ("penalty_factors", ctypes.POINTER(ctypes.c_double)),
        ("lower_bounds", ctypes.POINTER(ctypes.c_double)),
        ("upper_bounds", ctypes.POINTER(ctypes.c_double)),
        ("warm_start", ctypes.POINTER(ctypes.c_double)),
    ]


class _CFitResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("n_iter", ctypes.c_uint64),
        ("out_coefs", ctypes.POINTER(ctypes.c_double)),
    ]


def _ptr(arr, keepalive: list):
    a = np.ascontiguousarray(arr, dtype=np.float64)
    keepalive.append(a)  # numpy array must outlive the call
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _build_opts(
    *,
    lambda_,
    alpha,
    tol,
    max_iter,
    n_lambda=0,
    lambda_min_ratio=0.0,
    penalty_factors=None,
    lower_bounds=None,
    upper_bounds=None,
    warm_start=None,
):
    opts = _CFitOptions()  # zero-initialized: all pointers start NULL
    opts.struct_size = ctypes.sizeof(_CFitOptions)
    opts.lambda_ = lambda_
    opts.alpha = alpha
    opts.tol = tol
    opts.max_iter = max_iter
    opts.n_lambda = n_lambda
    opts.lambda_min_ratio = lambda_min_ratio

    keepalive = []

    if penalty_factors is not None:
        opts.penalty_factors = _ptr(penalty_factors, keepalive)
    if lower_bounds is not None:
        opts.lower_bounds = _ptr(lower_bounds, keepalive)
    if upper_bounds is not None:
        opts.upper_bounds = _ptr(upper_bounds, keepalive)
    if warm_start is not None:
        opts.warm_start = _ptr(warm_start, keepalive)
    return opts, keepalive


def _build_result(n_features: int):
    result = _CFitResult()
    result.struct_size = ctypes.sizeof(_CFitResult)
    out_coefs = np.zeros(n_features, dtype=np.float64)

    keepalive = []

    result.out_coefs = _ptr(out_coefs, keepalive)

    return result, out_coefs


_lib.quarrel_fit.restype = ctypes.c_int
_lib.quarrel_fit.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,  # stream, y_array, y_schema
    ctypes.c_int,  # solver
    ctypes.POINTER(_CFitOptions),
    ctypes.POINTER(_CFitResult),
]


def quarrel_fit(df, target: str, solver: SOLVER, warm_start: NDArray | None = None):
    # call site:

    nw_df = nw.from_native(df)
    feature_names = [c for c in nw_df.columns if c != target]
    n_features = len(feature_names)

    if n_features == 0:
        raise ValueError("No feature columns found")

    # Select feature columns, cast to float64, convert to Arrow
    features_df = nw_df.select(
        *[nw.col(feature).cast(nw.Float64) for feature in feature_names]
    )
    features_arrow = features_df.to_native()

    # Get the Arrow stream pointer for features
    # Convert to pyarrow Table first if not already
    if hasattr(features_arrow, "__arrow_c_stream__"):
        features_pa = features_arrow
    else:
        features_pa = (
            pa.Table.from_pandas(features_arrow)
            if hasattr(features_arrow, "dtypes")
            else pa.table(features_arrow)
        )

    stream_ptr, _capsule = _extract_stream_pointer(features_pa)

    y_series = nw_df.get_column(target).cast(nw.Float64)
    y_native = y_series.to_native()

    if hasattr(y_native, "to_arrow"):
        y_pa = y_native.to_arrow()  # Polars Series → pyarrow ChunkedArray
        if isinstance(y_pa, pa.ChunkedArray):
            y_pa = y_pa.combine_chunks()  # Ensure single chunk
    elif isinstance(y_native, pa.Array):
        y_pa = y_native
    elif isinstance(y_native, pa.ChunkedArray):
        y_pa = y_native.combine_chunks()
    else:
        y_pa = pa.array(y_native, type=pa.float64())

    y_array_ptr, y_schema_ptr, _yac, _ysc = _extract_array_pointers(y_pa)

    opts, _keep_o = _build_opts(lambda_=0.05, alpha=1.0, tol=1e-8, max_iter=10_000)
    result, _keep_r = _build_result(n_features)
    rc = _lib.quarrel_fit(
        stream_ptr,
        y_array_ptr,
        y_schema_ptr,
        ctypes.c_int(solver),
        ctypes.byref(opts),
        ctypes.byref(result),
    )

    raise_for_code(rc)

    # TODO: convert result into a dataclass, based on solver
    return rc


# Old per solver calls


@dataclass
class OLSResult:
    """Result of an OLS fit."""

    coefficients: dict[str, float]  # feature_name -> coefficient
    feature_names: list[str]
    coef_array: np.ndarray  # raw coefficient vector


_lib.quarrel_ols_fit.restype = ctypes.c_int
_lib.quarrel_ols_fit.argtypes = [
    ctypes.c_void_p,  # ArrowArrayStream*
    ctypes.c_void_p,  # ArrowArray* (y)
    ctypes.c_void_p,  # ArrowSchema* (y)
    ctypes.c_void_p,  # f64* out_coeffs
    ctypes.c_int,  # n_features
]


def ols(df, target: str) -> OLSResult:
    """
    Fit OLS regression: target ~ all other columns.

    Parameters
    ----------
    df : Polars DataFrame, pandas DataFrame, or anything narwhals supports
        Input data. All columns except `target` are used as features.
        All columns must be numeric (float64).
    target : str
        Name of the response variable column.

    Returns
    -------
    OLSResult
        Fitted coefficients mapped to feature names.
    """
    # --- Step 1: Normalize via narwhals ---
    nw_df = nw.from_native(df)
    feature_names = [c for c in nw_df.columns if c != target]
    n_features = len(feature_names)

    if n_features == 0:
        raise ValueError("No feature columns found")

    # --- Step 2: Extract features as Arrow stream ---
    # Select feature columns, cast to float64, convert to Arrow
    features_df = nw_df.select(
        *[nw.col(feature).cast(nw.Float64) for feature in feature_names]
    )
    features_arrow = features_df.to_native()

    # Get the Arrow stream pointer for features
    # Convert to pyarrow Table first if not already
    if hasattr(features_arrow, "__arrow_c_stream__"):
        features_pa = features_arrow
    else:
        # pandas/other — go through pyarrow
        features_pa = (
            pa.Table.from_pandas(features_arrow)
            if hasattr(features_arrow, "dtypes")
            else pa.table(features_arrow)
        )

    stream_ptr, _capsule = _extract_stream_pointer(features_pa)

    # --- Step 3: Extract y as Arrow array ---
    y_series = nw_df.get_column(target).cast(nw.Float64)
    y_native = y_series.to_native()

    # Convert to pyarrow Array
    if hasattr(y_native, "to_arrow"):
        y_pa = y_native.to_arrow()  # Polars Series → pyarrow ChunkedArray
        if isinstance(y_pa, pa.ChunkedArray):
            y_pa = y_pa.combine_chunks()  # Ensure single chunk
    elif isinstance(y_native, pa.Array):
        y_pa = y_native
    elif isinstance(y_native, pa.ChunkedArray):
        y_pa = y_native.combine_chunks()
    else:
        y_pa = pa.array(y_native, type=pa.float64())

    y_array_ptr, y_schema_ptr, _yac, _ysc = _extract_array_pointers(y_pa)

    # --- Step 4: Allocate output buffer ---
    out_coeffs = np.zeros(n_features, dtype=np.float64)
    out_ptr = out_coeffs.ctypes.data_as(ctypes.c_void_p)

    # --- Step 5: Call Zig ---
    rc = _lib.quarrel_ols_fit(
        stream_ptr,
        y_array_ptr,
        y_schema_ptr,
        out_ptr,
        ctypes.c_int(n_features),
    )

    if rc != 0:
        error_map = {
            -1: "Wrong Arrow format (expected float64)",
            -2: "Data contains nulls (not supported yet)",
            -3: "Null data buffer",
            -4: "Arrow stream error",
            -5: "Arrow schema error",
            -6: "Dimension mismatch",
            -7: "Singular matrix (features are linearly dependent)",
        }
        msg = error_map.get(rc, f"Unknown error (code {rc})")
        raise RuntimeError(f"zigregress OLS fit failed: {msg}")

    # --- Step 6: Package results ---
    return OLSResult(
        coefficients=dict(zip(feature_names, out_coeffs.tolist())),
        feature_names=feature_names,
        coef_array=out_coeffs,
    )


_lib.quarrel_ols_fit_simd.restype = ctypes.c_int
_lib.quarrel_ols_fit_simd.argtypes = [
    ctypes.c_void_p,  # ArrowArrayStream*
    ctypes.c_void_p,  # ArrowArray* (y)
    ctypes.c_void_p,  # ArrowSchema* (y)
    ctypes.c_void_p,  # f64* out_coeffs
    ctypes.c_int,  # n_features
]


def ols_simd(df, target: str) -> OLSResult:
    """
    Fit OLS SIMD regression: target ~ all other columns.

    Parameters
    ----------
    df : Polars DataFrame, pandas DataFrame, or anything narwhals supports
        Input data. All columns except `target` are used as features.
        All columns must be numeric (float64).
    target : str
        Name of the response variable column.

    Returns
    -------
    OLSResult
        Fitted coefficients mapped to feature names.
    """
    # --- Step 1: Normalize via narwhals ---
    nw_df = nw.from_native(df)
    feature_names = [c for c in nw_df.columns if c != target]
    n_features = len(feature_names)

    if n_features == 0:
        raise ValueError("No feature columns found")

    # --- Step 2: Extract features as Arrow stream ---
    # Select feature columns, cast to float64, convert to Arrow
    features_df = nw_df.select(
        *[nw.col(feature).cast(nw.Float64) for feature in feature_names]
    )
    features_arrow = features_df.to_native()

    # Get the Arrow stream pointer for features
    # Convert to pyarrow Table first if not already
    if hasattr(features_arrow, "__arrow_c_stream__"):
        features_pa = features_arrow
    else:
        # pandas/other — go through pyarrow
        features_pa = (
            pa.Table.from_pandas(features_arrow)
            if hasattr(features_arrow, "dtypes")
            else pa.table(features_arrow)
        )

    stream_ptr, _capsule = _extract_stream_pointer(features_pa)

    # --- Step 3: Extract y as Arrow array ---
    y_series = nw_df.get_column(target).cast(nw.Float64)
    y_native = y_series.to_native()

    # Convert to pyarrow Array
    if hasattr(y_native, "to_arrow"):
        y_pa = y_native.to_arrow()  # Polars Series → pyarrow ChunkedArray
        if isinstance(y_pa, pa.ChunkedArray):
            y_pa = y_pa.combine_chunks()  # Ensure single chunk
    elif isinstance(y_native, pa.Array):
        y_pa = y_native
    elif isinstance(y_native, pa.ChunkedArray):
        y_pa = y_native.combine_chunks()
    else:
        y_pa = pa.array(y_native, type=pa.float64())

    y_array_ptr, y_schema_ptr, _yac, _ysc = _extract_array_pointers(y_pa)

    # --- Step 4: Allocate output buffer ---
    out_coeffs = np.zeros(n_features, dtype=np.float64)
    out_ptr = out_coeffs.ctypes.data_as(ctypes.c_void_p)

    # --- Step 5: Call Zig ---
    rc = _lib.quarrel_ols_fit_simd(
        stream_ptr,
        y_array_ptr,
        y_schema_ptr,
        out_ptr,
        ctypes.c_int(n_features),
    )

    if rc != 0:
        error_map = {
            -1: "Wrong Arrow format (expected float64)",
            -2: "Data contains nulls (not supported yet)",
            -3: "Null data buffer",
            -4: "Arrow stream error",
            -5: "Arrow schema error",
            -6: "Dimension mismatch",
            -7: "Singular matrix (features are linearly dependent)",
        }
        msg = error_map.get(rc, f"Unknown error (code {rc})")
        raise RuntimeError(f"zigregress OLS fit failed: {msg}")

    # --- Step 6: Package results ---
    return OLSResult(
        coefficients=dict(zip(feature_names, out_coeffs.tolist())),
        feature_names=feature_names,
        coef_array=out_coeffs,
    )


_lib.quarrel_enet_fit.restype = ctypes.c_int
_lib.quarrel_enet_fit.argtypes = [
    ctypes.c_void_p,  # ArrowArrayStream*
    ctypes.c_void_p,  # ArrowArray* (y)
    ctypes.c_void_p,  # ArrowSchema* (y)
    ctypes.c_void_p,  # f64* penalty_factors
    ctypes.c_void_p,  # f64* lower_bounds
    ctypes.c_void_p,  # f64* upper_bounds
    ctypes.c_void_p,  # f64* out_coeffs
    ctypes.c_int,  # n_features
    ctypes.c_double,  # lambda
    ctypes.c_double,  # alpha
    ctypes.c_double,  # tol
    ctypes.c_int,  # max_iter
]


@dataclass
class ElasticNetResult:
    coefficients: dict[str, float]  # original scale
    feature_names: list[str]  # feature names at fit
    penalty_factors: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    coef_array: np.ndarray  # original scale
    means: np.ndarray  # feature means from fit
    sds: np.ndarray  # feature sds from fit
    intercept: float  # recovered from unstandardization
    lambda_: float
    alpha: float
    n_iter: int


def enet(
    df,
    target: str,
    alpha: float,
    lambda_: float,
    max_iter: int,
    tol: float,
    penalty_factors: np.ndarray | None = None,
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
) -> ElasticNetResult:
    """
    Fit ElasticNet regression: target ~ all other columns.

    Parameters
    ----------
    df : Polars DataFrame, pandas DataFrame, or anything narwhals supports
        Input data. All columns except `target` are used as features.
        All columns must be numeric (float64).
    target : str
        Name of the response variable column.

    Returns
    -------
    OLSResult
        Fitted coefficients mapped to feature names.
    """
    # --- Step 1: Normalize via narwhals ---
    nw_df = nw.from_native(df)
    feature_names = [c for c in nw_df.columns if c != target]
    n_features = len(feature_names)

    if n_features == 0:
        raise ValueError("No feature columns found")

    if penalty_factors is None:
        penalty_factors = np.ones(n_features, dtype=np.float64)
    if lower_bounds is None:
        lower_bounds = np.full(n_features, -np.inf, dtype=np.float64)
    if upper_bounds is None:
        upper_bounds = np.full(n_features, np.inf, dtype=np.float64)

    pf_ptr = penalty_factors.ctypes.data_as(ctypes.c_void_p)
    lb_ptr = lower_bounds.ctypes.data_as(ctypes.c_void_p)
    ub_ptr = upper_bounds.ctypes.data_as(ctypes.c_void_p)

    # --- Step 2: Extract features as Arrow stream ---
    # Select feature columns, cast to float64, convert to Arrow
    features_df = nw_df.select(
        *[nw.col(feature).cast(nw.Float64) for feature in feature_names]
    )
    features_arrow = features_df.to_native()

    # Get the Arrow stream pointer for features
    # Convert to pyarrow Table first if not already
    if hasattr(features_arrow, "__arrow_c_stream__"):
        features_pa = features_arrow
    else:
        # pandas/other — go through pyarrow
        features_pa = (
            pa.Table.from_pandas(features_arrow)
            if hasattr(features_arrow, "dtypes")
            else pa.table(features_arrow)
        )

    stream_ptr, _capsule = _extract_stream_pointer(features_pa)

    # --- Step 3: Extract y as Arrow array ---
    y_series = nw_df.get_column(target).cast(nw.Float64)
    y_native = y_series.to_native()

    # Convert to pyarrow Array
    if hasattr(y_native, "to_arrow"):
        y_pa = y_native.to_arrow()  # Polars Series → pyarrow ChunkedArray
        if isinstance(y_pa, pa.ChunkedArray):
            y_pa = y_pa.combine_chunks()  # Ensure single chunk
    elif isinstance(y_native, pa.Array):
        y_pa = y_native
    elif isinstance(y_native, pa.ChunkedArray):
        y_pa = y_native.combine_chunks()
    else:
        y_pa = pa.array(y_native, type=pa.float64())

    y_array_ptr, y_schema_ptr, _yac, _ysc = _extract_array_pointers(y_pa)

    # --- Step 4: Allocate output buffer ---
    out_coeffs = np.zeros(n_features, dtype=np.float64)
    out_ptr = out_coeffs.ctypes.data_as(ctypes.c_void_p)

    # stream_ptr: *arrow.ArrowArrayStream,
    # y_array_ptr: *arrow.ArrowArray,
    # y_schema_ptr: *arrow.ArrowSchema,
    # penalty_factors: [*]f64,
    # lower_bounds: [*]f64,
    # upper_bounds: [*]f64,
    # out_coeffs: [*]f64,
    # n_features: c_int,
    # lambda: f64,
    # alpha: f64,
    # tol: f64,
    # max_iter: c_int,
    # --- Step 5: Call Zig ---
    rc = _lib.quarrel_enet_fit(
        stream_ptr,
        y_array_ptr,
        y_schema_ptr,
        pf_ptr,
        lb_ptr,
        ub_ptr,
        out_ptr,
        ctypes.c_int(n_features),
        ctypes.c_double(lambda_),
        ctypes.c_double(alpha),
        ctypes.c_double(tol),
        ctypes.c_int(max_iter),
    )

    if rc < 0:
        error_map = {
            -1: "Wrong Arrow format (expected float64)",
            -2: "Data contains nulls (not supported yet)",
            -3: "Null data buffer",
            -4: "Arrow stream error",
            -5: "Arrow schema error",
            -6: "Dimension mismatch",
            -7: "Singular matrix (features are linearly dependent)",
        }
        msg = error_map.get(rc, f"Unknown error (code {rc})")
        raise RuntimeError(f"ElasticNet fit failed: {msg}")

    # --- Step 6: Package results ---
    return ElasticNetResult(
        coefficients=dict(zip(feature_names, out_coeffs.tolist())),
        feature_names=feature_names,
        penalty_factors=penalty_factors,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        coef_array=out_coeffs,
        means=np.array([1, 2, 3]),
        sds=np.array([1, 2, 3]),
        intercept=0,
        lambda_=lambda_,
        alpha=alpha,
        n_iter=rc,
    )


_lib.quarrel_enet_path.restype = ctypes.c_int
_lib.quarrel_enet_path.argtypes = [
    ctypes.c_void_p,  # stream
    ctypes.c_void_p,  # y array
    ctypes.c_void_p,  # y schema
    ctypes.c_void_p,  # penalty_factors
    ctypes.c_void_p,  # lower_bounds
    ctypes.c_void_p,  # upper_bounds
    ctypes.c_void_p,  # out_coef_matrix
    ctypes.c_void_p,  # out_lambdas
    ctypes.c_int,  # n_features
    ctypes.c_int,  # n_lambda
    ctypes.c_double,  # alpha
    ctypes.c_double,  # lambda_min_ratio
    ctypes.c_double,  # tol
    ctypes.c_int,  # max_iter
]


def enet_path(
    df,
    target: str,
    alpha: float = 1.0,
    n_lambda: int = 100,
    lambda_min_ratio: float = 1e-4,
    max_iter: int = 10000,
    tol: float = 1e-7,
    penalty_factors: np.ndarray | None = None,
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
):
    # --- Step 1: Normalize via narwhals ---
    nw_df = nw.from_native(df)
    feature_names = [c for c in nw_df.columns if c != target]
    n_features = len(feature_names)

    if n_features == 0:
        raise ValueError("No feature columns found")

    if penalty_factors is None:
        penalty_factors = np.ones(n_features, dtype=np.float64)
    if lower_bounds is None:
        lower_bounds = np.full(n_features, -np.inf, dtype=np.float64)
    if upper_bounds is None:
        upper_bounds = np.full(n_features, np.inf, dtype=np.float64)

    pf_ptr = penalty_factors.ctypes.data_as(ctypes.c_void_p)
    lb_ptr = lower_bounds.ctypes.data_as(ctypes.c_void_p)
    ub_ptr = upper_bounds.ctypes.data_as(ctypes.c_void_p)

    # --- Step 2: Extract features as Arrow stream ---
    # Select feature columns, cast to float64, convert to Arrow
    features_df = nw_df.select(
        *[nw.col(feature).cast(nw.Float64) for feature in feature_names]
    )
    features_arrow = features_df.to_native()

    # Get the Arrow stream pointer for features
    # Convert to pyarrow Table first if not already
    if hasattr(features_arrow, "__arrow_c_stream__"):
        features_pa = features_arrow
    else:
        # pandas/other — go through pyarrow
        features_pa = (
            pa.Table.from_pandas(features_arrow)
            if hasattr(features_arrow, "dtypes")
            else pa.table(features_arrow)
        )

    stream_ptr, _capsule = _extract_stream_pointer(features_pa)

    # --- Step 3: Extract y as Arrow array ---
    y_series = nw_df.get_column(target).cast(nw.Float64)
    y_native = y_series.to_native()

    # Convert to pyarrow Array
    if hasattr(y_native, "to_arrow"):
        y_pa = y_native.to_arrow()  # Polars Series → pyarrow ChunkedArray
        if isinstance(y_pa, pa.ChunkedArray):
            y_pa = y_pa.combine_chunks()  # Ensure single chunk
    elif isinstance(y_native, pa.Array):
        y_pa = y_native
    elif isinstance(y_native, pa.ChunkedArray):
        y_pa = y_native.combine_chunks()
    else:
        y_pa = pa.array(y_native, type=pa.float64())

    y_array_ptr, y_schema_ptr, _yac, _ysc = _extract_array_pointers(y_pa)

    # --- Step 4: Allocate output buffer ---
    out_lambdas = np.zeros(n_lambda, dtype=np.float64)
    out_coef_matrix = np.zeros(n_features * n_lambda, dtype=np.float64)

    rc = _lib.quarrel_enet_path(
        stream_ptr,
        y_array_ptr,
        y_schema_ptr,
        pf_ptr,
        lb_ptr,
        ub_ptr,
        out_coef_matrix.ctypes.data_as(ctypes.c_void_p),
        out_lambdas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(n_features),
        ctypes.c_int(n_lambda),
        ctypes.c_double(alpha),
        ctypes.c_double(lambda_min_ratio),
        ctypes.c_double(tol),
        ctypes.c_int(max_iter),
    )

    if rc < 0:
        raise RuntimeError(f"Path failed: {rc}")

    # Reshape to (p, n_lambda) — column j's path is coef_matrix[j, :]
    coef_matrix = out_coef_matrix.reshape(n_features, n_lambda)

    return {
        "coef_matrix": coef_matrix,
        "lambdas": out_lambdas,
        "feature_names": feature_names,
        "n_iter": rc,
    }
