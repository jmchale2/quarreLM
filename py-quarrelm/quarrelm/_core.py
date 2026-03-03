from pathlib import Path
import ctypes


import os
import sys
from dataclasses import dataclass
from typing import Any

import narwhals as nw
import numpy as np
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
        (libpath / f"zig-out/lib/libquarreLM{lib_suffix}"),
        (libpath / f"libquarreLM{lib_suffix}"),
    ]

    for path in candidates:
        if path.exists():
            return ctypes.CDLL(str(path))

    raise FileNotFoundError(
        f"Could not find libquarreLM.\nSearched: {[str(p) for p in candidates]}"
    )


_lib = _find_lib()


def get_version():
    return _lib.quarrel_version()


def ping():
    return _lib.quarrel_ping()


_lib.quarrel_ping.restype = ctypes.c_int

_lib.quarrel_ping.restype = ctypes.c_int
_lib.quarrel_ping.argtypes = []


_lib.quarrel_array_len.restype = ctypes.c_int
_lib.quarrel_array_len.argtypes = [
    ctypes.c_void_p,  # ArrowArray* (y)
    ctypes.c_void_p,  # ArrowSchema* (y)
]


_lib.quarrel_ols_fit.restype = ctypes.c_int
_lib.quarrel_ols_fit.argtypes = [
    ctypes.c_void_p,  # ArrowArrayStream*
    ctypes.c_void_p,  # ArrowArray* (y)
    ctypes.c_void_p,  # ArrowSchema* (y)
    ctypes.c_void_p,  # f64* out_coeffs
    ctypes.c_int,  # n_features
]

_lib.quarrel_ols_fit_simd.restype = ctypes.c_int
_lib.quarrel_ols_fit_simd.argtypes = [
    ctypes.c_void_p,  # ArrowArrayStream*
    ctypes.c_void_p,  # ArrowArray* (y)
    ctypes.c_void_p,  # ArrowSchema* (y)
    ctypes.c_void_p,  # f64* out_coeffs
    ctypes.c_int,  # n_features
]


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


def _get_array_len(arr):
    array_ptr, schema_ptr, _sc, _ac = _extract_array_pointers(arr)
    len = _lib.quarrel_array_len(array_ptr, schema_ptr)

    return len


@dataclass
class OLSResult:
    """Result of an OLS fit."""

    coefficients: dict[str, float]  # feature_name -> coefficient
    feature_names: list[str]
    coef_array: np.ndarray  # raw coefficient vector


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


def ols_simd(df, target: str) -> OLSResult:
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
