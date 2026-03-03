# tests/test_arrow_bridge.py
import polars as pl
import pyarrow as pa

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
from quarrelm._core import (
    _extract_stream_pointer,
    _extract_array_pointers,
    _get_array_len,
)


def test_arrow_array_len():
    arr = pa.array([1.0, 2.0, 3.0], type=pa.float64())
    arr_len = _get_array_len(arr)
    print(f"pa.array length: {len(arr.to_pylist())}, quarrel len: {arr_len}")
    assert arr_len == 3, "Array length is incorrect."
    print("PASS: array array len")


def test_stream_pointer():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    table = df.to_arrow()
    ptr, _capsule = _extract_stream_pointer(table)
    print(f"Stream pointer: {ptr}")
    assert ptr != 0, "Got null pointer"
    print("PASS: stream pointer extraction")


def test_array_pointer():
    arr = pa.array([1.0, 2.0, 3.0], type=pa.float64())
    array_ptr, schema_ptr, _sc, _ac = _extract_array_pointers(arr)
    print(f"Array pointer: {array_ptr}, Schema pointer: {schema_ptr}")
    assert array_ptr != 0, "Got null array pointer"
    assert schema_ptr != 0, "Got null schema pointer"
    print("PASS: array pointer extraction")


if __name__ == "__main__":
    test_arrow_array_len()
    test_stream_pointer()
    test_array_pointer()
