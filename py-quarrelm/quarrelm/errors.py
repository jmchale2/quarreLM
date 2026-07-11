from enum import IntEnum


class ErrorCode(IntEnum):
    """Mirror of the Zig ErrorCode enum."""

    Ok = 0
    WrongFormat = -1
    HasNulls = -2
    NullBuffer = -3
    StreamError = -4
    SchemaError = -5
    DimensionMismatch = -6
    SingularMatrix = -7
    OutOfMemory = -8
    DegenerateData = -9
    EmptyStream = -10
    ChunkedNotSupported = -11
    BatchSchemaError = -12
    ParameterError = -13
    StructSizeMismatch = -14
    Unknown = -99


class QuarrelError(RuntimeError):
    """Base class for all quarrelm errors."""

    def __init__(self, code: ErrorCode, detail: str = ""):
        self.code = code
        msg = _MESSAGES.get(code, code.name)
        super().__init__(f"{msg} [{code.name}]" + (f": {detail}" if detail else ""))


class DataError(QuarrelError):
    """The input data cannot be used: wrong dtype, nulls, empty or
    multi-chunk Arrow input. Fix the data and retry."""


class DimensionError(QuarrelError):
    """Shapes disagree: X vs y row counts, feature count vs schema,
    or a per-feature array with the wrong length."""


class ParameterError(QuarrelError, ValueError):
    """A hyperparameter is invalid (e.g. alpha outside [0, 1], bounds
    excluding zero, n_lambda < 2). Also catchable as ValueError."""


class SolverError(QuarrelError):
    """The problem is numerically unsolvable as posed: singular/degenerate
    data. Usually means collinear features or a constant y."""


class InternalError(QuarrelError):
    """Something is wrong with the library itself, not your input:
    ABI/struct-size disagreement, allocation failure, or an unmapped
    native error. Worth a bug report."""


_MESSAGES = {
    ErrorCode.WrongFormat: "expected float64 Arrow data",
    ErrorCode.HasNulls: "data contains nulls",
    ErrorCode.NullBuffer: "Arrow array has no data buffer",
    ErrorCode.StreamError: "Arrow stream failed while producing batches",
    ErrorCode.SchemaError: "Arrow schema does not match the expected features",
    ErrorCode.DimensionMismatch: "array lengths disagree (X rows vs y, or per-feature arrays vs n_features)",
    ErrorCode.SingularMatrix: "features are linearly dependent",
    ErrorCode.OutOfMemory: "native allocation failed",
    ErrorCode.DegenerateData: "no usable signal (all penalties zero, or y orthogonal to X)",
    ErrorCode.EmptyStream: "Arrow stream produced no batches",
    ErrorCode.ChunkedNotSupported: "multi-chunk input not supported yet — combine_chunks() first",
    ErrorCode.BatchSchemaError: "record batch does not match its schema",
    ErrorCode.ParameterError: "invalid hyperparameter",
    ErrorCode.StructSizeMismatch: "ABI mismatch between quarrelm and libquarrelm",
    ErrorCode.Unknown: "unmapped native error",
}

_EXC = {
    ErrorCode.WrongFormat: DataError,
    ErrorCode.HasNulls: DataError,
    ErrorCode.NullBuffer: DataError,
    ErrorCode.StreamError: DataError,
    ErrorCode.EmptyStream: DataError,
    ErrorCode.ChunkedNotSupported: DataError,
    ErrorCode.SchemaError: DimensionError,
    ErrorCode.DimensionMismatch: DimensionError,
    ErrorCode.BatchSchemaError: DimensionError,
    ErrorCode.ParameterError: ParameterError,
    ErrorCode.SingularMatrix: SolverError,
    ErrorCode.DegenerateData: SolverError,
    ErrorCode.OutOfMemory: InternalError,
    ErrorCode.StructSizeMismatch: InternalError,
    ErrorCode.Unknown: InternalError,
}


def raise_for_code(rc: int, detail: str = "", context: str = "") -> None:
    """Raise the mapped exception for a negative native return code."""
    if rc >= 0:
        return
    try:
        code = ErrorCode(rc)
    except ValueError:
        code = ErrorCode.Unknown

    raise _EXC.get(code, InternalError)(
        code, f"{detail}\nContext:{context}" if context else detail
    )
