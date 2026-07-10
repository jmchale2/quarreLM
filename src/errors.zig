const std = @import("std");
const arrow = @import("arrow.zig");

pub const QError = arrow.ArrowError || error{
    DimensionMismatch,
    SingularMatrix,
    OutOfMemory,
    DegenerateData,
    SchemaError,
    ChunkedNotSupported,
    BatchSchemaError,
    ParameterError,
    WrongAPICall,
    StructSizeMismatch,
    InvalidValue,
};

pub const ErrorCode = enum(c_int) {
    Ok = 0,
    WrongFormat = -1,
    HasNulls = -2,
    NullBuffer = -3,
    StreamError = -4,
    SchemaError = -5,
    DimensionMismatch = -6,
    SingularMatrix = -7,
    OutOfMemory = -8,
    DegenerateData = -9,
    EmptyStream = -10,
    ChunkedNotSupported = -11,
    BatchSchemaError = -12,
    ParameterError = -13,
    StructSizeMismatch = -14,
    WrongAPICall = -15,
    InvalidValue = -16,
    Unknown = -99,
};

pub fn errorToErrorCode(err: QError) ErrorCode {
    return switch (err) {
        inline else => |e| @field(ErrorCode, @errorName(e)),
    };
}
pub fn errorCodeFromInt(code: c_int) ErrorCode {
    return std.enums.fromInt(ErrorCode, code);
}
