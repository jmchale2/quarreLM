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
    NotPositiveDefinite,
    InvalidArgument,
    SolveFailed,
    NotImplemented,
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
    NotPositiveDefinite = -17,
    InvalidArgument = -18,
    SolveFailed = -19,
    NotImplemented = -20,
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

pub threadlocal var context_buf: [256]u8 = undefined;
pub threadlocal var context: [:0]const u8 = "";

pub fn setContext(comptime fmt: []const u8, args: anytype) void {
    // context = std.mem.printSentinel(&context_buf, fmt, args, 0) catch "context truncated"; // master
    context = std.fmt.bufPrintSentinel(&context_buf, fmt, args, 0) catch "context truncated"; // 0.16.0
}
pub fn clearContext() void {
    context = "";
}
