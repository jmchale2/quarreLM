const std = @import("std");
const arrow = @import("arrow.zig");
const bridge = @import("bridge.zig");

const build_options = @import("build_options");
// C ABI Exports - consumed from the python side of the fence

var last_error_buf: [128]u8 = undefined;
var last_error: [*:0]const u8 = "";

fn setLastError(err: anyerror) void {
    last_error = std.fmt.bufPrintZ(&last_error_buf, "{s}", .{@errorName(err)}) catch "error";
}

export fn quarrel_last_error() callconv(.c) [*:0]const u8 {
    return last_error;
}

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
    Unknown = -99,
};

fn errorToErrorCode(err: bridge.QuarrelError) ErrorCode {
    return switch (err) {
        inline else => |e| @field(ErrorCode, @errorName(e)),
    };
}
fn errorCodeFromInt(code: c_int) ?ErrorCode {
    inline for (@typeInfo(ErrorCode).@"enum".fields) |f| {
        if (code == f.value) return @enumFromInt(code);
    }
    return null;
}

fn errorToC(err: bridge.QuarrelError) c_int {
    setLastError(err);
    return @intFromEnum(errorToErrorCode(err));
}

export fn quarrel_error_name(code: c_int) callconv(.c) [*:0]const u8 {
    const ec = std.enums.fromInt(ErrorCode, code) orelse return "InvalidErrorCode";
    return @tagName(ec);
}

export fn quarrel_array_len(arr_ptr: *arrow.ArrowArray, arr_schema_ptr: *arrow.ArrowSchema) callconv(.c) c_int {
    const arr = arrow.asFloat64Slice(arr_ptr, arr_schema_ptr) catch |err| {
        std.debug.print("Error: {any}\n", .{err});
        return -1;
    };
    return @as(i32, @intCast(arr.len));
}

export fn quarrel_ols_fit(
    stream_ptr: *arrow.ArrowArrayStream,
    y_array_ptr: *arrow.ArrowArray,
    y_schema_ptr: *arrow.ArrowSchema,
    out_coeffs: [*]f64,
    n_features: c_int,
) callconv(.c) c_int {
    // Call the internal implementation, catching errors
    bridge.olsFit(
        stream_ptr,
        y_array_ptr,
        y_schema_ptr,
        out_coeffs,
        n_features,
    ) catch |err| {
        return errorToC(err);
    };
    return 0;
}

export fn quarrel_ols_fit_simd(
    stream_ptr: *arrow.ArrowArrayStream,
    y_array_ptr: *arrow.ArrowArray,
    y_schema_ptr: *arrow.ArrowSchema,
    out_coeffs: [*]f64,
    n_features: c_int,
) callconv(.c) c_int {

    // Call the internal implementation, catching errors
    bridge.olsFitVec(
        stream_ptr,
        y_array_ptr,
        y_schema_ptr,
        out_coeffs,
        n_features,
    ) catch |err| {
        return errorToC(err);
    };
    return 0;
}

export fn quarrel_enet_fit(
    stream_ptr: *arrow.ArrowArrayStream,
    y_array_ptr: *arrow.ArrowArray,
    y_schema_ptr: *arrow.ArrowSchema,
    penalty_factors: [*]f64,
    lower_bounds: [*]f64,
    upper_bounds: [*]f64,
    out_coeffs: [*]f64,
    n_features: c_int,
    lambda: f64,
    alpha: f64,
    tol: f64,
    max_iter: c_int,
) callconv(.c) c_int {
    const max_iter_usize: usize = @intCast(max_iter);

    // Call the internal implementation, catching errors
    const n_iter = bridge.elasticNetFit(
        stream_ptr,
        y_array_ptr,
        y_schema_ptr,
        penalty_factors,
        lower_bounds,
        upper_bounds,
        out_coeffs,
        n_features,
        lambda,
        alpha,
        tol,
        max_iter_usize,
    ) catch |err| {
        return errorToC(err);
    };
    return @intCast(n_iter);
}

export fn quarrel_enet_path(
    stream_ptr: *arrow.ArrowArrayStream,
    y_array_ptr: *arrow.ArrowArray,
    y_schema_ptr: *arrow.ArrowSchema,
    penalty_factors: [*]f64,
    lower_bounds: [*]f64,
    upper_bounds: [*]f64,
    out_coef_matrix: [*]f64,
    out_lambdas: [*]f64,
    n_features: c_int,
    n_lambda: c_int,
    alpha: f64,
    lambda_min_ratio: f64,
    tol: f64,
    max_iter: c_int,
) callconv(.c) c_int {
    const n_iter = bridge.elasticNetPath(
        stream_ptr,
        y_array_ptr,
        y_schema_ptr,
        penalty_factors,
        lower_bounds,
        upper_bounds,
        out_coef_matrix,
        out_lambdas,
        n_features,
        n_lambda,
        alpha,
        lambda_min_ratio,
        tol,
        @intCast(max_iter),
    ) catch |err| {
        return errorToC(err);
    };
    return @intCast(n_iter);
}

/// Simple health check — returns 42. Use to verify the library loads.
export fn quarrel_ping() callconv(.c) c_int {
    return 42;
}

/// Returns the library version as a static string.
export fn quarrel_version() callconv(.c) [*:0]const u8 {
    return build_options.version ++ "\x00";
}
