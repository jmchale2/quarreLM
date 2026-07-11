const std = @import("std");
const arrow = @import("arrow.zig");
const bridge = @import("bridge.zig");
const errors = @import("errors.zig");

const build_options = @import("build_options");

// capi  error management
fn errorToC(err: errors.QError) c_int {
    setLastError(err);
    return @intFromEnum(errors.errorToErrorCode(err));
}

threadlocal var last_error: [*:0]const u8 = "";

fn setLastError(err: anyerror) void {
    last_error = @errorName(err);
}

export fn quarrel_last_error() callconv(.c) [*:0]const u8 {
    return last_error;
}

export fn quarrel_last_error_context() callconv(.c) [*:0]const u8 {
    return errors.context;
}

export fn quarrel_error_name(code: c_int) callconv(.c) [*:0]const u8 {
    const ec = std.enums.fromInt(errors.ErrorCode, code) orelse return "InvalidErrorCode";
    return @tagName(ec);
}

/// Set's an error context and returns the error code. Can be used with string formatters.
/// Preferred way to set errors in capi.
fn fail(err: errors.QError, comptime fmt: []const u8, args: anytype) c_int {
    errors.setContext(fmt, args);
    return errorToC(err);
}

// general functions
/// Array length validation. Helpful when validating arrow contracts.
export fn quarrel_array_len(arr_ptr: *arrow.ArrowArray, arr_schema_ptr: *arrow.ArrowSchema) callconv(.c) c_int {
    const arr = arrow.asFloat64Slice(arr_ptr, arr_schema_ptr) catch |err| {
        std.debug.print("Error: {any}\n", .{err});
        return -1;
    };
    return @as(i32, @intCast(arr.len));
}

/// Simple health check — returns 42. Use to verify the library loads.
export fn quarrel_ping() callconv(.c) c_int {
    return 42;
}

/// Returns the library version as a static string.
export fn quarrel_version() callconv(.c) [*:0]const u8 {
    errors.clearContext();
    return build_options.version ++ "\x00";
}

/// ABI probe: returns how many args crossed intact. Expected: 5.
export fn quarrel_abi_probe(_: ?*anyopaque, _: ?*anyopaque, _: ?*anyopaque, _: ?*anyopaque, _: ?*anyopaque, _: ?*anyopaque, _: ?*anyopaque, n: c_int, a: f64, b: f64, c: f64, m: c_int) callconv(.c) c_int {
    errors.clearContext();
    var ok: c_int = 0;
    if (n == 42) ok += 1;
    if (a == 1.5) ok += 1;
    if (b == 2.5) ok += 1;
    if (c == 3.5) ok += 1;
    if (m == 7) ok += 1;
    return ok;
}

/// Convert Solver enum to a c_int
fn solverToCode(solver: bridge.Solver) c_int {
    return @intFromEnum(solver);
}
/// Gather Solver enum from a c_int
fn solverCodeFromInt(code: c_int) ?bridge.Solver {
    return std.enums.fromInt(bridge.Solver, code);
}

// objects
pub const CFitOptions = extern struct {
    struct_size: u64,
    lambda: f64,
    alpha: f64,
    tol: f64,
    max_iter: u64,
    n_lambda: u64,
    lambda_min_ratio: f64,
    penalty_factors: ?[*]const f64,
    lower_bounds: ?[*]const f64,
    upper_bounds: ?[*]const f64,
    warm_start: ?[*]const f64,
};

pub const CFitResult = extern struct {
    struct_size: u64,
    n_iter: u64,
    out_coeffs: ?[*]f64,
};

pub const CPathResult = extern struct {
    struct_size: u64,
    n_iters: ?[*]u64,
    lambda_paths: ?[*]f64,
    out_coeffs_matrix: ?[*]f64,
};

// fit functions
export fn quarrel_fit(
    stream: *arrow.ArrowArrayStream,
    y: *arrow.ArrowArray,
    y_schema: *arrow.ArrowSchema,
    n_features: c_int,
    solver: c_int,
    opts: *const CFitOptions,
    out: *CFitResult,
) callconv(.c) c_int {
    errors.clearContext();
    if (opts.struct_size != @sizeOf(CFitOptions)) return fail(errors.QError.StructSizeMismatch, "CFitOptions structs are not the same size!", .{});

    const solver_enum = solverCodeFromInt(solver) orelse return fail(
        errors.QError.ParameterError,
        "Solver Enum {d} did not resolve",
        .{solver},
    );

    if (solver_enum == bridge.Solver.enet_path) {
        return errorToC(error.WrongAPICall);
    }

    const out_coeffs = out.out_coeffs orelse return errorToC(errors.QError.ParameterError);

    const fit_opts = bridge.FitOptions{
        .alpha = opts.alpha,
        .lambda = opts.lambda,
        .penalty_factors = opts.penalty_factors,
        .lower_bounds = opts.lower_bounds,
        .upper_bounds = opts.upper_bounds,
        .warm_start = opts.warm_start,
        .tol = opts.tol,
        .max_iter = opts.max_iter,

        // path only
        .n_lambda = opts.n_lambda,
        .lambda_min_ratio = opts.lambda_min_ratio,
    };

    var fit_out = bridge.FitResult{
        .out_coeffs = out_coeffs,
        .n_iter = undefined,
    };

    const return_code = bridge.fit(stream, y, y_schema, n_features, solver_enum, fit_opts, &fit_out) catch |err| {
        return errorToC(err);
    };
    out.n_iter = fit_out.n_iter;

    return @intCast(return_code);
}

export fn quarrel_fit_path(
    stream: *arrow.ArrowArrayStream,
    y: *arrow.ArrowArray,
    y_schema: *arrow.ArrowSchema,
    n_features: c_int,
    solver: c_int,
    opts: *const CFitOptions,
    out: *CPathResult,
) callconv(.c) c_int {
    errors.clearContext();
    if (opts.struct_size != @sizeOf(CFitOptions)) return errorToC(errors.QError.StructSizeMismatch);

    const solver_enum = solverCodeFromInt(solver) orelse return errorToC(errors.QError.ParameterError);

    if (solver_enum != bridge.Solver.enet_path) {
        return errorToC(errors.QError.WrongAPICall);
    }

    const out_coeffs_matrix = out.out_coeffs_matrix orelse return errorToC(errors.QError.ParameterError);
    const lambda_paths = out.lambda_paths orelse return errorToC(errors.QError.ParameterError);
    const n_iters = out.n_iters orelse return errorToC(errors.QError.ParameterError);

    const lambda_min_ratio: ?f64 =
        if (opts.lambda_min_ratio == -1.0) null else if (opts.lambda_min_ratio <= 0 or opts.lambda_min_ratio >= 1)
            return errorToC(error.ParameterError)
        else
            opts.lambda_min_ratio;

    const n_lambda = if (opts.n_lambda == 0) 100 else opts.n_lambda;
    if (n_lambda < 2) return fail(errors.QError.ParameterError, "n_lambda must be >= 2, received n_lambda={d}", .{n_lambda});

    const fit_opts = bridge.FitOptions{
        .alpha = opts.alpha,
        .lambda = opts.lambda,
        .penalty_factors = opts.penalty_factors,
        .lower_bounds = opts.lower_bounds,
        .upper_bounds = opts.upper_bounds,
        .warm_start = opts.warm_start,
        .tol = opts.tol,
        .max_iter = opts.max_iter,

        // path only
        .n_lambda = n_lambda,
        .lambda_min_ratio = lambda_min_ratio,
    };

    var path_out = bridge.PathResult{
        .out_coeffs_matrix = out_coeffs_matrix,
        .n_iters = n_iters,
        .lambda_paths = lambda_paths,
    };

    const return_code = bridge.fit_path(stream, y, y_schema, n_features, solver_enum, fit_opts, &path_out) catch |err| {
        return errorToC(err);
    };

    return @intCast(return_code);
}

test "quarrel_fit matches bridge.fit (capi plumbing)" {
    const inf_ = std.math.inf(f64);
    var pf = [_]f64{ 1.0, 1.0 };
    var lb = [_]f64{ -inf_, -inf_ };
    var ub = [_]f64{ inf_, inf_ };
    var c_out_coefs = [_]f64{ 0, 0 };
    var out_coefs = [_]f64{ 0, 0 };

    // --- through the C ABI export ---
    const copts = CFitOptions{
        .struct_size = @sizeOf(CFitOptions),
        .lambda = 0.01,
        .alpha = 0.5,
        .tol = 1e-7,
        .max_iter = 1000,
        .n_lambda = 0, // wire sentinel: unset
        .lambda_min_ratio = -1.0, // wire sentinel: unset
        .penalty_factors = &pf,
        .lower_bounds = &lb,
        .upper_bounds = &ub,
        .warm_start = null,
    };
    var c_out = CFitResult{
        .struct_size = @sizeOf(CFitResult),
        .n_iter = 0,
        .out_coeffs = &c_out_coefs,
    };

    var s1 = bridge.mock.makeStream();
    const rc = quarrel_fit(&s1, &bridge.mock.y_array, &bridge.mock.y_schema, 2, @intFromEnum(bridge.Solver.enet), &copts, &c_out);
    try std.testing.expect(rc > 0);

    // --- reference: same fit through bridge directly ---
    const bopts = bridge.FitOptions{
        .lambda = 0.01,
        .alpha = 0.5,
        .tol = 1e-7,
        .max_iter = 1000,
        .n_lambda = null,
        .lambda_min_ratio = null,
        .penalty_factors = &pf,
        .lower_bounds = &lb,
        .upper_bounds = &ub,
        .warm_start = null,
    };
    var b_out = bridge.FitResult{ .n_iter = 0, .out_coeffs = &out_coefs };

    var s2 = bridge.mock.makeStream();
    const n_iter = try bridge.fit(&s2, &bridge.mock.y_array, &bridge.mock.y_schema, 2, .enet, bopts, &b_out);

    // identical computation, not just similar results
    try std.testing.expectEqual(@as(c_int, @intCast(n_iter)), rc);
    try std.testing.expectEqual(@as(u64, @intCast(n_iter)), c_out.n_iter); // see note

    try std.testing.expectApproxEqRel(c_out_coefs[0], out_coefs[0], 1e-8);
    try std.testing.expectApproxEqRel(c_out_coefs[1], out_coefs[1], 1e-8);
}

test "quarrel_fit: null pf/lb/ub produce the defaults" {
    var explicit_pf = [_]f64{ 1.0, 1.0 };
    const inf_ = std.math.inf(f64);
    var explicit_lb = [_]f64{ -inf_, -inf_ };
    var explicit_ub = [_]f64{ inf_, inf_ };
    var coefs_explicit = [_]f64{ 0, 0 };
    var coefs_defaulted = [_]f64{ 0, 0 };

    var copts = CFitOptions{
        .struct_size = @sizeOf(CFitOptions),
        .lambda = 0.01,
        .alpha = 0.5,
        .tol = 1e-7,
        .max_iter = 1000,
        .n_lambda = 0,
        .lambda_min_ratio = -1.0,
        .penalty_factors = &explicit_pf,
        .lower_bounds = &explicit_lb,
        .upper_bounds = &explicit_ub,
        .warm_start = null,
    };
    var out = CFitResult{ .struct_size = @sizeOf(CFitResult), .n_iter = 0, .out_coeffs = &coefs_explicit };

    var s1 = bridge.mock.makeStream();
    _ = quarrel_fit(&s1, &bridge.mock.y_array, &bridge.mock.y_schema, 2, @intFromEnum(bridge.Solver.enet), &copts, &out);

    // explicit values above ARE the documented defaults — nulls must match exactly
    copts.penalty_factors = null;
    copts.lower_bounds = null;
    copts.upper_bounds = null;
    out.out_coeffs = &coefs_defaulted;

    var s2 = bridge.mock.makeStream();
    _ = quarrel_fit(&s2, &bridge.mock.y_array, &bridge.mock.y_schema, 2, @intFromEnum(bridge.Solver.enet), &copts, &out);

    try std.testing.expectEqual(coefs_explicit[0], coefs_defaulted[0]);
    try std.testing.expectEqual(coefs_explicit[1], coefs_defaulted[1]);
}

test "quarrel_fit: struct_size mismatch is rejected" {
    var coefs = [_]f64{ 0, 0 };
    var copts = CFitOptions{
        .struct_size = 0, // wrong on purpose
        .lambda = 0.01,
        .alpha = 0.5,
        .tol = 1e-7,
        .max_iter = 1000,
        .n_lambda = 0,
        .lambda_min_ratio = -1.0,
        .penalty_factors = null,
        .lower_bounds = null,
        .upper_bounds = null,
        .warm_start = null,
    };
    var out = CFitResult{ .struct_size = @sizeOf(CFitResult), .n_iter = 0, .out_coeffs = &coefs };

    var s = bridge.mock.makeStream();
    const rc = quarrel_fit(&s, &bridge.mock.y_array, &bridge.mock.y_schema, 2, @intFromEnum(bridge.Solver.enet), &copts, &out);
    try std.testing.expectEqual(@intFromEnum(errors.ErrorCode.StructSizeMismatch), rc);
}
