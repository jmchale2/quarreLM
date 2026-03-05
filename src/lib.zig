const std = @import("std");
const arrow = @import("arrow.zig");
const regression = @import("regression.zig");

const build_options = @import("build_options");

// C ABI Exports - consumed from the python side of the fence

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
    olsFitInternal(
        stream_ptr,
        y_array_ptr,
        y_schema_ptr,
        out_coeffs,
        n_features,
    ) catch |err| {
        return switch (err) {
            arrow.ArrowError.WrongFormat => -1,
            arrow.ArrowError.HasNulls => -2,
            arrow.ArrowError.NullBuffer => -3,
            arrow.ArrowError.StreamError => -4,
            arrow.ArrowError.SchemaError => -5,
            error.DimensionMismatch => -6,
            error.SingularMatrix => -7,
            else => -99,
        };
    };
    return 0;
}

fn olsFitInternal(
    stream_ptr: *arrow.ArrowArrayStream,
    y_array_ptr: *arrow.ArrowArray,
    y_schema_ptr: *arrow.ArrowSchema,
    out_coeffs: [*]f64,
    n_features: c_int,
) !void {
    const p: usize = @intCast(n_features);

    // Extract y
    const y = try arrow.asFloat64Slice(y_array_ptr, y_schema_ptr);

    // Get schema from stream to validate
    var schema: arrow.ArrowSchema = undefined;
    const schema_rc = stream_ptr.get_schema.?(stream_ptr, &schema);
    if (schema_rc != 0) return arrow.ArrowError.SchemaError;
    defer {
        if (schema.release) |release_fn| release_fn(&schema);
    }

    // Read the batch from the stream
    var batch: arrow.ArrowArray = undefined;
    const batch_rc = stream_ptr.get_next.?(stream_ptr, &batch);
    if (batch_rc != 0) return arrow.ArrowError.StreamError;
    defer {
        if (batch.release) |release_fn| release_fn(&batch);
    }

    // batch.children contains one ArrowArray per column
    // schema.children contains one ArrowSchema per column
    const allocator = std.heap.page_allocator;
    const columns = try allocator.alloc([]const f64, p);
    defer allocator.free(columns);

    for (0..p) |i| {
        const child_array: *arrow.ArrowArray = @ptrCast(batch.children[i]);
        const child_schema: *arrow.ArrowSchema = @ptrCast(schema.children[i]);
        columns[i] = try arrow.asFloat64Slice(child_array, child_schema);
    }

    // Call the solver
    try regression.olsFit(columns, y, out_coeffs[0..p]);
}

export fn quarrel_ols_fit_simd(
    stream_ptr: *arrow.ArrowArrayStream,
    y_array_ptr: *arrow.ArrowArray,
    y_schema_ptr: *arrow.ArrowSchema,
    out_coeffs: [*]f64,
    n_features: c_int,
) callconv(.c) c_int {

    // Call the internal implementation, catching errors
    olsFitVecInternal(
        stream_ptr,
        y_array_ptr,
        y_schema_ptr,
        out_coeffs,
        n_features,
    ) catch |err| {
        return switch (err) {
            arrow.ArrowError.WrongFormat => -1,
            arrow.ArrowError.HasNulls => -2,
            arrow.ArrowError.NullBuffer => -3,
            arrow.ArrowError.StreamError => -4,
            arrow.ArrowError.SchemaError => -5,
            error.DimensionMismatch => -6,
            error.SingularMatrix => -7,
            else => -99,
        };
    };
    return 0;
}

fn olsFitVecInternal(
    stream_ptr: *arrow.ArrowArrayStream,
    y_array_ptr: *arrow.ArrowArray,
    y_schema_ptr: *arrow.ArrowSchema,
    out_coeffs: [*]f64,
    n_features: c_int,
) !void {
    const p: usize = @intCast(n_features);

    // Extract y
    const y = try arrow.asFloat64Slice(y_array_ptr, y_schema_ptr);

    // Get schema from stream to validate
    var schema: arrow.ArrowSchema = undefined;
    const schema_rc = stream_ptr.get_schema.?(stream_ptr, &schema);
    if (schema_rc != 0) return arrow.ArrowError.SchemaError;
    defer {
        if (schema.release) |release_fn| release_fn(&schema);
    }

    // Read the batch from the stream
    var batch: arrow.ArrowArray = undefined;
    const batch_rc = stream_ptr.get_next.?(stream_ptr, &batch);
    if (batch_rc != 0) return arrow.ArrowError.StreamError;
    defer {
        if (batch.release) |release_fn| release_fn(&batch);
    }

    // batch.children contains one ArrowArray per column
    // schema.children contains one ArrowSchema per column
    const allocator = std.heap.page_allocator;
    const columns = try allocator.alloc([]const f64, p);
    defer allocator.free(columns);

    for (0..p) |i| {
        const child_array: *arrow.ArrowArray = @ptrCast(batch.children[i]);
        const child_schema: *arrow.ArrowSchema = @ptrCast(schema.children[i]);
        columns[i] = try arrow.asFloat64Slice(child_array, child_schema);
    }

    // Call the solver
    try regression.olsFitVec(columns, y, out_coeffs[0..p]);
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
    const n_iter = elasticNetFitInternal(
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
        return switch (err) {
            arrow.ArrowError.WrongFormat => -1,
            arrow.ArrowError.HasNulls => -2,
            arrow.ArrowError.NullBuffer => -3,
            arrow.ArrowError.StreamError => -4,
            arrow.ArrowError.SchemaError => -5,
            else => -99,
        };
    };
    return @intCast(n_iter);
}

fn elasticNetFitInternal(
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
    max_iter: usize,
) !usize {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const p: usize = @intCast(n_features);

    // Extract y
    const y = try arrow.asFloat64Slice(y_array_ptr, y_schema_ptr);

    // Get schema from stream to validate
    var schema: arrow.ArrowSchema = undefined;
    const schema_rc = stream_ptr.get_schema.?(stream_ptr, &schema);
    if (schema_rc != 0) return arrow.ArrowError.SchemaError;
    defer {
        if (schema.release) |release_fn| release_fn(&schema);
    }

    // Read the batch from the stream
    var batch: arrow.ArrowArray = undefined;
    const batch_rc = stream_ptr.get_next.?(stream_ptr, &batch);
    if (batch_rc != 0) return arrow.ArrowError.StreamError;
    defer {
        if (batch.release) |release_fn| release_fn(&batch);
    }

    // batch.children contains one ArrowArray per column
    // schema.children contains one ArrowSchema per column
    const allocator = std.heap.page_allocator;
    const columns = try allocator.alloc([]const f64, p);
    defer allocator.free(columns);

    for (0..p) |i| {
        const child_array: *arrow.ArrowArray = @ptrCast(batch.children[i]);
        const child_schema: *arrow.ArrowSchema = @ptrCast(schema.children[i]);
        columns[i] = try arrow.asFloat64Slice(child_array, child_schema);
    }

    // Call the solver
    // columns: []const []const f64,
    // y: []const f64,
    // lambda: f64,
    // alpha: f64,
    // penalty_factors: []const f64,
    // lower_bounds: []const f64,
    // upper_bounds: []const f64,
    // out_coefs: []f64,
    // max_iter: usize,
    // tol: f64,
    const n_iter = try regression.elasticNetFit(
        alloc,
        columns,
        y,
        lambda,
        alpha,
        penalty_factors[0..p],
        lower_bounds[0..p],
        upper_bounds[0..p],
        out_coeffs[0..p],
        max_iter,
        tol,
    );

    return n_iter;
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
    const n_iter = elasticNetPathInternal(
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
        return switch (err) {
            arrow.ArrowError.WrongFormat => -1,
            arrow.ArrowError.HasNulls => -2,
            arrow.ArrowError.NullBuffer => -3,
            arrow.ArrowError.StreamError => -4,
            arrow.ArrowError.SchemaError => -5,
            else => -99,
        };
    };
    return @intCast(n_iter);
}

fn elasticNetPathInternal(
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
    max_iter: usize,
) !usize {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const p: usize = @intCast(n_features);
    const nl: usize = @intCast(n_lambda);

    const y = try arrow.asFloat64Slice(y_array_ptr, y_schema_ptr);

    var schema: arrow.ArrowSchema = undefined;
    const schema_rc = stream_ptr.get_schema.?(stream_ptr, &schema);
    if (schema_rc != 0) return arrow.ArrowError.SchemaError;
    defer {
        if (schema.release) |release_fn| release_fn(&schema);
    }

    var batch: arrow.ArrowArray = undefined;
    const batch_rc = stream_ptr.get_next.?(stream_ptr, &batch);
    if (batch_rc != 0) return arrow.ArrowError.StreamError;
    defer {
        if (batch.release) |release_fn| release_fn(&batch);
    }

    const columns = try alloc.alloc([]const f64, p);
    for (0..p) |i| {
        const child_array: *arrow.ArrowArray = @ptrCast(batch.children[i]);
        const child_schema: *arrow.ArrowSchema = @ptrCast(schema.children[i]);
        columns[i] = try arrow.asFloat64Slice(child_array, child_schema);
    }

    const n_iter = try regression.elasticNetPath(
        alloc,
        columns,
        y,
        alpha,
        penalty_factors[0..p],
        lower_bounds[0..p],
        upper_bounds[0..p],
        nl,
        lambda_min_ratio,
        out_coef_matrix[0 .. p * nl],
        out_lambdas[0..nl],
        max_iter,
        tol,
    );

    return n_iter;
}
/// Simple health check — returns 42. Use to verify the library loads.
export fn quarrel_ping() callconv(.c) c_int {
    return 42;
}

/// Returns the library version as a static string.
export fn quarrel_version() callconv(.c) [*:0]const u8 {
    return build_options.version ++ "\x00";
}
