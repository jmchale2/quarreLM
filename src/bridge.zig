const arrow = @import("arrow.zig");
const std = @import("std");
const regression = @import("regression.zig");

pub fn olsFit(
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

pub fn olsFitVec(
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

pub fn elasticNetFit(
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

pub fn elasticNetPath(
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
