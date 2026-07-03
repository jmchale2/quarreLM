const arrow = @import("arrow.zig");
const std = @import("std");
const regression = @import("regression.zig");

pub const QuarrelError = arrow.ArrowError || error{
    DimensionMismatch,
    SingularMatrix,
    OutOfMemory,
    DegenerateData,
    SchemaError,
    ChunkedNotSupported,
    BatchSchemaError,
};

pub const Table = struct {
    schema: arrow.ArrowSchema,
    batches: std.ArrayList(arrow.ArrowArray),
    columns: [][]const f64,
    n_rows: usize,

    pub fn deinit(self: *Table) void {
        for (self.batches.items) |*b| {
            if (b.release) |release| release(b);
        }
        if (self.schema.release) |release| release(&self.schema);
    }
};

pub fn importStream(alloc: std.mem.Allocator, stream: *arrow.ArrowArrayStream, p: usize) !Table {
    var schema: arrow.ArrowSchema = undefined;
    const schema_rc = stream.get_schema.?(stream, &schema);
    if (schema_rc != 0) return arrow.ArrowError.SchemaError;

    var batches: std.ArrayList(arrow.ArrowArray) = .empty;

    errdefer {
        for (batches.items) |*b| {
            if (b.release) |release| release(b);
        }
        if (schema.release) |release| release(&schema);
    }
    while (true) {
        var batch: arrow.ArrowArray = undefined;
        if (stream.get_next.?(stream, &batch) != 0) return arrow.ArrowError.StreamError;
        if (batch.release == null) break;
        batches.append(alloc, batch) catch |err| {
            // if batch fails to append, batch not in batches, so errdefer does NOT release.
            if (batch.release) |rel| rel(&batch);
            return err;
        };
    }

    var n_rows: usize = 0;

    // Currently do no support chunkes
    if (batches.items.len == 0) return arrow.ArrowError.EmptyStream;
    if (batches.items.len > 1) return error.ChunkedNotSupported;

    const batch = batches.items[0];

    // FUTURE:  can iterate over batches to handle chunks
    // for (batches.items) |batch| {
    n_rows += @intCast(batch.length);

    if (schema.n_children != p) return error.SchemaError;
    if (batch.n_children != p) return error.BatchSchemaError;

    const columns = try alloc.alloc([]const f64, p);

    for (0..p) |i| {
        const child_array: *arrow.ArrowArray = @ptrCast(batch.children[i]);
        const child_schema: *arrow.ArrowSchema = @ptrCast(schema.children[i]);
        columns[i] = try arrow.asFloat64Slice(child_array, child_schema);
    }
    // }

    return Table{
        .schema = schema,
        .batches = batches,
        .columns = columns,
        .n_rows = n_rows,
    };
}

test "import stream compilation" {
    _ = &importStream;
}

pub fn olsFit(
    stream_ptr: *arrow.ArrowArrayStream,
    y_array_ptr: *arrow.ArrowArray,
    y_schema_ptr: *arrow.ArrowSchema,
    out_coeffs: [*]f64,
    n_features: c_int,
) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const p: usize = @intCast(n_features);

    // Extract y
    const y = try arrow.asFloat64Slice(y_array_ptr, y_schema_ptr);

    var table = try importStream(alloc, stream_ptr, p);
    defer table.deinit();

    if (y.len != table.n_rows) return error.DimensionMismatch;

    // Call the solver
    try regression.olsFit(table.columns, y, out_coeffs[0..p]);
}

pub fn olsFitVec(
    stream_ptr: *arrow.ArrowArrayStream,
    y_array_ptr: *arrow.ArrowArray,
    y_schema_ptr: *arrow.ArrowSchema,
    out_coeffs: [*]f64,
    n_features: c_int,
) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const p: usize = @intCast(n_features);

    // Extract y
    const y = try arrow.asFloat64Slice(y_array_ptr, y_schema_ptr);

    var table = try importStream(alloc, stream_ptr, p);
    defer table.deinit();

    if (y.len != table.n_rows) return error.DimensionMismatch;

    // Call the solver
    try regression.olsFitVec(table.columns, y, out_coeffs[0..p]);
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

    var table = try importStream(alloc, stream_ptr, p);
    defer table.deinit();

    if (y.len != table.n_rows) return error.DimensionMismatch;
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
        table.columns,
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

    var table = try importStream(alloc, stream_ptr, p);
    defer table.deinit();

    if (y.len != table.n_rows) return error.DimensionMismatch;

    const n_iter = try regression.elasticNetPath(
        alloc,
        table.columns,
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
