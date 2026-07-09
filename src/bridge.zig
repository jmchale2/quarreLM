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
    ParameterError,
    WrongAPICall,
    StructSizeMismatch,
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
fn sliceOrFill(alloc: std.mem.Allocator, ptr: ?[*]const f64, p: usize, fill: f64) ![]const f64 {
    if (ptr) |raw| return raw[0..p];
    const buf = try alloc.alloc(f64, p);
    @memset(buf, fill);
    return buf;
}

// =========================================
// Coverting to a fit and fit_path structure
// =========================================
//

pub const Solver = enum(c_int) {
    ols = 0,
    enet = 1,
    enet_path = 2,
};

pub const FitOptions = struct {
    alpha: f64,
    lambda: f64,
    tol: f64,
    max_iter: u64,
    n_lambda: ?u64,
    lambda_min_ratio: ?f64,
    penalty_factors: ?[*]const f64,
    lower_bounds: ?[*]const f64,
    upper_bounds: ?[*]const f64,
    warm_start: ?[*]const f64,
};

pub const FitResult = struct {
    n_iter: u64,
    out_coeffs: [*]f64,
};

pub const PathResult = struct {
    n_iters: [*]u64,
    lambda_paths: [*]f64,
    out_coeffs_matrix: [*]f64,
};

pub fn fit(
    stream_ptr: *arrow.ArrowArrayStream,
    y_array_ptr: *arrow.ArrowArray,
    y_schema_ptr: *arrow.ArrowSchema,
    n_features: c_int,
    solver_enum: Solver,
    opts: FitOptions,
    out: *FitResult,
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

    var n_iters: usize = undefined;
    switch (solver_enum) {
        .ols => {
            _ = try regression.olsFitVec(table.columns, y, out.out_coeffs[0..p]);
            n_iters = 0;
        },
        .enet => {
            const penalty_factors = try sliceOrFill(alloc, opts.penalty_factors, p, 1.0);
            const lower_bounds = try sliceOrFill(alloc, opts.lower_bounds, p, -std.math.inf(f64));
            const upper_bounds = try sliceOrFill(alloc, opts.upper_bounds, p, std.math.inf(f64));

            //TODO: Handle Warm Starts here!

            const enet_opts: regression.EnetOptions = .{
                .lambda = opts.lambda,
                .alpha = opts.alpha,
                .penalty_factors = penalty_factors,
                .lower_bounds = lower_bounds,
                .upper_bounds = upper_bounds,
                .max_iter = opts.max_iter,
                .tol = opts.tol,
            };

            n_iters = try regression.elasticNetFit(alloc, table.columns, y, out.out_coeffs[0..p], enet_opts);
        },
        .enet_path => return error.ParameterError,
    }

    return n_iters;
}

test {
    _ = &fit;
}

pub fn fit_path(solver_enum: Solver) !void {
    _ = solver_enum;
    // const params: FitParams = switch (solver_enum) {
    //     .ols => return error.ParameterError,
    //     .enet => return error.ParameterError,
    //     .enet_path => .{ .enet_path = .{} },
    // };

}

//===========================================
//Individual Fit Calls
//===========================================

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

    // marshall params
    const params = regression.EnetOptions{
        .lambda = lambda,
        .alpha = alpha,
        .penalty_factors = penalty_factors[0..p],
        .lower_bounds = lower_bounds[0..p],
        .upper_bounds = upper_bounds[0..p],
        .tol = tol,
        .max_iter = max_iter,
    };

    const n_iter = try regression.elasticNetFit(
        alloc,
        table.columns,
        y,
        out_coeffs[0..p],
        params,
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

    const params = regression.PathOptions{
        .alpha = alpha,
        .penalty_factors = penalty_factors[0..p],
        .lower_bounds = lower_bounds[0..p],
        .upper_bounds = upper_bounds[0..p],
        .n_lambda = nl,
        .lambda_min_ratio = lambda_min_ratio,
        .max_iter = max_iter,
        .tol = tol,
    };

    const n_iter = try regression.elasticNetPath(
        alloc,
        table.columns,
        y,
        out_coef_matrix[0 .. p * nl],
        out_lambdas[0..nl],
        params,
    );

    return n_iter;
}

// mock a stream
pub const mock = struct {
    const n = 8;
    var col0 = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var col1 = [_]f64{ 8, 7, 6, 5, 4, 3, 2, 1 };
    var yv = [_]f64{ 10, 11, 12, 13, 14, 15, 16, 17 }; // 2*x0 + x1

    fn releaseSchema(s: *arrow.ArrowSchema) callconv(.c) void {
        s.release = null;
    }
    fn releaseArray(a: *arrow.ArrowArray) callconv(.c) void {
        a.release = null;
    }

    var child_schemas = [_]arrow.ArrowSchema{
        .{ .format = "g", .name = null, .metadata = null, .flags = 0, .n_children = 0, .children = null, .dictionary = null, .release = null, .private_data = null },
        .{ .format = "g", .name = null, .metadata = null, .flags = 0, .n_children = 0, .children = null, .dictionary = null, .release = null, .private_data = null },
    };
    var schema_children = [_][*c]arrow.ArrowSchema{ &child_schemas[0], &child_schemas[1] };

    var bufs0 = [_]?*const anyopaque{ null, &col0 };
    var bufs1 = [_]?*const anyopaque{ null, &col1 };
    var child_arrays = [_]arrow.ArrowArray{
        .{ .length = n, .null_count = 0, .offset = 0, .n_buffers = 2, .n_children = 0, .buffers = &bufs0, .children = null, .dictionary = null, .release = null, .private_data = null },
        .{ .length = n, .null_count = 0, .offset = 0, .n_buffers = 2, .n_children = 0, .buffers = &bufs1, .children = null, .dictionary = null, .release = null, .private_data = null },
    };
    var batch_children = [_][*c]arrow.ArrowArray{ &child_arrays[0], &child_arrays[1] };

    fn getSchema(_: *arrow.ArrowArrayStream, out: *arrow.ArrowSchema) callconv(.c) c_int {
        out.* = .{ .format = "+s", .name = null, .metadata = null, .flags = 0, .n_children = 2, .children = &schema_children, .dictionary = null, .release = releaseSchema, .private_data = null };
        return 0;
    }

    var served: bool = false;
    fn getNext(_: *arrow.ArrowArrayStream, out: *arrow.ArrowArray) callconv(.c) c_int {
        if (served) {
            out.release = null;
            return 0;
        } // end of stream
        served = true;
        out.* = .{ .length = n, .null_count = 0, .offset = 0, .n_buffers = 0, .n_children = 2, .buffers = null, .children = &batch_children, .dictionary = null, .release = releaseArray, .private_data = null };
        return 0;
    }

    pub fn makeStream() arrow.ArrowArrayStream {
        served = false;
        return .{ .get_schema = getSchema, .get_next = getNext, .get_last_error = null, .release = null, .private_data = null };
    }

    var y_bufs = [_]?*const anyopaque{ null, &yv };
    pub var y_array = arrow.ArrowArray{ .length = n, .null_count = 0, .offset = 0, .n_buffers = 2, .n_children = 0, .buffers = &y_bufs, .children = null, .dictionary = null, .release = null, .private_data = null };
    pub var y_schema = arrow.ArrowSchema{ .format = "g", .name = null, .metadata = null, .flags = 0, .n_children = 0, .children = null, .dictionary = null, .release = null, .private_data = null };
};

test "bridge path: alpha actually reaches the solver (lambda_max scales as 1/alpha)" {
    const inf_ = std.math.inf(f64);
    var pf = [_]f64{ 1.0, 1.0 };
    var lb = [_]f64{ -inf_, -inf_ };
    var ub = [_]f64{ inf_, inf_ };
    const n_lambda = 5;
    var out_coefs: [2 * n_lambda]f64 = undefined;
    var out_lambdas: [n_lambda]f64 = undefined;

    var s1 = mock.makeStream();
    _ = try elasticNetPath(&s1, &mock.y_array, &mock.y_schema, &pf, &lb, &ub, &out_coefs, &out_lambdas, 2, n_lambda, 1.0, 1e-4, 1e-7, 1000);
    const lmax_at_1 = out_lambdas[0];

    var s2 = mock.makeStream();
    _ = try elasticNetPath(&s2, &mock.y_array, &mock.y_schema, &pf, &lb, &ub, &out_coefs, &out_lambdas, 2, n_lambda, 0.5, 1e-4, 1e-7, 1000);
    const lmax_at_half = out_lambdas[0];

    // lambda_max ∝ 1/alpha — halving alpha must double lambda_max
    try std.testing.expectApproxEqRel(lmax_at_1 * 2.0, lmax_at_half, 1e-12);
}
