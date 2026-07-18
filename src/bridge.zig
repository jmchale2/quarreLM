const arrow = @import("arrow.zig");
const std = @import("std");
const regression = @import("regression.zig");

const errors = @import("errors.zig");
const fixtures = @import("fixtures.zig");

const OLSMethod = @import("solvers/ols.zig").Method;
const OLSOptions = @import("solvers/ols.zig").Options;

const EnetOptions = @import("solvers/enet.zig").Options;
const PathOptions = @import("solvers/enet.zig").PathOptions;

const Solver = regression.Solver;
const StatsAccumulator = @import("solvers/common.zig").StatsAccumulator;
const SufficientStats = @import("solvers/common.zig").SufficientStats;
const StatsSpec = @import("solvers/common.zig").StatsSpec;

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

pub const StreamCursor = struct {
    stream: *arrow.ArrowArrayStream,
    schema: arrow.ArrowSchema,
    columns: [][]const f64,
    p: usize,

    pub const Batch = struct {
        array: arrow.ArrowArray,
        columns: []const []const f64,
        n_rows: usize,

        pub fn release(self: *Batch) void {
            if (self.array.release) |rel| rel(&self.array);
        }
    };

    /// Pull and validate the schema, allocate the column scratch.
    /// On any failure the stream (and schema, if pulled) is released before return.
    pub fn open(alloc: std.mem.Allocator, stream: *arrow.ArrowArrayStream, p: usize) !StreamCursor {
        var schema: arrow.ArrowSchema = undefined;
        if (stream.get_schema.?(stream, &schema) != 0) {
            if (stream.release) |rel| rel(stream);
            return arrow.ArrowError.SchemaError;
        }
        errdefer {
            if (schema.release) |rel| rel(&schema);
            if (stream.release) |rel| rel(stream);
        }

        if (schema.n_children != @as(i64, @intCast(p))) return errors.QError.SchemaError;

        const columns = try alloc.alloc([]const f64, p);

        return .{
            .stream = stream,
            .schema = schema,
            .columns = columns,
            .p = p,
        };
    }

    /// Advance to the next batch. Returns null at end of stream.
    /// The returned batch's `columns` alias cursor scratch (valid until the next call).
    pub fn next(self: *StreamCursor) !?Batch {
        var array: arrow.ArrowArray = undefined;
        if (self.stream.get_next.?(self.stream, &array) != 0) return arrow.ArrowError.StreamError;
        if (array.release == null) return null; // end of stream — nothing to release

        errdefer if (array.release) |rel| rel(&array);

        if (array.n_children != @as(i64, @intCast(self.p))) return errors.QError.BatchSchemaError;

        for (0..self.p) |i| {
            const child_array: *arrow.ArrowArray = @ptrCast(array.children[i]);
            const child_schema: *arrow.ArrowSchema = @ptrCast(self.schema.children[i]);
            self.columns[i] = try arrow.asFloat64Slice(child_array, child_schema);
        }

        return .{
            .array = array,
            .columns = self.columns,
            .n_rows = @intCast(array.length),
        };
    }

    /// Release schema and stream. Call once, after the final `next`.
    pub fn deinit(self: *StreamCursor) void {
        if (self.schema.release) |rel| rel(&self.schema);
        if (self.stream.release) |rel| rel(self.stream);
    }
};
fn accumulateStream(
    alloc: std.mem.Allocator,
    stream: *arrow.ArrowArrayStream,
    y: []const f64,
    p: usize,
    spec: StatsSpec,
) !SufficientStats {
    var cursor = try StreamCursor.open(alloc, stream, p);
    defer cursor.deinit();

    var acc = try StatsAccumulator.init(alloc, p, spec);

    var offset: usize = 0;
    while (try cursor.next()) |batch_val| {
        var batch = batch_val;
        defer batch.release();
        const end = offset + batch.n_rows;
        if (end > y.len) return errors.QError.DimensionMismatch;
        acc.update(batch.columns, y[offset..end]);
        offset = end;
    }

    if (acc.n_seen == 0) return arrow.ArrowError.EmptyStream; // finalize asserts n_seen > 0
    if (offset != y.len) return errors.QError.DimensionMismatch;

    return acc.finalize();
}

pub fn importStream(alloc: std.mem.Allocator, stream: *arrow.ArrowArrayStream, p: usize) !Table {
    var schema: arrow.ArrowSchema = undefined;
    const schema_rc = stream.get_schema.?(stream, &schema);
    defer if (stream.release) |rel| rel(stream);
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
    if (batches.items.len > 1) return errors.QError.ChunkedNotSupported;

    const batch = batches.items[0];

    // FUTURE:  can iterate over batches to handle chunks
    // for (batches.items) |batch| {
    n_rows += @intCast(batch.length);

    if (schema.n_children != p) return errors.QError.SchemaError;
    if (batch.n_children != p) return errors.QError.BatchSchemaError;

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

pub const Ingested = struct {
    table: ?Table,
    stats: SufficientStats,

    pub fn deinit(self: *Ingested) void {
        if (self.table) |*t| t.deinit();
    }
};

pub fn ingest(
    alloc: std.mem.Allocator,
    stream: *arrow.ArrowArrayStream,
    y: []const f64,
    p: usize,
    plan: regression.IngestPlan,
) !Ingested {
    switch (plan.mode) {
        .stream => return .{
            .table = null,
            .stats = try accumulateStream(alloc, stream, y, p, plan.spec),
        },
        .materialize => {
            var table = try importStream(alloc, stream, p);
            errdefer table.deinit();
            if (y.len != table.n_rows) return errors.QError.DimensionMismatch;

            var acc = try StatsAccumulator.init(alloc, p, plan.spec);
            acc.update(table.columns, y);
            return .{ .table = table, .stats = acc.finalize() };
        },
    }
}

fn sliceOrFill(alloc: std.mem.Allocator, ptr: ?[*]const f64, p: usize, fill: f64) ![]const f64 {
    if (ptr) |raw| return raw[0..p];
    const buf = try alloc.alloc(f64, p);
    @memset(buf, fill);
    return buf;
}

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
    ols_method: OLSMethod = OLSMethod.auto,
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

    const ingest_plan = regression.planIngest(solver_enum, p);
    //TODO: stream mode does not have a consumer yet (solveFromStats)
    // ingest_plan.mode = .materialize;

    var ing = try ingest(alloc, stream_ptr, y, p, ingest_plan);

    defer ing.deinit();

    if (ing.table != null) {
        if (y.len != ing.table.?.n_rows) return errors.QError.StreamError;
    }
    var regopts: regression.SolverOptions = undefined;
    switch (solver_enum) {
        .ols => {
            regopts = .{ .ols = .{
                .method = opts.ols_method,
            } };
            // _ = try regression.olsFit(alloc, table.columns, y, out.out_coeffs[0..p], ols_opts);
        },
        .enet => {
            const penalty_factors = try sliceOrFill(alloc, opts.penalty_factors, p, 1.0);
            const lower_bounds = try sliceOrFill(alloc, opts.lower_bounds, p, -std.math.inf(f64));
            const upper_bounds = try sliceOrFill(alloc, opts.upper_bounds, p, std.math.inf(f64));

            regopts = .{ .enet = .{
                .lambda = opts.lambda,
                .alpha = opts.alpha,
                .penalty_factors = penalty_factors,
                .lower_bounds = lower_bounds,
                .upper_bounds = upper_bounds,
                .warm_start = if (opts.warm_start) |w| w[0..p] else null,
                .max_iter = opts.max_iter,
                .tol = opts.tol,
            } };

            // const table = ing.table orelse return errors.QError.DimensionMismatch;
            // n_iters = try regression.elasticNetFit(alloc, table.columns, y, out.out_coeffs[0..p], enet_opts);
            // out.n_iter = n_iters;
        },
        .enet_path => return errors.QError.ParameterError,
    }

    const cols: ?[]const []const f64 = if (ing.table) |t| t.columns else null;
    const n_iter = try regression.solveFromStats(
        alloc,
        cols,
        y,
        ing.stats,
        regopts,
        out.out_coeffs[0..p],
    );
    out.n_iter = n_iter;

    return n_iter;
}

test {
    _ = &fit;
}

pub fn fit_path(
    stream_ptr: *arrow.ArrowArrayStream,
    y_array_ptr: *arrow.ArrowArray,
    y_schema_ptr: *arrow.ArrowSchema,
    n_features: c_int,
    solver_enum: Solver,
    opts: FitOptions,
    out: *PathResult,
) !usize {
    if (opts.lambda_min_ratio != null) std.debug.assert((opts.lambda_min_ratio.? > 0) and (opts.lambda_min_ratio.? < 1));

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const p: usize = @intCast(n_features);
    // Extract y
    const y = try arrow.asFloat64Slice(y_array_ptr, y_schema_ptr);

    var table = try importStream(alloc, stream_ptr, p);
    defer table.deinit();

    if (y.len != table.n_rows) return errors.QError.DimensionMismatch;

    var total_iters: usize = undefined;
    switch (solver_enum) {
        .ols => {
            return errors.QError.WrongAPICall;
        },
        .enet => {
            return errors.QError.WrongAPICall;
        },
        .enet_path => {
            const penalty_factors = try sliceOrFill(alloc, opts.penalty_factors, p, 1.0);
            const lower_bounds = try sliceOrFill(alloc, opts.lower_bounds, p, -std.math.inf(f64));
            const upper_bounds = try sliceOrFill(alloc, opts.upper_bounds, p, std.math.inf(f64));
            const lambda_min_ratio: f64 = opts.lambda_min_ratio orelse
                if (table.n_rows >= p) 1e-4 else 1e-2;

            const n_lambda = opts.n_lambda orelse return errors.QError.ParameterError;

            const path_opts: PathOptions = .{
                .alpha = opts.alpha,
                .penalty_factors = penalty_factors,
                .lower_bounds = lower_bounds,
                .upper_bounds = upper_bounds,
                .n_lambda = n_lambda,
                .lambda_min_ratio = lambda_min_ratio,
                .warm_start = if (opts.warm_start) |w| w[0..p] else null,
                .max_iter = opts.max_iter,
                .tol = opts.tol,
            };

            total_iters = try regression.elasticNetPath(
                alloc,
                table.columns,
                y,
                out.out_coeffs_matrix[0 .. p * n_lambda],
                out.lambda_paths[0..n_lambda],
                out.n_iters[0..n_lambda],
                path_opts,
            );
        },
    }

    return total_iters;
}

test {
    _ = &fit_path;
}

test "bridge path: alpha actually reaches the solver (lambda_max scales as 1/alpha)" {
    const inf_ = std.math.inf(f64);
    const pf = [_]f64{ 1.0, 1.0 };
    const lb = [_]f64{ -inf_, -inf_ };
    const ub = [_]f64{ inf_, inf_ };
    const n_lambda = 5;
    var opts: FitOptions = .{
        .alpha = 1,
        .lambda = 1e-4,
        .tol = 1e-7,
        .max_iter = 1000,
        .n_lambda = n_lambda,
        .lambda_min_ratio = null,
        .penalty_factors = &pf,
        .lower_bounds = &lb,
        .upper_bounds = &ub,
        .warm_start = null,
        .ols_method = OLSMethod.auto,
    };

    var out_lambdas: [n_lambda]f64 = undefined;
    @memset(&out_lambdas, 0);
    var out_coef_matrix: [2 * n_lambda]f64 = undefined;
    @memset(&out_coef_matrix, 0);
    var out_iters: [n_lambda]u64 = undefined;
    @memset(&out_iters, 0);

    var result = PathResult{ .lambda_paths = &out_lambdas, .n_iters = &out_iters, .out_coeffs_matrix = &out_coef_matrix };

    const mock = fixtures.mock;
    var s1 = mock.makeStream();
    _ = try fit_path(&s1, &mock.y_array, &mock.y_schema, 2, Solver.enet_path, opts, &result);
    const lmax_at_1 = out_lambdas[0];

    var out_lambdas2: [n_lambda]f64 = undefined;
    @memset(&out_lambdas2, 0);
    var out_coef_matrix2: [2 * n_lambda]f64 = undefined;
    @memset(&out_coef_matrix2, 0);
    var out_iters2: [n_lambda]u64 = undefined;
    @memset(&out_iters2, 0);

    opts.alpha = 0.5;

    var result2 = PathResult{ .lambda_paths = &out_lambdas2, .n_iters = &out_iters2, .out_coeffs_matrix = &out_coef_matrix2 };

    var s2 = mock.makeStream();
    _ = try fit_path(&s2, &mock.y_array, &mock.y_schema, 2, Solver.enet_path, opts, &result2);
    const lmax_at_half = out_lambdas2[0];

    // lambda_max ∝ 1/alpha — halving alpha must double lambda_max
    try std.testing.expectApproxEqRel(lmax_at_1 * 2.0, lmax_at_half, 1e-6);
}

test {
    std.testing.refAllDecls(@This());
}
