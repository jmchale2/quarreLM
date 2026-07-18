const std = @import("std");
const errors = @import("errors.zig");
const fixtures = @import("fixtures.zig");

const blas = @import("blas.zig");

const ols = @import("solvers/ols.zig");
const enet = @import("solvers/enet.zig");

const inf = std.math.inf(f64);
const clamp = std.math.clamp;

const dotProduct = @import("solvers/common.zig").dotProduct;

const axpy = @import("solvers/common.zig").axpy;

const StatsSpec = @import("solvers/common.zig").StatsSpec;
const SufficientStats = @import("solvers/common.zig").SufficientStats;
const GRAM_P_THRESHOLD = @import("solvers/enet.zig").GRAM_P_THRESHOLD;

pub const Solver = enum(c_int) {
    ols = 0,
    enet = 1,
    enet_path = 2,
};

pub fn olsFit(
    alloc: std.mem.Allocator,
    columns: []const []const f64,
    y: []const f64,
    out_coefs: []f64,
    regopts: ols.Options,
) !void {
    const p = columns.len;
    const n = y.len;

    for (columns) |col| {
        if (col.len != n) return errors.QError.DimensionMismatch;
    }
    if (out_coefs.len != p) return errors.QError.DimensionMismatch;

    switch (regopts.method) {
        ols.Method.auto => {
            try ols.choleskyDecomp(
                alloc,
                columns,
                y,
                out_coefs,
            );
        },
        ols.Method.cholesky => {
            try ols.choleskyDecomp(
                alloc,
                columns,
                y,
                out_coefs,
            );
        },

        ols.Method.gaussian_elimination => {
            try ols.gaussianElim(
                alloc,
                columns,
                y,
                out_coefs,
            );
        },
    }
}

test "OLSFit - auto recovers known coefficients" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    // y = 2*x1 + 3*x2, no noise: recover exactly [2, 3]
    var coefs: [2]f64 = undefined;

    const regopts = fixtures.ols_defaults;

    try olsFit(alloc, &fixtures.exact_2col.cols, &fixtures.exact_2col.y, &coefs, regopts);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), coefs[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), coefs[1], 1e-10);
}

test "OLSFit - cholesky recovers known coefficients" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    // y = 2*x1 + 3*x2, no noise: recover exactly [2, 3]
    var coefs: [2]f64 = undefined;

    var regopts = fixtures.ols_defaults;
    regopts.method = ols.Method.cholesky;

    try olsFit(alloc, &fixtures.exact_2col.cols, &fixtures.exact_2col.y, &coefs, regopts);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), coefs[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), coefs[1], 1e-10);
}

test "OLSFit - GE recovers known coefficients" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    // y = 2*x1 + 3*x2, no noise: recover exactly [2, 3]
    var coefs: [2]f64 = undefined;

    var regopts = fixtures.ols_defaults;
    regopts.method = ols.Method.gaussian_elimination;

    try olsFit(alloc, &fixtures.exact_2col.cols, &fixtures.exact_2col.y, &coefs, regopts);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), coefs[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), coefs[1], 1e-10);
}

pub fn elasticNetFit(
    alloc: std.mem.Allocator,
    columns: []const []const f64,
    y: []const f64,
    out_coefs: []f64,
    regopts: enet.Options,
) !usize {
    const p = columns.len;
    const n = y.len;

    //check shapes
    if (out_coefs.len != p or regopts.penalty_factors.len != p) return errors.QError.DimensionMismatch;
    for (0..p) |j| {
        if (columns[j].len != n) {
            return errors.QError.DimensionMismatch;
        }
    }

    const total_passes = enet.fit(alloc, columns, y, out_coefs, regopts);

    return total_passes;
}

test "elasticNet warm start converges faster" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const x1 = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const x2 = [_]f64{ 2, 1, 3, 2, 4, 3, 5, 4, 6, 5 };
    const y = [_]f64{ 7, 7, 15, 14, 22, 21, 29, 28, 36, 35 };

    const cols = [_][]const f64{ &x1, &x2 };
    var coefs = [_]f64{ 0.0, 0.0 };

    // enet_defaults (alpha=0.5, lambda=0.01) is exactly what this test wants.
    const regopts = fixtures.enet_defaults;

    const iter_cold = try elasticNetFit(alloc, &cols, &y, &coefs, regopts);

    var warm_seed = coefs;
    // reset coefs to prove warm start is being used
    coefs = [_]f64{ 0.0, 0.0 };

    var regopts_warm = regopts;
    regopts_warm.warm_start = &warm_seed;

    const iter_warm = try elasticNetFit(alloc, &cols, &y, &coefs, regopts_warm);

    try std.testing.expect(iter_warm <= iter_cold);
}

pub fn elasticNetPath(
    alloc: std.mem.Allocator,
    columns: []const []const f64,
    y: []const f64,
    out_coefs_matrix: []f64, // coef j, offset at lambda k = [j*n_lambda + k]
    out_lambdas: []f64,
    out_iters: []u64,
    regopts: enet.PathOptions,
) !usize {
    const p = columns.len;
    const n = y.len;
    const n_f: f64 = @floatFromInt(n);

    const lambda_min_ratio: f64 = regopts.lambda_min_ratio orelse
        if (n >= p) 1e-4 else 1e-2;

    if (lambda_min_ratio <= 0 or lambda_min_ratio > 1) {
        errors.setContext("elasticNetPath lambda_min_ratio must be between 0 and 1, {d}", .{lambda_min_ratio});
        return errors.QError.ParameterError;
    }

    var lambda_max: f64 = 0.0;

    std.debug.assert(regopts.alpha >= 0 and regopts.alpha <= 1);
    const alpha_safe = @max(regopts.alpha, 1e-3);

    if (regopts.n_lambda < 2) {
        errors.setContext("elasticNetPath requires n_lambda >= 2, received: {d}", .{regopts.n_lambda});
        return errors.QError.ParameterError;
    }

    for (0..p) |j| {
        if (regopts.penalty_factors[j] == 0.0) continue;

        const xty_j = @abs(dotProduct(columns[j], y)) / (n_f * alpha_safe * regopts.penalty_factors[j]);
        if (xty_j > lambda_max) lambda_max = xty_j;
    }
    if (lambda_max < 1e-10) return errors.QError.DegenerateData;

    const log_lambda_max = @log(lambda_max);
    const log_lambda_min = @log(lambda_max * lambda_min_ratio);
    for (0..regopts.n_lambda) |k| {
        const k_f: f64 = @floatFromInt(k);
        const n_l2: f64 = @floatFromInt(regopts.n_lambda - 1);
        out_lambdas[k] = @exp(log_lambda_max + k_f * (log_lambda_min - log_lambda_max) / n_l2);
    }

    const total_iters = try enet.path(alloc, columns, y, out_coefs_matrix, out_lambdas, out_iters, regopts);

    return total_iters;
}

// ============================================
// elasticNetPath Tests
// ============================================
//
test "elasticNetPath has positive n_iters" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const data = fixtures.sinCos(500).init();
    const cols = [_][]const f64{ &data.x1, &data.x2 };

    // path_defaults (alpha=1, n_lambda=20, lambda_min_ratio=1e-4, tol=1e-10) fits as-is.
    const regopts = fixtures.path_defaults;
    const n_lambda = regopts.n_lambda;

    var out_lambdas: [n_lambda]f64 = undefined;
    var out_coef_matrix: [2 * n_lambda]f64 = undefined;
    var out_iters = [_]u64{0} ** n_lambda;

    _ = try elasticNetPath(alloc, &cols, &data.y, &out_coef_matrix, &out_lambdas, &out_iters, regopts);

    // All fits should have a positive number if n_iters
    for (0..n_lambda) |k| {
        try std.testing.expect(out_iters[k] > 0);
    }
}
test "elasticNetPath produces decreasing lambda sequence" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const data = fixtures.sinCos(500).init();
    const cols = [_][]const f64{ &data.x1, &data.x2 };

    const regopts = fixtures.path_defaults;
    const n_lambda = regopts.n_lambda;

    var out_lambdas: [n_lambda]f64 = undefined;
    var out_coef_matrix: [2 * n_lambda]f64 = undefined;
    var out_iters: [n_lambda]u64 = undefined;

    _ = try elasticNetPath(alloc, &cols, &data.y, &out_coef_matrix, &out_lambdas, &out_iters, regopts);

    // Lambda sequence should be strictly decreasing
    for (0..n_lambda - 1) |k| {
        try std.testing.expect(out_lambdas[k] > out_lambdas[k + 1]);
    }

    // First lambda should produce all zeros (lambda_max)
    for (0..2) |j| {
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), out_coef_matrix[j * n_lambda + 0], 1e-10);
    }

    // Last lambda (smallest) should have nonzero coefficients
    var any_nonzero = false;
    for (0..2) |j| {
        if (@abs(out_coef_matrix[j * n_lambda + n_lambda - 1]) > 1e-6) {
            any_nonzero = true;
        }
    }
    try std.testing.expect(any_nonzero);
}

test "elasticNetPath warm starts reduce total iterations" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const p = 3;
    const n_lambda = 50;
    // x3 is irrelevant here (true coef 0): y = 2*x1 + 3*x2 + noise.
    const data = fixtures.sinCos(500).init();
    const cols = [_][]const f64{ &data.x1, &data.x2, &data.x3 };

    var regopts = fixtures.path_defaults;
    regopts.n_lambda = n_lambda;
    regopts.penalty_factors = &fixtures.pf_ones_3;
    regopts.lower_bounds = &fixtures.lb_open_3;
    regopts.upper_bounds = &fixtures.ub_open_3;

    var out_lambdas: [n_lambda]f64 = undefined;
    var out_coef_matrix: [p * n_lambda]f64 = undefined;
    var out_iters: [n_lambda]u64 = undefined;

    const path_iters = try elasticNetPath(alloc, &cols, &data.y, &out_coef_matrix, &out_lambdas, &out_iters, regopts);

    const avg_iters = @as(f64, @floatFromInt(path_iters)) / @as(f64, @floatFromInt(n_lambda));
    // std.debug.print("\n  Path avg iterations per lambda: {d:.2}\n", .{avg_iters});
    try std.testing.expect(avg_iters < 20.0);

    // Coefficients should grow as lambda decreases
    for (0..p) |j| {
        const at_max = @abs(out_coef_matrix[j * n_lambda + 0]);
        const at_min = @abs(out_coef_matrix[j * n_lambda + n_lambda - 1]);
        try std.testing.expect(at_min >= at_max);
    }
}

pub const IngestMode = enum { stream, materialize };

pub const IngestPlan = struct {
    mode: IngestMode,
    spec: StatsSpec,
};
pub const SolverOptions = union(enum) {
    ols: ols.Options,
    enet: enet.Options,
};

pub fn planIngest(solver: Solver, p: usize) IngestPlan {
    return switch (solver) {
        .ols => .{ .mode = .stream, .spec = .{ .gram = true, .xty = true } },
        .enet, .enet_path => if (p <= GRAM_P_THRESHOLD)
            .{ .mode = .stream, .spec = .{ .gram = true, .xty = true } }
        else
            .{ .mode = .materialize, .spec = .{ .gram = false, .xty = true } },
    };
}

/// Calls a solver based on the SolverOptions
/// All allocations are workspace only, all results are written to the callers
/// pre-allocated out_coefs that are passed in.
pub fn solveFromStats(
    alloc: std.mem.Allocator,
    columns: ?[]const []const f64,
    y: ?[]const f64,
    stats: SufficientStats,
    regopts: SolverOptions,
    out_coefs: []f64,
) !usize {
    if (out_coefs.len != stats.p) {
        errors.setContext("solveFromStats out_coef.len and stats.p disagree: {d}, {d}", .{ out_coefs.len, stats.p });
        return errors.QError.DimensionMismatch;
    }
    switch (regopts) {
        .ols => {
            try ols.choleskyDecompGram(
                alloc,
                stats,
                out_coefs,
            );
            return 1;
        },
        .enet => |opts| {
            if (stats.gram == null) {
                const cols = columns orelse return errors.QError.ParameterError;
                const y_p = y orelse return errors.QError.ParameterError;

                if (cols.len != stats.p) {
                    errors.setContext("solveFromStats columns.len and stats.p disagree: {d}, {d}", .{ cols.len, stats.p });
                    return errors.QError.DimensionMismatch;
                }
                for (cols) |col| {
                    if (col.len != y_p.len) {
                        errors.setContext("solveFromStats column length and y.len disagree: {d}, {d}", .{ col.len, y_p.len });
                        return errors.QError.DimensionMismatch;
                    }
                }

                if (y_p.len != stats.n) {
                    errors.setContext("solveFromStats y.len and stats.n disagree: {d}, {d}", .{ y_p.len, stats.n });
                    return errors.QError.DimensionMismatch;
                }

                return enet.fit(alloc, cols, y_p, out_coefs, opts);
            } else {
                return enet.fitGram(alloc, out_coefs, opts, stats);
            }
        },
    }
}

/// All allocations are workspace only, all results are written to the callers
/// pre-allocated out_coefs that are passed in.
pub fn solvePathFromStats(
    alloc: std.mem.Allocator,
    columns: ?[]const []const f64,
    y: ?[]const f64,
    out_coefs_matrix: []f64, // coef j, offset at lambda k = [j*n_lambda + k]
    out_lambdas: []f64,
    out_iters: []u64,
    regopts: enet.PathOptions,
    stats: SufficientStats,
) !usize {
    if (out_coefs_matrix.len != stats.p * regopts.n_lambda) {
        errors.setContext("solvePathFromStats out_coef_matrix.len and stats.p disagree: {d}, {d}", .{ out_coefs_matrix.len, stats.p });
        return errors.QError.DimensionMismatch;
    }

    const xty = stats.xty orelse {
        errors.setContext("solvePathFromStats requires xty, got null.", .{});
        return errors.QError.ParameterError;
    };

    const p = stats.p;
    const n = stats.n;

    const lambda_min_ratio: f64 = regopts.lambda_min_ratio orelse
        if (n >= p) 1e-4 else 1e-2;

    if (lambda_min_ratio <= 0 or lambda_min_ratio > 1) {
        errors.setContext("elasticNetPath lambda_min_ratio must be between 0 and 1, {d}", .{lambda_min_ratio});
        return errors.QError.ParameterError;
    }

    var lambda_max: f64 = 0.0;
    std.debug.assert(regopts.alpha >= 0 and regopts.alpha <= 1);
    const alpha_safe = @max(regopts.alpha, 1e-3);

    if (regopts.n_lambda < 2) {
        errors.setContext("solvePathFromStats requires n_lambda >= 2, received: {d}", .{regopts.n_lambda});
        return errors.QError.ParameterError;
    }

    for (0..p) |j| {
        if (regopts.penalty_factors[j] == 0.0) continue;

        const lambda_j = @abs(xty[j]) / (alpha_safe * regopts.penalty_factors[j]);

        if (lambda_j > lambda_max) lambda_max = lambda_j;
    }
    if (lambda_max < 1e-10) {
        errors.setContext("solvePathFromStats lambda_max < 1e-10 : {d}. Check if all(pf == 0)?", .{lambda_max});
        return errors.QError.DegenerateData;
    }

    const log_lambda_max = @log(lambda_max);
    const log_lambda_min = @log(lambda_max * lambda_min_ratio);
    for (0..regopts.n_lambda) |k| {
        const k_f: f64 = @floatFromInt(k);
        const n_f2: f64 = @floatFromInt(regopts.n_lambda - 1);
        out_lambdas[k] = @exp(log_lambda_max + k_f * (log_lambda_min - log_lambda_max) / n_f2);
    }

    if (stats.gram == null) {
        const cols = columns orelse {
            errors.setContext("solvePathFromStats requires columns if stats.gram == null", .{});
            return errors.QError.ParameterError;
        };

        if (cols.len != stats.p) {
            errors.setContext("solvePathFromStats columns.len and stats.p disagree: {d}, {d}", .{ cols.len, stats.p });
            return errors.QError.DimensionMismatch;
        }
        const y_p = y orelse {
            errors.setContext("solvePathFromStats requires  y if stats.gram == null", .{});
            return errors.QError.ParameterError;
        };

        if (y_p.len != stats.n) {
            errors.setContext("solvePathFromStats y.len and stats.n disagree: {d}, {d}", .{ y_p.len, stats.n });
            return errors.QError.DimensionMismatch;
        }

        for (cols) |col| {
            if (col.len != y_p.len) {
                errors.setContext("solvePathFromStats column length and y.len disagree: {d}, {d}", .{ col.len, y_p.len });
                return errors.QError.DimensionMismatch;
            }
        }
        return enet.path(alloc, cols, y_p, out_coefs_matrix, out_lambdas, out_iters, regopts);
    } else {
        return enet.pathGram(alloc, out_coefs_matrix, out_lambdas, out_iters, regopts, stats);
    }
}
test {
    std.testing.refAllDecls(@This());
}
