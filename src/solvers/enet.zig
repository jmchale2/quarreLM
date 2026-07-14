const std = @import("std");

const errors = @import("../errors.zig");
const fixtures = @import("../fixtures.zig");

const dotProduct = @import("common.zig").dotProduct;
const axpy = @import("common.zig").axpy;

const inf = std.math.inf(f64);
const clamp = std.math.clamp;

pub const Options = struct {
    lambda: f64,
    alpha: f64,
    penalty_factors: []const f64,
    lower_bounds: []const f64,
    upper_bounds: []const f64,
    warm_start: ?[]const f64 = null,
    max_iter: usize = 10_000,
    tol: f64 = 1e-7,
};

pub const PathOptions = struct {
    alpha: f64,
    penalty_factors: []const f64,
    lower_bounds: []const f64,
    upper_bounds: []const f64,
    n_lambda: usize,
    lambda_min_ratio: f64, // 1e-4 if n>=p, 1e-2 if n<p
    warm_start: ?[]const f64 = null,
    max_iter: usize = 10_000,
    tol: f64 = 1e-7,
};

fn softThreshold(z: f64, gamma: f64) f64 {
    if (z > gamma) return z - gamma;
    if (z < -gamma) return z + gamma;
    return 0.0;
}
test "softThreshold positive above gamma" {
    // z > gamma: return z - gamma
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), softThreshold(3.5, 2.0), 1e-10);
}

test "softThreshold negative below neg gamma" {
    // z < -gamma: return z + gamma
    try std.testing.expectApproxEqAbs(@as(f64, -1.5), softThreshold(-3.5, 2.0), 1e-10);
}

test "softThreshold within dead zone" {
    // |z| <= gamma: return 0
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), softThreshold(0.5, 2.0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), softThreshold(-0.5, 2.0), 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), softThreshold(0.0, 2.0), 1e-10);
}

fn fitInner(
    columns: []const []const f64,
    r: []f64, // residual — PERSISTENT, not re-created
    coefs: []f64, // warm-started coefficients
    col_norms_squared: []const f64, // precomputed
    active: []bool, // pre-allocated
    gram: ?[]const f64,
    xty: ?[]const f64,
    regopts: Options,
) usize {
    const p = columns.len;
    const n = r.len;
    const n_f: f64 = @floatFromInt(n);

    // Just the coordinate descent loops

    var total_passes: usize = 0;
    var any_new = true;

    while (any_new and total_passes < regopts.max_iter) {
        var max_change: f64 = 0.0;
        while (total_passes < regopts.max_iter) {
            total_passes += 1;
            max_change = 0.0;

            for (0..p) |j| {
                if (!active[j]) continue;

                const beta_old = coefs[j];

                const rho_j = blk: {
                    if (gram) |g| {
                        var xjr: f64 = xty.?[j];
                        for (0..p) |k| {
                            xjr -= g[j * p + k] * coefs[k];
                        }
                        break :blk xjr + col_norms_squared[j] * coefs[j];
                    } else {
                        break :blk dotProduct(columns[j], r) / n_f + col_norms_squared[j] * coefs[j];
                    }
                };

                const beta_new = softThreshold(rho_j, regopts.lambda * regopts.alpha * regopts.penalty_factors[j]) /
                    (col_norms_squared[j] + regopts.lambda * (1.0 - regopts.alpha) * regopts.penalty_factors[j]);
                const beta_clamped = clamp(beta_new, regopts.lower_bounds[j], regopts.upper_bounds[j]);

                if (beta_clamped == 0.0) active[j] = false;

                const delta = beta_clamped - beta_old;
                if (delta != 0.0) {
                    if (gram == null) {
                        axpy(-delta, columns[j], r);
                    }
                    const change = col_norms_squared[j] * delta * delta;
                    if (change > max_change) max_change = change;
                }
                coefs[j] = beta_clamped;
            }
            if (max_change < regopts.tol) break;
        }

        any_new = false;
        max_change = 0.0;
        for (0..p) |j| {
            if (active[j]) continue;
            const rho_j = blk: {
                if (gram) |g| {
                    var xjr: f64 = xty.?[j];
                    for (0..p) |k| {
                        xjr -= g[j * p + k] * coefs[k];
                    }
                    break :blk xjr + col_norms_squared[j] * coefs[j];
                } else {
                    break :blk dotProduct(columns[j], r) / n_f + col_norms_squared[j] * coefs[j];
                }
            };
            const beta_new = softThreshold(rho_j, regopts.lambda * regopts.alpha * regopts.penalty_factors[j]) /
                (col_norms_squared[j] + regopts.lambda * (1.0 - regopts.alpha) * regopts.penalty_factors[j]);
            const beta_clamped = clamp(beta_new, regopts.lower_bounds[j], regopts.upper_bounds[j]);

            if (beta_clamped != 0.0) {
                active[j] = true;
                any_new = true;
                coefs[j] = beta_clamped;
                const delta = beta_clamped;
                if (gram == null) {
                    axpy(-delta, columns[j], r);
                }
                const change = col_norms_squared[j] * delta * delta;
                if (change > max_change) max_change = change;
            }
        }
        if (max_change < regopts.tol) break;
    }

    return total_passes;
}

pub fn fit(
    alloc: std.mem.Allocator,
    columns: []const []const f64,
    y: []const f64,
    out_coefs: []f64,
    regopts: Options,
) !usize {
    const p = columns.len;
    const n = y.len;
    const n_f: f64 = @floatFromInt(n);

    //residual
    const r = try alloc.alloc(f64, n);
    @memcpy(r, y);
    for (0..p) |j| {
        if (out_coefs[j] != 0.0) {
            // subtract warm starts.
            axpy(-out_coefs[j], columns[j], r);
        }
    }

    // column squared norms
    const col_norms_squared = try alloc.alloc(f64, p);
    for (0..p) |j| {
        col_norms_squared[j] = dotProduct(columns[j], columns[j]) / n_f;
    }

    const active = try alloc.alloc(bool, p);
    for (0..p) |j| {
        active[j] = out_coefs[j] != 0.0;
    }

    const gram: ?[]f64 = null;
    const xty: ?[]f64 = null;

    const total_passes = fitInner(columns, r, out_coefs, col_norms_squared, active, gram, xty, regopts);

    return total_passes;
}

pub fn path(
    alloc: std.mem.Allocator,
    columns: []const []const f64,
    y: []const f64,
    //outputs, px n_lamba
    out_coefs_matrix: []f64, // coef j, offset at lambda k = [j*n_lambda + k]
    out_lambdas: []f64,
    out_iters: []u64,
    regopts: PathOptions,
) !usize {
    const p = columns.len;
    const n = y.len;
    const n_f: f64 = @floatFromInt(n);

    //path loop, warm starts
    const coefs = try alloc.alloc(f64, p);

    if (regopts.warm_start) |w| {
        if (w.len != p) return error.DimensionMismatch;
        @memcpy(coefs, w);
    } else {
        @memset(coefs, 0);
    }

    // Precompute
    const col_norms_squared = try alloc.alloc(f64, p);
    for (0..p) |j| {
        col_norms_squared[j] = dotProduct(columns[j], columns[j]) / n_f;
    }

    //tunable p breakpoint
    const p_breakpoint: usize = 300;
    const n_breakpoint: usize = p;
    const use_gram = n > n_breakpoint and p < p_breakpoint;
    var gram: ?[]f64 = null;
    var xty: ?[]f64 = null;

    if (use_gram) {
        gram = try alloc.alloc(f64, p * p);
        for (0..p) |i| {
            for (i..p) |j| {
                const dot = dotProduct(columns[i], columns[j]);
                gram.?[i * p + j] = dot / n_f;
                gram.?[j * p + i] = dot / n_f;
            }
        }

        xty = try alloc.alloc(f64, p);
        for (0..p) |j| {
            xty.?[j] = dotProduct(columns[j], y) / n_f;
        }
    }

    // Persistent residual
    const r = try alloc.alloc(f64, n);
    @memcpy(r, y);

    for (0..p) |j| {
        if (coefs[j] != 0.0) {
            // subtract warm startsa
            axpy(-coefs[j], columns[j], r);
        }
    }

    const active = try alloc.alloc(bool, p);

    var total_iters: usize = 0;
    for (0..regopts.n_lambda) |k| {
        // Initialize active from current warm start
        for (0..p) |j| {
            active[j] = coefs[j] != 0.0;
        }

        const iters = fitInner(columns, r, coefs, col_norms_squared, active, gram, xty, .{
            .lambda = out_lambdas[k],
            .alpha = regopts.alpha,
            .penalty_factors = regopts.penalty_factors,
            .lower_bounds = regopts.lower_bounds,
            .upper_bounds = regopts.upper_bounds,
            .max_iter = regopts.max_iter,
            .tol = regopts.tol,
        });
        out_iters[k] = iters;

        total_iters += iters;

        for (0..p) |j| {
            out_coefs_matrix[j * regopts.n_lambda + k] = coefs[j];
        }
    }
    return total_iters;
}

// fit tests
//

// ============================================
// elasticNet Tests
// ============================================
// High level, broad correctness tests, NOT specific proofs of correctness
// More smoke tests to make sure the wheels didn't fall off than anything else
test "elasticNet recovers known coefficients" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var coefs = [_]f64{ 0.0, 0.0 };

    var regopts = fixtures.enet_defaults;
    regopts.alpha = 0.0;
    regopts.lambda = 0.0;

    const n_iter = try fit(alloc, &fixtures.exact_2col.cols, &fixtures.exact_2col.y, &coefs, regopts);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), coefs[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), coefs[1], 1e-4);
    try std.testing.expect(n_iter <= regopts.max_iter);
}

test "elasticNet lasso zeros out irrelevant features" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var coefs = [_]f64{ 0.0, 0.0 };

    var regopts = fixtures.enet_defaults;
    regopts.alpha = 1.0;
    regopts.lambda = 0.5;

    _ = try fit(alloc, &fixtures.sparse_2col.cols, &fixtures.sparse_2col.y, &coefs, regopts);

    try std.testing.expect(@abs(coefs[0]) > 1.0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), coefs[1], 0.1);
}

test "elasticNet ridge shrinks but keeps all nonzero" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var coefs = [_]f64{ 0.0, 0.0 };

    var regopts = fixtures.enet_defaults;
    regopts.alpha = 0.001;
    regopts.lambda = 0.5;

    _ = try fit(alloc, &fixtures.collinear_2col.cols, &fixtures.collinear_2col.y, &coefs, regopts);

    try std.testing.expect(@abs(coefs[0]) > 0.01);
    try std.testing.expect(@abs(coefs[1]) > 0.01);
}
// Constraint-specific tests
test "elasticNet lower bound prevents negative coefficients" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    // y = 2*x1 - 3*x2
    // constrain both coefficients >= 0
    const x1 = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const x2 = [_]f64{ 2, 1, 3, 2, 4, 3, 5, 4, 6, 5 };
    var y: [10]f64 = undefined;
    for (0..10) |i| {
        y[i] = 2.0 * x1[i] - 3.0 * x2[i];
    }

    const cols = [_][]const f64{ &x1, &x2 };
    var coefs = [_]f64{ 0.0, 0.0 };
    const lb = [_]f64{ 0.0, 0.0 };

    var regopts = fixtures.enet_defaults;
    regopts.alpha = 0.0;
    regopts.lambda = 0.0;
    regopts.lower_bounds = &lb;

    _ = try fit(alloc, &cols, &y, &coefs, regopts);

    try std.testing.expect(coefs[0] >= 0.0);
    try std.testing.expect(coefs[1] >= 0.0);
}

test "elasticNet upper bound caps coefficients" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    // y = 10*x1
    // x1<=2.0
    const x1 = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var y: [10]f64 = undefined;
    for (0..10) |i| {
        y[i] = 10.0 * x1[i];
    }

    const cols = [_][]const f64{&x1};
    var coefs = [_]f64{0.0};
    const ub = [_]f64{2.0};

    var regopts = fixtures.enet_defaults;
    regopts.alpha = 0.0;
    regopts.lambda = 0.0;
    regopts.penalty_factors = &fixtures.pf_ones_1;
    regopts.lower_bounds = &fixtures.lb_open_1;
    regopts.upper_bounds = &ub;

    _ = try fit(alloc, &cols, &y, &coefs, regopts);

    // Should be clamped at 2.0
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), coefs[0], 1e-10);
}

test "elasticNet penalty factor zero forces variable in" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    // With penalty_factor=0, variable is unpenalized (always enters)
    var regopts = fixtures.enet_defaults;
    regopts.alpha = 1.0;
    regopts.lambda = 1.0;

    // High lambda with equal penalty ->  x2 should be zeroed
    var coefs_penalized = [_]f64{ 0.0, 0.0 };
    _ = try fit(alloc, &fixtures.sparse_2col.cols, &fixtures.sparse_2col.y, &coefs_penalized, regopts);

    // Now with penalty_factor=0 on x2 -> should be nonzero even at high lambda
    const pf_forced = [_]f64{ 1.0, 0.0 };
    var coefs_forced = [_]f64{ 0.0, 0.0 };

    var regopts_forced = regopts;
    regopts_forced.penalty_factors = &pf_forced;

    _ = try fit(alloc, &fixtures.sparse_2col.cols, &fixtures.sparse_2col.y, &coefs_forced, regopts_forced);

    // x2 was zero with penalty, should be nonzero without
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), coefs_penalized[1], 0.01);
    try std.testing.expect(@abs(coefs_forced[1]) > @abs(coefs_penalized[1]));
}

test "elasticNet high penalty factor increases shrinkage" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var regopts = fixtures.enet_defaults;
    regopts.alpha = 1.0;
    regopts.lambda = 0.1;

    // Equal penalty (enet_defaults.penalty_factors is all ones)
    var coefs_equal = [_]f64{ 0.0, 0.0 };
    _ = try fit(alloc, &fixtures.collinear_2col.cols, &fixtures.collinear_2col.y, &coefs_equal, regopts);

    // Heavy penalty on x2
    const pf_heavy = [_]f64{ 1.0, 5.0 };
    var coefs_heavy = [_]f64{ 0.0, 0.0 };

    var regopts_heavy = regopts;
    regopts_heavy.penalty_factors = &pf_heavy;

    _ = try fit(alloc, &fixtures.collinear_2col.cols, &fixtures.collinear_2col.y, &coefs_heavy, regopts_heavy);

    // x2 should be more shrunk with higher penalty factor
    try std.testing.expect(@abs(coefs_heavy[1]) < @abs(coefs_equal[1]));
}

test "elasticNet box constraints with regularization" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    // Combine bounds (alpha=0.5 lasso/ridge mix) with a box constraint.
    var coefs = [_]f64{ 0.0, 0.0 };
    const lb = [_]f64{ 0.5, -inf }; // x1 >= 0.5
    const ub = [_]f64{ inf, 0.3 }; // x2 <= 0.3

    var regopts = fixtures.enet_defaults; // alpha = 0.5
    regopts.lambda = 0.1;
    regopts.lower_bounds = &lb;
    regopts.upper_bounds = &ub;

    _ = try fit(alloc, &fixtures.collinear_2col.cols, &fixtures.collinear_2col.y, &coefs, regopts);

    try std.testing.expect(coefs[0] >= 0.5 - 1e-10);
    try std.testing.expect(coefs[1] <= 0.3 + 1e-10);
}
/// Verify coefs satisfy the elastic-net KKT (optimality) conditions.
/// Objective: (1/2n)||y-Xb||² + λαΣpf|b| + (λ(1-α)/2)Σpf·b²
fn checkKKT(
    alloc: std.mem.Allocator,
    columns: []const []const f64,
    y: []const f64,
    coefs: []const f64,
    params: Options,
    kkt_tol: f64,
) !void {
    const p = columns.len;
    const n = y.len;
    const n_f: f64 = @floatFromInt(n);

    const r = try alloc.alloc(f64, n);
    defer alloc.free(r);
    @memcpy(r, y);
    for (0..p) |j| {
        if (coefs[j] != 0.0) axpy(-coefs[j], columns[j], r);
    }

    for (0..p) |j| {
        const grad = dotProduct(columns[j], r) / n_f; // X_j'r/n
        const l1 = params.lambda * params.alpha * params.penalty_factors[j];
        const l2 = params.lambda * (1.0 - params.alpha) * params.penalty_factors[j];
        const b = coefs[j];
        const at_lb = b <= params.lower_bounds[j] + 1e-12;
        const at_ub = b >= params.upper_bounds[j] - 1e-12;

        if (b == 0.0) {
            // zero coord: |X_j'r/n| must be under the L1 threshold
            try std.testing.expect(@abs(grad) <= l1 + kkt_tol);
        } else if (at_ub) {
            // pinned at upper bound: gradient may push further up, never down
            try std.testing.expect(grad - l2 * b - l1 * std.math.sign(b) >= -kkt_tol);
        } else if (at_lb) {
            try std.testing.expect(grad - l2 * b - l1 * std.math.sign(b) <= kkt_tol);
        } else {
            // interior nonzero: stationarity, exactly
            try std.testing.expectApproxEqAbs(l1 * std.math.sign(b), grad - l2 * b, kkt_tol);
        }
    }
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

    _ = try path(alloc, &cols, &data.y, &out_coef_matrix, &out_lambdas, &out_iters, regopts);

    // All fits should have a positive number if n_iters
    for (0..n_lambda) |k| {
        try std.testing.expect(out_iters[k] > 0);
    }
}

test "elasticNet solutions satisfy KKT across lambda/alpha grid" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const n = 200;
    const p = 4;
    var xs: [p][n]f64 = undefined;
    var y: [n]f64 = undefined;
    for (0..n) |i| {
        const t: f64 = @floatFromInt(i);
        xs[0][i] = @sin(t * 0.13) * 2.0;
        xs[1][i] = @cos(t * 0.31) + @sin(t * 0.05);
        xs[2][i] = @sin(t * 0.71) * 0.5 + @cos(t * 0.11);
        xs[3][i] = @cos(t * 0.97) * 1.5;
        y[i] = 2.0 * xs[0][i] - 1.0 * xs[1][i] + 0.3 * xs[3][i] + @sin(t * 3.1) * 0.05;
    }
    const cols = [_][]const f64{ &xs[0], &xs[1], &xs[2], &xs[3] };

    var base = fixtures.enet_defaults;
    base.penalty_factors = &fixtures.pf_ones_4;
    base.lower_bounds = &fixtures.lb_open_4;
    base.upper_bounds = &fixtures.ub_open_4;
    base.tol = 1e-14;
    base.max_iter = 100_000;

    const lambdas = [_]f64{ 0.5, 0.1, 0.01 };
    const alphas = [_]f64{ 1.0, 0.5, 0.05 };
    for (lambdas) |lam| {
        for (alphas) |a| {
            var coefs = [_]f64{ 0, 0, 0, 0 };
            var params = base;
            params.lambda = lam;
            params.alpha = a;
            _ = try fit(alloc, &cols, &y, &coefs, params);
            try checkKKT(alloc, &cols, &y, &coefs, params, 1e-6);
        }
    }
}
test "elasticNetPath warm start does not change solutions (naive branch)" {
    // Pins the warm-start residual contract: elasticNetPath must subtract the
    // warm start from r before entering the naive branch, which assumes
    // r = y - X*beta throughout. n <= p disables the gram branch
    // (use_gram = n > p and ...), so this test exercises exactly the branch
    // the gram path masks.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const n = 20;
    const p = 20;
    var xs: [p][n]f64 = undefined;
    var y: [n]f64 = undefined;
    for (0..n) |i| {
        const t: f64 = @floatFromInt(i + 1);
        for (0..p) |j| {
            const u: f64 = @floatFromInt(j + 1);
            xs[j][i] = @sin(t * u * 0.17) + 0.3 * @cos(t * (u + 2.0) * 0.09);
        }
        y[i] = 1.5 * xs[0][i] - 2.0 * xs[1][i] + 0.7 * xs[2][i] + @sin(t * 2.3) * 0.05;
    }
    var cols: [p][]const f64 = undefined;
    for (0..p) |j| cols[j] = &xs[j];

    const pf = [_]f64{1.0} ** p;
    const lb = [_]f64{-inf} ** p;
    const ub = [_]f64{inf} ** p;

    const n_lambda = 8;
    var regopts = fixtures.path_defaults;
    regopts.alpha = 0.5; // strictly convex: unique solution to compare against
    regopts.n_lambda = n_lambda;
    regopts.lambda_min_ratio = 1e-3;
    regopts.penalty_factors = &pf;
    regopts.lower_bounds = &lb;
    regopts.upper_bounds = &ub;
    regopts.tol = 1e-12;
    regopts.max_iter = 100_000;

    var lambdas_cold: [n_lambda]f64 = undefined;
    var coefs_cold: [p * n_lambda]f64 = undefined;
    var iters_cold: [n_lambda]u64 = undefined;
    _ = try path(alloc, &cols, &y, &coefs_cold, &lambdas_cold, &iters_cold, regopts);

    // Seed the warm run with the densest (smallest-lambda) cold solution.
    var seed: [p]f64 = undefined;
    for (0..p) |j| seed[j] = coefs_cold[j * n_lambda + n_lambda - 1];
    regopts.warm_start = &seed;

    var lambdas_warm: [n_lambda]f64 = undefined;
    var coefs_warm: [p * n_lambda]f64 = undefined;
    var iters_warm: [n_lambda]u64 = undefined;
    _ = try path(alloc, &cols, &y, &coefs_warm, &lambdas_warm, &iters_warm, regopts);

    // Warm starts change iteration counts, never solutions.
    for (0..n_lambda) |k| {
        try std.testing.expectEqual(lambdas_cold[k], lambdas_warm[k]);
        for (0..p) |j| {
            try std.testing.expectApproxEqAbs(coefs_cold[j * n_lambda + k], coefs_warm[j * n_lambda + k], 1e-5);
        }
    }

    // Self-contained check (does not trust the cold run): every warm-run
    // solution must satisfy KKT at its own lambda.
    for (0..n_lambda) |k| {
        var beta_k: [p]f64 = undefined;
        for (0..p) |j| beta_k[j] = coefs_warm[j * n_lambda + k];
        try checkKKT(alloc, &cols, &y, &beta_k, .{
            .lambda = lambdas_warm[k],
            .alpha = regopts.alpha,
            .penalty_factors = &pf,
            .lower_bounds = &lb,
            .upper_bounds = &ub,
            .tol = regopts.tol,
            .max_iter = regopts.max_iter,
        }, 1e-5);
    }
}
test "elasticNetFitInner gram and naive branches agree from a warm start" {
    // Both rho_j formulations solve the same problem; from an identical
    // (nonzero) starting point they must land on the same solution. Guards
    // the covariance-update/BLAS work against branch drift.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const n = 60;
    const p = 4;
    const n_f: f64 = @floatFromInt(n);
    var xs: [p][n]f64 = undefined;
    var y: [n]f64 = undefined;
    for (0..n) |i| {
        const t: f64 = @floatFromInt(i);
        xs[0][i] = @sin(t * 0.13) * 2.0;
        xs[1][i] = @cos(t * 0.31) + @sin(t * 0.05);
        xs[2][i] = @sin(t * 0.71) * 0.5 + @cos(t * 0.11);
        xs[3][i] = @cos(t * 0.97) * 1.5;
        y[i] = 1.5 * xs[0][i] - 2.0 * xs[1][i] + 0.7 * xs[3][i] + @sin(t * 2.9) * 0.05;
    }
    const cols = [_][]const f64{ &xs[0], &xs[1], &xs[2], &xs[3] };

    const w = [_]f64{ 0.5, -0.25, 0.0, 1.0 };

    var col_norms_squared: [p]f64 = undefined;
    for (0..p) |j| col_norms_squared[j] = dotProduct(cols[j], cols[j]) / n_f;

    var opts = fixtures.enet_defaults;
    opts.lambda = 0.05;
    opts.alpha = 0.7;
    opts.penalty_factors = &fixtures.pf_ones_4;
    opts.lower_bounds = &fixtures.lb_open_4;
    opts.upper_bounds = &fixtures.ub_open_4;
    opts.tol = 1e-12;
    opts.max_iter = 100_000;

    // --- naive branch: needs a residual consistent with the warm start ---
    var r_naive: [n]f64 = undefined;
    @memcpy(&r_naive, &y);
    for (0..p) |j| {
        if (w[j] != 0.0) axpy(-w[j], cols[j], &r_naive);
    }
    var coefs_naive = w;
    var active_naive: [p]bool = undefined;
    for (0..p) |j| active_naive[j] = w[j] != 0.0;
    _ = fitInner(&cols, &r_naive, &coefs_naive, &col_norms_squared, &active_naive, null, null, opts);

    // --- gram branch: same start, residual is unused (length only) ---
    var gram: [p * p]f64 = undefined;
    var xty_: [p]f64 = undefined;
    for (0..p) |i| {
        for (i..p) |j| {
            const dot = dotProduct(cols[i], cols[j]) / n_f;
            gram[i * p + j] = dot;
            gram[j * p + i] = dot;
        }
        xty_[i] = dotProduct(cols[i], &y) / n_f;
    }
    var r_gram: [n]f64 = undefined;
    @memcpy(&r_gram, &y);
    var coefs_gram = w;
    var active_gram: [p]bool = undefined;
    for (0..p) |j| active_gram[j] = w[j] != 0.0;
    _ = fitInner(&cols, &r_gram, &coefs_gram, &col_norms_squared, &active_gram, &gram, &xty_, opts);

    for (0..p) |j| {
        try std.testing.expectApproxEqAbs(coefs_naive[j], coefs_gram[j], 1e-8);
    }
    try checkKKT(alloc, &cols, &y, &coefs_naive, opts, 1e-6);
    try checkKKT(alloc, &cols, &y, &coefs_gram, opts, 1e-6);
}
