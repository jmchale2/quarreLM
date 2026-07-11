const std = @import("std");
const errors = @import("errors.zig");
const fixtures = @import("fixtures.zig");

const inf = std.math.inf(f64);
const clamp = std.math.clamp;

pub const EnetOptions = struct {
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

/// Gaussian Elimination OLS
pub fn olsFit(
    columns: []const []const f64,
    y: []const f64,
    out_coefs: []f64,
) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const p = columns.len;
    const n = y.len;

    for (columns) |col| {
        if (col.len != n) return errors.QError.DimensionMismatch;
    }
    if (out_coefs.len != p) return errors.QError.DimensionMismatch;

    // X'X (p x p symmetric matrix)
    const xtx = try alloc.alloc(f64, p * p);
    for (0..p) |i| {
        for (i..p) |j| {
            var dot: f64 = 0;
            for (0..n) |k| {
                dot += columns[i][k] * columns[j][k];
            }
            xtx[i * p + j] = dot;
            xtx[j * p + i] = dot;
        }
    }

    // X'y (p x 1 matrix)
    const xty = try alloc.alloc(f64, p);
    for (0..p) |i| {
        var dot: f64 = 0;
        for (0..n) |k| {
            dot += columns[i][k] * y[k];
        }
        xty[i] = dot;
    }

    // X'X*B = X'y
    // [X'x | X'y']
    const aug = try alloc.alloc(f64, p * (p + 1));
    for (0..p) |i| {
        for (0..p) |j| {
            aug[i * (p + 1) + j] = xtx[i * p + j];
        }

        aug[i * (p + 1) + p] = xty[i];
    }

    // forward elimination
    //partial pivoting
    for (0..p) |col| {
        // pivot point
        var max_val: f64 = @abs(aug[col * (p + 1) + col]);
        var max_row: usize = col;
        for (col + 1..p) |row| {
            const val = @abs(aug[row * (p + 1) + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }

        //swap
        if (max_row != col) {
            for (0..p + 1) |j| {
                const tmp = aug[col * (p + 1) + j];
                aug[col * (p + 1) + j] = aug[max_row * (p + 1) + j];
                aug[max_row * (p + 1) + j] = tmp;
            }
        }

        // Eliminate below
        const pivot = aug[col * (p + 1) + col];
        if (@abs(pivot) < 1e-12) return errors.QError.SingularMatrix;

        for (col + 1..p) |row| {
            const factor = aug[row * (p + 1) + col] / pivot;
            for (col..p + 1) |j| {
                aug[row * (p + 1) + j] -= factor * aug[col * (p + 1) + j];
            }
        }
    }

    // Back substitution
    var i: usize = p;
    while (i > 0) {
        i -= 1;
        var sum: f64 = aug[i * (p + 1) + p];
        for (i + 1..p) |j| {
            sum -= aug[i * (p + 1) + j] * out_coefs[j];
        }
        out_coefs[i] = sum / aug[i * (p + 1) + i];
    }
}

test "OLS recovers known coefficients" {
    // y = 2*x1 + 3*x2, no noise: recover exactly [2, 3]
    var coefs: [2]f64 = undefined;

    try olsFit(&fixtures.exact_2col.cols, &fixtures.exact_2col.y, &coefs);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), coefs[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), coefs[1], 1e-10);
}

fn axpy(dst: []f64, src: []const f64, scalar: f64) void {
    const vec_len = 4;
    const n = dst.len;
    const delta_vec: @Vector(vec_len, f64) = @splat(scalar);

    var i: usize = 0;
    while (i + vec_len <= n) : (i += vec_len) {
        const d: @Vector(vec_len, f64) = dst[i..][0..vec_len].*;
        const s: @Vector(vec_len, f64) = src[i..][0..vec_len].*;
        dst[i..][0..vec_len].* = d - s * delta_vec;
    }

    while (i < n) : (i += 1) {
        dst[i] -= src[i] * scalar;
    }
}

pub fn dotProduct(a: []const f64, b: []const f64) f64 {
    const vec_len: u32 = std.simd.suggestVectorLength(f64) orelse 4; // per-target lane count; fallback 4 (AVX2-ish)
    const n = a.len;

    // SIMD accumulator
    var acc: @Vector(vec_len, f64) = @splat(0.0);

    var i: usize = 0;
    while (i + vec_len <= n) : (i += vec_len) {
        const va: @Vector(vec_len, f64) = a[i..][0..vec_len].*;
        const vb: @Vector(vec_len, f64) = b[i..][0..vec_len].*;
        acc += va * vb;
    }

    var sum: f64 = @reduce(.Add, acc);

    // Scalar remainder
    while (i < n) : (i += 1) {
        sum += a[i] * b[i];
    }

    return sum;
}

test "dotProduct basic" {
    const a = [_]f64{ 1, 2, 3 };
    const b = [_]f64{ 4, 5, 6 };
    // 1*4 + 2*5 + 3*6 = 32
    try std.testing.expectApproxEqAbs(@as(f64, 32.0), dotProduct(&a, &b), 1e-10);
}

test "dotProduct exact vec_len (no remainder)" {
    const a = [_]f64{ 1, 2, 3, 4 };
    const b = [_]f64{ 5, 6, 7, 8 };
    // 5 + 12 + 21 + 32 = 70
    try std.testing.expectApproxEqAbs(@as(f64, 70.0), dotProduct(&a, &b), 1e-10);
}

test "dotProduct crosses vec boundary with remainder" {
    // 7 elements: one full SIMD pass (4) + 3 scalar remainder
    const a = [_]f64{ 1, 2, 3, 4, 5, 6, 7 };
    const b = [_]f64{ 7, 6, 5, 4, 3, 2, 1 };
    // 7 + 12 + 15 + 16 + 15 + 12 + 7 = 84
    try std.testing.expectApproxEqAbs(@as(f64, 84.0), dotProduct(&a, &b), 1e-10);
}

test "dotProduct single element" {
    const a = [_]f64{3.5};
    const b = [_]f64{2.0};
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), dotProduct(&a, &b), 1e-10);
}

test "dotProduct zeros" {
    const a = [_]f64{ 1, 2, 3, 4, 5 };
    const z = [_]f64{ 0, 0, 0, 0, 0 };
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), dotProduct(&a, &z), 1e-10);
}

test "dotProduct orthogonal" {
    const a = [_]f64{ 1, 0, 0, 0 };
    const b = [_]f64{ 0, 1, 0, 0 };
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), dotProduct(&a, &b), 1e-10);
}

test "dotProduct self is squared norm" {
    const a = [_]f64{ 3, 4 };
    // 9 + 16 = 25
    try std.testing.expectApproxEqAbs(@as(f64, 25.0), dotProduct(&a, &a), 1e-10);
}

test "dotProduct negative values" {
    const a = [_]f64{ -1, 2, -3, 4, -5 };
    const b = [_]f64{ 5, -4, 3, -2, 1 };
    // -5 + -8 + -9 + -8 + -5 = -35
    try std.testing.expectApproxEqAbs(@as(f64, -35.0), dotProduct(&a, &b), 1e-10);
}

pub fn olsFitVec(
    columns: []const []const f64,
    y: []const f64,
    out_coefs: []f64,
) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const p = columns.len;
    const n = y.len;

    for (columns) |col| {
        if (col.len != n) return errors.QError.DimensionMismatch;
    }
    if (out_coefs.len != p) return errors.QError.DimensionMismatch;

    // X'X (p x p symmetric matrix)
    const xtx = try alloc.alloc(f64, p * p);
    for (0..p) |i| {
        for (i..p) |j| {
            const dot = dotProduct(columns[i], columns[j]);
            xtx[i * p + j] = dot;
            xtx[j * p + i] = dot;
        }
    }

    // X'y (p x 1 matrix)
    const xty = try alloc.alloc(f64, p);
    for (0..p) |i| {
        const dot = dotProduct(columns[i], y);
        xty[i] = dot;
    }

    // X'X*B = X'y
    // [X'x | X'y']
    const aug = try alloc.alloc(f64, p * (p + 1));
    for (0..p) |i| {
        for (0..p) |j| {
            aug[i * (p + 1) + j] = xtx[i * p + j];
        }

        aug[i * (p + 1) + p] = xty[i];
    }

    // forward elimination
    //partial pivoting
    for (0..p) |col| {
        // pivot point
        var max_val: f64 = @abs(aug[col * (p + 1) + col]);
        var max_row: usize = col;
        for (col + 1..p) |row| {
            const val = @abs(aug[row * (p + 1) + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }

        //swap
        if (max_row != col) {
            for (0..p + 1) |j| {
                const tmp = aug[col * (p + 1) + j];
                aug[col * (p + 1) + j] = aug[max_row * (p + 1) + j];
                aug[max_row * (p + 1) + j] = tmp;
            }
        }

        // Eliminate below
        const pivot = aug[col * (p + 1) + col];
        if (@abs(pivot) < 1e-12) return errors.QError.SingularMatrix;

        for (col + 1..p) |row| {
            const factor = aug[row * (p + 1) + col] / pivot;
            for (col..p + 1) |j| {
                aug[row * (p + 1) + j] -= factor * aug[col * (p + 1) + j];
            }
        }
    }

    // Back substitution
    var i: usize = p;
    while (i > 0) {
        i -= 1;
        var sum: f64 = aug[i * (p + 1) + p];
        for (i + 1..p) |j| {
            sum -= aug[i * (p + 1) + j] * out_coefs[j];
        }
        out_coefs[i] = sum / aug[i * (p + 1) + i];
    }
}

test "OLSVec recovers known coefficients" {
    // y = 2*x1 + 3*x2, no noise: recover exactly [2, 3]
    var coefs: [2]f64 = undefined;

    try olsFitVec(&fixtures.exact_2col.cols, &fixtures.exact_2col.y, &coefs);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), coefs[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), coefs[1], 1e-10);
}

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

fn softThresholdVec(comptime N: usize, z: @Vector(N, f64), gamma: @Vector(N, f64)) @Vector(N, f64) {
    const zero: @Vector(N, f64) = @splat(0.0);
    const one: @Vector(N, f64) = @splat(1.0);
    const neg_one: @Vector(N, f64) = @splat(-1.0);
    const abs_z = @abs(z);
    const sign_z = @select(f64, z > zero, one, neg_one);
    const shrunk = abs_z - gamma;
    return @select(f64, shrunk > zero, sign_z * shrunk, zero);
}

pub fn elasticNetFit(
    alloc: std.mem.Allocator,
    columns: []const []const f64,
    y: []const f64,
    out_coefs: []f64,
    regopts: EnetOptions,
) !usize {
    const p = columns.len;
    const n = y.len;
    const n_f: f64 = @floatFromInt(n);

    //check shapes
    if (out_coefs.len != p or regopts.penalty_factors.len != p) return errors.QError.DimensionMismatch;
    for (0..p) |j| {
        if (columns[j].len != n) {
            return errors.QError.DimensionMismatch;
        }
    }

    if (regopts.warm_start) |w| {
        if (w.len != p) return errors.QError.DimensionMismatch;
        if (w.ptr != out_coefs.ptr) @memcpy(out_coefs, w);
    } else {
        @memset(out_coefs, 0);
    }

    //residual
    const r = try alloc.alloc(f64, n);
    @memcpy(r, y);
    for (0..p) |j| {
        if (out_coefs[j] != 0.0) {
            // subtract warm starts.
            axpy(r, columns[j], out_coefs[j]);
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

    const total_passes = elasticNetFitInner(columns, r, out_coefs, col_norms_squared, active, gram, xty, regopts);

    return total_passes;
}

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

    const n_iter = try elasticNetFit(alloc, &fixtures.exact_2col.cols, &fixtures.exact_2col.y, &coefs, regopts);

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

    _ = try elasticNetFit(alloc, &fixtures.sparse_2col.cols, &fixtures.sparse_2col.y, &coefs, regopts);

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

    _ = try elasticNetFit(alloc, &fixtures.collinear_2col.cols, &fixtures.collinear_2col.y, &coefs, regopts);

    try std.testing.expect(@abs(coefs[0]) > 0.01);
    try std.testing.expect(@abs(coefs[1]) > 0.01);
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

    _ = try elasticNetFit(alloc, &cols, &y, &coefs, regopts);

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

    _ = try elasticNetFit(alloc, &cols, &y, &coefs, regopts);

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
    _ = try elasticNetFit(alloc, &fixtures.sparse_2col.cols, &fixtures.sparse_2col.y, &coefs_penalized, regopts);

    // Now with penalty_factor=0 on x2 -> should be nonzero even at high lambda
    const pf_forced = [_]f64{ 1.0, 0.0 };
    var coefs_forced = [_]f64{ 0.0, 0.0 };

    var regopts_forced = regopts;
    regopts_forced.penalty_factors = &pf_forced;

    _ = try elasticNetFit(alloc, &fixtures.sparse_2col.cols, &fixtures.sparse_2col.y, &coefs_forced, regopts_forced);

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
    _ = try elasticNetFit(alloc, &fixtures.collinear_2col.cols, &fixtures.collinear_2col.y, &coefs_equal, regopts);

    // Heavy penalty on x2
    const pf_heavy = [_]f64{ 1.0, 5.0 };
    var coefs_heavy = [_]f64{ 0.0, 0.0 };

    var regopts_heavy = regopts;
    regopts_heavy.penalty_factors = &pf_heavy;

    _ = try elasticNetFit(alloc, &fixtures.collinear_2col.cols, &fixtures.collinear_2col.y, &coefs_heavy, regopts_heavy);

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

    _ = try elasticNetFit(alloc, &fixtures.collinear_2col.cols, &fixtures.collinear_2col.y, &coefs, regopts);

    try std.testing.expect(coefs[0] >= 0.5 - 1e-10);
    try std.testing.expect(coefs[1] <= 0.3 + 1e-10);
}

fn elasticNetFitInner(
    columns: []const []const f64,
    r: []f64, // residual — PERSISTENT, not re-created
    coefs: []f64, // warm-started coefficients
    col_norms_squared: []const f64, // precomputed
    active: []bool, // pre-allocated
    gram: ?[]const f64,
    xty: ?[]const f64,
    regopts: EnetOptions,
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
                        axpy(r, columns[j], delta);
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
                    axpy(r, columns[j], delta);
                }
                const change = col_norms_squared[j] * delta * delta;
                if (change > max_change) max_change = change;
            }
        }
        if (max_change < regopts.tol) break;
    }

    return total_passes;
}
pub fn elasticNetPath(
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

    var lambda_max: f64 = 0.0;
    const alpha_safe = @max(regopts.alpha, 1e-3);

    for (0..p) |j| {
        if (regopts.penalty_factors[j] == 0.0) continue;

        const xty_j = @abs(dotProduct(columns[j], y)) / (n_f * alpha_safe * regopts.penalty_factors[j]);
        if (xty_j > lambda_max) lambda_max = xty_j;
    }
    if (lambda_max < 1e-10) return errors.QError.DegenerateData;

    const log_lambda_max = @log(lambda_max);
    const log_lambda_min = @log(lambda_max * regopts.lambda_min_ratio);
    for (0..regopts.n_lambda) |k| {
        const k_f: f64 = @floatFromInt(k);
        const n_f2: f64 = @floatFromInt(regopts.n_lambda - 1);
        out_lambdas[k] = @exp(log_lambda_max + k_f * (log_lambda_min - log_lambda_max) / n_f2);
    }

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

    const use_gram = n > p and p < 300;
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

    const active = try alloc.alloc(bool, p);

    var total_iters: usize = 0;
    for (0..regopts.n_lambda) |k| {
        // Initialize active from current warm start
        for (0..p) |j| {
            active[j] = coefs[j] != 0.0;
        }

        const iters = elasticNetFitInner(columns, r, coefs, col_norms_squared, active, gram, xty, .{
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

/// Verify coefs satisfy the elastic-net KKT (optimality) conditions.
/// Objective: (1/2n)||y-Xb||² + λαΣpf|b| + (λ(1-α)/2)Σpf·b²
fn checkKKT(
    alloc: std.mem.Allocator,
    columns: []const []const f64,
    y: []const f64,
    coefs: []const f64,
    params: EnetOptions,
    kkt_tol: f64,
) !void {
    const p = columns.len;
    const n = y.len;
    const n_f: f64 = @floatFromInt(n);

    const r = try alloc.alloc(f64, n);
    defer alloc.free(r);
    @memcpy(r, y);
    for (0..p) |j| {
        if (coefs[j] != 0.0) axpy(r, columns[j], coefs[j]);
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
            _ = try elasticNetFit(alloc, &cols, &y, &coefs, params);
            try checkKKT(alloc, &cols, &y, &coefs, params, 1e-6);
        }
    }
}
