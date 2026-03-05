const std = @import("std");

const inf = std.math.inf(f64);
const clamp = std.math.clamp;

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
        if (col.len != n) return error.DimensionMismatch;
    }
    if (out_coefs.len != p) return error.DimensionMismatch;

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
        if (@abs(pivot) < 1e-12) return error.SingularMatrix;

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
    // y = 2*x1 + 3*x2 + noise(0)
    // With no noise, should recover exactly [2, 3]
    const x1 = [_]f64{ 1, 2, 3, 4, 5 };
    const x2 = [_]f64{ 2, 1, 3, 2, 4 };
    const y = [_]f64{
        2 * 1 + 3 * 2, // 8
        2 * 2 + 3 * 1, // 7
        2 * 3 + 3 * 3, // 15
        2 * 4 + 3 * 2, // 14
        2 * 5 + 3 * 4, // 22
    };

    const cols = [_][]const f64{ &x1, &x2 };
    var coefs: [2]f64 = undefined;

    try olsFit(&cols, &y, &coefs);

    // std.debug.print("{any}", .{coefs});

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

fn dotProduct(a: []const f64, b: []const f64) f64 {
    const vec_len = 4; // AVX2: 4 × f64 = 256 bits
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
        if (col.len != n) return error.DimensionMismatch;
    }
    if (out_coefs.len != p) return error.DimensionMismatch;

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
        if (@abs(pivot) < 1e-12) return error.SingularMatrix;

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
    // y = 2*x1 + 3*x2 + noise(0)
    // With no noise, should recover exactly [2, 3]
    const x1 = [_]f64{ 1, 2, 3, 4, 5 };
    const x2 = [_]f64{ 2, 1, 3, 2, 4 };
    const y = [_]f64{
        2 * 1 + 3 * 2, // 8
        2 * 2 + 3 * 1, // 7
        2 * 3 + 3 * 3, // 15
        2 * 4 + 3 * 2, // 14
        2 * 5 + 3 * 4, // 22
    };

    const cols = [_][]const f64{ &x1, &x2 };
    var coefs: [2]f64 = undefined;

    try olsFitVec(&cols, &y, &coefs);

    // std.debug.print("{any}\n", .{coefs});

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
    lambda: f64,
    alpha: f64,
    penalty_factors: []const f64,
    lower_bounds: []const f64,
    upper_bounds: []const f64,
    out_coefs: []f64,
    max_iter: usize,
    tol: f64,
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
            for (0..n) |i| {
                r[i] -= columns[j][i] * out_coefs[j];
            }
        }
    }

    // column squared norms
    const col_norms_squared = try alloc.alloc(f64, p);
    for (0..p) |j| {
        col_norms_squared[j] = dotProduct(columns[j], columns[j]) / n_f;
    }

    const active = try alloc.alloc(bool, p);
    @memset(active, false);
    for (0..p) |j| {
        active[j] = out_coefs[j] != 0.0;
    }

    var total_passes: usize = 0;
    var any_new = true;

    while (any_new and total_passes < max_iter) {
        var max_change: f64 = 0.0;
        while (total_passes < max_iter) {
            total_passes += 1;
            max_change = 0.0;

            for (0..p) |j| {
                if (!active[j]) continue;

                const beta_old = out_coefs[j];

                // rho_j = (1/N) * X_j^T r + beta_j (partial residuals)
                const rho_j = dotProduct(columns[j], r) / n_f + col_norms_squared[j] * beta_old;

                // soft threshold
                const beta_new = softThreshold(rho_j, lambda * alpha * penalty_factors[j]) /
                    (col_norms_squared[j] + lambda * (1.0 - alpha) * penalty_factors[j]);
                const beta_clamped = clamp(beta_new, lower_bounds[j], upper_bounds[j]);

                if (beta_clamped == 0.0) {
                    active[j] = false;
                }

                const delta = beta_clamped - beta_old;
                if (delta != 0.0) {
                    //Update residuals
                    // r -= X_j * delta
                    for (0..n) |i| {
                        r[i] -= columns[j][i] * delta;
                    }

                    const change = col_norms_squared[j] * delta * delta;
                    if (change > max_change) max_change = change;
                }

                out_coefs[j] = beta_clamped;
            }

            if (max_change < tol) break;
        }

        // check for new candidates
        any_new = false;
        // total_passes += 1;
        max_change = 0.0;
        for (0..p) |j| {
            if (active[j]) continue;

            const rho_j = dotProduct(columns[j], r) / n_f;

            // soft threshold
            const beta_new = softThreshold(rho_j, lambda * alpha * penalty_factors[j]);
            const beta_clamped = std.math.clamp(beta_new, lower_bounds[j], upper_bounds[j]);

            if (beta_clamped != 0.0) {
                active[j] = true;
                any_new = true;
                // Actually perform the update
                out_coefs[j] = beta_clamped / (col_norms_squared[j] + lambda * (1.0 - alpha));
                const delta = out_coefs[j];
                if (delta != 0.0) {
                    //Update residuals
                    // r -= X_j * delta
                    for (0..n) |i| {
                        r[i] -= columns[j][i] * delta;
                    }

                    const change = col_norms_squared[j] * delta * delta;
                    if (change > max_change) max_change = change;
                }
            }
        }

        if (max_change < tol) break;
    }

    return total_passes;
}

// ============================================
// elasticNet Tests
// ============================================
test "elasticNet recovers known coefficients" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const x1 = [_]f64{ 1, 2, 3, 4, 5 };
    const x2 = [_]f64{ 2, 1, 3, 2, 4 };
    const y = [_]f64{ 8, 7, 15, 14, 22 }; // y = 2*x1 + 3*x2

    const cols = [_][]const f64{ &x1, &x2 };
    var coefs = [_]f64{ 0.0, 0.0 };
    const pf = [_]f64{ 1.0, 1.0 };
    const lb = [_]f64{ -inf, -inf };
    const ub = [_]f64{ inf, inf };

    const n_iter = try elasticNetFit(alloc, &cols, &y, 0.0, 0.0, &pf, &lb, &ub, &coefs, 10000, 1e-10);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), coefs[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), coefs[1], 1e-4);
    try std.testing.expect(n_iter <= 10000);
}

test "elasticNet lasso zeros out irrelevant features" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const x1 = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const x2 = [_]f64{ 0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1 };
    const y = [_]f64{ 3, 6, 9, 12, 15, 18, 21, 24, 27, 30 };

    const cols = [_][]const f64{ &x1, &x2 };
    var coefs = [_]f64{ 0.0, 0.0 };
    const pf = [_]f64{ 1.0, 1.0 };
    const lb = [_]f64{ -inf, -inf };
    const ub = [_]f64{ inf, inf };

    _ = try elasticNetFit(alloc, &cols, &y, 0.5, 1.0, &pf, &lb, &ub, &coefs, 10000, 1e-10);

    try std.testing.expect(@abs(coefs[0]) > 1.0);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), coefs[1], 0.1);
}

test "elasticNet ridge shrinks but keeps all nonzero" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const x1 = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const x2 = [_]f64{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    const y = [_]f64{ 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 };

    const cols = [_][]const f64{ &x1, &x2 };
    var coefs = [_]f64{ 0.0, 0.0 };
    const pf = [_]f64{ 1.0, 1.0 };
    const lb = [_]f64{ -inf, -inf };
    const ub = [_]f64{ inf, inf };

    _ = try elasticNetFit(alloc, &cols, &y, 0.5, 0.001, &pf, &lb, &ub, &coefs, 10000, 1e-10);

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
    const pf = [_]f64{ 1.0, 1.0 };
    const lb = [_]f64{ -inf, -inf };
    const ub = [_]f64{ inf, inf };

    var coefs_cold = [_]f64{ 0.0, 0.0 };
    const iter_cold = try elasticNetFit(alloc, &cols, &y, 0.01, 0.5, &pf, &lb, &ub, &coefs_cold, 10000, 1e-10);

    var coefs_warm = coefs_cold;
    const iter_warm = try elasticNetFit(alloc, &cols, &y, 0.01, 0.5, &pf, &lb, &ub, &coefs_warm, 10000, 1e-10);

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
    const pf = [_]f64{ 1.0, 1.0 };
    const lb = [_]f64{ 0.0, 0.0 };
    const ub = [_]f64{ inf, inf };

    _ = try elasticNetFit(alloc, &cols, &y, 0.0, 0.0, &pf, &lb, &ub, &coefs, 10000, 1e-10);

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
    const pf = [_]f64{1.0};
    const lb = [_]f64{-inf};
    const ub = [_]f64{2.0};

    _ = try elasticNetFit(alloc, &cols, &y, 0.0, 0.0, &pf, &lb, &ub, &coefs, 10000, 1e-10);

    // Should be clamped at 2.0
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), coefs[0], 1e-10);
}

test "elasticNet penalty factor zero forces variable in" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    // With penalty_factor=0, variable is unpenalized (always enters)
    const x1 = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const x2 = [_]f64{ 0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1 };
    const y = [_]f64{ 3, 6, 9, 12, 15, 18, 21, 24, 27, 30 };

    const cols = [_][]const f64{ &x1, &x2 };
    const pf = [_]f64{ 1.0, 1.0 };
    const lb = [_]f64{ -inf, -inf };
    const ub = [_]f64{ inf, inf };

    // High lambda with equal penalty ->  x2 should be zeroed
    var coefs_penalized = [_]f64{ 0.0, 0.0 };
    _ = try elasticNetFit(alloc, &cols, &y, 1.0, 1.0, &pf, &lb, &ub, &coefs_penalized, 10000, 1e-10);

    // Now with penalty_factor=0 on x2 -> should be nonzero even at high lambda
    const pf_forced = [_]f64{ 1.0, 0.0 };
    var coefs_forced = [_]f64{ 0.0, 0.0 };
    _ = try elasticNetFit(alloc, &cols, &y, 1.0, 1.0, &pf_forced, &lb, &ub, &coefs_forced, 10000, 1e-10);

    // x2 was zero with penalty, should be nonzero without
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), coefs_penalized[1], 0.01);
    try std.testing.expect(@abs(coefs_forced[1]) > @abs(coefs_penalized[1]));
}

test "elasticNet high penalty factor increases shrinkage" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const x1 = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const x2 = [_]f64{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    // y = 1*x1 + 1*x2
    const y = [_]f64{ 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 };

    const cols = [_][]const f64{ &x1, &x2 };
    const lb = [_]f64{ -inf, -inf };
    const ub = [_]f64{ inf, inf };

    // Equal penalty
    const pf_equal = [_]f64{ 1.0, 1.0 };
    var coefs_equal = [_]f64{ 0.0, 0.0 };
    _ = try elasticNetFit(alloc, &cols, &y, 0.1, 1.0, &pf_equal, &lb, &ub, &coefs_equal, 10000, 1e-10);

    // Heavy penalty on x2
    const pf_heavy = [_]f64{ 1.0, 5.0 };
    var coefs_heavy = [_]f64{ 0.0, 0.0 };
    _ = try elasticNetFit(alloc, &cols, &y, 0.1, 1.0, &pf_heavy, &lb, &ub, &coefs_heavy, 10000, 1e-10);

    // x2 should be more shrunk with higher penalty factor
    try std.testing.expect(@abs(coefs_heavy[1]) < @abs(coefs_equal[1]));
}

test "elasticNet box constraints with regularization" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    // Combine bounds with lasso penalty
    const x1 = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const x2 = [_]f64{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    const y = [_]f64{ 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 };

    const cols = [_][]const f64{ &x1, &x2 };
    var coefs = [_]f64{ 0.0, 0.0 };
    const pf = [_]f64{ 1.0, 1.0 };
    const lb = [_]f64{ 0.5, -inf }; // x1 >= 0.5
    const ub = [_]f64{ inf, 0.3 }; // x2 <= 0.3

    _ = try elasticNetFit(alloc, &cols, &y, 0.1, 0.5, &pf, &lb, &ub, &coefs, 10000, 1e-10);

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
    lambda: f64,
    alpha: f64,
    penalty_factors: []const f64,
    lower_bounds: []const f64,
    upper_bounds: []const f64,
    max_iter: usize,
    tol: f64,
) usize {
    const p = columns.len;
    const n = r.len;
    const n_f: f64 = @floatFromInt(n);

    // Just the coordinate descent loops

    var total_passes: usize = 0;
    var any_new = true;

    while (any_new and total_passes < max_iter) {
        var max_change: f64 = 0.0;
        while (total_passes < max_iter) {
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

                const beta_new = softThreshold(rho_j, lambda * alpha * penalty_factors[j]) /
                    (col_norms_squared[j] + lambda * (1.0 - alpha) * penalty_factors[j]);
                const beta_clamped = clamp(beta_new, lower_bounds[j], upper_bounds[j]);

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
            if (max_change < tol) break;
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
            const beta_new = softThreshold(rho_j, lambda * alpha * penalty_factors[j]);
            const beta_clamped = clamp(beta_new, lower_bounds[j], upper_bounds[j]);

            if (beta_clamped != 0.0) {
                active[j] = true;
                any_new = true;
                coefs[j] = beta_clamped / (col_norms_squared[j] + lambda * (1.0 - alpha) * penalty_factors[j]);
                const delta = coefs[j];
                if (delta != 0.0) {
                    if (gram == null) {
                        axpy(r, columns[j], delta);
                    }
                    const change = col_norms_squared[j] * delta * delta;
                    if (change > max_change) max_change = change;
                }
            }
        }
        if (max_change < tol) break;
    }

    return total_passes;
}
pub fn elasticNetPath(
    alloc: std.mem.Allocator,
    columns: []const []const f64,
    y: []const f64,
    alpha: f64,
    penalty_factors: []const f64,
    lower_bounds: []const f64,
    upper_bounds: []const f64,
    //path_params
    n_lambda: usize,
    lambda_min_ratio: f64, // 1e-4 if n>=p, 1e-2 if n<p
    //outputs, px n_lamba
    out_coefs_matrix: []f64, // coef j, offset at lambda k = [j*n_lambda + k]
    out_lambdas: []f64,
    max_iter: usize,
    tol: f64,
) !usize {
    const p = columns.len;
    const n = y.len;
    const n_f: f64 = @floatFromInt(n);

    var lambda_max: f64 = 0.0;
    const alpha_safe = clamp(alpha, 1e-3, alpha);

    for (0..p) |j| {
        if (penalty_factors[j] == 0.0) continue;

        const xty_j = @abs(dotProduct(columns[j], y)) / (n_f * alpha_safe * penalty_factors[j]);
        if (xty_j > lambda_max) lambda_max = xty_j;
    }
    if (lambda_max < 1e-10) return error.DegenerateData;

    const log_lambda_max = @log(lambda_max);
    const log_lambda_min = @log(lambda_max * lambda_min_ratio);
    for (0..n_lambda) |k| {
        const k_f: f64 = @floatFromInt(k);
        const n_f2: f64 = @floatFromInt(n_lambda - 1);
        out_lambdas[k] = @exp(log_lambda_max + k_f * (log_lambda_min - log_lambda_max) / n_f2);
    }

    //path loop, warm starts
    const coefs = try alloc.alloc(f64, p);
    @memset(coefs, 0.0);

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
    for (0..n_lambda) |k| {
        // Initialize active from current warm start
        for (0..p) |j| {
            active[j] = coefs[j] != 0.0;
        }

        const iters = elasticNetFitInner(
            columns,
            r,
            coefs,
            col_norms_squared,
            active,
            gram,
            xty,
            out_lambdas[k],
            alpha,
            penalty_factors,
            lower_bounds,
            upper_bounds,
            max_iter,
            tol,
        );
        total_iters += iters;

        for (0..p) |j| {
            out_coefs_matrix[j * n_lambda + k] = coefs[j];
        }
    }
    return total_iters;
}

// ============================================
// elasticNetPath Tests
// ============================================
test "elasticNetPath produces decreasing lambda sequence" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const n = 500;

    // Generate simple synthetic data: y = 2*x1 + 3*x2 +  noise
    var x1: [n]f64 = undefined;
    var x2: [n]f64 = undefined;
    var y: [n]f64 = undefined;

    // Simple deterministic "random" data
    for (0..n) |i| {
        const t: f64 = @floatFromInt(i);
        x1[i] = @sin(t * 0.1) * 3.0 + t * 0.01;
        x2[i] = @cos(t * 0.07) * 2.0 - t * 0.005;
        y[i] = 2.0 * x1[i] + 3.0 * x2[i] + @sin(t * 1.7) * 0.1;
    }

    const cols = [_][]const f64{
        &x1,
        &x2,
    };
    const pf = [_]f64{ 1.0, 1.0, 1.0 };
    const lb = [_]f64{ -inf, -inf, -inf };
    const ub = [_]f64{ inf, inf, inf };

    const n_lambda = 20;
    var out_lambdas: [n_lambda]f64 = undefined;
    var out_coef_matrix: [2 * n_lambda]f64 = undefined;

    _ = try elasticNetPath(
        alloc,
        &cols,
        &y,
        1.0,
        &pf,
        &lb,
        &ub,
        n_lambda,
        1e-4,
        &out_coef_matrix,
        &out_lambdas,
        10000,
        1e-10,
    );

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

    const n = 500;
    const p = 3;

    // Generate simple synthetic data: y = 2*x1 + 3*x2 + 0*x3 + noise
    var x1: [n]f64 = undefined;
    var x2: [n]f64 = undefined;
    var x3: [n]f64 = undefined;
    var y: [n]f64 = undefined;

    // Simple deterministic "random" data
    for (0..n) |i| {
        const t: f64 = @floatFromInt(i);
        x1[i] = @sin(t * 0.1) * 3.0 + t * 0.01;
        x2[i] = @cos(t * 0.07) * 2.0 - t * 0.005;
        x3[i] = @sin(t * 0.3) * 0.5;
        y[i] = 2.0 * x1[i] + 3.0 * x2[i] + @sin(t * 1.7) * 0.1;
    }

    const cols = [_][]const f64{ &x1, &x2, &x3 };
    const pf = [_]f64{ 1.0, 1.0, 1.0 };
    const lb = [_]f64{ -inf, -inf, -inf };
    const ub = [_]f64{ inf, inf, inf };

    const n_lambda = 50;
    var out_lambdas: [n_lambda]f64 = undefined;
    var out_coef_matrix: [p * n_lambda]f64 = undefined;

    const path_iters = try elasticNetPath(
        alloc,
        &cols,
        &y,
        1.0,
        &pf,
        &lb,
        &ub,
        n_lambda,
        1e-4,
        &out_coef_matrix,
        &out_lambdas,
        10000,
        1e-7,
    );

    const avg_iters = @as(f64, @floatFromInt(path_iters)) / @as(f64, @floatFromInt(n_lambda));
    std.debug.print("\n  Path avg iterations per lambda: {d:.2}\n", .{avg_iters});
    try std.testing.expect(avg_iters < 20.0);

    // Coefficients should grow as lambda decreases
    for (0..p) |j| {
        const at_max = @abs(out_coef_matrix[j * n_lambda + 0]);
        const at_min = @abs(out_coef_matrix[j * n_lambda + n_lambda - 1]);
        try std.testing.expect(at_min >= at_max);
    }
}
