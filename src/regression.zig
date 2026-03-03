const std = @import("std");

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

//TODO: Add multiple accumulators
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
