const std = @import("std");
const blas = @import("../blas.zig");

const errors = @import("../errors.zig");

pub fn axpy(a: f64, x: []const f64, y: []f64) void {
    std.debug.assert(x.len == y.len);
    const vec_len: u32 = std.simd.suggestVectorLength(f64) orelse 4; // per-target lane count; fallback 4 (AVX2-ish)
    const a_vec: @Vector(vec_len, f64) = @splat(a);

    var i: usize = 0;
    while (i + vec_len <= y.len) : (i += vec_len) {
        const xv: @Vector(vec_len, f64) = x[i..][0..vec_len].*;
        const yv: @Vector(vec_len, f64) = y[i..][0..vec_len].*;
        y[i..][0..vec_len].* = yv + xv * a_vec;
    }

    while (i < y.len) : (i += 1) {
        y[i] += a * x[i];
    }
}

test "axpy basic" {
    const x = [_]f64{ 1, 2, 3 };
    var y = [_]f64{ 10, 20, 30 };
    axpy(2.0, &x, &y);
    // y + 2x = {12, 24, 36}
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), y[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 24.0), y[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 36.0), y[2], 1e-10);
}

test "axpy exact vec width (no remainder)" {
    // len 8: whole multiple of any plausible lane count (4 or 8)
    const x = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var y = [_]f64{ 8, 7, 6, 5, 4, 3, 2, 1 };
    axpy(0.5, &x, &y);
    // {8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5}

    const want = [_]f64{ 8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5 };
    for (0..8) |i| {
        try std.testing.expectApproxEqAbs(want[i], y[i], 1e-10);
    }
}

test "axpy crosses vec boundary with remainder" {
    // 11 elements: at least one full SIMD pass + scalar remainder for lane count 4 or 8
    var x: [11]f64 = undefined;
    var y: [11]f64 = undefined;
    for (0..11) |i| {
        x[i] = @floatFromInt(i + 1); // 1..11
        y[i] = 100.0;
    }
    axpy(3.0, &x, &y);
    for (0..11) |i| {
        const want = 100.0 + 3.0 * @as(f64, @floatFromInt(i + 1));
        try std.testing.expectApproxEqAbs(want, y[i], 1e-10);
    }
}

test "axpy negative a is the residual update idiom" {
    // r -= coef * x  ==  axpy(-coef, x, r): aligns with blas implementation
    const x = [_]f64{ 1, 2, 3, 4 };
    var r = [_]f64{ 2, 4, 6, 8 }; // exactly 2*x
    axpy(-2.0, &x, &r);
    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), r[i], 1e-10);
    }
}

test "axpy zero a is a no-op" {
    const x = [_]f64{ 5, 6, 7, 8, 9 };
    var y = [_]f64{ 1, 2, 3, 4, 5 };
    axpy(0.0, &x, &y);
    const want = [_]f64{ 1, 2, 3, 4, 5 };
    for (0..5) |i| {
        try std.testing.expectApproxEqAbs(want[i], y[i], 1e-10);
    }
}

test "axpy add then subtract round-trips exactly" {
    // integer-valued doubles: y + 3x - 3x is exact in f64, pins the sign convention
    const x = [_]f64{ 1, 2, 3, 4, 5, 6, 7 };
    var y = [_]f64{ 10, 20, 30, 40, 50, 60, 70 };
    axpy(3.0, &x, &y);
    axpy(-3.0, &x, &y);
    const want = [_]f64{ 10, 20, 30, 40, 50, 60, 70 };
    for (0..7) |i| {
        try std.testing.expectEqual(want[i], y[i]); // exact, not approx
    }
}

test "axpy single element" {
    const x = [_]f64{2.0};
    var y = [_]f64{1.5};
    axpy(4.0, &x, &y);
    try std.testing.expectApproxEqAbs(@as(f64, 9.5), y[0], 1e-10);
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

/// Compute Gram matrix: result = (1/n) * X'X
/// X is n×p column-major. Result is p×p symmetric (upper triangle filled).
pub fn gramMatrix(X_col_major: []const f64, n: usize, p: usize, result: []f64) void {
    const n_f: f64 = @floatFromInt(n);
    // Trick: column-major n×p matrix is identical memory layout to row-major p×n matrix.
    // So tell BLAS it's RowMajor, p×n, and compute A * A' = p×p
    blas.cblas_dsyrk(
        .RowMajor,
        .Upper,
        .NoTrans,
        @intCast(p), // n (rows of result)
        @intCast(n), // k (inner dimension)
        1.0 / n_f,
        X_col_major.ptr,
        @intCast(n), // lda = n (row stride in row-major = columns per row)
        0.0,
        result.ptr,
        @intCast(p),
    );

    // Fill lower from upper (row-major: upper means j >= i)
    for (0..p) |i| {
        for (i + 1..p) |j| {
            result[j * p + i] = result[i * p + j];
        }
    }
}

pub fn xty(X_col_major: []const f64, y: []const f64, n: usize, p: usize, result: []f64) void {
    const n_f: f64 = @floatFromInt(n);
    // RowMajor: X is p×n, we want X * y (p×n times n×1 = p×1)
    blas.cblas_dgemv(
        .RowMajor,
        .NoTrans,
        @intCast(p), // rows
        @intCast(n), // cols
        1.0 / n_f,
        X_col_major.ptr,
        @intCast(n), // lda
        y.ptr,
        1,
        0.0,
        result.ptr,
        1,
    );
}

test "gramMatrix matches manual computation" {
    // 3×2 matrix, columns packed contiguously (column-major):
    // col0 = [1, 2, 3], col1 = [4, 5, 6]
    // Visually:   X = [1  4]
    //                 [2  5]
    //                 [3  6]
    const X = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const n: usize = 3;
    const p: usize = 2;

    var result: [4]f64 = undefined;
    gramMatrix(&X, n, p, &result);

    // X'X / N = [[1+4+9, 4+10+18], [4+10+18, 16+25+36]] / 3
    //         = [[14, 32], [32, 77]] / 3
    // Result is row-major: result[row * p + col]
    try std.testing.expectApproxEqAbs(@as(f64, 14.0 / 3.0), result[0 * p + 0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 32.0 / 3.0), result[0 * p + 1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 32.0 / 3.0), result[1 * p + 0], 1e-10); // symmetry
    try std.testing.expectApproxEqAbs(@as(f64, 77.0 / 3.0), result[1 * p + 1], 1e-10);
}

test "gramMatrix matches manual dotProduct computation" {
    const n: usize = 5;
    const p: usize = 3;

    // Individual columns (our Arrow format)
    const col0 = [_]f64{ 1, 2, 3, 4, 5 };
    const col1 = [_]f64{ 5, 4, 3, 2, 1 };
    const col2 = [_]f64{ 1, 0, 1, 0, 1 };

    // Packed column-major for BLAS: columns end-to-end (correct by construction).
    const X = col0 ++ col1 ++ col2;

    var blas_result: [9]f64 = undefined;
    gramMatrix(&X, n, p, &blas_result);

    // Compare against hand-rolled dotProduct for each (i, j)
    const n_f: f64 = @floatFromInt(n);
    const cols = [_][]const f64{ &col0, &col1, &col2 };
    for (0..p) |i| {
        for (0..p) |j| {
            const manual = dotProduct(cols[i], cols[j]) / n_f;
            // BLAS result is row-major: element (i, j) at result[i * p + j]
            try std.testing.expectApproxEqAbs(manual, blas_result[i * p + j], 1e-10);
        }
    }
}

test "xty matches manual computation" {
    // X = [1 4]    y = [1]
    //     [2 5]        [2]
    //     [3 6]        [3]
    // Column-major: col0 = [1,2,3], col1 = [4,5,6]
    const X = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const y = [_]f64{ 1, 2, 3 };
    const n: usize = 3;
    const p: usize = 2;

    var result: [2]f64 = undefined;
    xty(&X, &y, n, p, &result);

    // X'y / N = [1*1+2*2+3*3, 4*1+5*2+6*3] / 3 = [14, 32] / 3
    try std.testing.expectApproxEqAbs(@as(f64, 14.0 / 3.0), result[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 32.0 / 3.0), result[1], 1e-10);
}

test "gramMatrix diagonal matches column norms" {
    const n: usize = 4;
    const p: usize = 2;
    // Column-major: col0 = [1,2,3,4], col1 = [10,20,30,40]
    const X = [_]f64{ 1, 2, 3, 4, 10, 20, 30, 40 };

    var result: [4]f64 = undefined;
    gramMatrix(&X, n, p, &result);

    // Diagonal = (1/N) * ||col||^2
    const n_f: f64 = @floatFromInt(n);
    try std.testing.expectApproxEqAbs((1 + 4 + 9 + 16) / n_f, result[0 * p + 0], 1e-10);
    try std.testing.expectApproxEqAbs((100 + 400 + 900 + 1600) / n_f, result[1 * p + 1], 1e-10);

    // Off-diagonal should also be correct
    const expected_01 = (1 * 10 + 2 * 20 + 3 * 30 + 4 * 40) / n_f; // 300/4 = 75
    try std.testing.expectApproxEqAbs(expected_01, result[0 * p + 1], 1e-10);
    try std.testing.expectApproxEqAbs(expected_01, result[1 * p + 0], 1e-10);
}

test "xty matches manual dotProduct" {
    const n: usize = 5;
    const p: usize = 3;

    const col0 = [_]f64{ 1, 2, 3, 4, 5 };
    const col1 = [_]f64{ 5, 4, 3, 2, 1 };
    const col2 = [_]f64{ 1, 0, 1, 0, 1 };
    const y = [_]f64{ 2, 4, 6, 8, 10 };

    // Column-major for BLAS: columns end-to-end (correct by construction).
    const X = col0 ++ col1 ++ col2;

    var blas_result: [3]f64 = undefined;
    xty(&X, &y, n, p, &blas_result);

    // Compare against dotProduct
    const n_f: f64 = @floatFromInt(n);
    const cols = [_][]const f64{ &col0, &col1, &col2 };
    for (0..p) |j| {
        const manual = dotProduct(cols[j], &y) / n_f;
        try std.testing.expectApproxEqAbs(manual, blas_result[j], 1e-10);
    }
}

// Cholesky

pub fn choleskySolve(gram: []f64, b: []f64, p: usize) !void {
    const info_f = blas.LAPACKE_dpotrf(
        .ColMajor, // 102 — but G is symmetric+full, so forgiving here
        'L',
        @intCast(p),
        gram.ptr,
        @intCast(p),
    );
    if (info_f > 0) return errors.QError.NotPositiveDefinite; // collinear features → QR fallback
    if (info_f < 0) return errors.QError.InvalidArgument; // marshalling bug (your side)

    const info_s = blas.LAPACKE_dpotrs(
        .ColMajor,
        'L',
        @intCast(p),
        1, // nrhs
        gram.ptr, // the L from dpotrf
        @intCast(p),
        b.ptr, // in: X'y ; out: β
        @intCast(p),
    );
    if (info_s != 0) return errors.QError.SolveFailed;
}
