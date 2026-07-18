const std = @import("std");
const blas = @import("../blas.zig");

const errors = @import("../errors.zig");
const fixtures = @import("../fixtures.zig");

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

pub fn sumV(a: []const f64) f64 {
    const vec_len: u32 = std.simd.suggestVectorLength(f64) orelse 4;
    const V = @Vector(vec_len, f64);
    const n = a.len;
    const stride = vec_len * 4;

    var acc0: V = @splat(0.0);
    var acc1: V = @splat(0.0);
    var acc2: V = @splat(0.0);
    var acc3: V = @splat(0.0);

    var i: usize = 0;
    while (i + stride <= n) : (i += stride) {
        acc0 += a[i..][0..vec_len].*;
        acc1 += a[i + vec_len ..][0..vec_len].*;
        acc2 += a[i + vec_len * 2 ..][0..vec_len].*;
        acc3 += a[i + vec_len * 3 ..][0..vec_len].*;
    }
    while (i + vec_len <= n) : (i += vec_len) {
        acc0 += @as(V, a[i..][0..vec_len].*);
    }

    var sum: f64 = @reduce(.Add, (acc0 + acc1) + (acc2 + acc3));
    while (i < n) : (i += 1) sum += a[i];
    return sum;
}

test "sumV sums correctly" {
    const a: [6]f64 = .{ 1, 2, 3, 4, 5, 6 };

    const b = sumV(&a);
    try std.testing.expectEqual(21, b);
}

pub fn dotProduct(a: []const f64, b: []const f64) f64 {
    std.debug.assert(a.len == b.len);
    const vec_len: u32 = std.simd.suggestVectorLength(f64) orelse 4;
    const V = @Vector(vec_len, f64);
    const n = a.len;
    const stride = vec_len * 4;

    var acc0: V = @splat(0.0);
    var acc1: V = @splat(0.0);
    var acc2: V = @splat(0.0);
    var acc3: V = @splat(0.0);

    var i: usize = 0;
    while (i + stride <= n) : (i += stride) {
        acc0 = @mulAdd(V, a[i..][0..vec_len].*, b[i..][0..vec_len].*, acc0);
        acc1 = @mulAdd(V, a[i + vec_len ..][0..vec_len].*, b[i + vec_len ..][0..vec_len].*, acc1);
        acc2 = @mulAdd(V, a[i + vec_len * 2 ..][0..vec_len].*, b[i + vec_len * 2 ..][0..vec_len].*, acc2);
        acc3 = @mulAdd(V, a[i + vec_len * 3 ..][0..vec_len].*, b[i + vec_len * 3 ..][0..vec_len].*, acc3);
    }

    // vector cleanup: single chain, at most 3 iterations
    while (i + vec_len <= n) : (i += vec_len) {
        acc0 = @mulAdd(V, a[i..][0..vec_len].*, b[i..][0..vec_len].*, acc0);
    }

    var sum: f64 = @reduce(.Add, (acc0 + acc1) + (acc2 + acc3));

    // scalar remainder: < vec_len iterations
    while (i < n) : (i += 1) {
        sum = @mulAdd(f64, a[i], b[i], sum);
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

pub const Moments1 = struct { sum: f64, sum_sq: f64 };

pub fn sumAndSumSq(a: []const f64) Moments1 {
    const vec_len: u32 = std.simd.suggestVectorLength(f64) orelse 4;
    const V = @Vector(vec_len, f64);
    const n = a.len;
    const stride = vec_len * 4;

    var s0: V = @splat(0.0);
    var s1: V = @splat(0.0);
    var s2: V = @splat(0.0);
    var s3: V = @splat(0.0);
    var q0: V = @splat(0.0);
    var q1: V = @splat(0.0);
    var q2: V = @splat(0.0);
    var q3: V = @splat(0.0);

    var i: usize = 0;
    while (i + stride <= n) : (i += stride) {
        const v0: V = a[i..][0..vec_len].*;
        const v1: V = a[i + vec_len ..][0..vec_len].*;
        const v2: V = a[i + vec_len * 2 ..][0..vec_len].*;
        const v3: V = a[i + vec_len * 3 ..][0..vec_len].*;
        s0 += v0;
        s1 += v1;
        s2 += v2;
        s3 += v3;
        q0 = @mulAdd(V, v0, v0, q0);
        q1 = @mulAdd(V, v1, v1, q1);
        q2 = @mulAdd(V, v2, v2, q2);
        q3 = @mulAdd(V, v3, v3, q3);
    }
    while (i + vec_len <= n) : (i += vec_len) {
        const v: V = a[i..][0..vec_len].*;
        s0 += v;
        q0 = @mulAdd(V, v, v, q0);
    }

    var sum: f64 = @reduce(.Add, (s0 + s1) + (s2 + s3));
    var sum_sq: f64 = @reduce(.Add, (q0 + q1) + (q2 + q3));
    while (i < n) : (i += 1) {
        sum += a[i];
        sum_sq = @mulAdd(f64, a[i], a[i], sum_sq);
    }
    return .{ .sum = sum, .sum_sq = sum_sq };
}

test "sumAndSumSq empty slice" {
    const a = [_]f64{};
    const m = sumAndSumSq(&a);
    try std.testing.expectEqual(@as(f64, 0.0), m.sum);
    try std.testing.expectEqual(@as(f64, 0.0), m.sum_sq);
}

test "sumAndSumSq single element" {
    const a = [_]f64{-3.0};
    const m = sumAndSumSq(&a);
    try std.testing.expectApproxEqAbs(@as(f64, -3.0), m.sum, 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), m.sum_sq, 1e-15);
}

test "sumAndSumSq known exact values" {
    // Small integers: every intermediate is exact in f64 regardless of
    // summation order, so equality here is exact, not approximate.
    // sum = 1+2+3+4+5 = 15; sum_sq = 1+4+9+16+25 = 55
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const m = sumAndSumSq(&a);
    try std.testing.expectEqual(@as(f64, 15.0), m.sum);
    try std.testing.expectEqual(@as(f64, 55.0), m.sum_sq);
}

test "sumAndSumSq mixed signs: sum cancels, sum_sq does not" {
    // Distinguishes the two outputs — a copy-paste bug wiring sum_sq to the
    // sum accumulators (or vice versa) fails here and nowhere obvious else.
    const a = [_]f64{ 2.0, -2.0, 1.0, -1.0, 3.0, -3.0 };
    const m = sumAndSumSq(&a);
    try std.testing.expectEqual(@as(f64, 0.0), m.sum);
    try std.testing.expectEqual(@as(f64, 28.0), m.sum_sq);
}

test "sumAndSumSq length spanning all three loop regimes exactly" {
    // n = 4*vec_len + vec_len + 1: one full stride iteration, one vector
    // cleanup iteration, one scalar-tail element — all three loops execute.
    // Constant 1.0 makes the expected values exact: sum = n, sum_sq = n.
    const vec_len: u32 = std.simd.suggestVectorLength(f64) orelse 4;
    const n = vec_len * 4 + vec_len + 1;

    var buf: [128]f64 = undefined;
    for (0..n) |i| buf[i] = 1.0;

    const m = sumAndSumSq(buf[0..n]);
    const n_f: f64 = @floatFromInt(n);
    try std.testing.expectEqual(n_f, m.sum);
    try std.testing.expectEqual(n_f, m.sum_sq);
}

test "sumAndSumSq large-magnitude values stay finite and scale correctly" {
    // sum_sq squares the inputs — guard against a kernel change that
    // introduces premature overflow (e.g. squaring a partial reduction).
    const a = [_]f64{ 1e150, 1e150, -1e150 };
    const m = sumAndSumSq(&a);
    try std.testing.expectApproxEqRel(@as(f64, 1e150), m.sum, 1e-12);
    try std.testing.expectApproxEqRel(@as(f64, 3e300), m.sum_sq, 1e-12);
    try std.testing.expect(std.math.isFinite(m.sum_sq));
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
pub fn choleskyFactor(a: []f64, p: usize) !void {
    std.debug.assert(a.len == p * p);

    for (0..p) |j| {
        // row_j = L[j][0..j], already computed (row-major: contiguous)
        const row_j = a[j * p .. j * p + j];

        // diagonal: d = A[j][j] − Σ L[j][k]²
        var d = a[j * p + j] - dotProduct(row_j, row_j);
        if (d <= 0.0) return errors.QError.NotPositiveDefinite; // rank-deficient gram
        d = @sqrt(d);
        a[j * p + j] = d;

        // below-diagonal: L[i][j] = (A[i][j] − <L[i][0..j], L[j][0..j]>) / L[j][j]
        for (j + 1..p) |i| {
            const row_i = a[i * p .. i * p + j];
            const s = a[i * p + j] - dotProduct(row_i, row_j);
            a[i * p + j] = s / d;
        }
    }
}

pub fn choleskySolve(l: []const f64, b: []f64, p: usize) void {
    std.debug.assert(l.len == p * p);
    std.debug.assert(b.len == p);

    // forward substitution: L·y = b (y overwrites b)
    for (0..p) |i| {
        const s = b[i] - dotProduct(l[i * p .. i * p + i], b[0..i]);
        b[i] = s / l[i * p + i];
    }

    // back substitution: L'·x = y (x overwrites y)
    var i: usize = p;
    while (i > 0) {
        i -= 1;
        var s = b[i];
        for (i + 1..p) |k| {
            s -= l[k * p + i] * b[k];
        }
        b[i] = s / l[i * p + i];
    }
}

// Cholesky Tests

test "choleskyFactor known 2x2" {
    // A = [4 2; 2 3]  →  L = [2 0; 1 √2]
    var a = [_]f64{ 4, 2, 2, 3 };
    try choleskyFactor(&a, 2);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), a[0], 1e-12); // L[0][0]
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), a[2], 1e-12); // L[1][0]
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 2.0)), a[3], 1e-12); // L[1][1]
    // upper triangle is stale A by contract — a[1] still 2
    try std.testing.expectEqual(@as(f64, 2.0), a[1]);
}

test "choleskyFactor round-trips a constructed L" {
    // Build A = L_true·L_true', factor it, expect L_true back exactly.
    // Chosen so every pivot is a perfect square: d values are 4, 9, 2.25.
    const l_true = [_]f64{
        2.0, 0,    0,
        1.0, 3.0,  0,
        0.5, -1.0, 1.5,
    };
    // A = L·L' (hand-computed, exact)
    var a = [_]f64{
        4.0, 2.0,  1.0,
        2.0, 10.0, -2.5,
        1.0, -2.5, 3.5,
    };
    try choleskyFactor(&a, 3);

    for (0..3) |i| {
        for (0..i + 1) |j| { // lower triangle only
            try std.testing.expectApproxEqAbs(l_true[i * 3 + j], a[i * 3 + j], 1e-12);
        }
    }
}

test "choleskyFactor rejects non-positive-definite" {
    // A = [1 2; 2 1]: symmetric, eigenvalues {3, -1} → not PD.
    // Second pivot: 1 − 2² = −3, deterministically ≤ 0 (no FP-borderline).
    var a = [_]f64{ 1, 2, 2, 1 };
    try std.testing.expectError(errors.QError.NotPositiveDefinite, choleskyFactor(&a, 2));
}
test "choleskySolve recovers known solution" {
    // A = [4 2; 2 3], x_true = [1, 2]  →  b = A·x_true = [8, 8]
    var a = [_]f64{ 4, 2, 2, 3 };
    try choleskyFactor(&a, 2);

    var b = [_]f64{ 8, 8 };
    choleskySolve(&a, &b, 2);

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), b[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), b[1], 1e-12);
}

test "choleskySolve identity is a no-op" {
    // L = I → forward and back substitution both leave b untouched.
    const eye = [_]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    var b = [_]f64{ 3.5, -2.0, 7.25 };
    choleskySolve(&eye, &b, 3);
    try std.testing.expectEqual(@as(f64, 3.5), b[0]);
    try std.testing.expectEqual(@as(f64, -2.0), b[1]);
    try std.testing.expectEqual(@as(f64, 7.25), b[2]);
}

pub fn packX(columns: []const []const f64, X: []f64, n: usize) void {
    std.debug.assert(X.len == columns.len * n);
    for (columns, 0..) |col, j| {
        std.debug.assert(col.len == n);
        @memcpy(X[j * n .. (j + 1) * n], col);
    }
}

test "packX lays columns end-to-end (column-major)" {
    const col0 = [_]f64{ 1, 2, 3 };
    const col1 = [_]f64{ 4, 5, 6 };
    const cols = [_][]const f64{ &col0, &col1 };

    var X: [6]f64 = undefined;
    packX(&cols, &X, 3);

    const want = [_]f64{ 1, 2, 3, 4, 5, 6 };
    for (0..6) |i| {
        try std.testing.expectEqual(want[i], X[i]);
    }
}

test "packX feeds gramMatrix identically to dotProduct" {
    const col0 = [_]f64{ 1, 2, 3, 4, 5 };
    const col1 = [_]f64{ 5, 4, 3, 2, 1 };
    const col2 = [_]f64{ 1, 0, 1, 0, 1 };
    const cols = [_][]const f64{ &col0, &col1, &col2 };
    const n = 5;
    const p = 3;
    const n_f: f64 = @floatFromInt(n);

    var X: [n * p]f64 = undefined;
    packX(&cols, &X, n);

    var gram: [p * p]f64 = undefined;
    gramMatrix(&X, n, p, &gram);

    for (0..p) |i| {
        for (0..p) |j| {
            const manual = dotProduct(cols[i], cols[j]) / n_f;
            try std.testing.expectApproxEqAbs(manual, gram[i * p + j], 1e-10);
        }
    }
}

test "pack -> gram -> factor -> solve recovers OLS coefficients" {
    // Full normal-equations pipeline on the exact_2col fixture:
    // (X'X/n)·β = X'y/n, y = 2·x1 + 3·x2 exactly → β = [2, 3].
    // Gram is [11 8.2; 8.2 6.8] (PD, det 7.56); X'y/n = [46.6, 36.8].
    const cols = fixtures.exact_2col.cols;
    const y = fixtures.exact_2col.y;
    const n = y.len;
    const p = 2;

    var X: [n * p]f64 = undefined;
    packX(&cols, &X, n);

    var gram: [p * p]f64 = undefined;
    gramMatrix(&X, n, p, &gram);

    var beta: [p]f64 = undefined;
    xty(&X, &y, n, p, &beta);

    try choleskyFactor(&gram, p);
    choleskySolve(&gram, &beta, p);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), beta[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), beta[1], 1e-10);
}

pub fn packColRange(slice: []const f64, pack: []f64, t_rows: usize, j: usize) void {
    std.debug.assert(slice.len == t_rows);
    std.debug.assert(pack.len >= (j + 1) * t_rows);
    @memcpy(pack[j * t_rows .. (j + 1) * t_rows], slice);
}

pub fn packRange(columns: []const []const f64, X: []f64, start: usize, end: usize) void {
    const rows = end - start;
    std.debug.assert(X.len >= columns.len * rows);
    for (columns, 0..) |col, j| {
        std.debug.assert(end <= col.len);
        @memcpy(X[j * rows .. (j + 1) * rows], col[start..end]);
    }
}

test "packRange lays columns end-to-end (column-major)" {
    const col0 = [_]f64{ 1, 2, 3 };
    const col1 = [_]f64{ 4, 5, 6 };
    const cols = [_][]const f64{ &col0, &col1 };

    var X: [4]f64 = undefined;
    @memset(&X, 0);
    packRange(&cols, &X, 0, 2);

    const want = [_]f64{ 1, 2, 4, 5 };
    for (0..4) |i| {
        try std.testing.expectEqual(want[i], X[i]);
    }
}

test "packRange middle window uses block coordinates" {
    const col0 = [_]f64{ 1, 2, 3 };
    const col1 = [_]f64{ 4, 5, 6 };
    const cols = [_][]const f64{ &col0, &col1 };

    var X: [4]f64 = undefined;
    packRange(&cols, &X, 1, 3); // rows 1..3
    // destination starts at 0 even though source window didn't
    const want = [_]f64{ 2, 3, 5, 6 };
    for (0..4) |i| try std.testing.expectEqual(want[i], X[i]);
}

test "packRange runt tile packs densely at front of larger scratch" {
    const col0 = [_]f64{ 1, 2, 3 };
    const col1 = [_]f64{ 4, 5, 6 };
    const cols = [_][]const f64{ &col0, &col1 };

    var X: [6]f64 = .{ -1, -1, -1, -1, -1, -1 }; // scratch bigger than block
    packRange(&cols, &X, 0, 2);
    const want = [_]f64{ 1, 2, 4, 5 };
    for (0..4) |i| try std.testing.expectEqual(want[i], X[i]);
    try std.testing.expectEqual(@as(f64, -1), X[4]); // tail untouched
    try std.testing.expectEqual(@as(f64, -1), X[5]);
}

pub const SufficientStats = struct {
    n: usize,
    p: usize,
    sum_x: []f64,
    sum_xx: []f64,
    sum_y: f64,
    y_bar: f64,
    yty: f64,
    xty: ?[]f64,
    gram: ?[]f64,
};

pub const StatsSpec = struct {
    gram: bool,
    xty: bool,
};

fn allocZeroed(alloc: std.mem.Allocator, n: usize) ![]f64 {
    const buf = try alloc.alloc(f64, n);
    @memset(buf, 0);
    return buf;
}
fn accumulateGram(pack: []const f64, rows_blk: usize, p: usize, gram: []f64) void {
    blas.cblas_dsyrk(
        .RowMajor,
        .Upper,
        .NoTrans,
        @intCast(p),
        @intCast(rows_blk),
        1.0,
        pack.ptr,
        @intCast(rows_blk),
        1.0,
        gram.ptr,
        @intCast(p),
    );
}

pub const StatsAccumulator = struct {
    spec: StatsSpec,
    p: usize,
    tile_rows: usize,
    n_seen: usize = 0,

    sum_x: []f64,
    sum_xx: []f64,
    sum_y: f64 = 0,
    yty: f64 = 0,
    xty: ?[]f64,
    gram: ?[]f64,
    pack: ?[]f64,

    pub fn init(alloc: std.mem.Allocator, p: usize, spec: StatsSpec) !StatsAccumulator {
        const tile_rows = @max(64, (256 * 1024 / @sizeOf(f64)) / p);

        const sum_x: []f64 = try allocZeroed(alloc, p);
        const sum_xx: []f64 = try allocZeroed(alloc, p);
        const xty_: ?[]f64 = if (spec.xty) try allocZeroed(alloc, p) else null;
        const gram: ?[]f64 = if (spec.gram) try allocZeroed(alloc, p * p) else null;
        const pack: ?[]f64 = if (spec.gram) try allocZeroed(alloc, p * tile_rows) else null;

        return .{
            .spec = spec,
            .p = p,
            .tile_rows = tile_rows,
            .sum_x = sum_x,
            .sum_xx = sum_xx,
            .gram = gram,
            .xty = xty_,
            .pack = pack,
        };
    }
    pub fn update(self: *StatsAccumulator, columns: []const []const f64, y: []const f64) void {
        std.debug.assert(columns.len == self.p);
        const rows = y.len;

        var start: usize = 0;

        while (start < rows) : (start += self.tile_rows) {
            const end = @min(start + self.tile_rows, rows);
            const t_rows = end - start;
            const y_t = y[start..end];
            // per-column stats
            for (columns, 0..) |col, j| {
                std.debug.assert(col.len == rows);
                const col_slice = col[start..end];

                const m = sumAndSumSq(col_slice);
                self.sum_x[j] += m.sum;
                self.sum_xx[j] += m.sum_sq;

                if (self.xty) |xt| xt[j] += dotProduct(col_slice, y_t);

                if (self.gram != null) packColRange(col_slice, self.pack.?, t_rows, j);
            }

            // per-tile stats
            self.sum_y += sumV(y_t);
            self.yty += dotProduct(y_t, y_t);

            if (self.gram) |g| accumulateGram(self.pack.?, end - start, self.p, g);
        }
        self.n_seen += rows;
    }

    pub fn finalize(self: *StatsAccumulator) SufficientStats {
        std.debug.assert(self.n_seen > 0);
        const n_f: f64 = @floatFromInt(self.n_seen);

        if (self.gram) |g| {
            for (0..self.p) |i| {
                for (i..self.p) |j| {
                    const v = g[i * self.p + j] / n_f;
                    g[i * self.p + j] = v;
                    g[j * self.p + i] = v;
                }
            }
        }
        if (self.xty) |xt| {
            for (xt) |*v| v.* /= n_f;
        }

        return .{
            .n = self.n_seen,
            .p = self.p,
            .sum_x = self.sum_x,
            .sum_xx = self.sum_xx,
            .sum_y = self.sum_y,
            .y_bar = self.sum_y / @as(f64, @floatFromInt(self.n_seen)),
            .yty = self.yty,
            .xty = self.xty,
            .gram = self.gram,
        };
    }
};

test "StatsAccumulator: two uneven chunks match one-shot computation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const n = 100;
    const p = 3;
    const data = fixtures.sinCos(n).init();
    const cols = [_][]const f64{ &data.x1, &data.x2, &data.x3 };

    // --- reference: one-shot over the full data ---
    var X: [n * p]f64 = undefined;
    packX(&cols, &X, n);

    var gram_ref: [p * p]f64 = undefined;
    gramMatrix(&X, n, p, &gram_ref); // X'X / n

    var xty_ref: [p]f64 = undefined;
    xty(&X, &data.y, n, p, &xty_ref); // X'y / n

    var sum_x_ref: [p]f64 = .{ 0, 0, 0 };
    var sum_y_ref: f64 = 0;
    var yty_ref: f64 = 0;
    for (0..n) |i| {
        for (0..p) |j| sum_x_ref[j] += cols[j][i];
        sum_y_ref += data.y[i];
        yty_ref += data.y[i] * data.y[i];
    }

    // --- accumulated: same data, two uneven chunks, tiny tiles ---
    var acc = try StatsAccumulator.init(alloc, p, .{
        .gram = true,
        .xty = true,
    });
    acc.tile_rows = 7; // force runt (37 = 5*7+2) and exact (63 = 9*7) tile paths

    const split = 37;
    const cols_a = [_][]const f64{ data.x1[0..split], data.x2[0..split], data.x3[0..split] };
    const cols_b = [_][]const f64{ data.x1[split..], data.x2[split..], data.x3[split..] };

    acc.update(&cols_a, data.y[0..split]);
    acc.update(&cols_b, data.y[split..]);
    const stats = acc.finalize();

    // --- identity checks ---
    try std.testing.expectEqual(@as(usize, n), stats.n);
    try std.testing.expectEqual(@as(usize, p), stats.p);

    // gram/xty: normalized by n on both sides; tolerance covers the
    // different summation order (tiled dsyrk vs one-shot)
    for (0..p * p) |i| {
        try std.testing.expectApproxEqAbs(gram_ref[i], stats.gram.?[i], 1e-9);
    }
    for (0..p) |j| {
        try std.testing.expectApproxEqAbs(xty_ref[j], stats.xty.?[j], 1e-9);
    }

    // moments: RAW by convention (consumers divide)
    for (0..p) |j| {
        try std.testing.expectApproxEqAbs(sum_x_ref[j], stats.sum_x[j], 1e-9);
    }
    try std.testing.expectApproxEqAbs(sum_y_ref, stats.sum_y, 1e-9);
    try std.testing.expectApproxEqAbs(yty_ref, stats.yty, 1e-9);
}

test "StatsAccumulator: two even chunks match one-shot computation" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const n = 100;
    const p = 3;
    const data = fixtures.sinCos(n).init();
    const cols = [_][]const f64{ &data.x1, &data.x2, &data.x3 };

    // --- reference: one-shot over the full data ---
    var X: [n * p]f64 = undefined;
    packX(&cols, &X, n);

    var gram_ref: [p * p]f64 = undefined;
    gramMatrix(&X, n, p, &gram_ref); // X'X / n

    var xty_ref: [p]f64 = undefined;
    xty(&X, &data.y, n, p, &xty_ref); // X'y / n

    var sum_x_ref: [p]f64 = .{ 0, 0, 0 };
    var sum_xx_ref: [p]f64 = .{ 0, 0, 0 };
    var sum_y_ref: f64 = 0;
    var yty_ref: f64 = 0;
    for (0..n) |i| {
        for (0..p) |j| {
            sum_x_ref[j] += cols[j][i];
            sum_xx_ref[j] += cols[j][i] * cols[j][i];
        }
        sum_y_ref += data.y[i];
        yty_ref += data.y[i] * data.y[i];
    }

    // --- accumulated: same data, two even chunks, tiny tiles ---
    var acc = try StatsAccumulator.init(alloc, p, .{
        .gram = true,
        .xty = true,
    });
    acc.tile_rows = 10;

    const split = 50;
    const cols_a = [_][]const f64{ data.x1[0..split], data.x2[0..split], data.x3[0..split] };
    const cols_b = [_][]const f64{ data.x1[split..], data.x2[split..], data.x3[split..] };

    acc.update(&cols_a, data.y[0..split]);
    acc.update(&cols_b, data.y[split..]);
    const stats = acc.finalize();

    // --- identity checks ---
    try std.testing.expectEqual(@as(usize, n), stats.n);
    try std.testing.expectEqual(@as(usize, p), stats.p);

    // gram/xty: normalized by n on both sides; tolerance covers the
    // different summation order (tiled dsyrk vs one-shot)
    for (0..p * p) |i| {
        try std.testing.expectApproxEqAbs(gram_ref[i], stats.gram.?[i], 1e-9);
    }
    for (0..p) |j| {
        try std.testing.expectApproxEqAbs(xty_ref[j], stats.xty.?[j], 1e-9);
    }

    // moments: RAW by convention (consumers divide)
    for (0..p) |j| {
        try expectRelClose(sum_x_ref[j], stats.sum_x[j], 1e-12);
        try expectRelClose(sum_xx_ref[j], stats.sum_xx[j], 1e-12);
    }
    try std.testing.expectApproxEqAbs(sum_y_ref, stats.sum_y, 1e-9);
    try std.testing.expectApproxEqAbs(yty_ref, stats.yty, 1e-9);
}

test "StatsAccumulator: spec flags off mean null, not garbage" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const data = fixtures.sinCos(20).init();
    const cols = [_][]const f64{ &data.x1, &data.x2, &data.x3 };

    var acc = try StatsAccumulator.init(alloc, 3, .{
        .gram = false,
        .xty = true,
    });
    acc.update(&cols, &data.y);
    const stats = acc.finalize();

    try std.testing.expectEqual(@as(?[]f64, null), stats.gram);
    try std.testing.expect(stats.xty != null);
    try std.testing.expectEqual(@as(usize, 20), stats.n);
}

fn refSum(a: []const f64) f64 {
    var s: f64 = 0.0;
    for (a) |x| s += x;
    return s;
}

fn refDot(a: []const f64, b: []const f64) f64 {
    var s: f64 = 0.0;
    for (a, b) |x, y| s += x * y;
    return s;
}

fn refSumSq(a: []const f64) f64 {
    var s: f64 = 0.0;
    for (a) |x| s += x * x;
    return s;
}

fn expectRelClose(expected: f64, actual: f64, rel_tol: f64) !void {
    // Relative comparison with an absolute floor: reassociation + FMA mean
    // the kernels won't match a scalar reference bitwise, and near-zero
    // sums make pure relative comparison meaningless.
    const scale = @max(@abs(expected), 1.0);
    try std.testing.expect(@abs(expected - actual) <= rel_tol * scale);
}

test "kernel property sweep: sumV/dotProduct/sumAndSumSq vs scalar reference across all remainder classes" {
    const vec_len: u32 = std.simd.suggestVectorLength(f64) orelse 4;
    const stride = vec_len * 4;
    // Two full stride periods + tail guarantees: 0 and ≥1 stride iterations
    // × {0,1,2,3} vector-cleanup iterations × every scalar-tail length.
    const max_n = stride * 2 + vec_len;

    var buf_a: [512]f64 = undefined;
    var buf_b: [512]f64 = undefined;
    std.debug.assert(max_n <= buf_a.len);

    var prng = std.Random.DefaultPrng.init(0xdead_beef);
    const rand = prng.random();

    var n: usize = 0;
    while (n <= max_n) : (n += 1) {
        for (0..n) |i| {
            // Mixed-sign, mixed-magnitude, no pathological cancellation.
            buf_a[i] = (rand.float(f64) - 0.5) * 4.0;
            buf_b[i] = (rand.float(f64) - 0.5) * 4.0;
        }
        const a = buf_a[0..n];
        const b = buf_b[0..n];

        try expectRelClose(refSum(a), sumV(a), 1e-12);
        try expectRelClose(refDot(a, b), dotProduct(a, b), 1e-12);

        const m = sumAndSumSq(a);
        try expectRelClose(refSum(a), m.sum, 1e-12);
        try expectRelClose(refSumSq(a), m.sum_sq, 1e-12);
    }
}

test "kernel property: sumAndSumSq agrees with sumV and dotProduct(a,a) at every remainder class" {
    // Ties the fused kernel to the two it replaces — this is the invariant
    // the accumulator's moments block actually depends on.
    const vec_len: u32 = std.simd.suggestVectorLength(f64) orelse 4;
    const max_n = vec_len * 4 * 2 + vec_len;

    var buf: [512]f64 = undefined;
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rand = prng.random();

    var n: usize = 0;
    while (n <= max_n) : (n += 1) {
        for (0..n) |i| buf[i] = (rand.float(f64) - 0.5) * 10.0;
        const a = buf[0..n];

        const m = sumAndSumSq(a);
        try expectRelClose(sumV(a), m.sum, 1e-12);
        try expectRelClose(dotProduct(a, a), m.sum_sq, 1e-12);
    }
}

test "sum_xx/n agrees with gram diagonal when both are computed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const n = 100;
    const p = 3;
    const data = fixtures.sinCos(n).init();

    var acc = try StatsAccumulator.init(alloc, p, .{
        .gram = true,
        .xty = true,
    });
    acc.tile_rows = 10;

    const split = 50;
    const cols_a = [_][]const f64{ data.x1[0..split], data.x2[0..split], data.x3[0..split] };
    const cols_b = [_][]const f64{ data.x1[split..], data.x2[split..], data.x3[split..] };

    acc.update(&cols_a, data.y[0..split]);
    acc.update(&cols_b, data.y[split..]);
    const stats = acc.finalize();

    const n_f: f64 = @floatFromInt(stats.n);
    for (0..p) |j| {
        // gram is /n, sum_xx is raw — this test also pins the convention.
        try expectRelClose(stats.sum_xx[j] / n_f, stats.gram.?[j * p + j], 1e-12);
    }
}
test "gram-less spec still produces full moments including sum_xx" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const n = 100;
    const p = 3;
    const data = fixtures.sinCos(n).init();

    const cols = [_][]const f64{ &data.x1, &data.x2, &data.x3 };

    var X: [n * p]f64 = undefined;
    packX(&cols, &X, n);

    var sum_x_ref: [p]f64 = .{ 0, 0, 0 };
    var sum_xx_ref: [p]f64 = .{ 0, 0, 0 };
    for (0..n) |i| {
        for (0..p) |j| {
            sum_x_ref[j] += cols[j][i];
            sum_xx_ref[j] += cols[j][i] * cols[j][i];
        }
    }
    var acc = try StatsAccumulator.init(alloc, p, .{
        .gram = false,
        .xty = true,
    });
    acc.tile_rows = 10;

    const split = 50;
    const cols_a = [_][]const f64{ data.x1[0..split], data.x2[0..split], data.x3[0..split] };
    const cols_b = [_][]const f64{ data.x1[split..], data.x2[split..], data.x3[split..] };

    acc.update(&cols_a, data.y[0..split]);
    acc.update(&cols_b, data.y[split..]);
    const stats = acc.finalize();
    try std.testing.expect(stats.gram == null);
    try std.testing.expect(stats.xty != null);
    for (0..p) |j| {
        try expectRelClose(sum_xx_ref[j], stats.sum_xx[j], 1e-12);
    }
}

test {
    std.testing.refAllDecls(@This());
}
