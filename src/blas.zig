const std = @import("std");

// CBLAS enums
pub const CBLAS_ORDER = enum(c_int) { RowMajor = 101, ColMajor = 102 };
pub const CBLAS_TRANSPOSE = enum(c_int) { NoTrans = 111, Trans = 112 };
pub const CBLAS_UPLO = enum(c_int) { Upper = 121, Lower = 122 };

// CBLAS extern declarations
pub extern "c" fn cblas_dsyrk(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: c_int,
    k: c_int,
    alpha: f64,
    a: [*]const f64,
    lda: c_int,
    beta: f64,
    c_mat: [*]f64,
    ldc: c_int,
) void;

pub extern "c" fn cblas_dgemv(
    order: CBLAS_ORDER,
    trans: CBLAS_TRANSPOSE,
    m: c_int,
    n: c_int,
    alpha: f64,
    a: [*]const f64,
    lda: c_int,
    x: [*]const f64,
    incx: c_int,
    beta: f64,
    y: [*]f64,
    incy: c_int,
) void;

pub extern "c" fn cblas_dgemm(
    order: CBLAS_ORDER,
    transA: CBLAS_TRANSPOSE,
    transB: CBLAS_TRANSPOSE,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f64,
    a: [*]const f64,
    lda: c_int,
    b: [*]const f64,
    ldb: c_int,
    beta: f64,
    c_mat: [*]f64,
    ldc: c_int,
) void;

/// Compute Gram matrix: result = (1/n) * X'X
/// X is n×p column-major. Result is p×p symmetric (upper triangle filled).
pub fn gramMatrix(X_col_major: []const f64, n: usize, p: usize, result: []f64) void {
    const n_f: f64 = @floatFromInt(n);
    // Trick: column-major n×p matrix is identical memory layout to row-major p×n matrix.
    // So tell BLAS it's RowMajor, p×n, and compute A * A' = p×p
    cblas_dsyrk(
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
    cblas_dgemv(
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

// ===== Tests =====

const regression = @import("regression.zig");

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
            const manual = regression.dotProduct(cols[i], cols[j]) / n_f;
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
        const manual = regression.dotProduct(cols[j], &y) / n_f;
        try std.testing.expectApproxEqAbs(manual, blas_result[j], 1e-10);
    }
}
