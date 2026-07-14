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
