const std = @import("std");
const errors = @import("../errors.zig");

const fixtures = @import("../fixtures.zig");
const dotProduct = @import("common.zig").dotProduct;
const axpy = @import("common.zig").axpy;

pub const Options = struct {
    method: Method = Method.auto,
};

pub const Method = enum(c_int) {
    auto = 0,
    cholesky = 1,
    gaussian_elimination = 2, // reference implementation; kept as bench baseline + oracle
};

pub fn gaussianElim(
    alloc: std.mem.Allocator,
    columns: []const []const f64,
    y: []const f64,
    out_coefs: []f64,
) !void {
    const p = columns.len;
    std.debug.assert(out_coefs.len == p);

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

test "gaussian elimination recovers known coefficients" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    // y = 2*x1 + 3*x2, no noise: recover exactly [2, 3]
    var coefs: [2]f64 = undefined;

    try gaussianElim(alloc, &fixtures.exact_2col.cols, &fixtures.exact_2col.y, &coefs);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), coefs[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), coefs[1], 1e-10);
}
