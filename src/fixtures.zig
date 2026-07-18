// fixtures.zig
const regression = @import("regression.zig");
const arrow = @import("arrow.zig");
const std = @import("std");
const common = @import("solvers/common.zig");

const OLSMethods = @import("solvers/ols.zig").Method;
const OLSOptions = @import("solvers/ols.zig").Options;

const EnetOptions = @import("solvers/enet.zig").Options;
const PathOptions = @import("solvers/enet.zig").PathOptions;

const StatsSpec = common.StatsSpec;
const SufficientStats = common.SufficientStats;
const StatsAccumulator = common.StatsAccumulator;

pub const inf = std.math.inf(f64);

// Unpenalized penalty factors and open (unconstrained) box bounds, by width.
pub const pf_ones_1: [1]f64 = @splat(1.0);
pub const lb_open_1: [1]f64 = @splat(-inf);
pub const ub_open_1: [1]f64 = @splat(inf);

pub const pf_ones_2: [2]f64 = @splat(1.0);
pub const lb_open_2: [2]f64 = @splat(-inf);
pub const ub_open_2: [2]f64 = @splat(inf);

pub const pf_ones_3: [3]f64 = @splat(1.0);
pub const lb_open_3: [3]f64 = @splat(-inf);
pub const ub_open_3: [3]f64 = @splat(inf);

pub const pf_ones_4: [4]f64 = @splat(1.0);
pub const lb_open_4: [4]f64 = @splat(-inf);
pub const ub_open_4: [4]f64 = @splat(inf);

pub const ols_defaults = OLSOptions{
    .method = OLSMethods.auto,
};

pub const enet_defaults = EnetOptions{
    .lambda = 0.01,
    .alpha = 0.5,
    .penalty_factors = &pf_ones_2,
    .lower_bounds = &lb_open_2,
    .upper_bounds = &ub_open_2,
    .tol = 1e-10,
};
pub const path_defaults = PathOptions{
    .alpha = 1.0,
    .penalty_factors = &pf_ones_2,
    .lower_bounds = &lb_open_2,
    .upper_bounds = &ub_open_2,
    .n_lambda = 20,
    .lambda_min_ratio = 1e-4,
    .tol = 1e-10,
    .max_iter = 10_000,
};

pub const exact_2col = struct {
    // y = 2*x1 + 3*x2 exactly (no noise): recovery tests expect [2, 3]
    pub const x1 = [_]f64{ 1, 2, 3, 4, 5 };
    pub const x2 = [_]f64{ 2, 1, 3, 2, 4 };
    pub const y = [_]f64{ 8, 7, 15, 14, 22 };
    pub const cols = [_][]const f64{ &x1, &x2 };
};

pub const sparse_2col = struct {
    // y = 3*x1 exactly; x2 is small and truly irrelevant (coef 0).
    // Drives sparsity/penalty tests: lasso should zero x2.
    pub const x1 = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    pub const x2 = [_]f64{ 0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1 };
    pub const y = [_]f64{ 3, 6, 9, 12, 15, 18, 21, 24, 27, 30 };
    pub const cols = [_][]const f64{ &x1, &x2 };
};

pub const collinear_2col = struct {
    // x1 + x2 = 11 on every row, so y = x1 + x2 is the constant 11.
    // Drives ridge/shrinkage and box-constraint tests.
    pub const x1 = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    pub const x2 = [_]f64{ 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };
    pub const y = [_]f64{ 11, 11, 11, 11, 11, 11, 11, 11, 11, 11 };
    pub const cols = [_][]const f64{ &x1, &x2 };
};

/// Deterministic sin/cos regression data with n rows and up to 3 columns.
/// y = 2*x1 + 3*x2 + small deterministic noise; x3 is irrelevant (true coef 0).
/// Larger, path-oriented fixture — instantiate with `sinCos(n).init()`.
pub fn sinCos(comptime n: usize) type {
    return struct {
        x1: [n]f64,
        x2: [n]f64,
        x3: [n]f64,
        y: [n]f64,

        pub fn init() @This() {
            var self: @This() = undefined;
            for (0..n) |i| {
                const t: f64 = @floatFromInt(i);
                self.x1[i] = @sin(t * 0.1) * 3.0 + t * 0.01;
                self.x2[i] = @cos(t * 0.07) * 2.0 - t * 0.005;
                self.x3[i] = @sin(t * 0.3) * 0.5;
                self.y[i] = 2.0 * self.x1[i] + 3.0 * self.x2[i] + @sin(t * 1.7) * 0.1;
            }
            return self;
        }
    };
}
pub fn statsFrom(alloc: std.mem.Allocator, cols: []const []const f64, y: []const f64, spec: StatsSpec) !SufficientStats {
    var acc = try StatsAccumulator.init(alloc, cols.len, spec);
    acc.update(cols, y);
    return acc.finalize();
}

// mock a stream
pub const mock = struct {
    const n = 8;
    var col0 = [_]f64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var col1 = [_]f64{ 8, 7, 6, 5, 4, 3, 2, 1 };
    var yv = [_]f64{ 10, 11, 12, 13, 14, 15, 16, 17 }; // 2*x0 + x1

    fn releaseSchema(s: *arrow.ArrowSchema) callconv(.c) void {
        s.release = null;
    }
    fn releaseArray(a: *arrow.ArrowArray) callconv(.c) void {
        a.release = null;
    }

    var child_schemas = [_]arrow.ArrowSchema{
        .{ .format = "g", .name = null, .metadata = null, .flags = 0, .n_children = 0, .children = null, .dictionary = null, .release = null, .private_data = null },
        .{ .format = "g", .name = null, .metadata = null, .flags = 0, .n_children = 0, .children = null, .dictionary = null, .release = null, .private_data = null },
    };
    var schema_children = [_][*c]arrow.ArrowSchema{ &child_schemas[0], &child_schemas[1] };

    var bufs0 = [_]?*const anyopaque{ null, &col0 };
    var bufs1 = [_]?*const anyopaque{ null, &col1 };
    var child_arrays = [_]arrow.ArrowArray{
        .{ .length = n, .null_count = 0, .offset = 0, .n_buffers = 2, .n_children = 0, .buffers = &bufs0, .children = null, .dictionary = null, .release = null, .private_data = null },
        .{ .length = n, .null_count = 0, .offset = 0, .n_buffers = 2, .n_children = 0, .buffers = &bufs1, .children = null, .dictionary = null, .release = null, .private_data = null },
    };
    var batch_children = [_][*c]arrow.ArrowArray{ &child_arrays[0], &child_arrays[1] };

    fn getSchema(_: *arrow.ArrowArrayStream, out: *arrow.ArrowSchema) callconv(.c) c_int {
        out.* = .{ .format = "+s", .name = null, .metadata = null, .flags = 0, .n_children = 2, .children = &schema_children, .dictionary = null, .release = releaseSchema, .private_data = null };
        return 0;
    }

    var served: bool = false;
    fn getNext(_: *arrow.ArrowArrayStream, out: *arrow.ArrowArray) callconv(.c) c_int {
        if (served) {
            out.release = null;
            return 0;
        } // end of stream
        served = true;
        out.* = .{ .length = n, .null_count = 0, .offset = 0, .n_buffers = 0, .n_children = 2, .buffers = null, .children = &batch_children, .dictionary = null, .release = releaseArray, .private_data = null };
        return 0;
    }

    pub fn makeStream() arrow.ArrowArrayStream {
        served = false;
        return .{ .get_schema = getSchema, .get_next = getNext, .get_last_error = null, .release = null, .private_data = null };
    }

    var y_bufs = [_]?*const anyopaque{ null, &yv };
    pub var y_array = arrow.ArrowArray{ .length = n, .null_count = 0, .offset = 0, .n_buffers = 2, .n_children = 0, .buffers = &y_bufs, .children = null, .dictionary = null, .release = null, .private_data = null };
    pub var y_schema = arrow.ArrowSchema{ .format = "g", .name = null, .metadata = null, .flags = 0, .n_children = 0, .children = null, .dictionary = null, .release = null, .private_data = null };
};
