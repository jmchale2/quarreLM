// fixtures.zig
const regression = @import("regression.zig");
const arrow = @import("arrow.zig");
const std = @import("std");

pub const inf = std.math.inf(f64);

pub const pf_ones_2: [2]f64 = @splat(1.0);
pub const lb_open_3: [3]f64 = @splat(-inf);

pub const pf_ones_3: [3]f64 = @splat(1.0);
pub const lb_open_2: [2]f64 = @splat(-inf);
pub const ub_open_2: [2]f64 = @splat(inf);
pub const ub_open_3: [3]f64 = @splat(inf);

pub const enet_defaults = regression.EnetOptions{
    .lambda = 0.01,
    .alpha = 0.5,
    .penalty_factors = &pf_ones_2,
    .lower_bounds = &lb_open_2,
    .upper_bounds = &ub_open_2,
    .tol = 1e-10,
};
pub const path_defaults = regression.PathOptions{
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
