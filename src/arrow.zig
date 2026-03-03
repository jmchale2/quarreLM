const std = @import("std");

// https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions

pub const ArrowSchema = extern struct {
    format: [*c]const u8, // Arrow format string ("g" = float64, "l" = int64, etc.)
    name: [*c]const u8, // Column name (optional, may be null)
    metadata: [*c]const u8, // Metadata (optional, may be null)
    flags: i64, // Bitfield (nullable, dict-ordered, map-keys-sorted)
    n_children: i64,
    children: [*c][*c]ArrowSchema,
    dictionary: [*c]ArrowSchema,
    release: ?*const fn (*ArrowSchema) callconv(.c) void,
    private_data: ?*anyopaque,
};

pub const ArrowArray = extern struct {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: [*c]const ?*const anyopaque,
    children: [*c][*c]ArrowArray,
    dictionary: [*c]ArrowArray,
    release: ?*const fn (*ArrowArray) callconv(.c) void, // line 24
    private_data: ?*anyopaque,
};

// Arrow C Stream Interface — for iterating batches/columns
pub const ArrowArrayStream = extern struct {
    get_schema: ?*const fn (*ArrowArrayStream, *ArrowSchema) callconv(.c) c_int,
    get_next: ?*const fn (*ArrowArrayStream, *ArrowArray) callconv(.c) c_int,
    get_last_error: ?*const fn (*ArrowArrayStream) callconv(.c) [*c]const u8,
    release: ?*const fn (*ArrowArrayStream) callconv(.c) void,
    private_data: ?*anyopaque,
};

// ---------------------------------------------------------------
// Consumer helpers
// ---------------------------------------------------------------

pub const ArrowError = error{
    WrongFormat,
    NullBuffer,
    StreamError,
    SchemaError,
    HasNulls,
};

/// Extract a read-only f64 slice from an ArrowArray.
/// Expects format string "g" (float64, IEEE 754 double).
/// Rejects arrays with nulls (for now — regression needs complete data).
pub fn asFloat64Slice(array: *ArrowArray, schema: *ArrowSchema) ArrowError![]const f64 {
    // Check format
    const fmt = std.mem.span(schema.format);
    if (!std.mem.eql(u8, fmt, "g")) return ArrowError.WrongFormat;

    // Check for nulls — we don't handle them yet
    if (array.null_count > 0) return ArrowError.HasNulls;

    // Float64 arrays have 2 buffers: [0] = validity bitmap, [1] = data
    if (array.n_buffers < 2) return ArrowError.NullBuffer;
    const raw_ptr = array.buffers[1] orelse return ArrowError.NullBuffer;

    const length: usize = @intCast(array.length);
    const offset: usize = @intCast(array.offset);
    const data: [*]const f64 = @ptrCast(@alignCast(raw_ptr));

    return data[offset..][0..length];
}

test "ArrowArray struct size and alignment" {
    // ArrowArray must match C layout exactly
    // On 64-bit: 10 fields × 8 bytes = 80 bytes
    try std.testing.expectEqual(@as(usize, 80), @sizeOf(ArrowArray));
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(ArrowSchema));
}
