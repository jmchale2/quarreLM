const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const git_response = b.run(
        &[_][]const u8{ "git", "describe", "--tags", "--always", "--dirty" },
    );
    const version_raw = std.mem.trim(u8, git_response, " \n\r");
    const version = b.allocator.dupe(u8, version_raw) catch unreachable;

    const options = b.addOptions();
    options.addOption([]const u8, "version", version);

    const lib = b.addLibrary(.{
        .name = "quarreLM",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });

    lib.root_module.addOptions("build_options", options);

    b.installArtifact(lib);

    const arrow_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/arrow.zig"),
        .target = target,
        .optimize = optimize,
    }) });

    const regression_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/regression.zig"),
        .target = target,
        .optimize = optimize,
    }) });

    const run_arrow_tests = b.addRunArtifact(arrow_tests);
    const run_regression_tests = b.addRunArtifact(regression_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_arrow_tests.step);
    test_step.dependOn(&run_regression_tests.step);
}
