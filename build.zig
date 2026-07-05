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

    const lib_module = b.createModule(.{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    lib_module.linkSystemLibrary("openblas", .{ .use_pkg_config = .yes });

    const lib = b.addLibrary(.{
        .name = "quarrelm",
        .linkage = .dynamic,
        .root_module = lib_module,
        .use_llvm = true, // currently non-llvm build fail to pass through ctypes correctly
    });

    lib.root_module.addOptions("build_options", options);

    b.installArtifact(lib);

    const blas_module = b.createModule(.{
        .root_source_file = b.path("src/blas.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    blas_module.linkSystemLibrary("openblas", .{ .use_pkg_config = .yes });

    const lib_tests = b.addTest(.{ .root_module = lib_module });

    const blas_tests = b.addTest(.{ .root_module = blas_module });

    const run_lib_tests = b.addRunArtifact(lib_tests);

    const run_blas_tests = b.addRunArtifact(blas_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&run_blas_tests.step);

    const blas_test_step = b.step("test-blas", "Run Blas Tests Directly");
    blas_test_step.dependOn(&run_blas_tests.step);
}
