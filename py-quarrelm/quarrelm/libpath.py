from enum import verify
import sys
import ctypes
from pathlib import Path

from os import environ


def _lib_name() -> str:
    if sys.platform == "win32":
        return "quarrelm.dll"
    elif sys.platform == "darwin":
        return "libquarrelm.dylib"
    return "libquarrelm.so"


def _find_lib() -> ctypes.CDLL:
    lib_name = _lib_name()
    libpath = Path(__file__).parents[2].resolve()

    env_var = environ.get("QUARRELM_LIB_PATH", None)

    if env_var:
        p = Path(env_var)
        if p.is_dir():
            p = p / lib_name
        candidates = [p]
    else:
        candidates = [
            # wheel build: prepend Path(__file__).parent / "lib" when packaging
            (libpath / f"zig-out/lib/{lib_name}"),
            (libpath / f"{lib_name}"),
        ]

    for path in candidates:
        if path.exists():
            return ctypes.CDLL(str(path))

    if env_var:
        raise FileNotFoundError(
            f"QUARRELM_LIB_PATH is set to {env_var!r} but {candidates[0]} does not exist"
        )
    raise FileNotFoundError(
        f"Could not find {lib_name}. Searched: {[str(p) for p in candidates]}. "
        "Build it with `zig build -Doptimize=ReleaseFast`, or set QUARRELM_LIB_PATH."
    )


def _verify_lib(lib: ctypes.CDLL) -> None:
    lib.quarrel_ping.restype = ctypes.c_int
    if lib.quarrel_ping() != 42:
        raise ImportError(
            "libquarrelm loaded but quarrel_ping() failed — wrong or corrupt library"
        )

    lib.quarrel_abi_probe.restype = ctypes.c_int
    lib.quarrel_abi_probe.argtypes = (
        [ctypes.c_void_p] * 7 + [ctypes.c_int] + [ctypes.c_double] * 3 + [ctypes.c_int]
    )
    buf = ctypes.c_double(0.0)
    p = ctypes.cast(ctypes.byref(buf), ctypes.c_void_p)
    ok = lib.quarrel_abi_probe(p, p, p, p, p, p, p, 42, 1.5, 2.5, 3.5, 7)
    if ok != 5:
        raise ImportError(
            f"libquarrelm ABI check failed ({ok}/5 args intact) — the library was likely "
            "built with a miscompiling backend; rebuild with -Doptimize=ReleaseFast or -fllvm "
            "(see https://codeberg.org/ziglang/zig/issues/36038)"
        )


_lib = _find_lib()
