from pathlib import Path


def repo_root() -> Path:
    for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (p / ".git").exists():
            return p
    raise RuntimeError("not inside the quarreLM repo")


DATA_DIR = repo_root() / "benchmarks" / "data"
