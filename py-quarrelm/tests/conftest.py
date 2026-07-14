import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def data_dir():
    for p in Path(__file__).resolve().parents:
        if (p / ".git").exists():
            return p / "benchmarks" / "data"
    raise RuntimeError("repo root not found")


@pytest.fixture(scope="session")
def fixtures_dir():
    return Path(__file__).parent / "fixtures"
