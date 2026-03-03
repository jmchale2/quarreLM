import numpy as np
import polars as pl
import pandas as pd
from sklearn.linear_model import LinearRegression

# Add parent to path so we can import zigregress
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
from quarrelm._core import ols, ols_simd, ping


def test_ping():
    """Verify Zig library loads."""
    assert ping() == 42, f"Expected 42, got {ping()}"
    print("PASS: ping")


def test_exact_recovery():
    """No noise — should recover exact coefficients."""
    df = pl.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [2.0, 1.0, 3.0, 2.0, 4.0],
            "y": [8.0, 7.0, 15.0, 14.0, 22.0],  # y = 2*x1 + 3*x2
        }
    )

    result = ols(df, target="y")
    print(f"  Coefficients: {result.coefficients}")
    assert abs(result.coefficients["x1"] - 2.0) < 1e-10
    assert abs(result.coefficients["x2"] - 3.0) < 1e-10
    print("PASS: exact_recovery")


def test_simd_exact_recovery():
    """No noise — should recover exact coefficients."""
    df = pl.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [2.0, 1.0, 3.0, 2.0, 4.0],
            "y": [8.0, 7.0, 15.0, 14.0, 22.0],  # y = 2*x1 + 3*x2
        }
    )

    result = ols_simd(df, target="y")
    print(f"  Coefficients: {result.coefficients}")
    assert abs(result.coefficients["x1"] - 2.0) < 1e-10
    assert abs(result.coefficients["x2"] - 3.0) < 1e-10
    print("PASS: simd_exact_recovery")


def test_vs_sklearn():
    """Compare with sklearn on noisy data."""
    np.random.seed(42)
    n = 1000
    p = 5
    true_coefs = np.array([1.5, -2.3, 0.7, 3.1, -0.5])

    X = np.random.randn(n, p)
    y = X @ true_coefs + np.random.randn(n) * 0.1

    # Fit with sklearn
    sk = LinearRegression(fit_intercept=False).fit(X, y)

    # Fit with zigregress via Polars
    data = {"y": y.tolist()}
    for i in range(p):
        data[f"x{i}"] = X[:, i].tolist()
    df = pl.DataFrame(data)

    result = ols(df, target="y")

    # Compare
    for i in range(p):
        zig_coef = result.coefficients[f"x{i}"]
        sk_coef = sk.coef_[i]
        diff = abs(zig_coef - sk_coef)
        print(f"  x{i}: zig={zig_coef:.6f}  sklearn={sk_coef:.6f}  diff={diff:.2e}")
        assert diff < 1e-6, f"Coefficient x{i} differs: {diff}"

    print("PASS: vs_sklearn")


def test_simd_vs_sklearn():
    """Compare with sklearn on noisy data."""
    np.random.seed(42)
    n = 1000
    p = 5
    true_coefs = np.array([1.5, -2.3, 0.7, 3.1, -0.5])

    X = np.random.randn(n, p)
    y = X @ true_coefs + np.random.randn(n) * 0.1

    # Fit with sklearn
    sk = LinearRegression(fit_intercept=False).fit(X, y)

    # Fit with zigregress via Polars
    data = {"y": y.tolist()}
    for i in range(p):
        data[f"x{i}"] = X[:, i].tolist()
    df = pl.DataFrame(data)

    result = ols_simd(df, target="y")

    # Compare
    for i in range(p):
        zig_coef = result.coefficients[f"x{i}"]
        sk_coef = sk.coef_[i]
        diff = abs(zig_coef - sk_coef)
        print(f"  x{i}: zig={zig_coef:.6f}  sklearn={sk_coef:.6f}  diff={diff:.2e}")
        assert diff < 1e-6, f"Coefficient x{i} differs: {diff}"

    print("PASS: simd_vs_sklearn")


def test_with_pandas():
    """Verify pandas DataFrames work via narwhals."""
    df = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [2.0, 1.0, 3.0, 2.0, 4.0],
            "y": [8.0, 7.0, 15.0, 14.0, 22.0],
        }
    )

    result = ols(df, target="y")
    assert abs(result.coefficients["x1"] - 2.0) < 1e-10
    assert abs(result.coefficients["x2"] - 3.0) < 1e-10
    print("PASS: with_pandas")


if __name__ == "__main__":
    test_ping()
    test_exact_recovery()
    test_simd_exact_recovery()
    test_vs_sklearn()
    test_simd_vs_sklearn()
    test_with_pandas()
    print("\nAll tests passed!")
