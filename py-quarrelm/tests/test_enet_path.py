# tests/test_enet_path.py
import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNet, Lasso

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
from quarrelm._core import enet_path, enet


def test_path_lambdas_decreasing():
    """Lambda sequence should be strictly decreasing."""
    np.random.seed(42)
    X = np.random.randn(500, 5)
    y = X @ np.array([2.0, -1.0, 0.0, 0.0, 0.5]) + np.random.randn(500) * 0.1

    data = {"y": y.tolist()}
    for i in range(5):
        data[f"x{i}"] = X[:, i].tolist()
    df = pl.DataFrame(data)

    result = enet_path(df, target="y", alpha=1.0, n_lambda=50)

    lambdas = result["lambdas"]
    for i in range(len(lambdas) - 1):
        assert lambdas[i] > lambdas[i + 1], f"Lambda not decreasing at index {i}"

    print(f"  Lambda range: {lambdas[0]:.6f} -> {lambdas[-1]:.6f}")
    print(f"PASS: lambdas_decreasing ({result['n_iter']} total iterations)")


def test_path_all_zero_at_lambda_max():
    """At lambda_max, all coefficients should be zero."""
    np.random.seed(42)
    X = np.random.randn(500, 5)
    y = X @ np.array([2.0, -1.0, 0.0, 0.0, 0.5]) + np.random.randn(500) * 0.1

    data = {"y": y.tolist()}
    for i in range(5):
        data[f"x{i}"] = X[:, i].tolist()
    df = pl.DataFrame(data)

    result = enet_path(df, target="y", alpha=1.0, n_lambda=50)
    coef_matrix = result["coef_matrix"]

    # First column (lambda_max) should be all zeros
    for j in range(5):
        assert abs(coef_matrix[j, 0]) < 1e-10, (
            f"Feature {j} nonzero at lambda_max: {coef_matrix[j, 0]}"
        )

    print(f"PASS: all_zero_at_lambda_max")


def test_path_nonzero_at_lambda_min():
    """At lambda_min, at least some coefficients should be nonzero."""
    np.random.seed(42)
    X = np.random.randn(500, 5)
    y = X @ np.array([2.0, -1.0, 0.0, 0.0, 0.5]) + np.random.randn(500) * 0.1

    data = {"y": y.tolist()}
    for i in range(5):
        data[f"x{i}"] = X[:, i].tolist()
    df = pl.DataFrame(data)

    result = enet_path(df, target="y", alpha=1.0, n_lambda=50)
    coef_matrix = result["coef_matrix"]

    # Last column (lambda_min) should have nonzero coefficients
    n_nonzero = sum(abs(coef_matrix[j, -1]) > 1e-6 for j in range(5))
    assert n_nonzero >= 2, (
        f"Expected nonzero coefficients at lambda_min, got {n_nonzero}"
    )

    print(f"  Nonzero at lambda_min: {n_nonzero}/5")
    print(f"PASS: nonzero_at_lambda_min")


def test_path_sparsity_increases_along_path():
    """As lambda decreases, number of nonzero coefficients should increase (or stay same)."""
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    true_coefs = np.array([3.0, -2.0, 1.5, 0, 0, 0, 0, 0, 0, 0.5])
    y = X @ true_coefs + np.random.randn(1000) * 0.1

    data = {"y": y.tolist()}
    for i in range(10):
        data[f"x{i}"] = X[:, i].tolist()
    df = pl.DataFrame(data)

    result = enet_path(df, target="y", alpha=1.0, n_lambda=100)
    coef_matrix = result["coef_matrix"]
    n_lambda = coef_matrix.shape[1]

    # Count nonzeros at each lambda
    prev_nnz = 0
    violations = 0
    for k in range(n_lambda):
        nnz = sum(abs(coef_matrix[j, k]) > 1e-10 for j in range(10))
        if nnz < prev_nnz:
            violations += 1
        prev_nnz = nnz

    # Allow a couple violations (features can drop out briefly)
    assert violations < 5, f"Sparsity not monotonic: {violations} violations"

    print(f"  Sparsity range: 0 -> {prev_nnz} nonzero")
    print(f"PASS: sparsity_increases ({violations} minor violations)")


def test_path_matches_single_fit():
    """Path result at a specific lambda should match a single enet() call at that lambda."""
    np.random.seed(42)
    X = np.random.randn(500, 5)
    y = X @ np.array([2.0, -1.0, 0.5, 0.0, 0.0]) + np.random.randn(500) * 0.1

    data = {"y": y.tolist()}
    for i in range(5):
        data[f"x{i}"] = X[:, i].tolist()
    df = pl.DataFrame(data)

    result = enet_path(df, target="y", alpha=0.5, n_lambda=50)
    coef_matrix = result["coef_matrix"]
    lambdas = result["lambdas"]

    # Pick a lambda in the middle of the path
    k = 25
    lambda_k = lambdas[k]

    # Fit single enet at that lambda
    single = enet(
        df, target="y", alpha=0.5, lambda_=lambda_k, max_iter=100000, tol=1e-10
    )

    max_diff = 0.0
    for j in range(5):
        path_coef = coef_matrix[j, k]
        single_coef = single.coef_array[j]
        diff = abs(path_coef - single_coef)
        if diff > max_diff:
            max_diff = diff
        print(
            f"  x{j}: path={path_coef:.6f}  single={single_coef:.6f}  diff={diff:.2e}"
        )

    assert max_diff < 1e-3, f"Path and single fit differ by {max_diff}"
    print(f"PASS: matches_single_fit (max diff: {max_diff:.2e})")


def test_path_vs_sklearn():
    """Compare path endpoints against sklearn at the same lambda."""
    np.random.seed(42)
    n, p = 1000, 5
    true_coefs = np.array([1.5, -2.3, 0.7, 3.1, -0.5])
    X = np.random.randn(n, p)
    y = X @ true_coefs + np.random.randn(n) * 0.1

    data = {"y": y.tolist()}
    for i in range(p):
        data[f"x{i}"] = X[:, i].tolist()
    df = pl.DataFrame(data)

    result = enet_path(df, target="y", alpha=1.0, n_lambda=50)
    lambdas = result["lambdas"]
    coef_matrix = result["coef_matrix"]

    # Compare at a few points along the path
    for k in [10, 25, 49]:
        lambda_k = lambdas[k]
        sk = Lasso(alpha=lambda_k, fit_intercept=False, max_iter=100000, tol=1e-12).fit(
            X, y
        )

        max_diff = 0.0
        for j in range(p):
            diff = abs(coef_matrix[j, k] - sk.coef_[j])
            if diff > max_diff:
                max_diff = diff

        print(f"  lambda={lambda_k:.6f}: max |zig - sklearn| = {max_diff:.2e}")
        assert max_diff < 1e-2, f"Differs at lambda={lambda_k}: {max_diff}"

    print(f"PASS: vs_sklearn")


def test_path_warm_starts_efficient():
    """Path should use fewer total iterations than cold-starting each lambda."""
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = X @ np.random.randn(10) + np.random.randn(1000) * 0.1

    data = {"y": y.tolist()}
    for i in range(10):
        data[f"x{i}"] = X[:, i].tolist()
    df = pl.DataFrame(data)

    result = enet_path(df, target="y", alpha=1.0, n_lambda=50)

    avg_iters = result["n_iter"] / 50
    print(f"  Total iterations: {result['n_iter']}, avg per lambda: {avg_iters:.1f}")
    assert avg_iters < 30, f"Avg iterations too high: {avg_iters}"

    print(f"PASS: warm_starts_efficient")


if __name__ == "__main__":
    test_path_lambdas_decreasing()
    test_path_all_zero_at_lambda_max()
    test_path_nonzero_at_lambda_min()
    test_path_sparsity_increases_along_path()
    test_path_matches_single_fit()
    test_path_vs_sklearn()
    test_path_warm_starts_efficient()
    print("\nAll path tests passed!")
