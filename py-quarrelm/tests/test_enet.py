import numpy as np
import polars as pl
from sklearn.linear_model import ElasticNet, Lasso, Ridge

import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
from quarrelm._core import enet


def test_enet_zero_penalty_recovers_ols():
    """With lambda=0, elastic net should recover OLS coefficients."""
    np.random.seed(42)
    n = 500
    p = 5
    true_coefs = np.array([2.0, -1.5, 3.0, 0.0, -0.5])

    X = np.random.randn(n, p)
    y = X @ true_coefs + np.random.randn(n) * 0.1

    data = {"y": y.tolist()}
    for i in range(p):
        data[f"x{i}"] = X[:, i].tolist()
    df = pl.DataFrame(data)

    result = enet(df, target="y", alpha=1.0, lambda_=0.0, max_iter=10000, tol=1e-10)

    from sklearn.linear_model import LinearRegression

    sk = LinearRegression(fit_intercept=False).fit(X, y)

    for i in range(p):
        diff = abs(result.coefficients[f"x{i}"] - sk.coef_[i])
        print(
            f"  x{i}: enet={result.coefficients[f'x{i}']:.6f}  ols={sk.coef_[i]:.6f}  diff={diff:.2e}"
        )
        assert diff < 1e-4, f"x{i} differs from OLS: {diff}"

    print(f"PASS: zero_penalty_recovers_ols ({result.n_iter} iterations)")


def test_enet_lasso_sparsity():
    """Lasso (alpha=1) with large lambda should zero out weak features."""
    np.random.seed(42)
    n = 500
    p = 5
    # Only first two features matter
    true_coefs = np.array([3.0, -2.0, 0.0, 0.0, 0.0])

    X = np.random.randn(n, p)
    y = X @ true_coefs + np.random.randn(n) * 0.1

    data = {"y": y.tolist()}
    for i in range(p):
        data[f"x{i}"] = X[:, i].tolist()
    df = pl.DataFrame(data)

    # Lambda large enough to kill the zero coefficients
    result = enet(df, target="y", alpha=1.0, lambda_=0.1, max_iter=10000, tol=1e-10)

    print(f"  Coefficients: {result.coefficients}")

    # The truly zero features should be near zero (or exactly zero)
    for i in [2, 3, 4]:
        val = abs(result.coefficients[f"x{i}"])
        print(f"  x{i} (should be ~0): {val:.6f}")
        assert val < 0.05, f"x{i} should be sparse but got {val}"

    # The real features should still be nonzero
    assert abs(result.coefficients["x0"]) > 1.0, "x0 should be large"
    assert abs(result.coefficients["x1"]) > 1.0, "x1 should be large"

    print(f"PASS: lasso_sparsity ({result.n_iter} iterations)")


def test_enet_ridge_no_zeros():
    """Ridge (alpha=0) should shrink but never zero out coefficients."""
    np.random.seed(42)
    n = 500
    p = 5
    true_coefs = np.array([1.0, -1.0, 0.5, -0.5, 0.1])

    X = np.random.randn(n, p)
    y = X @ true_coefs + np.random.randn(n) * 0.1

    data = {"y": y.tolist()}
    for i in range(p):
        data[f"x{i}"] = X[:, i].tolist()
    df = pl.DataFrame(data)

    # alpha=0 is pure ridge — use small epsilon to avoid div by zero in lambda*alpha
    result = enet(df, target="y", alpha=0.001, lambda_=0.5, max_iter=10000, tol=1e-10)

    print(f"  Coefficients: {result.coefficients}")

    # All coefficients should be nonzero (ridge shrinks but doesn't zero)
    for i in range(p):
        val = abs(result.coefficients[f"x{i}"])
        print(f"  x{i}: {val:.6f}")
        assert val > 1e-6, f"x{i} should be nonzero under ridge"

    print(f"PASS: ridge_no_zeros ({result.n_iter} iterations)")


def test_enet_vs_sklearn_lasso():
    """Compare lasso coefficients against sklearn."""
    np.random.seed(42)
    n = 1000
    p = 5
    true_coefs = np.array([1.5, -2.3, 0.7, 3.1, -0.5])

    X = np.random.randn(n, p)
    y = X @ true_coefs + np.random.randn(n) * 0.1

    # sklearn Lasso: objective is (1/2N)||y-Xb||^2 + alpha*||b||_1
    # so sklearn alpha = our lambda
    sk_lambda = 0.05
    sk = Lasso(alpha=sk_lambda, fit_intercept=False, max_iter=100000, tol=1e-12).fit(
        X, y
    )

    data = {"y": y.tolist()}
    for i in range(p):
        data[f"x{i}"] = X[:, i].tolist()
    df = pl.DataFrame(data)

    result = enet(
        df, target="y", alpha=1.0, lambda_=sk_lambda, max_iter=100000, tol=1e-12
    )

    for i in range(p):
        zig_coef = result.coefficients[f"x{i}"]
        sk_coef = sk.coef_[i]
        diff = abs(zig_coef - sk_coef)
        print(f"  x{i}: zig={zig_coef:.6f}  sklearn={sk_coef:.6f}  diff={diff:.2e}")
        assert diff < 1e-3, f"x{i} differs: {diff}"

    print(f"PASS: vs_sklearn_lasso ({result.n_iter} iterations)")


def test_enet_vs_sklearn_elasticnet():
    """Compare elastic net coefficients against sklearn."""
    np.random.seed(42)
    n = 1000
    p = 5
    true_coefs = np.array([1.5, -2.3, 0.7, 3.1, -0.5])

    X = np.random.randn(n, p)
    y = X @ true_coefs + np.random.randn(n) * 0.1

    our_lambda = 0.05
    our_alpha = 0.5

    # sklearn ElasticNet: (1/2N)||y-Xb||^2 + alpha*l1_ratio*||b||_1 + alpha*(1-l1_ratio)/2*||b||^2
    # so sklearn alpha = our lambda, sklearn l1_ratio = our alpha
    sk = ElasticNet(
        alpha=our_lambda,
        l1_ratio=our_alpha,
        fit_intercept=False,
        max_iter=100000,
        tol=1e-12,
    ).fit(X, y)

    data = {"y": y.tolist()}
    for i in range(p):
        data[f"x{i}"] = X[:, i].tolist()
    df = pl.DataFrame(data)

    result = enet(
        df, target="y", alpha=our_alpha, lambda_=our_lambda, max_iter=100000, tol=1e-12
    )

    for i in range(p):
        zig_coef = result.coefficients[f"x{i}"]
        sk_coef = sk.coef_[i]
        diff = abs(zig_coef - sk_coef)
        print(f"  x{i}: zig={zig_coef:.6f}  sklearn={sk_coef:.6f}  diff={diff:.2e}")
        assert diff < 1e-3, f"x{i} differs: {diff}"

    print(f"PASS: vs_sklearn_elasticnet ({result.n_iter} iterations)")


def test_enet_convergence():
    """Verify convergence is reached before max_iter on easy problems."""
    np.random.seed(42)
    n = 500
    p = 3
    X = np.random.randn(n, p)
    y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(n) * 0.1

    data = {"y": y.tolist()}
    for i in range(p):
        data[f"x{i}"] = X[:, i].tolist()
    df = pl.DataFrame(data)

    result = enet(df, target="y", alpha=0.5, lambda_=0.01, max_iter=10000, tol=1e-7)

    print(f"  Converged in {result.n_iter} iterations")
    assert result.n_iter < 1000, f"Too many iterations: {result.n_iter}"

    print(f"PASS: convergence ({result.n_iter} iterations)")


if __name__ == "__main__":
    test_enet_zero_penalty_recovers_ols()
    test_enet_lasso_sparsity()
    test_enet_ridge_no_zeros()
    test_enet_vs_sklearn_lasso()
    test_enet_vs_sklearn_elasticnet()
    test_enet_convergence()
    print("\nAll elastic net tests passed!")
