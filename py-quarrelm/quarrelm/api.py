from numpy._core import ndarray

from quarrelm._core import quarrel_fit, quarrel_fit_path, SOLVER
from quarrelm._params import FitOptions, OLSResult, ElasticNetResult
from dataclasses import dataclass

import numpy as np
from narwhals.typing import IntoDataFrame


def ols(df: IntoDataFrame, target: str) -> OLSResult:
    solver = SOLVER.OLS
    fitopts = FitOptions()  # no valid params
    results = quarrel_fit(df, target, solver, fitopts)
    return results


def enet(
    df: IntoDataFrame,
    target: str,
    alpha: float,
    lambda_: float,
    max_iter: int,
    tol: float,
    penalty_factors: np.ndarray | None = None,
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
) -> ElasticNetResult:

    solver = SOLVER.ENET

    fitopts = FitOptions(
        alpha=alpha,
        lambda_=lambda_,
        max_iter=max_iter,
        tol=tol,
        penalty_factors=penalty_factors,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    results = quarrel_fit(df, target, solver, fitopts)

    return results


def enet_path(
    df: IntoDataFrame,
    target: str,
    alpha: float = 1.0,
    n_lambda: int = 100,
    lambda_min_ratio: float = 1e-4,
    max_iter: int = 10000,
    tol: float = 1e-7,
    penalty_factors: np.ndarray | None = None,
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
) -> dict:

    solver = SOLVER.ENET_PATH

    fitopts = FitOptions(
        alpha=alpha,
        n_lambda=n_lambda,
        lambda_min_ratio=lambda_min_ratio,
        max_iter=max_iter,
        tol=tol,
        penalty_factors=penalty_factors,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
    )

    results = quarrel_fit_path(df, target, solver, n_lambda, fitopts)

    return {
        "coef_matrix": results.coef_matrix,
        "lambdas": results.lambda_,
        "feature_names": results.feature_names,
        "n_iter": results.total_iters,
    }
