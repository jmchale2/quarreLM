# TODO:  Parameter dataclasses that turn to a global superset to pass to quarrel_fit
# capi will convert to _CFitOptions internally.

from dataclasses import dataclass
import numpy as np


@dataclass
class OLSResult:
    """Result of an OLS fit."""

    coefficients: dict[str, float]  # feature_name -> coefficient
    feature_names: list[str]
    coef_array: np.ndarray  # raw coefficient vector


@dataclass
class ElasticNetResult:
    coefficients: dict[str, float]  # original scale
    feature_names: list[str]  # feature names at fit
    penalty_factors: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    coef_array: np.ndarray  # original scale
    lambda_: float
    alpha: float
    n_iter: int

    means: np.ndarray | None = None  # feature means from fit
    sds: np.ndarray | None = None  # feature sds from fit
    intercept: float | None = None  # recovered from unstandardization


@dataclass
class ElasticNetPathResult:
    coefficients: dict[str, np.ndarray]  # original scale
    feature_names: list[str]  # feature names at fit
    penalty_factors: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    coef_matrix: np.ndarray  # original scale
    lambda_: np.ndarray
    alpha: float
    n_iters: np.ndarray
    total_iters: int

    means: np.ndarray | None = None  # feature means from fit
    sds: np.ndarray | None = None  # feature sds from fit
    intercept: float | None = None  # recovered from unstandardization


@dataclass
class FitOptions:
    lambda_: float | None = 0.01
    alpha: float = 0.5
    n_lambda: int | None = 100
    lambda_min_ratio: float | None = -1
    penalty_factors: np.ndarray | None = None
    lower_bounds: np.ndarray | None = None
    upper_bounds: np.ndarray | None = None
    warm_start: np.ndarray | None = None

    tol: float = 1e-8
    max_iter: int = 10_000
