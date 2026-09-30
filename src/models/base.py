from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Computes standard regression benchmark metrics:
      - R^2 (Coefficient of Determination)
      - RMSE (Root Mean Squared Error)
      - MAE (Mean Absolute Error)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if len(y_true) == 0:
        return {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan")}

    # Check for NaNs
    valid_mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if not np.all(valid_mask):
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

    if len(y_true) < 2:
        return {"r2": 0.0, "rmse": 0.0, "mae": 0.0}

    # Variance check
    variance = np.var(y_true)
    if variance == 0.0 or np.isclose(variance, 0.0):
        r2 = 1.0 if np.allclose(y_true, y_pred) else 0.0
    else:
        r2 = float(r2_score(y_true, y_pred))

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
    }


class Log10TargetTransformer:
    """
    Transforms strictly positive phenotypic target traits (such as body mass,
    gestation length, litter size) into logarithmic log10 scale and back.
    In macroecology, phenotypic life-history traits span multiple orders of magnitude.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float64)
        clipped = np.clip(y_arr, self.eps, None)
        return np.log10(clipped)

    def inverse_transform(self, y_log: np.ndarray) -> np.ndarray:
        return np.power(10.0, np.asarray(y_log, dtype=np.float64))


class PhenixModel(ABC):
    """
    Unified abstract model interface for all PHENIX predictive architectures
    (Ridge, Random Forest, XGBoost, and PyTorch GNN).
    """

    def __init__(self, log_transform_target: bool = True):
        self.log_transform_target = log_transform_target
        self.transformer = Log10TargetTransformer() if log_transform_target else None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'PhenixModel':
        """Fits model to training features X and targets y."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predicts target trait values for features X."""
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Evaluates model predictions against ground truth y and returns metrics."""
        y_true = np.asarray(y, dtype=np.float64)
        if self.log_transform_target and self.transformer is not None:
            y_eval = self.transformer.fit_transform(y_true)
        else:
            y_eval = y_true

        y_pred = self.predict(X, **kwargs)
        return compute_regression_metrics(y_eval, y_pred)
