from typing import Optional, List, Dict, Any
import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from src.models.base import PhenixModel


class RidgeRegressionPipeline(PhenixModel):
    """
    Ridge Regression Model Pipeline with feature standardization and L2 regularization.
    Uses cross-validated hyperparameter tuning over regularization parameter alpha.
    """

    def __init__(
        self,
        alphas: Optional[List[float]] = None,
        log_transform_target: bool = True,
        standardize: bool = True,
        cv: int = 5,
    ):
        super().__init__(log_transform_target=log_transform_target)
        self.alphas = alphas or list(np.logspace(-3, 4, 30))
        self.standardize = standardize
        self.cv = cv
        self.scaler = StandardScaler() if standardize else None
        self.model: Optional[RidgeCV] = None
        self.best_alpha_: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'RidgeRegressionPipeline':
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        if self.log_transform_target and self.transformer is not None:
            y_fit = self.transformer.fit_transform(y_arr)
        else:
            y_fit = y_arr

        if self.standardize and self.scaler is not None:
            X_fit = self.scaler.fit_transform(X_arr)
        else:
            X_fit = X_arr

        # Handle cv folds vs sample count
        n_samples = len(X_fit)
        actual_cv = min(self.cv, max(2, n_samples // 5)) if n_samples >= 10 else None

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = RidgeCV(alphas=self.alphas, cv=actual_cv)
            self.model.fit(X_fit, y_fit)
        self.best_alpha_ = float(self.model.alpha_)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict() is called.")

        X_arr = np.asarray(X, dtype=np.float32)
        if self.standardize and self.scaler is not None:
            X_eval = self.scaler.transform(X_arr)
        else:
            X_eval = X_arr

        return self.model.predict(X_eval)
