from typing import Optional, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from src.models.base import PhenixModel


class RandomForestPipeline(PhenixModel):
    """
    Random Forest Regressor Pipeline for non-linear feature interactions.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 4,
        random_state: int = 42,
        log_transform_target: bool = True,
        n_jobs: int = -1,
    ):
        super().__init__(log_transform_target=log_transform_target)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'RandomForestPipeline':
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        if self.log_transform_target and self.transformer is not None:
            y_fit = self.transformer.fit_transform(y_arr)
        else:
            y_fit = y_arr

        self.model.fit(X_arr, y_fit)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict() is called.")
        X_arr = np.asarray(X, dtype=np.float32)
        return self.model.predict(X_arr)


class XGBoostPipeline(PhenixModel):
    """
    Gradient Boosted Decision Trees Pipeline powered by XGBoost.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        log_transform_target: bool = True,
        n_jobs: int = -1,
    ):
        super().__init__(log_transform_target=log_transform_target)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model: Optional[Any] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'XGBoostPipeline':
        import xgboost as xgb

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        if self.log_transform_target and self.transformer is not None:
            y_fit = self.transformer.fit_transform(y_arr)
        else:
            y_fit = y_arr

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X_arr, y_fit)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict() is called.")
        X_arr = np.asarray(X, dtype=np.float32)
        return self.model.predict(X_arr)
