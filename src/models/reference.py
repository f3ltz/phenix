"""
Reference (non-ML) predictors that every benchmark model should be compared against:
  - GlobalMeanModel: predicts the training-set mean (the R^2 = 0 reference).
  - PhyloBMImputer: phylogenetic imputation under Brownian motion, i.e. the conditional
    expectation of a test species' trait given the training species' traits and the tree
    (Garland & Ives 2000; the standard comparative-methods baseline for trait prediction).
"""

from typing import Optional
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from src.models.base import PhenixModel


class GlobalMeanModel(PhenixModel):
    """Predicts the mean log10 trait of the training species for every test species."""

    def __init__(self, log_transform_target: bool = True):
        super().__init__(log_transform_target=log_transform_target)
        self.mean_: Optional[float] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'GlobalMeanModel':
        y_arr = np.asarray(y, dtype=np.float64)
        y_fit = self.transformer.fit_transform(y_arr) if self.log_transform_target and self.transformer else y_arr
        self.mean_ = float(np.mean(y_fit))
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted or self.mean_ is None:
            raise RuntimeError("Model must be fitted before predict() is called.")
        return np.full(len(X), self.mean_)


class PhyloBMImputer(PhenixModel):
    """
    Brownian-motion phylogenetic imputation (ignores features X).

    On an ultrametric tree with root age T, the BM covariance between species i and j is the
    length of their shared root-to-MRCA path: V_ij = T - D_ij / 2, where D is the patristic matrix.
    Prediction for test species:
        y_hat_test = a + V_test,train  V_train,train^-1 (y_train - a)
    where a is the GLS (phylogenetically weighted) mean of the training species.
    """

    def __init__(
        self,
        patristic_matrix: np.ndarray,
        log_transform_target: bool = True,
        jitter: float = 1e-6,
    ):
        super().__init__(log_transform_target=log_transform_target)
        D = np.asarray(patristic_matrix, dtype=np.float64)
        self.root_age = float(D.max() / 2.0)
        self.V = self.root_age - D / 2.0
        self.jitter = jitter
        self._train_idx: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._gls_mean: Optional[float] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_indices: Optional[np.ndarray] = None,
        **kwargs,
    ) -> 'PhyloBMImputer':
        if train_indices is None:
            raise ValueError("PhyloBMImputer needs train_indices to locate species on the tree.")
        y_arr = np.asarray(y, dtype=np.float64)
        y_fit = self.transformer.fit_transform(y_arr) if self.log_transform_target and self.transformer else y_arr

        idx = np.asarray(train_indices, dtype=int)
        V_tr = self.V[np.ix_(idx, idx)] + self.jitter * self.root_age * np.eye(len(idx))
        factor = cho_factor(V_tr, lower=True)

        ones = np.ones(len(idx))
        V_inv_ones = cho_solve(factor, ones)
        self._gls_mean = float(V_inv_ones @ y_fit / (V_inv_ones @ ones))
        self._alpha = cho_solve(factor, y_fit - self._gls_mean)
        self._train_idx = idx
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, test_indices: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if not self.is_fitted or self._alpha is None:
            raise RuntimeError("Model must be fitted before predict() is called.")
        if test_indices is None:
            raise ValueError("PhyloBMImputer needs test_indices to locate species on the tree.")
        V_te_tr = self.V[np.ix_(np.asarray(test_indices, dtype=int), self._train_idx)]
        return self._gls_mean + V_te_tr @ self._alpha
