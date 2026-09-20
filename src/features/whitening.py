from typing import Tuple, Optional
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from src.data.schemas import ID_COL


class CholeskyWhitener:
    """
    Applies Cholesky whitening (X* = X L^-T) to decorrelate features into white-noise space,
    satisfying Cov(X*) = I.
    """

    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.L_inv: Optional[np.ndarray] = None
        self.feature_cols: list = []

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fits Cholesky whitening on input features and returns whitened dataframe and covariance.
        """
        if ID_COL not in df.columns:
            raise ValueError(f"Dataframe missing '{ID_COL}'")

        taxa = df[ID_COL].values
        self.feature_cols = [c for c in df.columns if c != ID_COL]
        X = df[self.feature_cols].values.astype(np.float64)

        N, P = X.shape
        # Center and scale
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0.0] = 1.0
        Z = (X - self.mean) / self.std

        # Sample covariance
        cov = (Z.T @ Z) / (N - 1)
        # Ridge regularize for numerical stability and positive definiteness
        cov_reg = cov + self.epsilon * np.eye(P)

        # Cholesky factor: cov_reg = L @ L.T
        try:
            L = cholesky(cov_reg, lower=True)
        except np.linalg.LinAlgError:
            # Fallback: eigenvalue clipping if strongly ill-conditioned
            eigvals, eigvecs = np.linalg.eigh(cov_reg)
            eigvals = np.maximum(eigvals, 1e-4)
            cov_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L = cholesky(cov_reg, lower=True)

        # L_inv = L^-1
        self.L_inv = solve_triangular(L, np.eye(P), lower=True)

        # Whitened features: Z* = Z @ (L^-1).T
        Z_whitened = Z @ self.L_inv.T

        col_names = [f"whiten_{c}" for c in self.feature_cols]
        df_whitened = pd.DataFrame(Z_whitened, columns=col_names)
        df_whitened.insert(0, ID_COL, taxa)

        # Verification covariance
        cov_whitened = (Z_whitened.T @ Z_whitened) / (N - 1)
        return df_whitened, cov_whitened

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms new data using fitted whitening parameters."""
        if self.L_inv is None or self.mean is None or self.std is None:
            raise RuntimeError("CholeskyWhitener is not fitted.")

        taxa = df[ID_COL].values
        X = df[self.feature_cols].values.astype(np.float64)
        Z = (X - self.mean) / self.std
        Z_whitened = Z @ self.L_inv.T

        col_names = [f"whiten_{c}" for c in self.feature_cols]
        df_whitened = pd.DataFrame(Z_whitened, columns=col_names)
        df_whitened.insert(0, ID_COL, taxa)
        return df_whitened
