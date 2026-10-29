"""
PHENIX Residual Diagnostics & Signal Attribution Engine (Stage 5)
Computes:
  1. Phylogenetic error autocorrelation (Blomberg's K and Pagel's lambda) on model residuals.
  2. Multi-modal variance partitioning across Environmental, Genomic, and Phylogenetic spaces.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


class ResidualDiagnosticsEngine:
    """
    Computes phylogenetic autocorrelation statistics (Blomberg's K, Pagel's lambda)
    on raw phenotypic traits and model out-of-fold residuals.
    """

    def __init__(self, patristic_matrix_path: Path, root_age_ma: float = 177.0):
        self.patristic_matrix_path = Path(patristic_matrix_path)
        self.root_age_ma = float(root_age_ma)
        self._D_full = np.load(self.patristic_matrix_path)

    def _get_covariance_spectral(self, D_sub: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Converts patristic distance submatrix D into Brownian motion covariance V,
        V_ij = T_max - 0.5 * D_ij, and computes spectral eigendecomposition.
        """
        V = self.root_age_ma - 0.5 * D_sub
        # Numerical safeguard for symmetry
        V = 0.5 * (V + V.T)
        evals, Q = np.linalg.eigh(V)
        # Ensure eigenvalues are bounded below for numerical stability
        evals = np.maximum(evals, 1e-4)
        return evals, Q, self.root_age_ma

    def compute_blomberg_k(self, y: np.ndarray, D_sub: np.ndarray) -> float:
        """
        Calculates Blomberg's K (Blomberg et al. 2003) for trait or residual vector y.
        K = (MSE_0 / MSE) / E[MSE_0 / MSE]
        """
        evals, Q, T_max = self._get_covariance_spectral(D_sub)
        N = len(y)
        ones = np.ones(N)
        Q_ones = Q.T @ ones
        y_tilde = Q.T @ y

        # GLS mean: hat_a = (1^T V^-1 1)^-1 (1^T V^-1 y)
        denom_a = np.sum(Q_ones**2 / evals)
        numer_a = np.sum(Q_ones * y_tilde / evals)
        hat_a = numer_a / denom_a

        diff = y - hat_a
        MSE_0 = np.sum(diff**2) / (N - 1)

        diff_tilde = Q.T @ diff
        MSE = np.sum((diff_tilde**2) / evals) / (N - 1)

        trace_V = np.sum(evals)
        expected_ratio = (trace_V - N / denom_a) / (N - 1)

        K = float((MSE_0 / MSE) / expected_ratio)
        return K

    def compute_pagel_lambda(self, y: np.ndarray, D_sub: np.ndarray) -> float:
        """
        Calculates Pagel's lambda (Pagel 1999) via maximum likelihood optimization.
        V(lambda) = lambda * V + (1 - lambda) * T_max * I
        Using spectral invariance: eigenvalues(V(lambda)) = lambda * evals + (1 - lambda) * T_max
        """
        evals, Q, T_max = self._get_covariance_spectral(D_sub)
        N = len(y)
        ones = np.ones(N)
        Q_ones = Q.T @ ones
        y_tilde = Q.T @ y

        def neg_log_lik(lam: float) -> float:
            lam = float(np.clip(lam, 1e-5, 1.0))
            eff_evals = lam * evals + (1.0 - lam) * T_max

            denom_a = np.sum(Q_ones**2 / eff_evals)
            numer_a = np.sum(Q_ones * y_tilde / eff_evals)
            hat_a = numer_a / denom_a

            diff_tilde = y_tilde - hat_a * Q_ones
            quad = np.sum(diff_tilde**2 / eff_evals)
            sigma2 = quad / N

            log_det = np.sum(np.log(eff_evals))
            nll = 0.5 * (N * np.log(2.0 * np.pi) + N * np.log(sigma2) + log_det + N)
            return float(nll)

        res = minimize_scalar(neg_log_lik, bounds=(1e-5, 1.0), method="bounded")
        return float(res.x)

    def evaluate_all_residuals(self, predictions_df: pd.DataFrame, min_valid_samples: int = 30) -> Dict[str, Any]:
        """
        Computes Blomberg's K and Pagel's lambda across all residual columns
        and the raw true trait in predictions_df.
        """
        results: Dict[str, Any] = {}

        # 1. Evaluate True Target Trait
        y_true = predictions_df["true_y_log10"].values
        k_true = self.compute_blomberg_k(y_true, self._D_full)
        lam_true = self.compute_pagel_lambda(y_true, self._D_full)
        results["true_target"] = {
            "trait": "true_y_log10",
            "N": len(y_true),
            "blomberg_k": round(k_true, 4),
            "pagel_lambda": round(lam_true, 4),
        }

        # 2. Evaluate Residual Columns
        residual_cols = [c for c in predictions_df.columns if c.startswith("resid__")]
        model_results: Dict[str, Any] = {}

        for col in residual_cols:
            series = predictions_df[col]
            valid_mask = ~series.isna()
            N_valid = int(valid_mask.sum())
            if N_valid < min_valid_samples:
                continue

            y_res = series[valid_mask].values
            if N_valid == len(predictions_df):
                D_sub = self._D_full
            else:
                idx = np.where(valid_mask.values)[0]
                D_sub = self._D_full[idx][:, idx]

            k_val = self.compute_blomberg_k(y_res, D_sub)
            lam_val = self.compute_pagel_lambda(y_res, D_sub)

            model_results[col] = {
                "N": N_valid,
                "blomberg_k": round(k_val, 4),
                "pagel_lambda": round(lam_val, 4),
            }

        results["residual_diagnostics"] = model_results
        return results


class VariancePartitioningEngine:
    """
    Performs variation partitioning (Legendre & Legendre; Peres-Neto et al. 2006)
    decomposing phenotypic variance into:
      [a] Pure Environmental (E)
      [b] Pure Genomic (G)
      [c] Pure Phylogenetic (P)
      and shared intersection components.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def run_partitioning(self, cv_folds: int = 5, random_state: int = 42) -> Dict[str, Any]:
        """
        Executes variance partitioning using 5-fold cross-validation
        with both Ridge and Random Forest estimators.
        """
        # Load features and targets
        pred_df = pd.read_csv(self.data_dir / "predictions_matrix.csv")
        y = pred_df["true_y_log10"].values

        env_df = pd.read_csv(self.data_dir / "environmental_clean.csv")
        gen_df = pd.read_csv(self.data_dir / "genomic_clean.csv")
        pcoa_df = pd.read_csv(self.data_dir / "features_phylo_pcoa.csv")

        E = env_df.drop(columns=["canonical_taxon_id"]).select_dtypes(include=[np.number]).values
        G = gen_df.drop(columns=["canonical_taxon_id"]).select_dtypes(include=[np.number]).values
        P = pcoa_df.drop(columns=["canonical_taxon_id"]).select_dtypes(include=[np.number]).values

        EG = np.hstack([E, G])
        EP = np.hstack([E, P])
        GP = np.hstack([G, P])
        EGP = np.hstack([E, G, P])

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        def eval_oof_rf(X: np.ndarray) -> float:
            oof = np.zeros(len(y))
            for train_idx, val_idx in kf.split(X):
                rf = RandomForestRegressor(
                    n_estimators=50, max_depth=12, random_state=random_state, n_jobs=-1
                )
                rf.fit(X[train_idx], y[train_idx])
                oof[val_idx] = rf.predict(X[val_idx])
            return float(r2_score(y, oof))

        def eval_oof_ridge(X: np.ndarray) -> float:
            oof = np.zeros(len(y))
            for train_idx, val_idx in kf.split(X):
                X_train = X[train_idx]
                X_val = X[val_idx]
                mu = X_train.mean(axis=0)
                std = np.maximum(X_train.std(axis=0), 1e-8)
                X_tr_std = (X_train - mu) / std
                X_val_std = (X_val - mu) / std

                reg = RidgeCV(alphas=np.logspace(-2, 4, 15))
                reg.fit(X_tr_std, y[train_idx])
                oof[val_idx] = reg.predict(X_val_std)
            return float(r2_score(y, oof))

        print("[Variance Partitioning] Evaluating Ridge regression partitions...")
        r2_ridge = {
            "E": eval_oof_ridge(E),
            "G": eval_oof_ridge(G),
            "P": eval_oof_ridge(P),
            "EG": eval_oof_ridge(EG),
            "EP": eval_oof_ridge(EP),
            "GP": eval_oof_ridge(GP),
            "EGP": eval_oof_ridge(EGP),
        }

        print("[Variance Partitioning] Evaluating Random Forest partitions...")
        r2_rf = {
            "E": eval_oof_rf(E),
            "G": eval_oof_rf(G),
            "P": eval_oof_rf(P),
            "EG": eval_oof_rf(EG),
            "EP": eval_oof_rf(EP),
            "GP": eval_oof_rf(GP),
            "EGP": eval_oof_rf(EGP),
        }

        def decompose(r2: Dict[str, float]) -> Dict[str, float]:
            pure_E = r2["EGP"] - r2["GP"]
            pure_G = r2["EGP"] - r2["EP"]
            pure_P = r2["EGP"] - r2["EG"]
            shared_all = r2["EGP"] - (pure_E + pure_G + pure_P)
            unexplained = 1.0 - max(0.0, r2["EGP"])
            return {
                "R2_total": round(r2["EGP"], 4),
                "pure_environmental": round(max(0.0, pure_E), 4),
                "pure_genomic": round(max(0.0, pure_G), 4),
                "pure_phylogenetic": round(max(0.0, pure_P), 4),
                "shared_intersection": round(shared_all, 4),
                "unexplained": round(unexplained, 4),
                "raw_R2": {k: round(v, 4) for k, v in r2.items()},
            }

        return {
            "ridge_linear_partitioning": decompose(r2_ridge),
            "random_forest_nonlinear_partitioning": decompose(r2_rf),
        }


def run_stage_5_diagnostics(data_dir: Path = Path("data/processed")) -> Dict[str, Any]:
    """
    Master runner for Stage 5 diagnostics.
    """
    print("\n" + "=" * 70)
    print("STAGE 5: RESIDUAL DIAGNOSTICS & MULTIMODAL VARIANCE PARTITIONING")
    print("=" * 70)

    data_dir = Path(data_dir)
    pred_path = data_dir / "predictions_matrix.csv"
    patristic_path = data_dir / "patristic_distance_matrix.npy"

    if not pred_path.exists() or not patristic_path.exists():
        raise FileNotFoundError(f"Missing required Stage 4 inputs: {pred_path} or {patristic_path}")

    # 1. Residual Diagnostics
    print("[Stage 5] Computing Blomberg's K and Pagel's Lambda on residuals...")
    diag_engine = ResidualDiagnosticsEngine(patristic_matrix_path=patristic_path)
    pred_df = pd.read_csv(pred_path)
    res_diagnostics = diag_engine.evaluate_all_residuals(pred_df)

    res_diag_out = data_dir / "residual_diagnostics.json"
    with open(res_diag_out, "w") as f:
        json.dump(res_diagnostics, f, indent=2)
    print(f"[Stage 5] Wrote residual diagnostics -> {res_diag_out}")

    # 2. Variance Partitioning
    print("[Stage 5] Computing Multimodal Variance Partitioning (E, G, P)...")
    part_engine = VariancePartitioningEngine(data_dir=data_dir)
    partitioning = part_engine.run_partitioning()

    part_out = data_dir / "variance_partitioning.json"
    with open(part_out, "w") as f:
        json.dump(partitioning, f, indent=2)
    print(f"[Stage 5] Wrote variance partitioning -> {part_out}")

    return {
        "residual_diagnostics": res_diagnostics,
        "variance_partitioning": partitioning,
    }


if __name__ == "__main__":
    run_stage_5_diagnostics()
