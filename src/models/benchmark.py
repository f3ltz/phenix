from typing import Dict, Any, List, Optional, Callable, Union
import numpy as np
import pandas as pd
from src.models.base import PhenixModel, compute_regression_metrics
from src.validation.cv import CVFold, RandomKFoldSplitter, PhyloCVSplitter


class ValidationEngine:
    """
    Validation Engine for executing cross-validation experiments across
    Random CV and Phylogenetic CV (Phylo-CV) protocols.
    """

    def run_fold(
        self,
        model: PhenixModel,
        X: np.ndarray,
        y: np.ndarray,
        fold: CVFold,
    ) -> Dict[str, Any]:
        """
        Executes training and evaluation for a single cross-validation fold.
        Guarantees evaluation is performed strictly on fold.test_indices.
        """
        X_train = X[fold.train_indices]
        y_train = y[fold.train_indices]
        X_test = X[fold.test_indices]
        y_test = y[fold.test_indices]

        # Pass train/test indices in case model is graph-aware (e.g. PhyloGNN)
        model.fit(X_train, y_train, train_indices=fold.train_indices)
        y_pred = model.predict(X_test, test_indices=fold.test_indices)

        # In log space if target was transformed
        if model.log_transform_target and model.transformer is not None:
            y_test_eval = model.transformer.fit_transform(y_test)
        else:
            y_test_eval = y_test

        metrics = compute_regression_metrics(y_test_eval, y_pred)
        return {
            "fold_id": fold.fold_id,
            "clade_name": fold.clade_name,
            "n_train": fold.n_train,
            "n_test": fold.n_test,
            "n_buffer": fold.n_buffer,
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "y_test_mean": float(np.mean(y_test_eval)),
            "y_pred_mean": float(np.mean(y_pred)),
        }

    def evaluate_cv(
        self,
        model_factory: Callable[[], PhenixModel],
        X: np.ndarray,
        y: np.ndarray,
        folds: List[CVFold],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs cross-validation across a pre-generated sequence of folds.
        Logs fold-level metrics and computes aggregated macro-averages.
        """
        fold_results: List[Dict[str, Any]] = []

        for fold in folds:
            model = model_factory()
            res = self.run_fold(model=model, X=X, y=y, fold=fold)
            fold_results.append(res)
            if verbose:
                print(
                    f"  Fold {res['clade_name'] or res['fold_id']} | "
                    f"N_test={res['n_test']} | R^2={res['r2']:.4f} | "
                    f"RMSE={res['rmse']:.4f} | MAE={res['mae']:.4f}"
                )

        r2_vals = [r["r2"] for r in fold_results if not np.isnan(r["r2"])]
        rmse_vals = [r["rmse"] for r in fold_results if not np.isnan(r["rmse"])]
        mae_vals = [r["mae"] for r in fold_results if not np.isnan(r["mae"])]

        summary = {
            "n_folds": len(fold_results),
            "r2_mean": float(np.mean(r2_vals)) if r2_vals else float("nan"),
            "r2_std": float(np.std(r2_vals)) if r2_vals else float("nan"),
            "rmse_mean": float(np.mean(rmse_vals)) if rmse_vals else float("nan"),
            "rmse_std": float(np.std(rmse_vals)) if rmse_vals else float("nan"),
            "mae_mean": float(np.mean(mae_vals)) if mae_vals else float("nan"),
            "mae_std": float(np.std(mae_vals)) if mae_vals else float("nan"),
            "fold_details": fold_results,
        }
        return summary
