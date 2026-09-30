import json
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd

from src.data.schemas import ID_COL
from src.validation.cv import CVFold, RandomKFoldSplitter, PhyloCVSplitter
from src.models import (
    RidgeRegressionPipeline,
    RandomForestPipeline,
    XGBoostPipeline,
    PhyloGNNPipeline,
    ValidationEngine,
    Log10TargetTransformer,
)
from src.phylogenetics.divergence import PhyloDivergenceTracker


class BenchmarkMatrixRunner:
    """
    Orchestrates the complete 16-configuration experimental benchmarking matrix:
      4 Models (Ridge, Random Forest, XGBoost, PhyloGNN)
      x 2 Feature Sets (Baseline [85 vars], Augmented [117 vars])
      x 2 Evaluation Protocols (Random 10-Fold CV [Interpolation], Phylo-CV [Extrapolation])
    """

    def __init__(
        self,
        df_pheno: pd.DataFrame,
        df_baseline: pd.DataFrame,
        df_augmented: pd.DataFrame,
        random_folds: List[CVFold],
        phylo_folds: List[CVFold],
        target_trait: str = "adult_body_mass_g",
        tree_graph_dict: Optional[Dict[str, Any]] = None,
        divergence_tracker: Optional[PhyloDivergenceTracker] = None,
    ):
        self.df_pheno = df_pheno
        self.df_baseline = df_baseline
        self.df_augmented = df_augmented
        self.random_folds = random_folds
        self.phylo_folds = phylo_folds
        self.target_trait = target_trait
        self.tree_graph_dict = tree_graph_dict
        self.divergence_tracker = divergence_tracker

        self.taxa = df_pheno[ID_COL].tolist()
        self.N = len(self.taxa)

        # Feature matrices
        base_cols = [c for c in df_baseline.columns if c != ID_COL]
        aug_cols = [c for c in df_augmented.columns if c != ID_COL]
        self.X_baseline = df_baseline[base_cols].values.astype(np.float32)
        self.X_augmented = df_augmented[aug_cols].values.astype(np.float32)

        # Target vector
        self.y_raw = df_pheno[target_trait].values.astype(np.float32)
        self.target_transformer = Log10TargetTransformer()
        self.y_log = self.target_transformer.fit_transform(self.y_raw)

        self.val_engine = ValidationEngine()

    def get_model_factory(
        self,
        model_name: str,
        feature_set_name: str,
    ) -> Callable[[], Any]:
        """Returns instantiated model pipeline factory with tuned hyperparameters."""
        if model_name == "Ridge":
            return lambda: RidgeRegressionPipeline(log_transform_target=True)
        elif model_name == "RandomForest":
            return lambda: RandomForestPipeline(
                n_estimators=50,
                max_depth=15,
                random_state=42,
                log_transform_target=True,
                n_jobs=-1,
            )
        elif model_name == "XGBoost":
            return lambda: XGBoostPipeline(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.08,
                random_state=42,
                log_transform_target=True,
                n_jobs=-1,
            )
        elif model_name == "PhyloGNN":
            graph = self.tree_graph_dict if feature_set_name == "Baseline" else None
            return lambda: PhyloGNNPipeline(
                graph_dict=graph,
                hidden_dim=64,
                epochs=25,
                lr=0.01,
                log_transform_target=True,
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def run_benchmark(
        self,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes all 16 benchmark configurations and generates predictions matrix.
        """
        models = ["Ridge", "RandomForest", "XGBoost", "PhyloGNN"]
        feature_sets = [
            ("Baseline", self.X_baseline),
            ("Augmented", self.X_augmented),
        ]
        protocols = [
            ("Random_CV", self.random_folds),
            ("Phylo_CV", self.phylo_folds),
        ]

        results: Dict[str, Any] = {}
        summary_rows: List[Dict[str, Any]] = []

        # Predictions matrix initialization
        pred_df = pd.DataFrame({
            ID_COL: self.taxa,
            "order": self.df_pheno.get("order", "Unknown"),
            "true_y_raw": self.y_raw,
            "true_y_log10": self.y_log,
        })

        # Run 16 permutations
        for model_name in models:
            for feat_name, X_mat in feature_sets:
                for proto_name, folds in protocols:
                    config_key = f"{model_name}__{feat_name}__{proto_name}"
                    if verbose:
                        print(f"  [Benchmarking] {config_key} ({len(folds)} folds)...")

                    factory = self.get_model_factory(model_name, feat_name)

                    # Vector to hold full out-of-fold predictions
                    oof_preds = np.full(self.N, np.nan, dtype=np.float32)
                    fold_evals: List[Dict[str, Any]] = []

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        for fold in folds:
                            model = factory()
                            res = self.val_engine.run_fold(model, X_mat, self.y_raw, fold)
                            fold_evals.append(res)

                            # Extract predictions for this test fold
                            y_pred_fold = model.predict(X_mat[fold.test_indices], test_indices=fold.test_indices)
                            oof_preds[fold.test_indices] = y_pred_fold

                    # Compute aggregated metrics
                    r2_vals = [r["r2"] for r in fold_evals if not np.isnan(r["r2"])]
                    rmse_vals = [r["rmse"] for r in fold_evals if not np.isnan(r["rmse"])]
                    mae_vals = [r["mae"] for r in fold_evals if not np.isnan(r["mae"])]

                    mean_r2 = float(np.mean(r2_vals)) if r2_vals else float("nan")
                    std_r2 = float(np.std(r2_vals)) if r2_vals else float("nan")
                    mean_rmse = float(np.mean(rmse_vals)) if rmse_vals else float("nan")
                    std_rmse = float(np.std(rmse_vals)) if rmse_vals else float("nan")
                    mean_mae = float(np.mean(mae_vals)) if mae_vals else float("nan")
                    std_mae = float(np.std(mae_vals)) if mae_vals else float("nan")

                    results[config_key] = {
                        "model": model_name,
                        "feature_set": feat_name,
                        "protocol": proto_name,
                        "n_folds": len(folds),
                        "r2_mean": round(mean_r2, 4),
                        "r2_std": round(std_r2, 4),
                        "rmse_mean": round(mean_rmse, 4),
                        "rmse_std": round(std_rmse, 4),
                        "mae_mean": round(mean_mae, 4),
                        "mae_std": round(std_mae, 4),
                        "fold_details": fold_evals,
                    }

                    summary_rows.append({
                        "model": model_name,
                        "feature_set": feat_name,
                        "protocol": proto_name,
                        "r2_mean": round(mean_r2, 4),
                        "r2_std": round(std_r2, 4),
                        "rmse_mean": round(mean_rmse, 4),
                        "rmse_std": round(std_rmse, 4),
                        "mae_mean": round(mean_mae, 4),
                        "mae_std": round(std_mae, 4),
                    })

                    # Record in prediction dataframe
                    pred_col = f"pred__{model_name}__{feat_name}__{proto_name}"
                    resid_col = f"resid__{model_name}__{feat_name}__{proto_name}"
                    pred_df[pred_col] = oof_preds
                    pred_df[resid_col] = self.y_log - oof_preds

                    if verbose:
                        print(
                            f"    -> R^2: {mean_r2:.4f} +/- {std_r2:.4f} | "
                            f"RMSE: {mean_rmse:.4f} | MAE: {mean_mae:.4f}"
                        )

        # 2. Compute Performance Inflation Gap & Phylo Gain
        df_summary = pd.DataFrame(summary_rows)

        # Add Inflation Gap: R^2(Random CV) - R^2(Phylo-CV)
        inflation_map: Dict[Tuple[str, str], float] = {}
        for (m, f), group in df_summary.groupby(["model", "feature_set"]):
            r_rand = group[group["protocol"] == "Random_CV"]["r2_mean"].values
            r_phylo = group[group["protocol"] == "Phylo_CV"]["r2_mean"].values
            if len(r_rand) > 0 and len(r_phylo) > 0:
                inflation_map[(m, f)] = round(float(r_rand[0] - r_phylo[0]), 4)

        df_summary["inflation_gap_r2"] = df_summary.apply(
            lambda row: inflation_map.get((row["model"], row["feature_set"]), float("nan")),
            axis=1,
        )

        # Add Phylogenetic Augmentation Gain: R^2(Augmented) - R^2(Baseline) under same protocol
        gain_map: Dict[Tuple[str, str], float] = {}
        for (m, p), group in df_summary.groupby(["model", "protocol"]):
            r_aug = group[group["feature_set"] == "Augmented"]["r2_mean"].values
            r_base = group[group["feature_set"] == "Baseline"]["r2_mean"].values
            if len(r_aug) > 0 and len(r_base) > 0:
                gain_map[(m, p)] = round(float(r_aug[0] - r_base[0]), 4)

        df_summary["phylo_gain_r2"] = df_summary.apply(
            lambda row: gain_map.get((row["model"], row["protocol"]), float("nan")),
            axis=1,
        )

        # 3. Divergence Degradation Tracking
        divergence_tracking: Dict[str, Any] = {}
        if self.divergence_tracker is not None:
            folds_dict = {f.clade_name: {"train_indices": f.train_indices, "test_indices": f.test_indices} for f in self.phylo_folds}
            for model_name in models:
                key = f"{model_name}__Augmented__Phylo_CV"
                if key in results:
                    div_res = self.divergence_tracker.track_divergence_vs_accuracy(
                        fold_results=results[key]["fold_details"],
                        folds_dict=folds_dict,
                    )
                    divergence_tracking[model_name] = div_res

        return {
            "matrix_results": results,
            "summary_df": df_summary,
            "predictions_df": pred_df,
            "divergence_tracking": divergence_tracking,
        }
