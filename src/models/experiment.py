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
    GlobalMeanModel,
    PhyloBMImputer,
    ValidationEngine,
    Log10TargetTransformer,
    compute_regression_metrics,
)
from src.phylogenetics.divergence import PhyloDivergenceTracker


REFERENCE_FEATURE_SET = "Reference"


class BenchmarkMatrixRunner:
    """
    Orchestrates the complete 16-configuration experimental benchmarking matrix:
      4 Models (Ridge, Random Forest, XGBoost, PhyloGNN)
      x 2 Feature Sets (Baseline: climate + sequence features, Augmented: + 32 Phylo-PCoA)
      x 2 Evaluation Protocols (Random K-Fold CV [Interpolation], Phylo-CV [Extrapolation])
    plus, when a patristic matrix is supplied, two feature-free reference predictors
    (GlobalMean and PhyloBM phylogenetic imputation) under both protocols.

    Headline metrics are POOLED over all out-of-fold predictions (r2_pooled etc.), so Random CV
    and Phylo-CV are scored against the same total variance. Per-fold means (r2_mean etc.) are
    kept for reference; under Phylo-CV each fold is a single order whose internal variance is
    small, which makes per-fold R^2 extremely negative and not comparable across protocols.
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
        patristic_matrix: Optional[np.ndarray] = None,
    ):
        self.df_pheno = df_pheno
        self.df_baseline = df_baseline
        self.df_augmented = df_augmented
        self.random_folds = random_folds
        self.phylo_folds = phylo_folds
        self.target_trait = target_trait
        self.tree_graph_dict = tree_graph_dict
        self.divergence_tracker = divergence_tracker
        self.patristic_matrix = patristic_matrix

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
            # Same tree topology for both feature sets; only the tip features differ
            X_mat = self.X_baseline if feature_set_name == "Baseline" else self.X_augmented
            graph = self._graph_with_tip_features(X_mat)
            return lambda: PhyloGNNPipeline(
                graph_dict=graph,
                hidden_dim=64,
                epochs=200,
                lr=0.01,
                log_transform_target=True,
            )
        elif model_name == "GlobalMean":
            return lambda: GlobalMeanModel(log_transform_target=True)
        elif model_name == "PhyloBM":
            return lambda: PhyloBMImputer(self.patristic_matrix, log_transform_target=True)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def _graph_with_tip_features(self, X_tips: np.ndarray) -> Optional[Dict[str, Any]]:
        """Copies the tree graph topology and sets tip node features to X_tips (internal nodes: tip mean)."""
        if self.tree_graph_dict is None:
            return None
        n_tips = self.tree_graph_dict["num_tips"]
        if n_tips != len(X_tips):
            raise ValueError(f"Tree graph has {n_tips} tips but feature matrix has {len(X_tips)} rows.")
        x_all = np.empty((self.tree_graph_dict["num_nodes"], X_tips.shape[1]), dtype=np.float32)
        x_all[:n_tips] = X_tips
        x_all[n_tips:] = X_tips.mean(axis=0)
        return {**self.tree_graph_dict, "x": x_all, "num_features": X_tips.shape[1]}

    def _run_configuration(
        self,
        model_name: str,
        feat_name: str,
        proto_name: str,
        X_mat: np.ndarray,
        folds: List[CVFold],
        verbose: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
        """Runs one (model, feature set, protocol) cell; returns (details, summary row, out-of-fold predictions)."""
        factory = self.get_model_factory(model_name, feat_name)

        oof_preds = np.full(self.N, np.nan, dtype=np.float64)
        fold_evals: List[Dict[str, Any]] = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for fold in folds:
                model = factory()
                res = self.val_engine.run_fold(model, X_mat, self.y_raw, fold)
                fold_evals.append(res)
                oof_preds[fold.test_indices] = model.predict(X_mat[fold.test_indices], test_indices=fold.test_indices)

        def mean_std(key: str) -> Tuple[float, float]:
            vals = [r[key] for r in fold_evals if not np.isnan(r[key])]
            return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

        mean_r2, std_r2 = mean_std("r2")
        mean_rmse, std_rmse = mean_std("rmse")
        mean_mae, std_mae = mean_std("mae")

        # Pooled metrics over every species that was in some test fold
        covered = ~np.isnan(oof_preds)
        pooled = compute_regression_metrics(self.y_log[covered], oof_preds[covered])

        summary = {
            "model": model_name,
            "feature_set": feat_name,
            "protocol": proto_name,
            "n_test_species": int(covered.sum()),
            "r2_pooled": round(pooled["r2"], 4),
            "rmse_pooled": round(pooled["rmse"], 4),
            "mae_pooled": round(pooled["mae"], 4),
            "r2_mean": round(mean_r2, 4),
            "r2_std": round(std_r2, 4),
            "r2_median": round(float(np.nanmedian([r["r2"] for r in fold_evals])), 4),
            "rmse_mean": round(mean_rmse, 4),
            "rmse_std": round(std_rmse, 4),
            "mae_mean": round(mean_mae, 4),
            "mae_std": round(std_mae, 4),
        }
        details = {**summary, "n_folds": len(folds), "fold_details": fold_evals}

        if verbose:
            print(
                f"    -> pooled R^2: {pooled['r2']:.4f} | RMSE: {pooled['rmse']:.4f} | MAE: {pooled['mae']:.4f} "
                f"(per-fold mean R^2: {mean_r2:.4f})"
            )
        return details, summary, oof_preds

    def run_benchmark(
        self,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes all 16 benchmark configurations (plus reference predictors when a patristic
        matrix is available) and generates the predictions matrix.
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

        configs = [
            (model_name, feat_name, X_mat)
            for model_name in models
            for feat_name, X_mat in feature_sets
        ]
        if self.patristic_matrix is not None:
            # Feature-free reference predictors (X is ignored)
            configs += [
                ("GlobalMean", REFERENCE_FEATURE_SET, self.X_baseline),
                ("PhyloBM", REFERENCE_FEATURE_SET, self.X_baseline),
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

        for model_name, feat_name, X_mat in configs:
            for proto_name, folds in protocols:
                config_key = f"{model_name}__{feat_name}__{proto_name}"
                if verbose:
                    print(f"  [Benchmarking] {config_key} ({len(folds)} folds)...")

                details, summary, oof_preds = self._run_configuration(
                    model_name, feat_name, proto_name, X_mat, folds, verbose
                )
                results[config_key] = details
                summary_rows.append(summary)

                pred_df[f"pred__{config_key}"] = oof_preds
                pred_df[f"resid__{config_key}"] = self.y_log - oof_preds

        # 2. Compute Performance Inflation Gap & Phylo Gain (on pooled R^2)
        df_summary = pd.DataFrame(summary_rows)

        # Add Inflation Gap: R^2(Random CV) - R^2(Phylo-CV)
        inflation_map: Dict[Tuple[str, str], float] = {}
        for (m, f), group in df_summary.groupby(["model", "feature_set"]):
            r_rand = group[group["protocol"] == "Random_CV"]["r2_pooled"].values
            r_phylo = group[group["protocol"] == "Phylo_CV"]["r2_pooled"].values
            if len(r_rand) > 0 and len(r_phylo) > 0:
                inflation_map[(m, f)] = round(float(r_rand[0] - r_phylo[0]), 4)

        df_summary["inflation_gap_r2"] = df_summary.apply(
            lambda row: inflation_map.get((row["model"], row["feature_set"]), float("nan")),
            axis=1,
        )

        # Add Phylogenetic Augmentation Gain: R^2(Augmented) - R^2(Baseline) under same protocol
        gain_map: Dict[Tuple[str, str], float] = {}
        for (m, p), group in df_summary.groupby(["model", "protocol"]):
            r_aug = group[group["feature_set"] == "Augmented"]["r2_pooled"].values
            r_base = group[group["feature_set"] == "Baseline"]["r2_pooled"].values
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
