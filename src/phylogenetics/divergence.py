from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.stats import pearsonr, spearmanr


class PhyloDivergenceTracker:
    """
    Evolutionary Divergence Tracking Engine.
    Quantifies patristic separation between evaluation test clades and training
    pools, and analyzes accuracy degradation as evolutionary distance increases.
    """

    def __init__(self, patristic_matrix: np.ndarray, all_taxa: List[str]):
        self.patristic_matrix = patristic_matrix
        self.all_taxa = list(all_taxa)
        self.taxa_to_idx = {t: i for i, t in enumerate(self.all_taxa)}

    def compute_fold_divergence(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> Dict[str, float]:
        """
        Computes summary evolutionary divergence metrics between a held-out test
        clade and the training species pool.
        """
        if len(train_indices) == 0 or len(test_indices) == 0:
            return {
                "min_divergence_ma": float("nan"),
                "mean_divergence_ma": float("nan"),
                "median_divergence_ma": float("nan"),
            }

        # Submatrix: test species (rows) x train species (cols)
        sub_D = self.patristic_matrix[np.ix_(test_indices, train_indices)]

        # For each test species, find nearest relative in training pool
        nearest_relative_dists = np.min(sub_D, axis=1)

        return {
            "min_divergence_ma": float(np.min(nearest_relative_dists)),
            "mean_divergence_ma": float(np.mean(nearest_relative_dists)),
            "median_divergence_ma": float(np.median(nearest_relative_dists)),
            "max_nearest_divergence_ma": float(np.max(nearest_relative_dists)),
        }

    def track_divergence_vs_accuracy(
        self,
        fold_results: List[Dict[str, Any]],
        folds_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Correlates divergence distances with predictive accuracy (RMSE / R^2)
        across cross-validation folds.
        """
        distances: List[float] = []
        rmses: List[float] = []
        r2s: List[float] = []
        clade_names: List[str] = []

        for res in fold_results:
            fold_id = res.get("clade_name") or res.get("fold_id")
            if fold_id not in folds_dict:
                continue

            fold_info = folds_dict[fold_id]
            train_idx = fold_info.get("train_indices")
            test_idx = fold_info.get("test_indices")

            if train_idx is None or test_idx is None:
                continue

            div_metrics = self.compute_fold_divergence(train_idx, test_idx)
            dist = div_metrics["mean_divergence_ma"]

            if not np.isnan(dist) and not np.isnan(res["rmse"]):
                distances.append(dist)
                rmses.append(res["rmse"])
                r2s.append(res["r2"])
                clade_names.append(str(fold_id))

        # Compute correlation between divergence distance and prediction error
        if len(distances) >= 3:
            pearson_corr, p_val = pearsonr(distances, rmses)
            spearman_corr, sp_val = spearmanr(distances, rmses)
        else:
            pearson_corr, p_val = float("nan"), float("nan")
            spearman_corr, sp_val = float("nan"), float("nan")

        return {
            "clades": clade_names,
            "mean_divergences_ma": [round(d, 2) for d in distances],
            "rmses": [round(r, 4) for r in rmses],
            "r2s": [round(r, 4) for r in r2s],
            "pearson_correlation_distance_vs_rmse": round(float(pearson_corr), 4) if not np.isnan(pearson_corr) else None,
            "spearman_correlation_distance_vs_rmse": round(float(spearman_corr), 4) if not np.isnan(spearman_corr) else None,
        }
