import unittest
import numpy as np
from src.phylogenetics.divergence import PhyloDivergenceTracker


class TestPhyloDivergence(unittest.TestCase):
    def setUp(self):
        # 4 taxa: sp0, sp1, sp2, sp3
        # sp0-sp1 dist = 20 Ma, sp2-sp3 dist = 30 Ma, across = 100 Ma
        self.taxa = ["sp_0", "sp_1", "sp_2", "sp_3"]
        self.D = np.array([
            [0.0, 20.0, 100.0, 100.0],
            [20.0, 0.0, 100.0, 100.0],
            [100.0, 100.0, 0.0, 30.0],
            [100.0, 100.0, 30.0, 0.0],
        ], dtype=np.float32)
        self.tracker = PhyloDivergenceTracker(patristic_matrix=self.D, all_taxa=self.taxa)

    def test_compute_fold_divergence(self):
        # Train: [0, 1], Test: [2, 3]
        metrics = self.tracker.compute_fold_divergence(
            train_indices=np.array([0, 1]),
            test_indices=np.array([2, 3]),
        )
        self.assertEqual(metrics["min_divergence_ma"], 100.0)
        self.assertEqual(metrics["mean_divergence_ma"], 100.0)
        self.assertEqual(metrics["median_divergence_ma"], 100.0)

    def test_track_divergence_vs_accuracy(self):
        fold_results = [
            {"clade_name": "clade_1", "rmse": 0.5, "r2": 0.8},
            {"clade_name": "clade_2", "rmse": 1.5, "r2": -0.2},
            {"clade_name": "clade_3", "rmse": 2.5, "r2": -1.5},
        ]
        folds_dict = {
            "clade_1": {"train_indices": np.array([0]), "test_indices": np.array([1])},   # dist = 20
            "clade_2": {"train_indices": np.array([0, 1]), "test_indices": np.array([2])}, # dist = 100
            "clade_3": {"train_indices": np.array([0]), "test_indices": np.array([3])},   # dist = 100
        }
        res = self.tracker.track_divergence_vs_accuracy(fold_results, folds_dict)
        self.assertEqual(len(res["clades"]), 3)
        self.assertIn("pearson_correlation_distance_vs_rmse", res)
        self.assertGreater(res["pearson_correlation_distance_vs_rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
