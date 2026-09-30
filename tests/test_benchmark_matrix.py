import unittest
import numpy as np
import pandas as pd
from src.data.schemas import ID_COL
from src.validation.cv import CVFold
from src.models.experiment import BenchmarkMatrixRunner


class TestBenchmarkMatrix(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.taxa = [f"sp_{i}" for i in range(30)]
        self.df_pheno = pd.DataFrame({
            ID_COL: self.taxa,
            "order": ["OrderA"] * 15 + ["OrderB"] * 15,
            "adult_body_mass_g": [10.0 + i for i in range(30)],
        })
        self.df_baseline = pd.DataFrame({
            ID_COL: self.taxa,
            "bio1": np.random.randn(30),
            "gen1": np.random.randn(30),
        })
        self.df_augmented = pd.DataFrame({
            ID_COL: self.taxa,
            "bio1": self.df_baseline["bio1"],
            "gen1": self.df_baseline["gen1"],
            "phylo1": np.linspace(0.1, 1.0, 30),
        })

        self.random_folds = [
            CVFold(fold_id=0, train_indices=np.arange(0, 20), test_indices=np.arange(20, 30), buffer_indices=np.array([], dtype=int)),
            CVFold(fold_id=1, train_indices=np.arange(10, 30), test_indices=np.arange(0, 10), buffer_indices=np.array([], dtype=int)),
        ]
        self.phylo_folds = [
            CVFold(fold_id="OrderA", train_indices=np.arange(15, 30), test_indices=np.arange(0, 15), buffer_indices=np.array([], dtype=int), clade_name="OrderA"),
            CVFold(fold_id="OrderB", train_indices=np.arange(0, 15), test_indices=np.arange(15, 30), buffer_indices=np.array([], dtype=int), clade_name="OrderB"),
        ]

    def test_benchmark_runner_execution(self):
        runner = BenchmarkMatrixRunner(
            df_pheno=self.df_pheno,
            df_baseline=self.df_baseline,
            df_augmented=self.df_augmented,
            random_folds=self.random_folds,
            phylo_folds=self.phylo_folds,
            target_trait="adult_body_mass_g",
        )

        outputs = runner.run_benchmark(verbose=False)
        self.assertIn("matrix_results", outputs)
        self.assertIn("summary_df", outputs)
        self.assertIn("predictions_df", outputs)

        summary_df = outputs["summary_df"]
        self.assertEqual(len(summary_df), 16)
        self.assertIn("inflation_gap_r2", summary_df.columns)
        self.assertIn("phylo_gain_r2", summary_df.columns)

        pred_df = outputs["predictions_df"]
        self.assertEqual(len(pred_df), 30)
        self.assertIn("true_y_log10", pred_df.columns)


if __name__ == "__main__":
    unittest.main()
