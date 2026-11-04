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

    def test_pooled_metrics_and_reference_models(self):
        # Two orders, 160 Ma apart; species within an order 20 Ma apart
        order_idx = np.array([0] * 15 + [1] * 15)
        D = np.where(order_idx[:, None] == order_idx[None, :], 20.0, 160.0)
        np.fill_diagonal(D, 0.0)

        runner = BenchmarkMatrixRunner(
            df_pheno=self.df_pheno,
            df_baseline=self.df_baseline,
            df_augmented=self.df_augmented,
            random_folds=self.random_folds,
            phylo_folds=self.phylo_folds,
            target_trait="adult_body_mass_g",
            patristic_matrix=D,
        )
        summary_df = runner.run_benchmark(verbose=False)["summary_df"]

        # 16 model configurations + 2 reference predictors x 2 protocols
        self.assertEqual(len(summary_df), 20)
        self.assertIn("r2_pooled", summary_df.columns)
        self.assertEqual(set(summary_df[summary_df["feature_set"] == "Reference"]["model"]), {"GlobalMean", "PhyloBM"})

        # Inflation gap is defined on pooled R^2
        ridge = summary_df[(summary_df["model"] == "Ridge") & (summary_df["feature_set"] == "Baseline")]
        r_rand = ridge[ridge["protocol"] == "Random_CV"]["r2_pooled"].iloc[0]
        r_phylo = ridge[ridge["protocol"] == "Phylo_CV"]["r2_pooled"].iloc[0]
        self.assertAlmostEqual(ridge["inflation_gap_r2"].iloc[0], round(r_rand - r_phylo, 4), places=3)


if __name__ == "__main__":
    unittest.main()
