import unittest
import numpy as np
from src.models import (
    compute_regression_metrics,
    Log10TargetTransformer,
    RidgeRegressionPipeline,
    RandomForestPipeline,
    XGBoostPipeline,
    PhyloGNNPipeline,
    ValidationEngine,
)
from src.validation.cv import CVFold


class TestModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.N = 60
        self.P = 8
        self.X = np.random.randn(self.N, self.P).astype(np.float32)
        # Target with non-linear exponential relation
        self.y = 10.0 ** (2.0 + 0.3 * self.X[:, 0] - 0.2 * self.X[:, 1] + np.random.randn(self.N) * 0.05)

        self.fold = CVFold(
            fold_id=0,
            train_indices=np.arange(0, 45),
            test_indices=np.arange(45, 60),
            buffer_indices=np.array([], dtype=int),
        )

    def test_regression_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_perfect = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics_perfect = compute_regression_metrics(y_true, y_pred_perfect)
        self.assertAlmostEqual(metrics_perfect["r2"], 1.0)
        self.assertAlmostEqual(metrics_perfect["rmse"], 0.0)
        self.assertAlmostEqual(metrics_perfect["mae"], 0.0)

        # Baseline mean prediction
        y_pred_mean = np.full(5, 3.0)
        metrics_mean = compute_regression_metrics(y_true, y_pred_mean)
        self.assertAlmostEqual(metrics_mean["r2"], 0.0)

    def test_log10_target_transformer(self):
        transformer = Log10TargetTransformer()
        y_raw = np.array([1.0, 10.0, 100.0, 1000.0])
        y_log = transformer.fit_transform(y_raw)
        np.testing.assert_allclose(y_log, [0.0, 1.0, 2.0, 3.0], rtol=1e-5)

        y_inv = transformer.inverse_transform(y_log)
        np.testing.assert_allclose(y_inv, y_raw, rtol=1e-5)

    def test_ridge_pipeline(self):
        model = RidgeRegressionPipeline(log_transform_target=True)
        model.fit(self.X[self.fold.train_indices], self.y[self.fold.train_indices])
        preds = model.predict(self.X[self.fold.test_indices])
        self.assertEqual(len(preds), 15)
        self.assertFalse(np.isnan(preds).any())

        metrics = model.evaluate(self.X[self.fold.test_indices], self.y[self.fold.test_indices])
        self.assertGreater(metrics["r2"], 0.5)

    def test_random_forest_pipeline(self):
        model = RandomForestPipeline(n_estimators=20, random_state=42)
        model.fit(self.X[self.fold.train_indices], self.y[self.fold.train_indices])
        preds = model.predict(self.X[self.fold.test_indices])
        self.assertEqual(len(preds), 15)

        metrics = model.evaluate(self.X[self.fold.test_indices], self.y[self.fold.test_indices])
        self.assertGreater(metrics["r2"], 0.3)

    def test_xgboost_pipeline(self):
        model = XGBoostPipeline(n_estimators=20, random_state=42)
        model.fit(self.X[self.fold.train_indices], self.y[self.fold.train_indices])
        preds = model.predict(self.X[self.fold.test_indices])
        self.assertEqual(len(preds), 15)

        metrics = model.evaluate(self.X[self.fold.test_indices], self.y[self.fold.test_indices])
        self.assertFalse(np.isnan(metrics["rmse"]))

    def test_phylo_gnn_pipeline(self):
        model = PhyloGNNPipeline(hidden_dim=32, epochs=10)
        model.fit(
            self.X[self.fold.train_indices],
            self.y[self.fold.train_indices],
        )
        preds = model.predict(self.X[self.fold.test_indices])
        self.assertEqual(len(preds), 15)
        self.assertFalse(np.isnan(preds).any())

    def test_validation_engine_fold(self):
        engine = ValidationEngine()
        model = RidgeRegressionPipeline()
        res = engine.run_fold(model, self.X, self.y, self.fold)

        self.assertEqual(res["n_train"], 45)
        self.assertEqual(res["n_test"], 15)
        self.assertIn("r2", res)
        self.assertIn("rmse", res)
        self.assertIn("mae", res)


if __name__ == "__main__":
    unittest.main()
