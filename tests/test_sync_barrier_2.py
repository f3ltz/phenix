import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.schemas import ID_COL
from src.validation.sync_barrier_2 import audit_sync_barrier_2


class TestSyncBarrier2(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)
        self.taxa = ["panthera_leo", "canis_lupus", "homo_sapiens"]

        # 1. Phenotypic
        self.pheno_path = self.tmp_path / "pheno.csv"
        pd.DataFrame({ID_COL: self.taxa, "mass": [150.0, 35.0, 65.0]}).to_csv(self.pheno_path, index=False)

        # 2. Baseline
        self.baseline_path = self.tmp_path / "baseline.csv"
        pd.DataFrame({ID_COL: self.taxa, "bio1": [20.0, 5.0, 18.0], "gen_1": [0.1, 0.2, 0.3]}).to_csv(self.baseline_path, index=False)

        # 3. Augmented
        self.augmented_path = self.tmp_path / "augmented.csv"
        pd.DataFrame({ID_COL: self.taxa, "bio1": [20.0, 5.0, 18.0], "gen_1": [0.1, 0.2, 0.3], "phylo_1": [0.5, -0.2, 0.1]}).to_csv(self.augmented_path, index=False)

        # 4. Tree
        self.tree_path = self.tmp_path / "tree.nwk"
        newick = "((panthera_leo:10.0,canis_lupus:10.0):20.0,homo_sapiens:30.0);"
        with open(self.tree_path, "w") as f:
            f.write(newick)

        # 5. Patristic
        self.patristic_path = self.tmp_path / "patristic.npy"
        D = np.array([
            [0.0, 20.0, 60.0],
            [20.0, 0.0, 60.0],
            [60.0, 60.0, 0.0],
        ], dtype=np.float32)
        np.save(self.patristic_path, D)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sync_barrier_2_pass(self):
        passed, report = audit_sync_barrier_2(
            pheno_path=self.pheno_path,
            baseline_path=self.baseline_path,
            augmented_path=self.augmented_path,
            tree_path=self.tree_path,
            patristic_path=self.patristic_path,
        )
        self.assertTrue(passed)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["architecture_freeze_locked"])
        self.assertEqual(report["taxa_count"], 3)

    def test_sync_barrier_2_fail_missing_taxon_in_tree(self):
        bad_tree_path = self.tmp_path / "bad_tree.nwk"
        with open(bad_tree_path, "w") as f:
            f.write("(panthera_leo:10.0,canis_lupus:10.0);")

        passed, report = audit_sync_barrier_2(
            pheno_path=self.pheno_path,
            baseline_path=self.baseline_path,
            augmented_path=self.augmented_path,
            tree_path=bad_tree_path,
            patristic_path=self.patristic_path,
        )
        self.assertFalse(passed)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("Tree is missing 1 taxa", report["errors"][0])

    def test_sync_barrier_2_fail_nan_in_features(self):
        nan_base_path = self.tmp_path / "nan_base.csv"
        df_nan = pd.DataFrame({ID_COL: self.taxa, "bio1": [20.0, np.nan, 18.0]})
        df_nan.to_csv(nan_base_path, index=False)

        passed, report = audit_sync_barrier_2(
            pheno_path=self.pheno_path,
            baseline_path=nan_base_path,
            augmented_path=self.augmented_path,
            tree_path=self.tree_path,
            patristic_path=self.patristic_path,
        )
        self.assertFalse(passed)
        self.assertEqual(report["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
