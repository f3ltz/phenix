import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.schemas import ID_COL
from src.validation.sync_barrier_3 import audit_sync_barrier_3


class TestSyncBarrier3(unittest.TestCase):
    def test_sync_barrier_3_pass_on_processed_data(self):
        """Tests that the real processed dataset passes the Sync Barrier 3 audit gate."""
        passed, report = audit_sync_barrier_3(
            data_dir="data/processed",
            d_buffer=140.0,
            min_clade_size=50,
            target_trait="adult_body_mass_g",
        )
        self.assertTrue(passed)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["clade_leakage_audit_locked"])
        self.assertTrue(report["buffer_isolation_verified"])
        self.assertEqual(report["taxa_count"], 3268)
        self.assertGreaterEqual(len(report["models_verified"]), 4)

    def test_sync_barrier_3_fail_missing_file(self):
        """Tests failure when a required artifact is missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            passed, report = audit_sync_barrier_3(
                data_dir=tmp_dir,
            )
            self.assertFalse(passed)
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["clade_leakage_audit_locked"])
            self.assertGreater(len(report["errors"]), 0)


if __name__ == "__main__":
    unittest.main()
