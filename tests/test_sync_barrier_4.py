import unittest
import tempfile
from pathlib import Path
from src.validation.sync_barrier_4 import audit_sync_barrier_4


class TestSyncBarrier4(unittest.TestCase):
    def test_sync_barrier_4_pass_on_processed_data(self):
        """Tests that the real processed benchmark dataset passes the Sync Barrier 4 audit gate."""
        passed, report = audit_sync_barrier_4(
            data_dir="data/processed",
        )
        self.assertTrue(passed)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["benchmark_review_locked"])
        self.assertEqual(report["configurations_tested_count"], 16)
        self.assertEqual(report["species_in_predictions"], 3268)
        self.assertTrue(report["inflation_gap_verified"])
        self.assertTrue(report["phylo_gain_verified"])
        self.assertTrue(report["divergence_tracking_verified"])

    def test_sync_barrier_4_fail_missing_file(self):
        """Tests failure when a required benchmark artifact is missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            passed, report = audit_sync_barrier_4(
                data_dir=tmp_dir,
            )
            self.assertFalse(passed)
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["benchmark_review_locked"])
            self.assertGreater(len(report["errors"]), 0)


if __name__ == "__main__":
    unittest.main()
