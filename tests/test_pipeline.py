import unittest
import tempfile
from pathlib import Path
import pandas as pd
from src.data.schemas import ID_COL
from src.data.pipeline import UnifiedDataPipeline
from src.validation.sync_barrier_1 import audit_sync_barrier_1


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_pipeline_end_to_end_and_sync_barrier_1(self):
        pipeline = UnifiedDataPipeline(
            output_dir=self.tmp_path,
            esm2_dims=8,
            snp_dims=6,
            func_dims=6,
        )
        res = pipeline.run_pipeline(use_reference_seed=True)

        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue((self.tmp_path / "phenotypic_clean.csv").exists())
        self.assertTrue((self.tmp_path / "environmental_clean.csv").exists())
        self.assertTrue((self.tmp_path / "genomic_clean.csv").exists())
        self.assertTrue((self.tmp_path / "alignment_manifest.json").exists())

        # Load and verify 1:1 matching
        df_pheno = pd.read_csv(self.tmp_path / "phenotypic_clean.csv")
        df_env = pd.read_csv(self.tmp_path / "environmental_clean.csv")
        df_genomic = pd.read_csv(self.tmp_path / "genomic_clean.csv")

        self.assertEqual(len(df_pheno), len(df_env))
        self.assertEqual(len(df_env), len(df_genomic))
        self.assertTrue(df_pheno[ID_COL].equals(df_env[ID_COL]))
        self.assertTrue(df_env[ID_COL].equals(df_genomic[ID_COL]))

        # Run Sync Barrier 1 audit gate
        report_path = self.tmp_path / "sync_barrier_1_report.json"
        passed, report = audit_sync_barrier_1(
            pheno_path=self.tmp_path / "phenotypic_clean.csv",
            env_path=self.tmp_path / "environmental_clean.csv",
            genomic_path=self.tmp_path / "genomic_clean.csv",
            report_out_path=report_path,
        )

        self.assertTrue(passed)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["alignment_verified"])
        self.assertEqual(report["overlap_count"], len(df_pheno))
        self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
