import unittest
import tempfile
from pathlib import Path
import pandas as pd
from src.data.schemas import ID_COL, BIOCLIM_VARS
from src.validation.sync_barrier_1 import audit_sync_barrier_1

class TestSyncBarrier1(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_sync_barrier_1_pass(self):
        taxa = ["species_a", "species_b", "species_c"]
        
        # Phenotypic
        df_pheno = pd.DataFrame({ID_COL: taxa, "trait_1": [1.0, 2.0, 3.0]})
        pheno_path = self.tmp_path / "pheno.csv"
        df_pheno.to_csv(pheno_path, index=False)
        
        # Environmental
        env_data = {ID_COL: taxa}
        for b in BIOCLIM_VARS:
            env_data[b] = [10.0, 20.0, 30.0]
        df_env = pd.DataFrame(env_data)
        env_path = self.tmp_path / "env.csv"
        df_env.to_csv(env_path, index=False)
        
        # Genomic
        df_gen = pd.DataFrame({ID_COL: taxa, "gen_1": [0.1, 0.2, 0.3], "gen_2": [1, 0, 1]})
        gen_path = self.tmp_path / "gen.csv"
        df_gen.to_csv(gen_path, index=False)
        
        passed, report = audit_sync_barrier_1(pheno_path, env_path, gen_path)
        self.assertTrue(passed)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["overlap_count"], 3)

    def test_sync_barrier_1_fail_mismatch_taxa(self):
        df_pheno = pd.DataFrame({ID_COL: ["species_a", "species_b"], "trait_1": [1.0, 2.0]})
        df_env = pd.DataFrame({ID_COL: ["species_b", "species_c"], "bio1": [10.0, 20.0]})
        df_gen = pd.DataFrame({ID_COL: ["species_a", "species_b"], "gen_1": [0.1, 0.2]})
        
        pheno_path = self.tmp_path / "pheno.csv"
        env_path = self.tmp_path / "env.csv"
        gen_path = self.tmp_path / "gen.csv"
        
        df_pheno.to_csv(pheno_path, index=False)
        df_env.to_csv(env_path, index=False)
        df_gen.to_csv(gen_path, index=False)
        
        passed, report = audit_sync_barrier_1(pheno_path, env_path, gen_path)
        self.assertFalse(passed)
        self.assertEqual(report["status"], "FAIL")

if __name__ == "__main__":
    unittest.main()
