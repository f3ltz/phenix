import unittest
import numpy as np
import pandas as pd
from src.data.schemas import ID_COL
from src.phylogenetics.otol import PhyloNode
from src.phylogenetics.withholding import MonophyleticWithholdingEngine


class TestMonophyleticWithholding(unittest.TestCase):
    def setUp(self):
        # Build a small toy ultrametric tree:
        # ((sp1:10.0,sp2:10.0):20.0,(sp3:15.0,sp4:15.0):15.0);
        # Total height = 30.0 Ma
        self.root = PhyloNode(name="Root", length=0.0)
        self.cladeA = PhyloNode(name="CladeA", length=10.0)
        self.cladeB = PhyloNode(name="CladeB", length=15.0)
        self.root.add_child(self.cladeA)
        self.root.add_child(self.cladeB)

        self.sp1 = PhyloNode(name="sp_1", length=20.0)
        self.sp2 = PhyloNode(name="sp_2", length=20.0)
        self.cladeA.add_child(self.sp1)
        self.cladeA.add_child(self.sp2)

        self.sp3 = PhyloNode(name="sp_3", length=15.0)
        self.sp4 = PhyloNode(name="sp_4", length=15.0)
        self.cladeB.add_child(self.sp3)
        self.cladeB.add_child(self.sp4)

        self.taxa = ["sp_1", "sp_2", "sp_3", "sp_4"]
        self.df_pheno = pd.DataFrame({
            ID_COL: self.taxa,
            "order": ["OrderA", "OrderA", "OrderB", "OrderB"],
            "adult_body_mass_g": [10.0, 15.0, 100.0, 120.0],
        })

        # Patristic matrix:
        # sp1-sp2: 2 * 20 = 40
        # sp3-sp4: 2 * 15 = 30
        # across CladeA and CladeB: 2 * 30 = 60
        self.D = np.array([
            [0.0, 40.0, 60.0, 60.0],
            [40.0, 0.0, 60.0, 60.0],
            [60.0, 60.0, 0.0, 30.0],
            [60.0, 60.0, 30.0, 0.0],
        ], dtype=np.float32)

        self.engine = MonophyleticWithholdingEngine(tree_root=self.root)

    def test_ensure_node_ages(self):
        age_root = self.engine._ensure_node_ages(self.root)
        self.assertAlmostEqual(age_root, 30.0)
        self.assertAlmostEqual(self.cladeA.age, 20.0)
        self.assertAlmostEqual(self.cladeB.age, 15.0)
        self.assertAlmostEqual(self.sp1.age, 0.0)

    def test_get_order_clades(self):
        clades = self.engine.get_order_clades(self.df_pheno, min_clade_size=2)
        self.assertEqual(len(clades), 2)
        self.assertIn("OrderA", clades)
        self.assertEqual(clades["OrderA"], ["sp_1", "sp_2"])
        self.assertEqual(clades["OrderB"], ["sp_3", "sp_4"])

        # Filter with min_clade_size > 2
        clades_empty = self.engine.get_order_clades(self.df_pheno, min_clade_size=3)
        self.assertEqual(len(clades_empty), 0)

    def test_get_t_cut_clusters(self):
        # Cut at T_cut = 25 Ma: Root age is 30 Ma > 25, CladeA is 20 <= 25, CladeB is 15 <= 25
        # Expected: 2 clusters (CladeA with [sp1, sp2], CladeB with [sp3, sp4])
        clusters_25 = self.engine.get_t_cut_clusters(t_cut_ma=25.0)
        self.assertEqual(len(clusters_25), 2)
        total_taxa = sum(len(v) for v in clusters_25.values())
        self.assertEqual(total_taxa, 4)

        # Cut at T_cut = 10 Ma: CladeA (20 > 10) splits into sp1, sp2; CladeB (15 > 10) splits into sp3, sp4
        # Expected: 4 singleton clusters
        clusters_10 = self.engine.get_t_cut_clusters(t_cut_ma=10.0)
        self.assertEqual(len(clusters_10), 4)

    def test_compute_buffer_quarantine(self):
        # Hold out sp_1. Min distance to sp_2 is 40.0, to sp_3 & sp_4 is 60.0.
        # If d_buffer = 50.0: sp_2 is quarantined (< 50), sp_3 & sp_4 remain (>= 50).
        quarantined = self.engine.compute_buffer_quarantine(
            test_taxa=["sp_1"],
            all_taxa=self.taxa,
            patristic_matrix=self.D,
            d_buffer=50.0,
        )
        self.assertEqual(quarantined, ["sp_2"])

        # If d_buffer = 30.0: nothing quarantined (< 30)
        quarantined_none = self.engine.compute_buffer_quarantine(
            test_taxa=["sp_1"],
            all_taxa=self.taxa,
            patristic_matrix=self.D,
            d_buffer=30.0,
        )
        self.assertEqual(len(quarantined_none), 0)

    def test_validate_monophyly(self):
        # [sp_1, sp_2] is monophyletic (descendants of CladeA)
        self.assertTrue(self.engine.validate_monophyly(["sp_1", "sp_2"]))

        # [sp_1, sp_3] is paraphyletic/polyphyletic
        self.assertFalse(self.engine.validate_monophyly(["sp_1", "sp_3"]))


if __name__ == "__main__":
    unittest.main()
