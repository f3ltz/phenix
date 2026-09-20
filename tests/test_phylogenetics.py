import unittest
import numpy as np
import pandas as pd
from src.data.schemas import ID_COL
from src.phylogenetics.otol import PhyloNode, OToLClient
from src.phylogenetics.calibration import TreeCalibrator
from src.phylogenetics.patristic import PatristicMatrixEngine


class TestPhylogenetics(unittest.TestCase):
    def setUp(self):
        self.taxa = ["panthera_leo", "canis_lupus", "homo_sapiens", "mus_musculus", "bos_taurus"]
        self.taxa_df = pd.DataFrame({
            ID_COL: self.taxa,
            "order": ["Carnivora", "Carnivora", "Primates", "Rodentia", "Cetartiodactyla"],
            "family": ["Felidae", "Canidae", "Hominidae", "Muridae", "Bovidae"],
        })

    def test_phylo_node_newick(self):
        newick = "((panthera_leo:1.5,canis_lupus:1.5)Carnivora:2.0,homo_sapiens:3.5)Root;"
        node = PhyloNode.from_newick(newick)
        leaves = node.get_leaf_names()
        self.assertIn("panthera_leo", leaves)
        self.assertIn("canis_lupus", leaves)
        self.assertIn("homo_sapiens", leaves)
        self.assertEqual(len(leaves), 3)

    def test_otol_backbone_topology(self):
        client = OToLClient()
        root = client.build_mammal_backbone_topology(self.taxa_df)
        leaves = root.get_leaf_names()
        for t in self.taxa:
            self.assertIn(t, leaves)
        self.assertEqual(len(leaves), len(self.taxa))

    def test_tree_calibrator_bladj(self):
        client = OToLClient()
        root = client.build_mammal_backbone_topology(self.taxa_df)
        calibrator = TreeCalibrator()
        calibrated = calibrator.calibrate_tree(root)

        leaves = calibrated.get_leaves()
        for leaf in leaves:
            self.assertEqual(leaf.age, 0.0)
            self.assertGreater(leaf.length, 0.0)

        self.assertEqual(calibrated.age, 177.0)

    def test_patristic_engine_and_pcoa(self):
        client = OToLClient()
        root = client.build_mammal_backbone_topology(self.taxa_df)
        calibrator = TreeCalibrator()
        calibrated = calibrator.calibrate_tree(root)

        engine = PatristicMatrixEngine()
        D = engine.compute_patristic_distance_matrix(calibrated, self.taxa)

        self.assertEqual(D.shape, (5, 5))
        self.assertTrue(np.allclose(D, D.T))
        self.assertTrue(np.allclose(np.diag(D), 0.0))
        # Lion and wolf (both Carnivora) should be closer than lion and human
        idx_lion = self.taxa.index("panthera_leo")
        idx_wolf = self.taxa.index("canis_lupus")
        idx_human = self.taxa.index("homo_sapiens")
        self.assertLess(D[idx_lion, idx_wolf], D[idx_lion, idx_human])

        B = engine.compute_double_centered_matrix(D)
        self.assertEqual(B.shape, (5, 5))
        # Centering property: row and column sums should be 0
        self.assertTrue(np.allclose(B.sum(axis=1), 0.0, atol=1e-3))

        df_pcoa, evals = engine.compute_phylo_pcoa_eigenmaps(B, self.taxa, n_components=3)
        self.assertEqual(len(df_pcoa), 5)
        self.assertEqual(df_pcoa[ID_COL].tolist(), self.taxa)
        self.assertIn("phylo_dim_1", df_pcoa.columns)
        self.assertFalse(df_pcoa.isnull().any().any())


if __name__ == "__main__":
    unittest.main()
