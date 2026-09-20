import unittest
import numpy as np
import pandas as pd
from src.data.schemas import ID_COL, BIOCLIM_VARS
from src.features.baseline import BaselineFeatureBuilder
from src.features.whitening import CholeskyWhitener
from src.features.graph import PhyloGraphConverter
from src.features.augmented import PhyloAugmentedFeatureBuilder
from src.phylogenetics.otol import OToLClient
from src.phylogenetics.calibration import TreeCalibrator


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.taxa = ["species_1", "species_2", "species_3", "species_4"]
        
        # Mock environmental
        env_dict = {ID_COL: self.taxa, "elevation": [100.0, 200.0, 300.0, 400.0]}
        for b in BIOCLIM_VARS:
            env_dict[b] = [15.0, 20.0, 25.0, 30.0]
        self.df_env = pd.DataFrame(env_dict)

        # Mock genomic
        self.df_genomic = pd.DataFrame({
            ID_COL: self.taxa,
            "esm2_dim_1": [0.1, 0.2, 0.3, 0.4],
            "snp_dim_1": [1.0, 0.0, 1.0, 0.0],
            "func_dim_1": [2.5, 3.5, 4.5, 5.5],
        })

    def test_baseline_feature_builder(self):
        builder = BaselineFeatureBuilder()
        df_base = builder.construct_baseline_features(self.df_env, self.df_genomic, target_taxa=self.taxa)

        self.assertEqual(len(df_base), 4)
        self.assertEqual(df_base[ID_COL].tolist(), self.taxa)
        self.assertIn("bio1", df_base.columns)
        self.assertIn("esm2_dim_1", df_base.columns)
        self.assertFalse(df_base.isnull().any().any())

    def test_cholesky_whitening(self):
        rng = np.random.default_rng(42)
        # Create correlated features
        N = 50
        taxa = [f"sp_{i}" for i in range(N)]
        x1 = rng.standard_normal(N)
        x2 = 0.8 * x1 + 0.2 * rng.standard_normal(N)
        x3 = 0.5 * x1 - 0.4 * x2 + rng.standard_normal(N)

        df = pd.DataFrame({ID_COL: taxa, "f1": x1, "f2": x2, "f3": x3})
        whitener = CholeskyWhitener(epsilon=1e-5)
        df_white, cov_white = whitener.fit_transform(df)

        self.assertEqual(len(df_white), N)
        self.assertEqual(df_white[ID_COL].tolist(), taxa)
        # Verify covariance of whitened data is close to identity
        self.assertTrue(np.allclose(cov_white, np.eye(3), atol=1e-2))

    def test_phylo_graph_converter(self):
        taxa_df = pd.DataFrame({
            ID_COL: self.taxa,
            "order": ["Carnivora", "Carnivora", "Primates", "Rodentia"],
            "family": ["Felidae", "Canidae", "Hominidae", "Muridae"],
        })
        client = OToLClient()
        root = client.build_mammal_backbone_topology(taxa_df)
        calibrated = TreeCalibrator().calibrate_tree(root)

        converter = PhyloGraphConverter()
        graph = converter.tree_to_pyg_graph(calibrated, self.df_genomic)

        self.assertEqual(graph["num_tips"], 4)
        self.assertGreater(graph["num_nodes"], 4)
        self.assertEqual(graph["edge_index"].shape[0], 2)
        self.assertGreater(graph["edge_index"].shape[1], 0)
        self.assertEqual(graph["edge_attr"].shape[0], graph["edge_index"].shape[1])

    def test_phylo_augmented_feature_builder(self):
        builder = BaselineFeatureBuilder()
        df_base = builder.construct_baseline_features(self.df_env, self.df_genomic, target_taxa=self.taxa)

        df_phylo = pd.DataFrame({
            ID_COL: self.taxa,
            "phylo_dim_1": [0.5, -0.2, 0.3, -0.6],
            "phylo_dim_2": [-0.1, 0.4, -0.3, 0.0],
        })

        aug_builder = PhyloAugmentedFeatureBuilder()
        df_aug = aug_builder.assemble_augmented_features(df_base, df_phylo)

        self.assertEqual(len(df_aug), 4)
        self.assertEqual(df_aug[ID_COL].tolist(), self.taxa)
        self.assertIn("bio1", df_aug.columns)
        self.assertIn("esm2_dim_1", df_aug.columns)
        self.assertIn("phylo_dim_1", df_aug.columns)
        self.assertFalse(df_aug.isnull().any().any())


if __name__ == "__main__":
    unittest.main()
