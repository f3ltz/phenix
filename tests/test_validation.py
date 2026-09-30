import unittest
import numpy as np
import pandas as pd
from src.data.schemas import ID_COL
from src.phylogenetics.otol import PhyloNode
from src.phylogenetics.withholding import MonophyleticWithholdingEngine
from src.validation.cv import CVFold, RandomKFoldSplitter, PhyloCVSplitter


class TestValidationSplitters(unittest.TestCase):
    def setUp(self):
        self.taxa = [f"taxon_{i}" for i in range(20)]
        self.df_pheno = pd.DataFrame({
            ID_COL: self.taxa,
            "order": ["Order1"] * 10 + ["Order2"] * 10,
            "adult_body_mass_g": [10.0 + i for i in range(20)],
        })

        # Toy tree with Order1 and Order2
        self.root = PhyloNode(name="Mammalia", length=0.0)
        self.o1 = PhyloNode(name="Order1", length=50.0)
        self.o2 = PhyloNode(name="Order2", length=50.0)
        self.root.add_child(self.o1)
        self.root.add_child(self.o2)

        for i in range(10):
            self.o1.add_child(PhyloNode(name=f"taxon_{i}", length=20.0))
        for i in range(10, 20):
            self.o2.add_child(PhyloNode(name=f"taxon_{i}", length=20.0))

        # Patristic matrix: within order = 40.0, between orders = 140.0
        self.D = np.zeros((20, 20), dtype=np.float32)
        for i in range(20):
            for j in range(20):
                if i != j:
                    if (i < 10 and j < 10) or (i >= 10 and j >= 10):
                        self.D[i, j] = 40.0
                    else:
                        self.D[i, j] = 140.0

        self.engine = MonophyleticWithholdingEngine(tree_root=self.root)

    def test_cv_fold_disjointness_and_properties(self):
        fold = CVFold(
            fold_id=0,
            train_indices=np.array([0, 1, 2]),
            test_indices=np.array([3, 4]),
            buffer_indices=np.array([5]),
        )
        self.assertTrue(fold.verify_disjointness())
        self.assertEqual(fold.n_train, 3)
        self.assertEqual(fold.n_test, 2)
        self.assertEqual(fold.n_buffer, 1)

        # Leaking fold
        bad_fold = CVFold(
            fold_id=1,
            train_indices=np.array([0, 1, 2]),
            test_indices=np.array([2, 3]),
            buffer_indices=np.array([]),
        )
        self.assertFalse(bad_fold.verify_disjointness())

    def test_random_kfold_splitter(self):
        splitter = RandomKFoldSplitter(n_splits=4, shuffle=True, random_state=42)
        folds = list(splitter.split(self.taxa))

        self.assertEqual(len(folds), 4)
        all_test_indices = []
        for fold in folds:
            self.assertTrue(fold.verify_disjointness())
            self.assertEqual(fold.n_buffer, 0)
            all_test_indices.extend(fold.test_indices.tolist())

        # Check full coverage
        self.assertEqual(sorted(all_test_indices), list(range(20)))

    def test_phylo_cv_splitter_order(self):
        phylo_cv = PhyloCVSplitter(
            all_taxa=self.taxa,
            patristic_matrix=self.D,
            withholding_engine=self.engine,
            mode="order",
            df_pheno=self.df_pheno,
            min_clade_size=10,
            d_buffer=100.0,
        )

        folds = list(phylo_cv.split())
        self.assertEqual(len(folds), 2)

        for fold in folds:
            self.assertTrue(fold.verify_disjointness())
            self.assertEqual(fold.n_test, 10)
            self.assertEqual(fold.n_train, 10)
            self.assertEqual(fold.n_buffer, 0)

    def test_phylo_cv_splitter_with_buffer(self):
        # If d_buffer is 150.0 Ma, holding out Order1 will quarantine Order2 (dist=140 < 150)
        phylo_cv_buf = PhyloCVSplitter(
            all_taxa=self.taxa,
            patristic_matrix=self.D,
            withholding_engine=self.engine,
            mode="order",
            df_pheno=self.df_pheno,
            min_clade_size=10,
            d_buffer=150.0,
        )
        folds = list(phylo_cv_buf.split())
        self.assertEqual(len(folds), 2)
        f = folds[0]
        self.assertEqual(f.n_test, 10)
        self.assertEqual(f.n_buffer, 10)
        self.assertEqual(f.n_train, 0)


if __name__ == "__main__":
    unittest.main()
