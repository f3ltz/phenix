import unittest
import pandas as pd
import numpy as np
from src.data.schemas import ID_COL
from src.data.genomic import (
    ESM2SequenceEmbedder,
    SNPMatrixProcessor,
    FunctionalGenomicProcessor,
    CompositeGenomicPipeline,
    generate_synthetic_genomics,
)


class TestGenomic(unittest.TestCase):
    def setUp(self):
        self.taxa = ["taxon_1", "taxon_2", "taxon_3", "taxon_4", "taxon_5"]

    def test_esm2_sequence_embedder(self):
        embedder = ESM2SequenceEmbedder(n_components=4, embedding_dim=32)
        # Test individual protein sequence embedding
        seq = "MKWVTFISLLLLFSSAYSRGVFRRDTHKSEIAHRFKDLGE"
        emb = embedder.embed_amino_acid_sequence(seq, "taxon_1")
        self.assertEqual(len(emb), 32)

        # Test processing raw embeddings
        raw_df = pd.DataFrame(
            np.random.randn(5, 20),
            columns=[f"dim_{i}" for i in range(20)]
        )
        raw_df.insert(0, ID_COL, self.taxa)
        processed = embedder.process_busco_embeddings(raw_df, fit_reducer=True)

        self.assertEqual(len(processed), 5)
        self.assertEqual(processed[ID_COL].tolist(), self.taxa)
        self.assertEqual(len([c for c in processed.columns if c.startswith("esm2_dim_")]), 4)
        self.assertFalse(processed.isnull().any().any())

    def test_snp_matrix_processor(self):
        # 5 taxa, 20 markers with some missing values
        rng = np.random.default_rng(42)
        genotypes = rng.choice([0.0, 1.0, 2.0, np.nan], size=(5, 20), p=[0.6, 0.2, 0.1, 0.1])
        snp_df = pd.DataFrame(genotypes, columns=[f"m_{i}" for i in range(20)])
        snp_df.insert(0, ID_COL, self.taxa)

        processor = SNPMatrixProcessor(n_components=3, min_maf=0.01)
        res = processor.fit_transform(snp_df)

        self.assertEqual(len(res), 5)
        self.assertEqual(res[ID_COL].tolist(), self.taxa)
        self.assertEqual(len([c for c in res.columns if c.startswith("snp_dim_")]), 3)
        self.assertFalse(res.isnull().any().any())

    def test_functional_genomic_processor(self):
        func_data = np.random.gamma(2.0, 2.0, size=(5, 10))
        func_df = pd.DataFrame(func_data, columns=[f"feat_{i}" for i in range(10)])
        func_df.insert(0, ID_COL, self.taxa)

        processor = FunctionalGenomicProcessor(n_components=3)
        res = processor.fit_transform(func_df)

        self.assertEqual(len(res), 5)
        self.assertEqual(res[ID_COL].tolist(), self.taxa)
        self.assertEqual(len([c for c in res.columns if c.startswith("func_dim_")]), 3)
        self.assertFalse(res.isnull().any().any())

    def test_composite_genomic_pipeline(self):
        pipeline = CompositeGenomicPipeline(
            esm2_components=4,
            snp_components=3,
            func_components=3,
        )
        genomic_table = pipeline.assemble(taxa=self.taxa)

        self.assertEqual(len(genomic_table), 5)
        self.assertEqual(genomic_table[ID_COL].tolist(), self.taxa)
        # 4 esm2 + 3 snp + 3 func = 10 feature columns + ID_COL = 11 total columns
        self.assertEqual(genomic_table.shape[1], 11)
        self.assertFalse(genomic_table.isnull().any().any())


if __name__ == "__main__":
    unittest.main()
