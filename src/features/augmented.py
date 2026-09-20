from typing import Optional, List
import pandas as pd
from src.data.schemas import ID_COL


class PhyloAugmentedFeatureBuilder:
    """
    Assembles the Phylogenetic-Augmented Feature Set by joining
    the Baseline Feature Set with Phylo-PCoA eigenmaps (or GNN representations).
    """

    def assemble_augmented_features(
        self,
        df_baseline: pd.DataFrame,
        df_phylo: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merges baseline features and phylogenetic eigenmaps on canonical_taxon_id.
        Guarantees strict 1:1 row index equality and zero missing values.
        """
        if ID_COL not in df_baseline.columns:
            raise ValueError(f"Baseline feature table missing '{ID_COL}'")
        if ID_COL not in df_phylo.columns:
            raise ValueError(f"Phylogenetic feature table missing '{ID_COL}'")

        taxa_base = df_baseline[ID_COL].tolist()
        taxa_phylo = set(df_phylo[ID_COL])

        # Check all baseline taxa are present in phylogenetic table
        missing_taxa = set(taxa_base) - taxa_phylo
        if missing_taxa:
            raise ValueError(f"Phylogenetic table missing {len(missing_taxa)} taxa: {list(missing_taxa)[:5]}")

        # Index and align strictly to baseline order
        phylo_aligned = df_phylo.set_index(ID_COL).loc[taxa_base].reset_index()

        phylo_cols = [c for c in phylo_aligned.columns if c != ID_COL]
        merged = pd.merge(df_baseline, phylo_aligned[[ID_COL] + phylo_cols], on=ID_COL, how="inner")

        if merged.isnull().any().any():
            nan_cols = merged.columns[merged.isnull().any()].tolist()
            raise ValueError(f"Augmented feature matrix contains NaNs in: {nan_cols}")

        return merged
