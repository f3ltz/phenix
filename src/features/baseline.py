from typing import Optional, List
import pandas as pd
from src.data.schemas import (
    ID_COL,
    validate_environmental_features,
    validate_genomic_features,
    validate_dataframe,
)


class BaselineFeatureBuilder:
    """
    Constructs the Baseline Feature Set by merging Environmental variables
    (WorldClim v2.1 bio1-bio19 + habitat parameters) and Genomic features
    (ESM-2 BUSCO sequence embeddings, SNP PCs, functional scores).
    """

    def construct_baseline_features(
        self,
        df_env: pd.DataFrame,
        df_genomic: pd.DataFrame,
        target_taxa: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Merges environmental and genomic tables on canonical_taxon_id.
        Enforces identical taxon ordering and zero missing values.
        """
        if ID_COL not in df_env.columns:
            raise ValueError(f"Environmental table missing '{ID_COL}'")
        if ID_COL not in df_genomic.columns:
            raise ValueError(f"Genomic table missing '{ID_COL}'")

        validate_environmental_features(df_env)
        validate_genomic_features(df_genomic)

        taxa_env = set(df_env[ID_COL])
        taxa_gen = set(df_genomic[ID_COL])
        common = taxa_env.intersection(taxa_gen)

        if not common:
            raise ValueError("Zero overlapping taxa between environmental and genomic tables.")

        if target_taxa is not None:
            ordered_taxa = [t for t in target_taxa if t in common]
        else:
            ordered_taxa = sorted(list(common))

        env_sub = df_env.set_index(ID_COL).loc[ordered_taxa].reset_index()
        gen_sub = df_genomic.set_index(ID_COL).loc[ordered_taxa].reset_index()

        # Combine features
        gen_cols = [c for c in gen_sub.columns if c != ID_COL]
        merged = pd.merge(env_sub, gen_sub[[ID_COL] + gen_cols], on=ID_COL, how="inner")

        # Zero NaN validation
        if merged.isnull().any().any():
            nan_cols = merged.columns[merged.isnull().any()].tolist()
            raise ValueError(f"Baseline feature matrix contains NaNs in: {nan_cols}")

        return merged
