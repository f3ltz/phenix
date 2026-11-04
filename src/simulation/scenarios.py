"""
Climate & Proteomic Perturbation Scenario Generator for PHENIX
Generates future environmental and genomic feature states:
  1. Bioclimatic warming rasters aligned with CMIP6 / IPCC SSP shared socioeconomic pathways.
  2. In silico molecular clock sequence mutations and drifted ESM-2 transformer embeddings.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.data.schemas import ID_COL, BIOCLIM_VARS
from src.data.genomic import ESM2SequenceEmbedder


class ClimateScenarioGenerator:
    """
    Generates perturbed WorldClim bioclimatic feature matrices under projected warming regimes.
    Adjusts both mean thermal parameters and seasonal extremes according to CMIP6 climatology.
    """

    # Pre-defined IPCC Sixth Assessment Report (AR6) scenarios for 2070
    SSP_PRESETS = {
        "SSP1-2.6": {"delta_temp_c": 1.5, "precip_factor": 1.02, "extreme_factor": 1.10},
        "SSP2-4.5": {"delta_temp_c": 3.0, "precip_factor": 1.05, "extreme_factor": 1.20},
        "SSP5-8.5": {"delta_temp_c": 5.0, "precip_factor": 1.08, "extreme_factor": 1.35},
    }

    def apply_warming_scenario(
        self,
        df_env: pd.DataFrame,
        delta_temp_c: float = 3.0,
        precip_factor: float = 1.05,
        extreme_factor: float = 1.20,
    ) -> pd.DataFrame:
        """
        Applies parameterized climate perturbation to environmental features:
          - Temperature variables (bio1, bio5, bio6, bio8, bio9, bio10, bio11) shifted by delta_temp.
          - Max/Min seasonal extremes (bio5, bio6) scaled by extreme_factor.
          - Precipitation variables (bio12-bio19) scaled by precip_factor.
        """
        df_future = df_env.copy()

        # Thermal variables
        temp_vars_mean = ["bio1", "bio8", "bio9", "bio10", "bio11"]
        for v in temp_vars_mean:
            if v in df_future.columns:
                df_future[v] = df_future[v] + delta_temp_c

        # Thermal extremes (polar/summer amplification)
        if "bio5" in df_future.columns:
            df_future["bio5"] = df_future["bio5"] + (delta_temp_c * extreme_factor)
        if "bio6" in df_future.columns:
            df_future["bio6"] = df_future["bio6"] + (delta_temp_c * extreme_factor)

        # Diurnal and annual ranges
        if "bio7" in df_future.columns and "bio5" in df_future.columns and "bio6" in df_future.columns:
            df_future["bio7"] = df_future["bio5"] - df_future["bio6"]

        # Precipitation variables
        precip_vars = ["bio12", "bio13", "bio14", "bio15", "bio16", "bio17", "bio18", "bio19"]
        for v in precip_vars:
            if v in df_future.columns:
                df_future[v] = np.maximum(df_future[v] * precip_factor, 0.0)

        # Habitat suitability adjustment under thermal stress
        if "habitat_suitability" in df_future.columns:
            # Thermal stress reduces habitat suitability for high warming
            decay = np.exp(-0.05 * max(delta_temp_c - 1.5, 0.0))
            df_future["habitat_suitability"] = np.clip(df_future["habitat_suitability"] * decay, 0.0, 1.0)

        return df_future

    def get_preset_scenario(self, df_env: pd.DataFrame, preset_name: str) -> pd.DataFrame:
        """Applies an IPCC preset scenario (SSP1-2.6, SSP2-4.5, SSP5-8.5)."""
        if preset_name not in self.SSP_PRESETS:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(self.SSP_PRESETS.keys())}")
        params = self.SSP_PRESETS[preset_name]
        return self.apply_warming_scenario(
            df_env=df_env,
            delta_temp_c=params["delta_temp_c"],
            precip_factor=params["precip_factor"],
            extreme_factor=params["extreme_factor"],
        )


class ProteomicMutationGenerator:
    """
    Simulates in silico molecular clock protein sequence evolution across forward time:
      - Introduces non-synonymous amino acid substitutions via transition probabilities.
      - Updates or drifts ESM-2 protein language representations (Z_genomic).
    """

    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.embedder = ESM2SequenceEmbedder(random_state=random_state)

    def mutate_amino_acid_sequence(
        self,
        sequence: str,
        years_forward: float = 100_000.0,
        mutation_rate_per_site_per_year: float = 1e-6,
    ) -> str:
        """
        Applies Poisson-distributed point mutations along an amino acid sequence:
          Expected mutations per site: lambda = mu * t
        """
        seq_list = list(sequence.upper())
        seq_len = len(seq_list)
        expected_subs = mutation_rate_per_site_per_year * years_forward * seq_len
        n_mutations = int(self.rng.poisson(lam=max(expected_subs, 0.0)))
        n_mutations = min(n_mutations, seq_len)

        if n_mutations > 0:
            mutation_indices = self.rng.choice(seq_len, size=n_mutations, replace=False)
            for idx in mutation_indices:
                orig_aa = seq_list[idx]
                candidates = [aa for aa in self.AMINO_ACIDS if aa != orig_aa]
                seq_list[idx] = str(self.rng.choice(candidates))

        return "".join(seq_list)

    def apply_genomic_drift(
        self,
        df_genomic: pd.DataFrame,
        years_forward_ma: float = 1.0,
        drift_rate: float = 0.05,
    ) -> pd.DataFrame:
        """
        Applies molecular clock diffusion to the genomic feature space (ESM-2 + SNP PCs):
          Z(t + dt) = Z(t) + N(0, drift_rate * sqrt(dt))
        Preserves canonical_taxon_id and feature column definitions.
        """
        df_mutated = df_genomic.copy()
        # Indicator columns (e.g. gen_has_sequence) are not sequence coordinates and must not drift
        feature_cols = [c for c in df_mutated.columns if c != ID_COL and not c.startswith("gen_has_")]

        dt = float(years_forward_ma)
        std = float(drift_rate * np.sqrt(max(dt, 1e-4)))
        noise = self.rng.normal(0.0, std, size=(len(df_mutated), len(feature_cols)))

        df_mutated[feature_cols] = df_mutated[feature_cols].values + noise.astype(np.float32)
        return df_mutated
