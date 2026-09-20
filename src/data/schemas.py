import re
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

ID_COL = "canonical_taxon_id"

# 19 WorldClim v2.1 bioclimatic variables
BIOCLIM_VARS = [f"bio{i}" for i in range(1, 20)]

# Standard habitat parameters
HABITAT_VARS = ["elevation", "biome_class", "habitat_suitability"]

# Standard feature prefixes for multimodal alignment
GENOMIC_PREFIXES = ("esm2_", "snp_", "func_", "gen_")

# Human-readable labels for WorldClim variables
BIOCLIM_DESCRIPTIONS = {
    "bio1": "Annual Mean Temperature",
    "bio2": "Mean Diurnal Range",
    "bio3": "Isothermality",
    "bio4": "Temperature Seasonality",
    "bio5": "Max Temperature of Warmest Month",
    "bio6": "Min Temperature of Coldest Month",
    "bio7": "Temperature Annual Range",
    "bio8": "Mean Temperature of Wettest Quarter",
    "bio9": "Mean Temperature of Driest Quarter",
    "bio10": "Mean Temperature of Warmest Quarter",
    "bio11": "Mean Temperature of Coldest Quarter",
    "bio12": "Annual Precipitation",
    "bio13": "Precipitation of Wettest Month",
    "bio14": "Precipitation of Driest Month",
    "bio15": "Precipitation Seasonality",
    "bio16": "Precipitation of Wettest Quarter",
    "bio17": "Precipitation of Driest Quarter",
    "bio18": "Precipitation of Warmest Quarter",
    "bio19": "Precipitation of Coldest Quarter",
}


def normalize_taxon_id(name: str) -> str:
    """Standardizes binomial species name to canonical lowercase snake_case identifier."""
    cleaned = name.strip()
    cleaned = re.sub(r"[\s\-]+", "_", cleaned)
    cleaned = re.sub(r"[^\w_]", "", cleaned)
    return cleaned.lower()


def validate_dataframe(
    df: pd.DataFrame,
    required_cols: List[str],
    id_col: str = ID_COL,
    allow_extra_cols: bool = True,
    allow_nan_cols: Optional[List[str]] = None,
    allow_duplicates: bool = False,
) -> None:
    """Validates dataframe for primary key uniqueness, required columns, and unexpected NaNs."""
    if id_col not in df.columns:
        raise ValueError(f"Missing required primary key identifier: '{id_col}'")

    if not allow_duplicates and df[id_col].duplicated().any():
        dups = df[id_col][df[id_col].duplicated()].unique().tolist()
        raise ValueError(f"Duplicate identifiers found in '{id_col}': {dups[:5]}")

    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(list(missing_cols))}")

    if not allow_extra_cols:
        extra_cols = set(df.columns) - set(required_cols) - {id_col}
        if extra_cols:
            raise ValueError(f"Unexpected extra columns found: {sorted(list(extra_cols))}")

    cols_to_check_nan = set(required_cols)
    if allow_nan_cols:
        cols_to_check_nan -= set(allow_nan_cols)

    for col in cols_to_check_nan:
        if df[col].isnull().any():
            nan_count = df[col].isnull().sum()
            raise ValueError(f"Column '{col}' contains {nan_count} NaN/null values. Non-null values required.")


def validate_occurrence_data(df: pd.DataFrame) -> None:
    """Validates occurrence coordinate data."""
    required = [ID_COL, "latitude", "longitude"]
    validate_dataframe(df, required_cols=required, allow_extra_cols=True, allow_duplicates=True)

    if not df["latitude"].between(-90, 90).all():
        raise ValueError("Latitude values must be within [-90, 90]")
    if not df["longitude"].between(-180, 180).all():
        raise ValueError("Longitude values must be within [-180, 180]")


def validate_environmental_features(
    df: pd.DataFrame,
    require_habitat: bool = False
) -> None:
    """Validates environmental feature dataframe containing WorldClim bioclimatic and habitat variables."""
    required = [ID_COL] + BIOCLIM_VARS
    if require_habitat:
        required.append("elevation")
    validate_dataframe(df, required_cols=required, allow_extra_cols=True)


def validate_genomic_features(
    df: pd.DataFrame,
    feature_prefix: Optional[str] = None
) -> None:
    """Validates genomic feature table containing ESM-2, SNP, or functional genomic dimensions."""
    if ID_COL not in df.columns:
        raise ValueError(f"Missing primary key: '{ID_COL}'")
    feature_cols = [c for c in df.columns if c != ID_COL]
    if len(feature_cols) == 0:
        raise ValueError("Genomic table contains no feature columns")
    if feature_prefix:
        non_matching = [c for c in feature_cols if not c.startswith(feature_prefix)]
        if non_matching:
            raise ValueError(f"Features missing expected prefix '{feature_prefix}': {non_matching[:5]}")
    validate_dataframe(df, required_cols=[ID_COL] + feature_cols, allow_extra_cols=True)


def validate_esm2_features(df: pd.DataFrame) -> None:
    """Validates ESM-2 sequence embedding feature table."""
    validate_genomic_features(df, feature_prefix="esm2_")


def validate_snp_features(df: pd.DataFrame) -> None:
    """Validates SNP marker feature table."""
    validate_genomic_features(df, feature_prefix="snp_")


def evaluate_data_completeness(
    df_pheno: pd.DataFrame,
    df_env: Optional[pd.DataFrame] = None,
    df_genomic: Optional[pd.DataFrame] = None,
    group_col: Optional[str] = "order",
) -> pd.DataFrame:
    """
    Evaluates data completeness across phenotypic, environmental, and genomic modalities.
    Used for Sync Barrier 1 to confirm target taxon group selection.
    """
    if ID_COL not in df_pheno.columns:
        raise ValueError(f"Phenotypic dataframe must contain '{ID_COL}'")

    records = []
    pheno_taxa = set(df_pheno[ID_COL])
    env_taxa = set(df_env[ID_COL]) if df_env is not None else set()
    gen_taxa = set(df_genomic[ID_COL]) if df_genomic is not None else set()

    for _, row in df_pheno.iterrows():
        t_id = row[ID_COL]
        group = row.get(group_col, "Unknown") if group_col and group_col in df_pheno.columns else "All"
        
        # Trait completeness in pheno
        trait_cols = [c for c in df_pheno.columns if c not in {ID_COL, group_col, "species", "genus", "family"}]
        pheno_completeness = (
            float((~row[trait_cols].isnull()).sum() / len(trait_cols))
            if trait_cols else 1.0
        )
        
        has_env = t_id in env_taxa if df_env is not None else True
        has_gen = t_id in gen_taxa if df_genomic is not None else True
        
        all_modalities = (pheno_completeness > 0.5) and has_env and has_gen

        records.append({
            ID_COL: t_id,
            "group": group,
            "phenotypic_completeness": pheno_completeness,
            "has_environmental": has_env,
            "has_genomic": has_gen,
            "all_modalities_complete": all_modalities,
        })

    summary_df = pd.DataFrame(records)
    return summary_df
