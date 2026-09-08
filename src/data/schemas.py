import re
from typing import List, Optional
import pandas as pd

ID_COL = "canonical_taxon_id"
BIOCLIM_VARS = [f"bio{i}" for i in range(1, 20)]

def normalize_taxon_id(name: str) -> str:
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
) -> None:
    if id_col not in df.columns:
        raise ValueError(f"Missing required primary key identifier: '{id_col}'")
    
    if df[id_col].duplicated().any():
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
    required = [ID_COL, "latitude", "longitude"]
    validate_dataframe(df, required_cols=required, allow_extra_cols=True)
    
    if not df["latitude"].between(-90, 90).all():
        raise ValueError("Latitude values must be within [-90, 90]")
    if not df["longitude"].between(-180, 180).all():
        raise ValueError("Longitude values must be within [-180, 180]")

def validate_environmental_features(df: pd.DataFrame) -> None:
    validate_dataframe(df, required_cols=[ID_COL] + BIOCLIM_VARS, allow_extra_cols=True)

def validate_genomic_features(df: pd.DataFrame, feature_prefix: str = "gen_") -> None:
    if ID_COL not in df.columns:
        raise ValueError(f"Missing primary key: '{ID_COL}'")
    feature_cols = [c for c in df.columns if c != ID_COL]
    if len(feature_cols) == 0:
        raise ValueError("Genomic table contains no feature columns")
    validate_dataframe(df, required_cols=[ID_COL] + feature_cols, allow_extra_cols=True)
