from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from src.data.schemas import ID_COL, validate_genomic_features

class GenomicFeatureProcessor:
    def __init__(
        self,
        n_components: Optional[int] = 32,
        method: str = "pca",
        random_state: int = 42
    ):
        self.n_components = n_components
        self.method = method
        self.random_state = random_state
        self.reducer = None
        self.feature_cols: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_genomic_features(df)
        taxa = df[ID_COL].values
        self.feature_cols = [c for c in df.columns if c != ID_COL]
        X = df[self.feature_cols].values
        
        if self.n_components is not None and self.n_components < X.shape[1]:
            if self.method == "pca":
                self.reducer = PCA(n_components=self.n_components, random_state=self.random_state)
            elif self.method == "svd":
                self.reducer = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
            else:
                raise ValueError(f"Unsupported reduction method: {self.method}")
            
            X_reduced = self.reducer.fit_transform(X)
            col_names = [f"gen_dim_{i+1}" for i in range(self.n_components)]
        else:
            X_reduced = X
            col_names = self.feature_cols

        res_df = pd.DataFrame(X_reduced, columns=col_names)
        res_df.insert(0, ID_COL, taxa)
        return res_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.reducer is None and self.n_components is not None:
            raise RuntimeError("Processor has not been fitted yet. Call fit_transform first.")
        
        validate_genomic_features(df)
        taxa = df[ID_COL].values
        X = df[self.feature_cols].values
        
        if self.reducer is not None:
            X_reduced = self.reducer.transform(X)
            col_names = [f"gen_dim_{i+1}" for i in range(self.n_components)]
        else:
            X_reduced = X
            col_names = self.feature_cols
            
        res_df = pd.DataFrame(X_reduced, columns=col_names)
        res_df.insert(0, ID_COL, taxa)
        return res_df

def generate_synthetic_genomics(
    taxa: List[str],
    n_markers: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    genotypes = rng.choice([0, 1, 2], size=(len(taxa), n_markers), p=[0.7, 0.2, 0.1])
    col_names = [f"marker_{i+1}" for i in range(n_markers)]
    df = pd.DataFrame(genotypes, columns=col_names)
    df.insert(0, ID_COL, taxa)
    return df
