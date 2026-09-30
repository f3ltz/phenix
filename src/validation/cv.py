from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterator, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from src.data.schemas import ID_COL
from src.phylogenetics.withholding import MonophyleticWithholdingEngine


@dataclass
class CVFold:
    """
    Data structure representing a cross-validation fold with strict partition tracking.
    """
    fold_id: Union[int, str]
    train_indices: np.ndarray
    test_indices: np.ndarray
    buffer_indices: np.ndarray
    clade_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return len(self.train_indices)

    @property
    def n_test(self) -> int:
        return len(self.test_indices)

    @property
    def n_buffer(self) -> int:
        return len(self.buffer_indices)

    def verify_disjointness(self) -> bool:
        """Verifies train, test, and buffer index sets are mutually disjoint."""
        s_train = set(self.train_indices.tolist())
        s_test = set(self.test_indices.tolist())
        s_buffer = set(self.buffer_indices.tolist())

        if s_train.intersection(s_test):
            return False
        if s_train.intersection(s_buffer):
            return False
        if s_test.intersection(s_buffer):
            return False
        return True


class RandomKFoldSplitter:
    """
    Standard Random K-Fold Cross-Validation generator.
    Represents the classical 'interpolation' regime where species are randomly
    assigned across folds without evolutionary structure.
    """

    def __init__(
        self,
        n_splits: int = 10,
        shuffle: bool = True,
        random_state: int = 42,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, all_taxa: List[str]) -> Iterator[CVFold]:
        """Generates Random K-Fold splits."""
        N = len(all_taxa)
        indices = np.arange(N)
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(indices)):
            yield CVFold(
                fold_id=fold_idx,
                train_indices=train_idx,
                test_indices=test_idx,
                buffer_indices=np.array([], dtype=int),
                clade_name=f"random_fold_{fold_idx}",
                metadata={"protocol": "random_k_fold", "n_splits": self.n_splits},
            )

    def get_n_splits(self) -> int:
        return self.n_splits


class PhyloCVSplitter:
    """
    Phylogenetic Cross-Validation (Phylo-CV) Generator.
    Evaluates true macroevolutionary extrapolation by withholding monophyletic
    taxonomic clades (orders, T_cut lineages) and quarantining sister taxa in
    patristic buffer zones.
    """

    def __init__(
        self,
        all_taxa: List[str],
        patristic_matrix: np.ndarray,
        withholding_engine: MonophyleticWithholdingEngine,
        mode: str = "order",
        df_pheno: Optional[pd.DataFrame] = None,
        min_clade_size: int = 50,
        t_cut_ma: Optional[float] = None,
        d_buffer: float = 0.0,
    ):
        self.all_taxa = list(all_taxa)
        self.patristic_matrix = patristic_matrix
        self.withholding_engine = withholding_engine
        self.mode = mode.lower()
        self.df_pheno = df_pheno
        self.min_clade_size = min_clade_size
        self.t_cut_ma = t_cut_ma
        self.d_buffer = float(d_buffer)

        self.taxa_to_idx = {t: i for i, t in enumerate(self.all_taxa)}
        self._clade_groups = self._prepare_clades()

    def _prepare_clades(self) -> Dict[str, List[str]]:
        """Identifies target monophyletic clades for holdout evaluation."""
        if self.mode == "order":
            if self.df_pheno is None:
                raise ValueError("df_pheno is required for order-level Phylo-CV.")
            return self.withholding_engine.get_order_clades(
                self.df_pheno, min_clade_size=self.min_clade_size
            )
        elif self.mode == "t_cut":
            t_cut = self.t_cut_ma or 65.0
            clusters = self.withholding_engine.get_t_cut_clusters(t_cut_ma=t_cut)
            # Filter for clusters with sufficient sample size
            return {
                name: leaves
                for name, leaves in clusters.items()
                if len(leaves) >= self.min_clade_size
            }
        else:
            raise ValueError(f"Unsupported Phylo-CV mode: {self.mode}. Must be 'order' or 't_cut'.")

    def split(self) -> Iterator[CVFold]:
        """
        Yields CVFold for each monophyletic evaluation clade with buffer quarantine.
        """
        all_indices = np.arange(len(self.all_taxa))

        for clade_name, test_taxa in self._clade_groups.items():
            test_idx = np.array(
                [self.taxa_to_idx[t] for t in test_taxa if t in self.taxa_to_idx],
                dtype=int,
            )
            if len(test_idx) == 0:
                continue

            # Compute buffer quarantine
            buffer_taxa = self.withholding_engine.compute_buffer_quarantine(
                test_taxa=test_taxa,
                all_taxa=self.all_taxa,
                patristic_matrix=self.patristic_matrix,
                d_buffer=self.d_buffer,
            )
            buffer_idx = np.array(
                [self.taxa_to_idx[t] for t in buffer_taxa if t in self.taxa_to_idx],
                dtype=int,
            )

            # Clean training set: All \ (Test U Buffer)
            excluded_set = set(test_idx.tolist()).union(set(buffer_idx.tolist()))
            train_idx = np.array(
                [i for i in all_indices if i not in excluded_set],
                dtype=int,
            )

            fold = CVFold(
                fold_id=clade_name,
                train_indices=train_idx,
                test_indices=test_idx,
                buffer_indices=buffer_idx,
                clade_name=clade_name,
                metadata={
                    "protocol": "phylo_cv",
                    "mode": self.mode,
                    "d_buffer": self.d_buffer,
                    "t_cut_ma": self.t_cut_ma,
                    "clade_size": len(test_idx),
                },
            )

            if not fold.verify_disjointness():
                raise RuntimeError(f"Clade leakage detected in fold '{clade_name}'!")

            yield fold

    def get_n_splits(self) -> int:
        return len(self._clade_groups)

    def get_clade_names(self) -> List[str]:
        return list(self._clade_groups.keys())
