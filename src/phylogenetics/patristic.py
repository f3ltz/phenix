from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from src.data.schemas import ID_COL
from src.phylogenetics.otol import PhyloNode


class PatristicMatrixEngine:
    """
    Computes:
      1. Patristic evolutionary distance matrix D along tree branches.
      2. Gower's double-centered inner-product matrix B.
      3. Phylo-PCoA eigenmaps capturing multiscale phylogenetic gradients.
    """

    def compute_patristic_distance_matrix(
        self, root: PhyloNode, target_taxa: List[str]
    ) -> np.ndarray:
        """
        Computes pairwise patristic evolutionary distance matrix D (N x N)
        for target taxa on an ultrametric phylogenetic tree.
        On an ultrametric tree, D[i, j] = 2 * age(MRCA(i, j)).
        """
        N = len(target_taxa)
        taxa_to_idx = {name: idx for idx, name in enumerate(target_taxa)}
        D = np.zeros((N, N), dtype=np.float32)

        # Post-order traversal to collect subtree leaves and assign MRCA ages
        def traverse(node: PhyloNode) -> List[int]:
            if node.is_leaf():
                if node.name in taxa_to_idx:
                    return [taxa_to_idx[node.name]]
                return []

            child_leaf_lists = []
            for child in node.children:
                leaves = traverse(child)
                if leaves:
                    child_leaf_lists.append(leaves)

            node_age = float(node.age) if node.age is not None else 10.0
            dist_val = float(2.0 * node_age)

            # Assign distance for all pairs across distinct children
            for i in range(len(child_leaf_lists)):
                for j in range(i + 1, len(child_leaf_lists)):
                    group_a = child_leaf_lists[i]
                    group_b = child_leaf_lists[j]
                    for idx_a in group_a:
                        for idx_b in group_b:
                            if D[idx_a, idx_b] == 0.0:
                                D[idx_a, idx_b] = dist_val
                                D[idx_b, idx_a] = dist_val

            all_leaves = []
            for cl in child_leaf_lists:
                all_leaves.extend(cl)
            return all_leaves

        traverse(root)
        np.fill_diagonal(D, 0.0)
        return D

    def compute_double_centered_matrix(self, D: np.ndarray) -> np.ndarray:
        """
        Computes Gower's double-centered inner-product matrix B from distance matrix D:
        A = -0.5 * D^2
        B = H * A * H = A - mean_row - mean_col + mean_all
        """
        A = -0.5 * (D.astype(np.float64) ** 2)
        mean_row = A.mean(axis=1, keepdims=True)
        mean_col = A.mean(axis=0, keepdims=True)
        mean_all = A.mean()
        B = A - mean_row - mean_col + mean_all
        # Symmetrize
        B = (B + B.T) / 2.0
        return B

    def compute_phylo_pcoa_eigenmaps(
        self,
        B: np.ndarray,
        taxa: List[str],
        n_components: int = 32,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Performs Principal Coordinates Analysis (PCoA) on inner-product matrix B.
        Returns dataframe with phylogenetic eigenvector maps (PVR) and eigenvalues.
        """
        N = B.shape[0]
        k = min(n_components, N - 1)

        # Compute top k eigenvalues and eigenvectors
        evals, evecs = eigh(B, subset_by_index=[N - k, N - 1])

        # Sort descending
        idx_desc = np.argsort(evals)[::-1]
        evals_sorted = evals[idx_desc]
        evecs_sorted = evecs[:, idx_desc]

        # Truncate negative eigenvalues (positive semi-definiteness)
        pos_mask = evals_sorted > 1e-6
        evals_pos = np.maximum(evals_sorted, 0.0)

        # Coordinates: X_pcoa = V * sqrt(Lambda)
        coords = evecs_sorted * np.sqrt(evals_pos)

        col_names = [f"phylo_dim_{i+1}" for i in range(k)]
        df_pcoa = pd.DataFrame(coords[:, :k], columns=col_names)
        df_pcoa.insert(0, ID_COL, taxa)

        return df_pcoa, evals_sorted
