from typing import List, Dict, Tuple, Set, Optional, Union
import numpy as np
import pandas as pd
from src.data.schemas import ID_COL
from src.phylogenetics.otol import PhyloNode


class MonophyleticWithholdingEngine:
    """
    Engine for defining monophyletic taxonomic withholdings, phylogenetic tree-cut
    depths (T_cut), and patristic distance buffer zones to eliminate clade leakage.
    """

    def __init__(self, tree_root: Optional[PhyloNode] = None):
        self.tree_root = tree_root
        if self.tree_root is not None:
            self._ensure_node_ages(self.tree_root)

    @staticmethod
    def _ensure_node_ages(root: PhyloNode) -> float:
        """
        Recursively calculates and assigns divergence ages (in Ma) to all nodes
        in an ultrametric tree from extant tips (age = 0.0) up to the root.
        """
        if root.is_leaf():
            root.age = 0.0
            return 0.0

        child_ages = []
        for child in root.children:
            c_age = MonophyleticWithholdingEngine._ensure_node_ages(child)
            child_ages.append(c_age + float(child.length))

        root.age = float(np.mean(child_ages)) if child_ages else 0.0
        return root.age

    def get_order_clades(
        self,
        df_pheno: pd.DataFrame,
        min_clade_size: int = 50,
    ) -> Dict[str, List[str]]:
        """
        Extracts monophyletic order-level withholdings from the phenotypic dataset.
        Filters for orders with at least min_clade_size taxa to ensure statistical
        power for out-of-clade extrapolation benchmarking.
        """
        if ID_COL not in df_pheno.columns or "order" not in df_pheno.columns:
            raise ValueError(f"Dataframe must contain '{ID_COL}' and 'order' columns.")

        order_groups: Dict[str, List[str]] = {}
        for order, group in df_pheno.groupby("order"):
            taxa = group[ID_COL].tolist()
            if len(taxa) >= min_clade_size:
                order_groups[str(order)] = taxa

        return order_groups

    def get_t_cut_clusters(
        self,
        t_cut_ma: float,
        root: Optional[PhyloNode] = None,
    ) -> Dict[str, List[str]]:
        """
        Partitions the ultrametric phylogenetic tree at divergence time depth T_cut (Ma).
        Identifies all maximal subtrees whose crown age <= T_cut.
        Every leaf in the tree belongs to exactly one monophyletic cluster diverged
        within the last T_cut Ma.
        """
        r = root or self.tree_root
        if r is None:
            raise ValueError("No phylogenetic tree root provided.")
        self._ensure_node_ages(r)

        clusters: Dict[str, List[str]] = {}
        cluster_counter = 0

        def traverse(node: PhyloNode):
            nonlocal cluster_counter
            node_age = float(node.age) if node.age is not None else 0.0
            if node_age <= t_cut_ma or node.is_leaf():
                leaves = node.get_leaf_names()
                cluster_name = node.name or f"clade_{cluster_counter}"
                if cluster_name in clusters:
                    cluster_name = f"{cluster_name}_{cluster_counter}"
                clusters[cluster_name] = leaves
                cluster_counter += 1
                return

            for child in node.children:
                traverse(child)

        traverse(r)
        return clusters

    def compute_buffer_quarantine(
        self,
        test_taxa: List[str],
        all_taxa: List[str],
        patristic_matrix: np.ndarray,
        d_buffer: float,
    ) -> List[str]:
        """
        Identifies taxa that must be quarantined from the training set due to
        insufficient patristic evolutionary distance to the test clade.
        
        Formula:
          Quarantine = { s in All \\ Test | min_{t in Test} D(s, t) < d_buffer }
          
        Guarantees that all remaining training taxa satisfy:
          D(s, t) >= d_buffer for all s in Train, t in Test.
        """
        if d_buffer <= 0.0:
            return []

        taxa_to_idx = {t: i for i, t in enumerate(all_taxa)}
        test_set = set(test_taxa)
        test_indices = [taxa_to_idx[t] for t in test_taxa if t in taxa_to_idx]

        if not test_indices:
            return []

        # Non-test taxa
        non_test_taxa = [t for t in all_taxa if t not in test_set]
        non_test_indices = [taxa_to_idx[t] for t in non_test_taxa]

        if not non_test_indices:
            return []

        # Submatrix: rows = non-test, cols = test
        sub_dist = patristic_matrix[np.ix_(non_test_indices, test_indices)]
        min_dist_to_test = np.min(sub_dist, axis=1)

        quarantined_taxa = [
            non_test_taxa[i]
            for i, d in enumerate(min_dist_to_test)
            if d < d_buffer
        ]
        return quarantined_taxa

    def find_mrca(
        self,
        clade_taxa: List[str],
        root: Optional[PhyloNode] = None,
    ) -> Optional[PhyloNode]:
        """
        Finds the Most Recent Common Ancestor (MRCA) node of a given set of taxa.
        """
        r = root or self.tree_root
        if r is None:
            raise ValueError("No phylogenetic tree root provided.")

        taxa_set = set(clade_taxa)
        if not taxa_set:
            return None

        mrca_node: Optional[PhyloNode] = None

        def post_order(node: PhyloNode) -> Set[str]:
            nonlocal mrca_node
            found: Set[str] = set()
            if node.is_leaf():
                if node.name in taxa_set:
                    found.add(node.name)
            else:
                for child in node.children:
                    found.update(post_order(child))

            if mrca_node is None and found == taxa_set:
                mrca_node = node
            return found

        post_order(r)
        return mrca_node

    def validate_monophyly(
        self,
        clade_taxa: List[str],
        root: Optional[PhyloNode] = None,
    ) -> bool:
        """
        Validates whether a given subset of taxa forms a strictly monophyletic
        group in the phylogenetic tree.
        A clade is monophyletic iff its MRCA contains only leaves from the clade.
        """
        r = root or self.tree_root
        if r is None:
            raise ValueError("No phylogenetic tree root provided.")

        mrca = self.find_mrca(clade_taxa, root=r)
        if mrca is None:
            return False

        mrca_leaves = set(mrca.get_leaf_names())
        return mrca_leaves == set(clade_taxa)
