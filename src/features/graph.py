from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from src.data.schemas import ID_COL
from src.phylogenetics.otol import PhyloNode


class PhyloGraphConverter:
    """
    Converts a calibrated phylogenetic tree topology and feature matrix
    into a PyTorch Geometric compatible graph structure G = (V, E).
    """

    def tree_to_pyg_graph(
        self,
        root: PhyloNode,
        df_features: pd.DataFrame,
        bidirectional: bool = True,
    ) -> Dict[str, Any]:
        """
        Builds graph dictionary representation:
          - Tip nodes [0 .. N_tips-1] strictly map to rows in df_features.
          - Internal nodes [N_tips .. |V|-1] represent ancestral evolutionary branch points.
          - edge_index: (2 x |E|) tensor of node index pairs.
          - edge_attr: (|E| x 1) tensor of branch lengths (divergence times).
          - x: (|V| x P) node feature matrix.
        """
        if ID_COL not in df_features.columns:
            raise ValueError(f"Features dataframe missing '{ID_COL}'")

        target_taxa = df_features[ID_COL].tolist()
        N_tips = len(target_taxa)
        feature_cols = [c for c in df_features.columns if c != ID_COL]
        X_tips = df_features[feature_cols].values.astype(np.float32)
        P = X_tips.shape[1]

        # Map leaf names to 0..N_tips-1
        node_to_idx: Dict[int, int] = {}
        idx_to_name: Dict[int, str] = {}
        current_internal_idx = N_tips

        # Map leaves first
        taxa_pos_map = {t: idx for idx, t in enumerate(target_taxa)}
        leaves = root.get_leaves()

        for leaf in leaves:
            leaf_id = id(leaf)
            if leaf.name in taxa_pos_map:
                idx = taxa_pos_map[leaf.name]
                node_to_idx[leaf_id] = idx
                idx_to_name[idx] = leaf.name

        # Assign internal node indices
        def index_internal_nodes(node: PhyloNode):
            nonlocal current_internal_idx
            node_id = id(node)
            if not node.is_leaf():
                if node_id not in node_to_idx:
                    node_to_idx[node_id] = current_internal_idx
                    idx_to_name[current_internal_idx] = node.name or f"internal_{current_internal_idx}"
                    current_internal_idx += 1
                for child in node.children:
                    index_internal_nodes(child)

        index_internal_nodes(root)
        total_nodes = current_internal_idx

        # Construct edges
        src_list = []
        dst_list = []
        attr_list = []

        def collect_edges(node: PhyloNode):
            u = node_to_idx[id(node)]
            for child in node.children:
                v = node_to_idx[id(child)]
                length = float(child.length)

                # Directed edge child -> parent
                src_list.append(v)
                dst_list.append(u)
                attr_list.append([length])

                if bidirectional:
                    # Parent -> child
                    src_list.append(u)
                    dst_list.append(v)
                    attr_list.append([length])

                collect_edges(child)

        collect_edges(root)

        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        edge_attr = np.array(attr_list, dtype=np.float32)

        # Initialize node feature matrix: tips have actual features, internal nodes have mean
        X_all = np.zeros((total_nodes, P), dtype=np.float32)
        X_all[:N_tips] = X_tips
        mean_feat = np.mean(X_tips, axis=0)
        X_all[N_tips:] = mean_feat

        graph_dict = {
            "num_nodes": total_nodes,
            "num_tips": N_tips,
            "num_features": P,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "x": X_all,
            "node_names": [idx_to_name.get(i, f"node_{i}") for i in range(total_nodes)],
            "taxa": target_taxa,
        }

        # If PyTorch Geometric is installed, wrap into Data object
        try:
            import torch
            from torch_geometric.data import Data
            pyg_data = Data(
                x=torch.from_numpy(X_all),
                edge_index=torch.from_numpy(edge_index),
                edge_attr=torch.from_numpy(edge_attr),
                num_nodes=total_nodes,
            )
            graph_dict["pyg_data"] = pyg_data
        except ImportError:
            pass

        return graph_dict
