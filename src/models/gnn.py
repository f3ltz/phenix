from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from src.models.base import PhenixModel


class PhyloGNNNet(nn.Module):
    """
    PyTorch Geometric Graph Neural Network for phylogenetic trait prediction.
    Aggregates multimodal features across calibrated tree branch lengths.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = F.relu(self.conv2(h, edge_index, edge_weight))
        out = self.head(h).squeeze(-1)
        return out


class PhyloGNNPipeline(PhenixModel):
    """
    Phylogenetic Graph Neural Network Pipeline.

    With a tree graph (graph_dict), training is transductive over the whole tree: tips are species,
    internal nodes are ancestors. Node inputs are standardized features plus two label channels
    (known log10 trait, is-known flag) so trait values of relatives can propagate along branches.
    To stop a node from simply reading back its own label, each epoch hides a random share of the
    training labels (mask_rate) and the loss is computed on those hidden nodes only.

    Without a graph it degrades to a per-sample network (self-loops only, no label channels).
    """

    def __init__(
        self,
        graph_dict: Optional[Dict[str, Any]] = None,
        hidden_dim: int = 64,
        epochs: int = 200,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        timescale_tau: float = 50.0,
        mask_rate: float = 0.5,
        log_transform_target: bool = True,
        random_state: int = 42,
        device: Optional[str] = None,
    ):
        super().__init__(log_transform_target=log_transform_target)
        self.graph_dict = graph_dict
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.timescale_tau = timescale_tau
        self.mask_rate = mask_rate
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[PhyloGNNNet] = None
        self._cached_preds: Optional[np.ndarray] = None
        self._feat_mean: Optional[np.ndarray] = None
        self._feat_std: Optional[np.ndarray] = None
        self._y_mean: float = 0.0

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return ((X - self._feat_mean) / self._feat_std).astype(np.float32)

    def _self_loop_graph(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Graph with only self-loops: each sample is predicted from its own features."""
        idx = torch.arange(n, dtype=torch.long, device=self.device)
        edge_index = torch.stack([idx, idx])
        return edge_index, torch.ones(n, dtype=torch.float32, device=self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_indices: Optional[np.ndarray] = None,
        **kwargs,
    ) -> 'PhyloGNNPipeline':
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        y_fit = self.transformer.fit_transform(y_arr) if self.log_transform_target and self.transformer else y_arr
        self._y_mean = float(np.mean(y_fit))
        y_centered = torch.from_numpy((y_fit - self._y_mean).astype(np.float32)).to(self.device)

        use_tree = self.graph_dict is not None and train_indices is not None
        if use_tree:
            train_idx = np.asarray(train_indices, dtype=np.int64)
            x_nodes = np.asarray(self.graph_dict["x"], dtype=np.float32)
            n_tips = self.graph_dict["num_tips"]
            # Scale with training-tip statistics only; internal nodes sit at the (scaled) mean
            self._feat_mean = x_nodes[train_idx].mean(axis=0)
            self._feat_std = x_nodes[train_idx].std(axis=0)
            self._feat_std[self._feat_std == 0] = 1.0
            x_scaled = self._standardize(x_nodes)
            x_scaled[n_tips:] = 0.0
            x_tensor = torch.from_numpy(x_scaled).to(self.device)

            edge_index = torch.from_numpy(self.graph_dict["edge_index"].astype(np.int64)).to(self.device)
            edge_attr = np.asarray(self.graph_dict["edge_attr"], dtype=np.float32).reshape(-1)
            edge_weight = torch.from_numpy(np.exp(-edge_attr / self.timescale_tau).astype(np.float32)).to(self.device)
            num_nodes = x_tensor.shape[0]
        else:
            self._feat_mean = X_arr.mean(axis=0)
            self._feat_std = X_arr.std(axis=0)
            self._feat_std[self._feat_std == 0] = 1.0
            x_tensor = torch.from_numpy(self._standardize(X_arr)).to(self.device)
            edge_index, edge_weight = self._self_loop_graph(len(X_arr))
            train_idx = np.arange(len(X_arr))
            num_nodes = len(X_arr)

        train_idx_t = torch.from_numpy(train_idx).to(self.device)

        def build_input(visible: np.ndarray) -> torch.Tensor:
            """Appends label channels: centered label and known-flag for visible training nodes."""
            if not use_tree:
                return x_tensor
            label = torch.zeros(num_nodes, device=self.device)
            known = torch.zeros(num_nodes, device=self.device)
            vis_t = torch.from_numpy(visible).to(self.device)
            label[train_idx_t[vis_t]] = y_centered[vis_t]
            known[train_idx_t[vis_t]] = 1.0
            return torch.cat([x_tensor, label.unsqueeze(1), known.unsqueeze(1)], dim=1)

        in_channels = x_tensor.shape[1] + (2 if use_tree else 0)
        self.model = PhyloGNNNet(in_channels=in_channels, hidden_dim=self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        n_train = len(train_idx)
        self.model.train()
        for _ in range(self.epochs):
            if use_tree:
                hidden = rng.random(n_train) < self.mask_rate
                if not hidden.any():
                    hidden[rng.integers(n_train)] = True
            else:
                hidden = np.ones(n_train, dtype=bool)
            optimizer.zero_grad()
            preds = self.model(build_input(~hidden), edge_index, edge_weight)
            hidden_t = torch.from_numpy(hidden).to(self.device)
            loss = F.mse_loss(preds[train_idx_t[hidden_t]], y_centered[hidden_t])
            loss.backward()
            optimizer.step()

        self.model.eval()
        if use_tree:
            with torch.no_grad():
                # At inference every training label is visible
                full_preds = self.model(build_input(np.ones(n_train, dtype=bool)), edge_index, edge_weight)
            self._cached_preds = full_preds.cpu().numpy() + self._y_mean
        else:
            self._cached_preds = None

        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        test_indices: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict() is called.")

        if self._cached_preds is not None:
            if test_indices is None:
                raise ValueError("Tree-graph PhyloGNN needs test_indices to locate species on the tree.")
            return self._cached_preds[np.asarray(test_indices, dtype=int)]

        X_arr = np.asarray(X, dtype=np.float32)
        x_tensor = torch.from_numpy(self._standardize(X_arr)).to(self.device)
        edge_index, edge_weight = self._self_loop_graph(len(X_arr))
        with torch.no_grad():
            preds = self.model(x_tensor, edge_index, edge_weight)
        return preds.cpu().numpy() + self._y_mean
