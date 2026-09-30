from typing import Optional, Dict, Any, Union, List, Tuple
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
    Supports both tree-based graph topology and standalone feature matrix execution.
    """

    def __init__(
        self,
        graph_dict: Optional[Dict[str, Any]] = None,
        hidden_dim: int = 64,
        epochs: int = 50,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        timescale_tau: float = 50.0,
        log_transform_target: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__(log_transform_target=log_transform_target)
        self.graph_dict = graph_dict
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.timescale_tau = timescale_tau
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[PhyloGNNNet] = None
        self._cached_preds: Optional[np.ndarray] = None

    def _build_standalone_graph(self, X: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Builds a simple nearest-neighbor / chain graph when no tree graph is provided."""
        N, P = X.shape
        x_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)

        if N <= 1:
            edge_index = torch.zeros((2, 1), dtype=torch.long, device=self.device)
            edge_weight = torch.ones(1, dtype=torch.float32, device=self.device)
            return x_tensor, edge_index, edge_weight

        # Chain graph edges between consecutive samples + self loops
        src = list(range(N - 1)) + list(range(1, N)) + list(range(N))
        dst = list(range(1, N)) + list(range(N - 1)) + list(range(N))
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        edge_weight = torch.ones(len(src), dtype=torch.float32, device=self.device)
        return x_tensor, edge_index, edge_weight

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_indices: Optional[np.ndarray] = None,
        **kwargs,
    ) -> 'PhyloGNNPipeline':
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        if self.log_transform_target and self.transformer is not None:
            y_fit = self.transformer.fit_transform(y_arr)
        else:
            y_fit = y_arr

        if self.graph_dict is not None:
            x_tensor = torch.from_numpy(self.graph_dict["x"].astype(np.float32)).to(self.device)
            edge_index = torch.from_numpy(self.graph_dict["edge_index"].astype(np.int64)).to(self.device)
            edge_attr = self.graph_dict["edge_attr"]
            edge_weight = np.exp(-edge_attr.squeeze() / self.timescale_tau).astype(np.float32)
            edge_weight_tensor = torch.from_numpy(edge_weight).to(self.device)

            if train_indices is not None:
                mask = torch.from_numpy(np.asarray(train_indices, dtype=np.int64)).to(self.device)
                y_target = torch.from_numpy(y_fit.astype(np.float32)).to(self.device)
            else:
                n_samples = len(y_fit)
                mask = torch.arange(n_samples, device=self.device)
                y_target = torch.from_numpy(y_fit.astype(np.float32)).to(self.device)
        else:
            x_tensor, edge_index, edge_weight_tensor = self._build_standalone_graph(X_arr)
            mask = torch.arange(len(y_fit), device=self.device)
            y_target = torch.from_numpy(y_fit.astype(np.float32)).to(self.device)

        in_channels = x_tensor.shape[1]
        self.model = PhyloGNNNet(in_channels=in_channels, hidden_dim=self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            preds = self.model(x_tensor, edge_index, edge_weight_tensor)
            loss = F.mse_loss(preds[mask], y_target)
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            full_preds = self.model(x_tensor, edge_index, edge_weight_tensor)
            self._cached_preds = full_preds.cpu().numpy()

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

        if self.graph_dict is not None and self._cached_preds is not None:
            if test_indices is not None:
                test_idx = np.asarray(test_indices, dtype=int)
                return self._cached_preds[test_idx]
            else:
                # If test size matches X rows
                return self._cached_preds[:len(X)]
        else:
            X_arr = np.asarray(X, dtype=np.float32)
            x_tensor, edge_index, edge_weight = self._build_standalone_graph(X_arr)
            self.model.eval()
            with torch.no_grad():
                preds = self.model(x_tensor, edge_index, edge_weight)
                return preds.cpu().numpy()
