"""
client.py – FedCVR Flower client.

Each client trains a local DNN with:
  1. FedProx proximal regularisation  – penalises deviation from the global
     model, helping to tame client drift under non-IID data (μ parameter).
  2. Differential Privacy via Opacus  – per-sample gradient clipping and
     calibrated Gaussian noise injection before the update is transmitted.
     When ``use_dp=False`` the client behaves as a standard FedProx client.

The client extends ``fl.client.NumPyClient`` for seamless Flower integration.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.client import NumPyClient
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

from .model import Net


class FedCVRClient(NumPyClient):
    """Flower client implementing FedProx + optional client-side DP.

    Parameters
    ----------
    model       : Initialised ``Net`` instance.
    train_loader: DataLoader for local training data.
    test_loader : DataLoader for local evaluation data.
    pos_weight  : Class-imbalance correction weight for BCEWithLogitsLoss.
    local_epochs: Number of local SGD epochs per communication round.
    use_dp      : Whether to activate Opacus Differential Privacy.
    dp_config   : Dict with keys ``noise_multiplier`` and ``max_grad_norm``.
                  Ignored when ``use_dp=False``.
    """

    def __init__(
        self,
        model: Net,
        train_loader: DataLoader,
        test_loader: DataLoader,
        pos_weight: torch.Tensor,
        local_epochs: int = 5,
        use_dp: bool = False,
        dp_config: Optional[Dict] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.pos_weight = pos_weight
        self.local_epochs = local_epochs
        self.use_dp = use_dp

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        if self.use_dp:
            if dp_config is None:
                raise ValueError("dp_config must be provided when use_dp=True")
            from opacus import PrivacyEngine

            privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.train_loader = (
                privacy_engine.make_private(
                    module=self.model,
                    optimizer=self.optimizer,
                    data_loader=self.train_loader,
                    noise_multiplier=dp_config["noise_multiplier"],
                    max_grad_norm=dp_config["max_grad_norm"],
                )
            )

    # ------------------------------------------------------------------
    # Flower NumPyClient interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)

        # Snapshot of global model parameters for the proximal term
        global_params = [p.clone().detach() for p in self.model.parameters()]
        mu: float = config.get("mu", 0.0)

        self.model.train()
        for _ in range(self.local_epochs):
            for features, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                # FedProx proximal term: (μ/2) ‖w − w_global‖²
                if mu > 0.0:
                    prox_loss = sum(
                        (local - global_).pow(2).sum()
                        for local, global_ in zip(self.model.parameters(), global_params)
                    )
                    loss = loss + (mu / 2.0) * prox_loss

                loss.backward()
                self.optimizer.step()

        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()

        all_labels: List[int] = []
        all_preds: List[int] = []
        total_loss = 0.0

        with torch.no_grad():
            for features, labels in self.test_loader:
                outputs = self.model(features)
                total_loss += self.criterion(outputs, labels).item() * len(labels)
                probs = torch.sigmoid(outputs)
                all_labels.extend(labels.numpy().flatten().tolist())
                all_preds.extend((probs >= 0.5).int().numpy().flatten().tolist())

        n = len(all_labels)
        if n == 0:
            return 0.0, 0, {}

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        avg_loss = total_loss / n

        return (
            float(avg_loss),
            n,
            {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            },
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_client(
    cid: str,
    client_train_data: list,
    client_test_data: list,
    batch_size: int = 32,
    local_epochs: int = 5,
    use_dp: bool = False,
    dp_config: Optional[Dict] = None,
) -> "FedCVRClient":
    """Construct a ``FedCVRClient`` from pre-split numpy arrays.

    Parameters
    ----------
    cid              : Client ID string (as provided by Flower simulation).
    client_train_data: List of (X_train, y_train) tuples, one per client.
    client_test_data : List of (X_test,  y_test)  tuples, one per client.
    batch_size       : Mini-batch size for training DataLoader.
    local_epochs     : SGD epochs per round.
    use_dp           : Activate Opacus DP.
    dp_config        : ``{"noise_multiplier": float, "max_grad_norm": float}``
    """
    idx = int(cid)
    X_train, y_train = client_train_data[idx]
    X_test, y_test = client_test_data[idx]

    model = Net(input_features=X_train.shape[1])

    def _to_loader(X, y, shuffle=False):
        return DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32).view(-1, 1),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    # Opacus requires non-trivially-small batches; reduce if DP is active
    eff_bs = max(8, batch_size // 2) if use_dp else batch_size
    train_loader = _to_loader(X_train, y_train, shuffle=True)
    # Rebuild with effective batch size when DP is on
    if use_dp:
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
            ),
            batch_size=eff_bs,
            shuffle=True,
        )
    test_loader = _to_loader(X_test, y_test)

    pos_count = float(y_train.sum())
    neg_count = float(len(y_train) - pos_count)
    pos_weight = torch.tensor(neg_count / pos_count if pos_count > 0 else 1.0)

    return FedCVRClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        pos_weight=pos_weight,
        local_epochs=local_epochs,
        use_dp=use_dp,
        dp_config=dp_config,
    )
