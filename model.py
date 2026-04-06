"""
model.py – Deep Neural Network for cardiovascular risk binary classification.

Architecture
------------
Input  →  Linear(n, 16)  →  ReLU
       →  Linear(16, 8)  →  ReLU
       →  Linear(8, 1)   →  (logits, no sigmoid)

The final sigmoid is intentionally omitted so that the model can be paired
with ``torch.nn.BCEWithLogitsLoss``, which is numerically more stable than
applying sigmoid first and then ``BCELoss``.
"""

import torch
import torch.nn as nn


class Net(nn.Module):
    """Three-layer DNN that returns raw logits.

    Parameters
    ----------
    input_features : int
        Number of input features (10 for the harmonised cardiovascular
        feature set used in this project).
    """

    def __init__(self, input_features: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_features, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.output(x)  # raw logits
