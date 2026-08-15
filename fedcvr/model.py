"""
model.py - DNN for cardiovascular risk binary classification.
Input(input_features) -> Linear(64) -> ReLU -> Linear(32) -> ReLU ->
Linear(1) -> Sigmoid. Returns calibrated P(Y=1|x) in [0, 1], paired with
``torch.nn.BCELoss``. See README "Datasets" for the input feature set.
"""

import torch
import torch.nn as nn


class Net(nn.Module):
    """Three-layer DNN for cardiovascular risk binary classification."""

    def __init__(self, input_features: int = 6) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_features, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.sigmoid(self.output(x))
