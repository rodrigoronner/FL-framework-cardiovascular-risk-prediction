"""
model.py - Deep Neural Network for cardiovascular risk binary classification.

Architecture
------------
Input(6)  ->  Linear(6, 64)  ->  ReLU
          ->  Linear(64, 32) ->  ReLU
          ->  Linear(32, 1)  ->  Sigmoid

The model returns calibrated probabilities P(Y=1|x) in [0, 1].
Paired with ``torch.nn.BCELoss`` during training.

The six harmonized input features are:
    age, sex, systolic BP (trestbps), diastolic BP (diaBP),
    cholesterol (chol), fasting blood glucose (fbs).

"""

import torch
import torch.nn as nn


class Net(nn.Module):
    """Three-layer DNN for cardiovascular risk binary classification.


    Parameters
    ----------
    input_features : int
        Number of input features. Defaults to 6, corresponding to the
        Harmonized cardiovascular feature set used in this project:
        age, sex, systolic BP, diastolic BP, cholesterol, and fasting blood glucose.
    """

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
