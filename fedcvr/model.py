"""
<<<<<<< HEAD
model.py – Deep Neural Network for cardiovascular risk binary classification.

Architecture
------------
Input  →  Linear(n, 16)  →  ReLU
       →  Linear(16, 8)  →  ReLU
       →  Linear(8, 1)   →  (logits, no sigmoid)

The final sigmoid is intentionally omitted so that the model can be paired
with ``torch.nn.BCEWithLogitsLoss``, which is numerically more stable than
applying sigmoid first and then ``BCELoss``.
=======
model.py - Deep Neural Network for cardiovascular risk binary classification.

Architecture
------------
Input(6)  ->  Linear(6, 64)  ->  ReLU
          ->  Linear(64, 32) ->  ReLU
          ->  Linear(32, 1)  ->  Sigmoid

The model returns calibrated probabilities P(Y=1|x) in [0, 1].
Paired with ``torch.nn.BCELoss`` during training.

The six harmonised input features are:
    age, sex, systolic BP (trestbps), diastolic BP (diaBP),
    cholesterol (chol), fasting blood glucose (fbs).

"""

import torch
import torch.nn as nn


class Net(nn.Module):
<<<<<<< HEAD
    """Three-layer DNN that returns raw logits.
=======
    """Three-layer DNN for cardiovascular risk binary classification.


    Parameters
    ----------
    input_features : int
<<<<<<< HEAD
        Number of input features (10 for the harmonised cardiovascular
        feature set used in this project).
    """

    def __init__(self, input_features: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_features, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)
=======
        Number of input features. Defaults to 6, corresponding to the
        harmonised cardiovascular feature set used in this project:
        age, sex, systolic BP, diastolic BP, cholesterol, fasting blood glucose.
    """

    def __init__(self, input_features: int = 6) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_features, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
>>>>>>> 3d539aa (fix: align code with paper (architecture, features, datasets, hyperparams))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
<<<<<<< HEAD
        return self.output(x)  # raw logits
=======
        return self.sigmoid(self.output(x))
>>>>>>> 3d539aa (fix: align code with paper (architecture, features, datasets, hyperparams))
