"""Plain MLP baseline for multi-label SDM: a lazy input projection followed by a
stack of Linear -> ReLU blocks and a per-species classifier head. Contrast with
``resnet.ResNet``, which adds batch norm and residual connections.
"""

import torch
from torch import nn

from src.models.location_encoding import encode_locations


class Block(nn.Module):
    """A single Linear -> ReLU block operating at fixed ``hidden_dim`` width."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SimpleMLP(nn.Module):
    """Plain feed-forward MLP producing one logit per species.

    Args:
        num_classes: Number of species (output logits).
        hidden_dim: Hidden layer width.
        num_blocks: Number of Linear -> ReLU blocks after the input projection.
        use_location: If True, concatenate a sin/cos encoding of ``locations`` to the predictors.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        use_location: bool = True,
    ):
        super().__init__()
        self.use_location = use_location
        self.net = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            *[Block(hidden_dim) for _ in range(num_blocks)]
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, locations, predictors):
        """Predict per-species logits from ``predictors`` (and optionally ``locations``).

        Args:
            locations: (batch, 2) [longitude, latitude]; used only if ``use_location``.
            predictors: (batch, n_features) environmental predictor tensor.

        Returns:
            (batch, num_classes) logit tensor (pre-sigmoid).
        """
        if self.use_location:
            x = torch.cat([encode_locations(locations), predictors], dim=-1)
        else:
            x = predictors
        x = self.net(x)
        return self.classifier(x)
