"""ResNet from Gorishniy et al., 'Revisiting Deep Learning Models for Tabular Data' (NeurIPS 2021).
arXiv:2106.11959

Exact block: BN → Linear(d_block→d_hidden) → ReLU → Dropout → Linear(d_hidden→d_block) → Dropout → skip
Output head: BN → ReLU → Linear(d_block→num_classes)
"""

import torch
from torch import nn

from src.models.location_encoding import encode_locations


class ResidualBlock(nn.Module):
    """One pre-norm residual block: BN -> Linear -> ReLU -> Dropout -> Linear -> Dropout, with a skip connection.

    Args:
        d_block: Width of the residual stream (block input/output dimension).
        d_hidden: Width of the inner Linear expansion.
        dropout1: Dropout rate after the first (expansion) Linear.
        dropout2: Dropout rate after the second (projection) Linear.
    """

    def __init__(self, d_block: int, d_hidden: int, dropout1: float, dropout2: float) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(d_block)
        self.fc1 = nn.Linear(d_block, d_hidden)
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(dropout1)
        self.fc2 = nn.Linear(d_hidden, d_block)
        self.drop2 = nn.Dropout(dropout2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return residual + x


class ResNet(nn.Module):
    """Residual MLP for multi-label species distribution modelling.

    Projects environmental predictors (optionally concatenated with a sin/cos
    location encoding) into a residual stream of ``num_blocks`` pre-norm blocks,
    followed by a BN -> ReLU -> Linear head producing one logit per species. The
    input projection is lazy (``nn.LazyLinear``), so the predictor dimension is
    inferred on the first forward pass.

    Args:
        num_classes: Number of species (output logits).
        hidden_dim: Width of the residual stream.
        num_blocks: Number of residual blocks.
        d_hidden_multiplier: Inner expansion factor (``d_hidden = hidden_dim * multiplier``).
        use_location: If True, concatenate a sin/cos encoding of ``locations`` to the predictors.
        dropout1: Dropout after each block's expansion Linear.
        dropout2: Dropout after each block's projection Linear.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        d_hidden_multiplier: float = 2.0,
        use_location: bool = True,
        dropout1: float = 0.0,
        dropout2: float = 0.0,
    ):
        super().__init__()
        self.use_location = use_location
        d_hidden = int(hidden_dim * d_hidden_multiplier)

        self.projection = nn.LazyLinear(hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, d_hidden, dropout1, dropout2) for _ in range(num_blocks)]
        )
        self.head_norm = nn.BatchNorm1d(hidden_dim)
        self.head_act = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, locations, predictors):
        """Predict per-species logits.

        Args:
            locations: (batch, 2) tensor of [longitude, latitude]; used only if ``use_location``.
            predictors: (batch, n_features) environmental predictor tensor.

        Returns:
            (batch, num_classes) logit tensor (pre-sigmoid).
        """
        if self.use_location:
            x = torch.cat([encode_locations(locations), predictors], dim=-1)
        else:
            x = predictors

        x = self.projection(x)
        x = self.blocks(x)
        x = self.head_norm(x)
        x = self.head_act(x)
        return self.classifier(x)
