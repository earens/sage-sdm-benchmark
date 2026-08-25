"""Logistic-regression model.

Expands the predictors into MaxEnt-style feature transforms (selected via
``maxent_features``) and applies a single lazy linear layer to produce per-species
logits, mirroring MaxEnt's feature classes inside a differentiable model.
"""

import torch
from torch import nn

from src.models.location_encoding import encode_locations


class Linear(nn.Module):
    """Linear model over MaxEnt-style feature expansions of the predictors.

    Args:
        num_classes: Number of species (output logits).
        maxent_features: String of MaxEnt feature-class codes to include:
            ``l`` linear, ``q`` quadratic, ``p`` product (all-feature product),
            ``t`` threshold (step at each of ``n_thresholds`` levels), ``h`` hinge
            (ReLU at each of ``n_hinges`` knots). E.g. ``"lqh"``.
        use_location: If True, concatenate a sin/cos encoding of ``locations`` to the predictors.
        n_hinges: Number of hinge knots (for the ``h`` feature class).
        n_thresholds: Number of threshold levels (for the ``t`` feature class).

    Raises:
        ValueError: If ``maxent_features`` contains a code outside ``l, q, p, t, h``.
    """

    def __init__(
        self,
        num_classes: int,
        maxent_features: str = "l",
        use_location: bool = True,
        n_hinges: int = 50,
        n_thresholds: int = 50,
    ):
        super().__init__()
        invalid = [f for f in (maxent_features or "") if f not in "lqpth"]
        if invalid:
            raise ValueError(
                f"Invalid maxent_features {maxent_features!r}: "
                f"unknown feature codes {invalid}. Allowed codes: l, q, p, t, h."
            )
        self.use_location = use_location
        self.maxent_features = maxent_features
        self.n_hinges = n_hinges
        self.n_thresholds = n_thresholds
        self.register_buffer("threshold_values", torch.linspace(-3, 3, self.n_thresholds))
        self.register_buffer("hinge_values", torch.linspace(-3, 3, self.n_hinges))
        self.classifier = nn.LazyLinear(num_classes)

    def forward(self, locations, predictors):
        """Expand predictors into the selected feature classes and predict logits.

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

        if self.maxent_features:
            features = []
            if "l" in self.maxent_features:
                features.append(x)
            if "q" in self.maxent_features:
                features.append(x ** 2)
            if "p" in self.maxent_features:
                features.append(x.prod(dim=-1, keepdim=True))
            if "t" in self.maxent_features:
                features.append((x.unsqueeze(-1) > self.threshold_values).float().view(x.shape[0], -1))
            if "h" in self.maxent_features:
                features.append(torch.relu(x.unsqueeze(-1) - self.hinge_values).view(x.shape[0], -1))
            x = features[0] if len(features) == 1 else torch.cat(features, dim=-1)

        return self.classifier(x)
