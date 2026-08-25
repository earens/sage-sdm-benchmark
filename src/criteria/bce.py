"""Weighted BCE loss for presence-only training with background points.

Implements the "Full Weighted Loss" of Zbinden et al. (adapted from
https://github.com/eceo-epfl/SDM-full-weighted-loss): the positive-observation
term and the target-group-background pseudo-absence term are weighted separately
(``lambda_1`` / ``lambda_2``), with optional per-species weights and range masking.
"""

import torch
from torch import nn


class BCE(nn.Module):
    """
    BCE loss with separate weighting for positive obs and background locations. (As proposed in Full Weighted Loss)

    Args:
        lambda_1: Weight on the positive-observation term (presences at observation
            rows).
        lambda_2: Weight on the target-group-background (pseudo-absence) term at
            observation rows; the random-background term is weighted by
            ``1 - lambda_2``.
    """

    def __init__(
        self,
        lambda_1: float = 1.0,
        lambda_2: float = 0.5,
    ):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        # will be set later
        self.species_weight: torch.Tensor | None = None
        self.species_weight_tgb: torch.Tensor | None = None
        self.species_weight_rb: torch.Tensor | None = None


    def set_species_weights(
        self,
        species_weight: torch.Tensor,
        species_weight_tgb: torch.Tensor | None = None,
        species_weight_rb: torch.Tensor | None = None):
        """Set species weights after initialization.

        """
        self.species_weight = species_weight
        if species_weight_tgb is not None:
            self.species_weight_tgb = species_weight_tgb
        else:
            self.species_weight_tgb = species_weight / (species_weight - 1 + 1e-5)
        if species_weight_rb is not None:
            self.species_weight_rb = species_weight_rb
        else:
            self.species_weight_rb = torch.ones_like(species_weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        obs_mask: torch.Tensor = None,
        range_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the weighted BCE loss.

        Args:
            logits: (batch, num_species) raw model outputs (pre-sigmoid).
            targets: (batch, num_species) binary targets (1 = present).
            obs_mask: (batch,) boolean mask selecting observation rows (vs.
                background rows). Observation rows contribute the positive and
                target-group-background terms; the rest contribute the background term.
            range_mask: Optional (batch, num_species) 0/1 mask; entries set to 0
                (out of range) are excluded from the loss.

        Returns:
            Scalar loss = ``lambda_1``·positive + ``lambda_2``·tgb + (1-``lambda_2``)·background.
        """
        probs = torch.sigmoid(logits)

        batch_size = logits.shape[0]

        if obs_mask is None:
            obs_mask = torch.ones(batch_size, dtype=torch.bool, device=logits.device)
        if self.species_weight is not None:
            species_weight = self.species_weight.to(logits.device)
            species_weight_tgb = self.species_weight_tgb.to(logits.device)
            species_weight_rb = self.species_weight_rb.to(logits.device)

            species_weight = species_weight.unsqueeze(0).expand(batch_size, -1)[obs_mask]
            species_weight_tgb = species_weight_tgb.unsqueeze(0).expand(batch_size, -1)[obs_mask]
        else:
            species_weight = 1.0
            species_weight_tgb = 1.0
            species_weight_rb = 1.0

        obs_probs = probs[obs_mask]
        obs_targets = targets[obs_mask]

        loss_obs_pos = (-torch.log(obs_probs + 1e-5) * obs_targets * species_weight)
        loss_tgb = (-torch.log(1 - obs_probs + 1e-5) * (1 - obs_targets) * species_weight_tgb)

        # Apply range mask
        if range_mask is not None:
            obs_range_mask = range_mask[obs_mask]
            loss_obs_pos = loss_obs_pos * obs_range_mask
            loss_tgb = loss_tgb * obs_range_mask

        # Background loss
        bg_mask = ~obs_mask
        if bg_mask.any():
            bg_probs = probs[bg_mask]
            if self.species_weight is not None:
                species_weight_rb = species_weight_rb.unsqueeze(0).expand(batch_size, -1)[bg_mask]
            loss_bg = (-torch.log(1 - bg_probs + 1e-5) * species_weight_rb)

            if range_mask is not None:
                loss_bg = loss_bg * range_mask[bg_mask]

            loss_bg = loss_bg.mean()
        else:
            loss_bg = torch.zeros((), device=logits.device)

        # Final loss computation
        loss = self.lambda_1 * loss_obs_pos.mean() + self.lambda_2 * loss_tgb.mean() + (1 - self.lambda_2) * loss_bg

        return loss
