"""z(x): learned coordinates beside the fixed KEGG ones - docs/DESIGN_HYBRID.md 1.

    p(x) = [ Phi_KEGG(x) ; z(x) ; 1 ]        z(x) = MLP(x) standardised per cell

The generators act on the whole of p; the decoder reads only how far p moved and applies
that to the original cell. Nothing is ever reconstructed from z, so the reconstruction
ceiling that cost scPKFM 45 % of its error (models/observables.py) does not arise: z only
adds coordinates.

NORMALISED PER CELL, not per batch. FM on z could otherwise be minimised by shrinking z
to zero; with each cell's z at mean 0 and variance 1 over its dimensions it cannot shrink.
Batch normalisation was the first design and is wrong here: training draws a control
batch and a perturbed batch separately, and normalising each to mean 0 would erase the
difference between them - the perturbation's effect itself.
"""

from __future__ import annotations

import torch
from torch import nn


class LatentEncoder(nn.Module):
    """log1p expression over every modelled gene -> z, [B, dim]."""

    def __init__(self, n_genes: int, hidden: int, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError(f"latent dim must be positive (got {dim})")
        self.dim = int(dim)
        self.net = nn.Sequential(nn.Linear(n_genes, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Divided by the cell's own spread, with no epsilon added to it. LayerNorm adds
        # 1e-5 beside the variance, so an encoder whose output shrank below that scale
        # would be divided by sqrt(1e-5) instead of by its own spread and z would go to
        # zero after all - tests/test_hybrid.py caught it, and 1e-12 still failed at an
        # output variance of 4e-12. The floor here only guards a cell whose every
        # coordinate is identical; above it the map is exactly invariant to the
        # encoder's scale, so no loss can pull z towards zero.
        h = self.net(x)
        h = h - h.mean(dim=1, keepdim=True)
        return h / h.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-30)
