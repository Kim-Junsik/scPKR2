"""The learned additive component: mu(x, S) = x * exp(u) + softplus(v).

WHAT THIS REPLACES. v2 predicted `x + sum_a w_a`, where w_a was a free gene-space vector
fitted in closed form by ridge and added to EVERY cell identically. Two measured defects
came out of that one choice.

  The learned part had nothing to do where it mattered most. The closed form carried
  88 % of the signal, and on Table 2 Single - the block where v2 scored best - the
  learned residual contributed exactly 0.000, because an interaction term is defined on
  pairs and a single perturbation has none.

  A population-level shift added to a cell where the gene is already zero goes negative,
  and 41 % of entries are exactly zero. 33.8 % of ComboSciPlex's per-cell predictions
  came out negative. Expression cannot be negative, so no non-negative realisation can
  reproduce them: flooring the predictions at zero and averaging scores 2.4226 against
  1.6220 for the mean itself, so non-negativity alone costs 0.80 there.

WHY THIS FORM. The two terms do different jobs and neither can produce a negative.

  x * exp(u)     modulates what is already expressed. exp is positive, so a gene at zero
                 stays at zero - the move that generated every negative in v2 is not
                 representable here.
  softplus(v)    turns a gene on. Positive by construction.

mu >= 0 holds by construction, so there is no clamp, no cap, and no post-hoc correction -
the three devices v2 needed and that each cost accuracy.

WHY ADDITIVITY SURVIVES, IN THE EMBEDDING. Dropping the additive structure entirely is a
measured failure path, not a hypothetical one: scDFM (1.7043) and CellFlow (1.7064) learn
the whole response and both lose to closed-form ridge (1.6690). The inductive bias is
right; only its location moves. e_S = sum_a e_a adds in the embedding, and what that sum
does to a cell is then computed jointly, which `w_A + w_B` could not do.

WHERE THIS CAN LOSE. ridge fits a free 5,000-dimensional vector per perturbation and does
it optimally. An embedding through a decoder is less expressive than that, so this wins
only if seeing the cell is worth more than the expressiveness it gives up. It is a
hypothesis; docs/DESIGN.md C1 is where it gets settled.
"""
import numpy as np
import torch
from torch import nn


class PerturbationModulation(nn.Module):
    """e_S -> (u, v, logit_q) -> mu = x exp(u) + softplus(v), and a per-cell q."""

    def __init__(self, config: dict, observable_dim: int, n_genes: int,
                 n_perturbations: int, detection: torch.Tensor | None = None,
                 dispersion: torch.Tensor | None = None):
        super().__init__()
        cfg = config["model"]
        width = int(cfg["decoder_width"])
        self.n_genes = int(n_genes)
        self.log_factor_max = float(cfg["log_factor_max"])

        self.embedding = nn.Embedding(int(n_perturbations), int(cfg["embed_dim"]))
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.trunk = nn.Sequential(
            nn.Linear(observable_dim + int(cfg["embed_dim"]), width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU())
        self.to_u = nn.Linear(width, n_genes)
        self.to_v = nn.Linear(width, n_genes)
        self.to_q = nn.Linear(width, n_genes)

        # ZERO weights on the three output maps, so at initialisation every output IS its
        # bias and the biases alone decide where training starts. Without this the start
        # is a random gene-space field, which is neither Control nor anything else that
        # can be named in a table.
        for layer in (self.to_u, self.to_v, self.to_q):
            nn.init.zeros_(layer.weight)
        nn.init.zeros_(self.to_u.bias)
        # softplus(-10) = 4.5e-5, so the turn-on term contributes about 0.003 of L2 across
        # 5,000 genes at the start - small enough that the untrained model is Control and
        # not Control-plus-something. -6 was the first choice and is 0.18, which is not.
        nn.init.constant_(self.to_v.bias, -10.0)

        # FROM THE DATA, like v2's intercept: each gene's observed detection rate. The
        # alternative, a constant, makes q wrong for every gene at once in a quantity that
        # divides the magnitude.
        rate = (torch.full((n_genes,), 0.5) if detection is None
                else detection.clamp(1e-3, 1.0 - 1e-3))
        with torch.no_grad():
            self.to_q.bias.copy_(torch.log(rate / (1.0 - rate)))

        # The magnitude's spread, per gene, from the data as v2 did it.
        self.log_scale = nn.Parameter(
            torch.zeros(n_genes) if dispersion is None
            else torch.log(dispersion.clamp(min=1e-3)))

    def embed(self, perturbations: list[int]) -> torch.Tensor:
        """e_S = sum_a e_a, [D]. Zero for the control, so mu = x there.

        THE SUM IS THE WHOLE INDUCTIVE BIAS. It is also what makes the model indifferent
        to the order of a combination, which a simultaneous perturbation requires and
        which no training signal would otherwise impose.
        """
        if not perturbations:
            return torch.zeros(self.embedding.embedding_dim,
                               device=self.embedding.weight.device,
                               dtype=self.embedding.weight.dtype)
        # SORTED, because float addition is not associative and summing the same
        # embeddings in two orders differs in the last bits. Order invariance is a
        # property this model is supposed to HAVE, not to have approximately: a
        # simultaneous perturbation has no order, and nothing in the data would impose
        # it. Sorting makes the sum exact rather than close.
        index = torch.as_tensor(sorted(perturbations), dtype=torch.long,
                                device=self.embedding.weight.device)
        return self.embedding(index).sum(dim=0)

    def forward(self, x: torch.Tensor, phi: torch.Tensor,
                perturbations: list[int]) -> dict[str, torch.Tensor]:
        """`x` is the control cell [B, G]; `phi` is its observable coordinates [B, K]."""
        e = self.embed(perturbations).expand(phi.shape[0], -1)
        h = self.trunk(torch.cat([phi, e], dim=1))
        # Bounded, because exp is not. At 3.0 the factor runs over [0.05, 20], which
        # covers anything log1p expression does; unbounded, one bad step puts exp(u) at
        # 1e9 and the run is lost rather than merely wrong.
        u = self.to_u(h).clamp(-self.log_factor_max, self.log_factor_max)
        mean = x * torch.exp(u) + torch.nn.functional.softplus(self.to_v(h))
        q = torch.sigmoid(self.to_q(h)).clamp(1e-3, 1.0 - 1e-6)
        return {"mean": mean, "q": q, "magnitude": mean / q,
                "log_scale": self.log_scale.expand_as(mean).clamp(-6.0, 2.0),
                "u": u}
