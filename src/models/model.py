"""scPKR2: a learned, cell-conditional perturbation response.

    mu(x, S) = x * exp(u) + softplus(v),   (u, v, logit_q) = g(Phi(x), sum_{a in S} e_a)

    cells ~ Bernoulli(q) * (mu / q),  whose mean is mu for any q

WHAT CHANGED FROM v2, AND WHY. v2 predicted `x + sum_a w_a + W (exp(B_S) - I) Phi(x)`,
where the first term was closed-form ridge, added identically to every cell. Both of its
measured defects come from that one term.

  THE LEARNED PART HAD NOTHING TO DO WHERE IT MATTERED. Closed-form ridge carried 88 % of
  the signal. On Table 2 Single - v2's best block - the learned residual contributed
  exactly 0.000, because the interaction is defined on pairs and a single perturbation
  has none. That was a theorem, and it was also the sentence a reviewer writes.

  A POPULATION SHIFT ADDED TO EVERY CELL GOES NEGATIVE. 41 % of entries are exactly zero,
  and a negative w_a drives those below zero: 33.8 % of ComboSciPlex's per-cell
  predictions came out negative. Flooring the predictions at zero and averaging scores
  2.4226 against 1.6220 for the mean itself, so non-negativity alone costs 0.80 - and no
  non-negative realisation can do better than that floor.

Here the dominant term is learned, it sees the cell, and mu >= 0 holds by construction.
There is no clamp, no cap and no post-hoc correction; v2 needed all three and each cost
accuracy.

WHAT IS KEPT. The sum over e_a. Dropping additivity altogether is a measured failure, not
a hypothetical one - scDFM (1.7043) and CellFlow (1.7064) learn the whole response and
both lose to ridge (1.6690). The bias is right; only its location moves.

WHAT IS NOT HERE YET. The Koopman operators and their anticommutator. The decoder already
sees the summed embedding and can represent an interaction, so running both would let two
terms explain one quantity - the coupling that broke scPKFM. docs/DESIGN.md stage 2 turns
them on only if C5 says they earn their place.

THE STARTING POINT IS NOT PROTECTED ANY MORE, and that is the risk this file takes on.
With the output maps at zero and v's bias at -10, the untrained model is mu = x, which is
Control: 3.9937. v2 started at ridge, 1.6690, and could only improve. scDFM and CellFlow
also start at Control and both finish behind ridge. The bet is that a non-negative
parameterisation and a decoder that sees the cell are worth what they did not have.
"""
import numpy as np
import torch
from torch import nn

from .modulation import PerturbationModulation
from .realise import HurdleRealisation


class PathwayModulation(nn.Module):
    """`observables` is fixed; everything else is learned."""

    def __init__(self, config: dict, observables, n_perturbations: int,
                 detection: torch.Tensor | None = None,
                 dispersion: torch.Tensor | None = None,
                 ceiling: torch.Tensor | None = None):
        super().__init__()
        self.observables = observables
        self.n_genes = len(observables.gene_names)
        self.n_perturbations = int(n_perturbations)
        self.modulation = PerturbationModulation(
            config, observables.dim, self.n_genes, n_perturbations,
            detection=detection, dispersion=dispersion, ceiling=ceiling)
        self.head = HurdleRealisation(config)

        # Stage 2. Off here, and the training loop skips its observable-space terms when
        # it is - with no operators there is nothing for them to supervise.
        self.interaction = bool(config["model"].get("interaction", False))
        self.operators = None
        self.readout = None

        # Carried from v2 unchanged. It multiplies the INTERACTION term, which stage 1
        # does not have, so it is inert here and every scale gives the same prediction.
        # It is kept rather than deleted so that stage 2 does not have to rebuild the
        # calibration machinery and the scripts that drive it.
        self.residual_scale = float(config["eval"].get("residual_scale", 1.0))
        coefficient = config["eval"].get("residual_coefficient")
        self.residual_coefficient = None if coefficient is None else float(coefficient)
        self.residual_power = float(config["eval"].get("residual_power", 0.0))
        self.residual_scale_max = float(config["eval"].get("residual_scale_max", 4.0))
        if self.residual_coefficient is not None:
            self.residual_scale = 1.0

    # ------------------------------------------------------------------ pieces

    def source_observables(self, x: torch.Tensor,
                           perturbations: list[int]) -> torch.Tensor:
        """Phi(x), [B, K]. Where the OT coupling measures distance.

        v2 used Phi(x + sum_a w_a) - the state after the additive move - so that the
        transport the coupling had to estimate was only the residual's part. There is no
        additive move to apply first here, so this is Phi of the control cell itself.

        It stays in observable space rather than gene space because a 418-dimensional
        cost matrix is what makes the plan affordable per batch, and because that space
        is standardised while raw log1p counts are not.
        """
        return self.observables(x)

    def mean(self, x: torch.Tensor, perturbations: list[int]) -> torch.Tensor:
        """mu(x, S), [B, G]. Non-negative by construction."""
        return self(x, perturbations)["mean"]

    def residual(self, x: torch.Tensor, perturbations: list[int]) -> torch.Tensor:
        """The interaction term in gene space. Identically zero in stage 1."""
        return torch.zeros_like(x)

    # The decomposition v2's diagnostics report on - an additive part and a residual on
    # top of it - DOES NOT EXIST HERE. The whole response is mu, and there is no seam to
    # read it apart at. These return zeros so the existing diagnostics run, and every
    # column derived from them (resid_R2, residual_cos, the observable displacement) is
    # therefore MEANINGLESS in stage 1 and must not be quoted. L2 is computed from
    # predict_cells and is unaffected.

    def additive(self, perturbations: list[int]) -> torch.Tensor:
        """Not a quantity in this model. See the note above."""
        return torch.zeros(self.n_genes, device=self.modulation.to_u.bias.device,
                           dtype=self.modulation.to_u.bias.dtype)

    def observable_displacement(self, x: torch.Tensor,
                                perturbations: list[int]) -> torch.Tensor:
        """Not a quantity in this model. See the note above."""
        return torch.zeros(x.shape[0], self.observables.dim,
                           device=x.device, dtype=x.dtype)

    def condition_scale(self, residual_mean: torch.Tensor) -> float:
        """s for one condition. Inert while there is no interaction term to scale."""
        if self.residual_coefficient is None:
            return self.residual_scale
        norm = float(torch.linalg.norm(residual_mean))
        if norm <= 1e-12:
            return 0.0
        return float(min(max(self.residual_coefficient * norm ** (-self.residual_power),
                             0.0), self.residual_scale_max))

    # ------------------------------------------------------------------ forward

    def forward(self, x: torch.Tensor,
                perturbations: list[int]) -> dict[str, torch.Tensor]:
        """Head parameters for the predicted population. `mean` is the prediction."""
        return self.modulation(x, self.observables(x), perturbations)

    def predict(self, x: torch.Tensor, perturbations: list[int]) -> torch.Tensor:
        """Realised cells, [B, G]. soft returns mu exactly; sample draws around it."""
        return self.head.point_estimate(self(x, perturbations))

    def predict_scaled(self, x: torch.Tensor, perturbations: list[int],
                       scale: float) -> torch.Tensor:
        """predict() with an explicit interaction scale. Identical to predict() here."""
        return self.predict(x, perturbations)

    # ------------------------------------------------------------- bookkeeping

    def learned_parameters(self) -> dict[str, int]:
        counts = {"embedding": int(self.modulation.embedding.weight.numel()),
                  "decoder": sum(p.numel() for n, p in self.modulation.named_parameters()
                                 if not n.startswith("embedding")),
                  "head": sum(p.numel() for p in self.head.parameters())}
        counts["total"] = sum(counts.values())
        return counts

    def extra_repr(self) -> str:
        return (f"n_genes={self.n_genes}, n_perturbations={self.n_perturbations}, "
                f"K={self.observables.dim}, interaction={self.interaction}")


# v2's name, so that a stale import fails loudly instead of resolving to a model with
# different semantics.
PathwayKoopmanResidual = None
