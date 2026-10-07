"""Scoring a predicted (mu, q) against real cells, and turning it back into cells.

This is what is left of v2's head once the decoder produces q. v2 computed it as
q = sigmoid(a_g mu + b_g) - an affine function of the mean, per gene, with no view of the
cell - and a_g barely moved during training, so q sat on its per-gene floor and
magnitude = mu/q reached a hundred times the mean. Uncapped, ComboSciPlex scored 465.66
against 1.60 for the mean the model states; capped, the truncation destroyed the mean
instead and it scored 2.27. Neither number is a realisation of anything.

So q is an output of the decoder here, which does see the cell, and this module holds no
opinion about where it came from. It owns the likelihood and the draw, nothing else.

mu is non-negative by construction upstream, which removes the clamp that biased v2's
realised mean upward by 0.40 of L2 and the cap that was needed to make an export valid at
all. Neither appears here. If a negative ever reaches this module the model is broken,
and `loss` says so rather than quietly flooring it.
"""
import torch
from torch import nn
import torch.nn.functional as F


class HurdleRealisation(nn.Module):
    """x = B * (mu / q), B ~ Bernoulli(q). Mean q * (mu/q) = mu, for any q."""

    def __init__(self, config: dict):
        super().__init__()
        cfg = config["model"]
        self.bce_weight = float(cfg["hurdle_bce_weight"])
        self.gate_mode = cfg["hurdle_gate"]
        self.realisation = str(config["eval"].get("realisation", "beta"))

    def loss(self, params: dict, x: torch.Tensor, **_) -> tuple[torch.Tensor, dict]:
        mean = params["mean"]
        if not torch.isfinite(mean).all():
            raise ValueError("the predicted mean is not finite")
        # C2 in docs/DESIGN.md. mu = x exp(u) + softplus(v) cannot be negative, so a
        # negative here is a bug in the parameterisation and not a property of the data -
        # it is raised rather than floored, because flooring is exactly how v2 hid this
        # class of error until cell-eval refused the export.
        if bool((mean < 0).any()):
            raise ValueError(
                f"mu went negative ({float(mean.min()):.4f}), which the "
                f"parameterisation makes impossible - see docs/DESIGN.md C2")

        observed = (x > 0).float()
        gate_loss = F.binary_cross_entropy(params["q"].clamp(1e-6, 1 - 1e-6), observed)
        # Supervised only where something was detected: training the magnitude on zeros
        # drags every prediction toward zero and undoes the split.
        denominator = observed.sum().clamp(min=1.0)
        residual = params["magnitude"] - x
        log_scale = params["log_scale"]
        nll = 0.5 * (residual / log_scale.exp()) ** 2 + log_scale
        magnitude_loss = (nll * observed).sum() / denominator
        total = magnitude_loss + self.bce_weight * gate_loss
        parts = {"gate_bce": float(gate_loss),
                 "rmse": float((((residual ** 2) * observed).sum()
                                / denominator).sqrt()),
                 "q_mean": float(params["q"].mean()),
                 "mu_min": float(mean.min()), "mu_max": float(mean.max())}
        return total, parts

    def point_estimate(self, params: dict, **_) -> torch.Tensor:
        """soft returns mu exactly; sample draws a cell whose mean is mu."""
        q, magnitude = params["q"], params["magnitude"]
        if self.gate_mode == "soft":
            return params["mean"]
        if self.gate_mode == "hard":
            gate = (q > 0.5).to(q.dtype)
        elif self.gate_mode == "sample":
            gate = torch.bernoulli(q)
        else:
            raise ValueError(f"unknown hurdle gate {self.gate_mode!r}")

        if self.gate_mode == "sample":
            spread = params["log_scale"].exp().clamp(min=1e-6)
            if self.realisation == "beta":
                # Supported on [0, ceiling] and with mean exactly m. The hurdle's m is
                # what a cell holds when the gene is detected, so it lives in the range
                # that gene actually occupies - and a distribution ON that range is the
                # honest way to say so. Capping a draw from an unbounded one says the
                # same thing and takes the mean with it, which cost v2 0.80 of L2 on
                # ComboSciPlex.
                #
                # Gamma is unbounded, and that is not academic here: with the spread
                # clamped at e^2 = 7.39 a gamma with mean 7 crosses 15 often enough that
                # cell-eval refused the export at 59.11.
                # Widened where the mean does not fit inside the gene's observed range.
                # A Beta on [0, c] cannot have a mean above c, so clamping m at 1 there
                # would silently replace the mean with c - the realisation would be
                # bounded and WRONG, which is the trade this whole design refuses. The
                # support grows to hold the mean instead, and the bound becomes
                # max(ceiling, m): still finite, still the data's scale wherever the mean
                # fits in it, and never a lie about the mean. A mu above what its gene
                # has ever reached is a defect in mu, and it belongs in the L2 rather
                # than hidden here.
                ceiling = torch.maximum(params["ceiling"], magnitude * 1.001
                                        ).clamp(min=1e-6)
                m = (magnitude / ceiling).clamp(1e-6, 1.0 - 1e-6)
                # A Beta cannot have more variance than m(1-m); asking for more is asking
                # for a distribution that does not exist, so the request is clipped there
                # rather than silently reinterpreted.
                var = ((spread / ceiling) ** 2).clamp(max=0.95 * m * (1.0 - m))
                nu = (m * (1.0 - m) / var.clamp(min=1e-12) - 1.0).clamp(min=1e-3)
                beta = torch.distributions.Beta((m * nu).clamp(min=1e-4),
                                                ((1.0 - m) * nu).clamp(min=1e-4))
                magnitude = beta.sample() * ceiling
            elif self.realisation == "gamma":
                # Mean m and variance s^2 exactly, from a distribution already positive:
                # Gamma(k, 1/theta) with k = (m/s)^2, theta = s^2/m has mean k theta = m.
                # v2 drew a gaussian and clamped it at zero, and the clamp biased the
                # realised mean upward by 0.40 of L2 - 1.8545 against 1.5348 for this.
                m = magnitude.clamp(min=1e-6)
                gamma = torch.distributions.Gamma(
                    ((m / spread) ** 2).clamp(min=1e-4),
                    (m / spread ** 2).clamp(min=1e-8))
                magnitude = gamma.sample()
            else:
                magnitude = (magnitude + torch.randn_like(magnitude)
                             * spread).clamp(min=0.0)
        return gate * magnitude
