"""Stage 2: a perturbation is a generator on the observables. docs/DESIGN_STAGE2.md.

    p    = [Phi(x); 1]                      the observables and one constant coordinate
    A_a  = U diag(c_a) V + P_a Q_a          one generator per perturbation, last row zero
    B_S  = sum_{a in S} A_a                 simultaneous perturbations superpose
    dp   = (exp(B_S) - I) p                 exact; one matrix exponential per condition
    u    = W_u dp,   v = v0 + W_v dp        log-gain and turn-on, read out per gene
    mu   = x exp(u) + softplus(v)           the same output form as stage 1

WHAT THIS REPLACES. Stage 1 produced u and v with an MLP over [Phi(x), sum_a e_a]. It
scores well - ahead of scDFM on four of six Table 1 columns under the published protocol
- and it is not a method: nothing in it says what a perturbation IS. Here a perturbation
is a linear generator on the observable space, its effect is that generator's
exponential, and the readout turns the movement of the observables into a fold change.

THE INTERACTION IS NOT A TERM. Summing two generators and exponentiating gives

    exp(A + B) - I = [exp(A) - I] + [exp(B) - I] + (AB + BA)/2 + O(3)

so what separates a double from the sum of its singles begins with the anticommutator -
the object v2 constructed by hand - and it is symmetric under exchange, which is what a
simultaneous perturbation has to be. Because u is a LOG-gain, generators that commute
give u_S = u_a + u_b: fold changes multiply, the null model genetic-interaction work
measures epistasis against.

THE CONSTANT COORDINATE IS NOT OPTIONAL. Phi is standardised to zero mean over the
training cells, so without it the average cell has p = 0 and therefore dp = 0: no
perturbation could have a mean effect at all. v2 never met this because ridge carried
the mean. With a last row of zeros the constant stays 1 under the flow and exp(B_S)
acts on Phi(x) as an affine map, whose offset is the mean effect and whose linear part
is how the effect depends on the cell.

THE CONTROL IS EXACT, at every point in training and not only at the start: with no
perturbation B = 0 and dp = 0, and neither readout carries a bias, so u = 0 and
mu = x + softplus(v0). Stage 1 only had this for an untrained model.

WHY SHARED MODES. Giving every perturbation free parameters of its own halved the
training loss in stage 1 and took test L2 from 1.68 to 2.49. Most of each generator
therefore lives in modes all perturbations share.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .modulation import assemble_output
from .operator import KoopmanOperators


class GeneratorResponse(nn.Module):
    """A_a -> B_S -> dp -> (u, v, logit_q) -> mu. Same outputs as stage 1's module."""

    def __init__(self, config: dict, observable_dim: int, n_genes: int,
                 n_perturbations: int, detection: torch.Tensor | None = None,
                 dispersion: torch.Tensor | None = None,
                 ceiling: torch.Tensor | None = None):
        super().__init__()
        cfg = config["model"]
        self.n_genes = int(n_genes)
        self.observable_dim = int(observable_dim)
        self.log_factor_max = float(cfg["log_factor_max"])
        self.bound_mean = bool(cfg.get("bound_mean", False))
        # The three arms of the ablation in docs/DESIGN_STAGE2.md section 7 that change
        # this module rather than its inputs.
        self.constant = bool(cfg.get("generator_constant", True))      # G: off
        self.linear = bool(cfg.get("generator_linear", False))         # C: on
        explicit = bool(cfg.get("generator_explicit_interaction", False))  # F: on
        self.dim = self.observable_dim + int(self.constant)

        # The parameterisation v2 already had - shared modes plus a small private part -
        # on one more coordinate. Composition is done here, so it is told `sum` and its
        # own pairwise forms are never used.
        self.operators = KoopmanOperators(
            {"model": {**cfg, "composition": "sum"}}, n_perturbations, self.dim)
        # An explicit weight on the anticommutator, beyond what the exponential already
        # supplies. ZERO at the start, so this arm begins as the plain model and the
        # comparison is nested rather than between two initialisations.
        self.kappa = nn.Parameter(torch.zeros(())) if explicit else None

        mask = torch.ones(self.dim, self.dim)
        if self.constant:
            mask[-1, :] = 0.0
        self.register_buffer("row_mask", mask, persistent=False)

        # NO BIAS on u or v. A bias is an effect that does not depend on dp, so the
        # control would receive it too, and "no perturbation changes nothing" would stop
        # being true the first time it moved.
        self.to_u = nn.Linear(self.dim, n_genes, bias=False)
        self.to_v = nn.Linear(self.dim, n_genes, bias=False)
        self.to_q = nn.Linear(self.dim, n_genes)
        # ZERO readouts, so an untrained model is the control. The generators must NOT
        # start at zero as well: dL/dW is proportional to dp, and with every generator
        # at zero dp is zero and nothing would ever move.
        for layer in (self.to_u, self.to_v, self.to_q):
            nn.init.zeros_(layer.weight)
        self.register_buffer(
            "turn_on", torch.tensor(float(cfg.get("turn_on_init", -10.0))),
            persistent=False)

        rate = (torch.full((n_genes,), 0.5) if detection is None
                else detection.clamp(1e-3, 1.0 - 1e-3))
        with torch.no_grad():
            self.to_q.bias.copy_(torch.log(rate / (1.0 - rate)))

        self.register_buffer("ceiling",
                             torch.full((n_genes,), float("inf")) if ceiling is None
                             else ceiling.clamp(min=1e-3).clone(), persistent=False)
        self.log_scale = nn.Parameter(
            torch.zeros(n_genes) if dispersion is None
            else torch.log(dispersion.clamp(min=1e-3)))

    # ------------------------------------------------------------- generators

    def augment(self, phi: torch.Tensor) -> torch.Tensor:
        """[Phi(x); 1], or Phi(x) itself in the arm without the constant."""
        if not self.constant:
            return phi
        return torch.cat([phi, torch.ones_like(phi[:, :1])], dim=1)

    def compose(self, perturbations: list[int]) -> torch.Tensor:
        """B_S, [D, D]. The zero matrix for the control.

        SORTED before summing. Float addition is not associative, so the same generators
        in two orders differ in the last bits, and order invariance is meant to be a
        property this model has rather than one it has approximately.
        """
        device, dtype = self.row_mask.device, self.row_mask.dtype
        if not perturbations:
            return torch.zeros(self.dim, self.dim, device=device, dtype=dtype)
        generators = [self.operators.matrix(pert) * self.row_mask
                      for pert in sorted(perturbations)]
        total = generators[0]
        for generator in generators[1:]:
            total = total + generator
        if self.kappa is not None:
            for i in range(len(generators)):
                for j in range(i + 1, len(generators)):
                    a, b = generators[i], generators[j]
                    total = total + self.kappa * (a @ b + b @ a)
        return total

    def displacement(self, phi: torch.Tensor,
                     perturbations: list[int]) -> torch.Tensor:
        """dp = (exp(B_S) - I) p, [B, D]. Exactly zero for the control."""
        p = self.augment(phi)
        if not perturbations:
            return torch.zeros_like(p)
        generator = self.compose(perturbations)
        if self.linear:
            # First order only: the model with the exponential's higher terms removed,
            # which is what the interaction claim is measured against.
            return p @ generator.T
        identity = torch.eye(self.dim, device=p.device, dtype=p.dtype)
        return p @ (torch.matrix_exp(generator) - identity).T

    def velocity(self, phi: torch.Tensor, perturbations: list[int]) -> torch.Tensor:
        """B_S p restricted to the observables, [B, K]: the field flow matching fits."""
        return (self.augment(phi) @ self.compose(perturbations).T)[
            :, :self.observable_dim]

    def flow(self, phi: torch.Tensor, perturbations: list[int]) -> torch.Tensor:
        """Where the observables end up, [B, K]: what the endpoint loss compares."""
        return phi + self.displacement(phi, perturbations)[:, :self.observable_dim]

    # ---------------------------------------------------------------- forward

    def forward(self, x: torch.Tensor, phi: torch.Tensor,
                perturbations: list[int]) -> dict[str, torch.Tensor]:
        dp = self.displacement(phi, perturbations)
        return assemble_output(
            x, self.to_u(dp), self.turn_on + self.to_v(dp), self.to_q(dp),
            self.ceiling, self.bound_mean, self.log_scale, self.log_factor_max)

    # ------------------------------------------------------------ bookkeeping

    def turn_on_offset(self) -> float:
        """softplus(v0): what an untrained model, and the control always, adds to x."""
        return float(torch.nn.functional.softplus(self.turn_on))

    @property
    def starts_from_ridge(self) -> bool:
        return False

    def report(self, perturbations: list[int]) -> str:
        """What to watch while this trains: the generator and both readouts."""
        with torch.no_grad():
            generator = float(self.compose(perturbations).abs().max())
            u_weight = float(self.to_u.weight.abs().max())
            v_weight = float(self.to_v.weight.abs().max())
        text = f"|B| {generator:.4f}  |Wu| {u_weight:.4f}  |Wv| {v_weight:.4f}"
        if self.kappa is not None:
            text += f"  kappa {float(self.kappa):+.4f}"
        return text
