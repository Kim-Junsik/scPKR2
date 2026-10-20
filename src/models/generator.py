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
simultaneous perturbation has to be. Because u is a LOG-gain, a double whose displacement
is the sum of its singles' gives u_S = u_a + u_b: fold changes multiply, the null model
genetic-interaction work measures epistasis against. That holds exactly when AB = BA = 0,
NOT merely when the generators commute - commuting gives exp(A+B) = exp(A) exp(B), i.e.
both at once equals one after the other, and the cross term (exp(A)-I)(exp(B)-I) p
remains. Arm C (generator_linear) imposes the multiplicative null everywhere.

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
                 ceiling: torch.Tensor | None = None,
                 graph_mask: torch.Tensor | None = None,
                 knowledge=None):
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
            {"model": {**cfg, "composition": "sum"}}, n_perturbations, self.dim,
            knowledge=knowledge)
        # docs/DESIGN_ENCODER.md 1: the probability, per perturbation per training step,
        # that its private parameters are switched off. Without it they would always be
        # there to fit the data, and the knowledge path - all an unseen perturbation
        # has - would never be asked to stand alone.
        self.dropout = (float(cfg.get("perturbation_dropout", 0.0))
                        if knowledge is not None else 0.0)
        # An explicit weight on the anticommutator, beyond what the exponential already
        # supplies. ZERO at the start, so this arm begins as the plain model and the
        # comparison is nested rather than between two initialisations.
        self.kappa = nn.Parameter(torch.zeros(())) if explicit else None

        mask = torch.ones(self.dim, self.dim)
        if self.constant:
            mask[-1, :] = 0.0
        self.register_buffer("row_mask", mask, persistent=False)

        # Which coordinate pairs a generator may couple - docs/DESIGN_GRAPH.md, built by
        # models/graph.py. All ones unless a graph is asked for, and asking for one
        # without supplying it is an error rather than a silent dense model. PERSISTENT,
        # so a scored run uses the graph it was trained on - the permuted control's
        # shuffle included - rather than one rebuilt and hoped to match.
        kind = str(cfg.get("generator_mask", "none"))
        if graph_mask is None:
            if kind != "none":
                raise ValueError(f"model.generator_mask={kind} but no graph was built; "
                                 f"construct the model through build_model")
            graph_mask = torch.ones(self.dim, self.dim)
        if tuple(graph_mask.shape) != (self.dim, self.dim):
            raise ValueError(f"graph mask is {tuple(graph_mask.shape)}, the generator "
                             f"is {(self.dim, self.dim)}")
        self.register_buffer("graph_mask", graph_mask.clone().float(), persistent=True)

        # A dense branch beside the masked one - docs/DESIGN_PRIOR.md. Its own shared modes
        # and private parts, the same ranks, and NO mask: what it adds outside the graph is
        # what the data needed that the graph did not give, and the training loop charges
        # it lambda * ||A_W||^2. No separate alpha: a learned scale on this branch would
        # let it take everything and leave the backbone meaningless.
        self.residual = None
        if bool(cfg.get("generator_residual", False)):
            if kind == "none":
                raise ValueError("model.generator_residual needs model.generator_mask: "
                                 "a residual with no backbone is a dense model twice "
                                 "the size")
            self.residual = KoopmanOperators(
                {"model": {**cfg, "composition": "sum"}}, n_perturbations, self.dim)
            # EXACTLY ZERO at the start, by zeroing one factor of each product - LoRA's
            # initialisation. Small random values, like the backbone's, put a dense layer
            # of noise outside the graph before any data is seen: the off-graph share read
            # 0.91 after one epoch, and H2 would have been measuring initialisation. With
            # one factor zero, A_W = 0 but dL/dV = (U c)^T dL/dA is not, so the branch
            # grows only where the data asks for it - and the model starts as the hard
            # mask.
            with torch.no_grad():
                if self.residual.shared_rank:
                    self.residual.basis_v.zero_()
                if self.residual.private_rank:
                    self.residual.private_v.zero_()

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

    def generator(self, pert: int) -> torch.Tensor:
        """A_a, [D, D]: the backbone on its graph, plus the residual if there is one."""
        private = True
        if self.operators.has_knowledge:
            if not bool(self.operators.seen[pert]):
                private = False          # nothing has trained its private parameters
            elif (self.training and self.dropout > 0
                  and float(torch.rand(())) < self.dropout):
                private = False
        out = self.operators.matrix(pert, private=private) * self.graph_mask
        if self.residual is not None:
            out = out + self.residual.matrix(pert)
        return out * self.row_mask

    def offgraph_share(self, perturbations: list[int]) -> float:
        """sum_a ||(1 - G) * A_a||^2 / sum_a ||A_a||^2 on the K x K block.

        docs/DESIGN_PRIOR.md H2. The constant's column is left out: it is open in every
        graph, so it would count the same for a backbone and its shuffle.
        """
        k = self.observable_dim
        outside = (1.0 - self.graph_mask)[:k, :k]
        off = total = 0.0
        for pert in sorted(set(perturbations)):
            a = self.generator(pert)[:k, :k]
            off += float((a * outside).square().sum())
            total += float(a.square().sum())
        return off / total if total > 0 else float("nan")

    def residual_penalty(self, perturbations: list[int]) -> torch.Tensor:
        """sum_a ||A_a^W||_F^2 over the perturbations given; zero without a residual."""
        if self.residual is None or not perturbations:
            return torch.zeros((), device=self.row_mask.device, dtype=self.row_mask.dtype)
        return sum((self.residual.matrix(pert) * self.row_mask).square().sum()
                   for pert in sorted(perturbations))

    def compose(self, perturbations: list[int]) -> torch.Tensor:
        """B_S, [D, D]. The zero matrix for the control.

        SORTED before summing. Float addition is not associative, so the same generators
        in two orders differ in the last bits, and order invariance is meant to be a
        property this model has rather than one it has approximately.
        """
        device, dtype = self.row_mask.device, self.row_mask.dtype
        if not perturbations:
            return torch.zeros(self.dim, self.dim, device=device, dtype=dtype)
        generators = [self.generator(pert) for pert in sorted(perturbations)]
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

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """All ones for checkpoints written before the graph existed.

        Every stage 2 run so far was trained with no mask, and all ones is exactly that,
        so this reproduces the saved model rather than approximating it. Same hook, same
        reason, as the whitener's in models/observables.py.
        """
        key = prefix + "graph_mask"
        if key not in state_dict:
            state_dict[key] = torch.ones_like(self.graph_mask)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

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
        if self.residual is not None:
            with torch.no_grad():
                share = self.offgraph_share(perturbations)
            text += f"  offgraph {share:.3f}"
        if self.kappa is not None:
            text += f"  kappa {float(self.kappa):+.4f}"
        return text
