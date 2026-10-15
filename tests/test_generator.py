"""What stage 2 guarantees by construction - docs/DESIGN_STAGE2.md section 3.

    python -m pytest tests/test_generator.py -v

Not accuracy tests. Each one checks something the design says cannot fail, so a failure
means the implementation is not the model in the document. The parameters are pushed
well away from their initial values first, because almost every one of these holds
trivially for an untrained model whose readouts are zero.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config as config_module
from src.models.model import PathwayModulation
from src.models.modulation import PerturbationModulation
from test_modulation import FakeObservables, K, N_GENES, N_PERTURBATIONS, control_cells


def build(overrides: list[str] | None = None, **kwargs) -> PathwayModulation:
    config = config_module.load(["model.response=generator"] + (overrides or []))
    torch.manual_seed(0)
    return PathwayModulation(config, FakeObservables(), N_PERTURBATIONS, **kwargs).eval()


def train_like(model: PathwayModulation, scale: float = 1.0) -> None:
    """Move every learnable piece off its starting value, as training would."""
    response = model.modulation
    with torch.no_grad():
        for layer in (response.to_u, response.to_v, response.to_q):
            layer.weight.normal_(0.0, scale)
        for parameter in response.operators.parameters():
            parameter.normal_(0.0, 0.3)
        if response.kappa is not None:
            response.kappa.fill_(0.7)


# ---------------------------------------------------------------- G1 .. G6

@pytest.mark.parametrize("perturbations", [[], [0], [1, 3], [0, 2, 5]])
def test_g1_the_mean_can_never_be_negative(perturbations):
    model = build()
    train_like(model, scale=5.0)
    with torch.no_grad():
        mean = model(control_cells(), perturbations)["mean"]
    assert torch.isfinite(mean).all()
    assert float(mean.min()) >= 0.0


def test_g2_the_control_is_unchanged_however_trained():
    """Stage 1 only had this for an untrained model. Here B = 0 gives dp = 0 and neither
    readout has a bias, so it holds at every point in training."""
    model = build()
    train_like(model, scale=5.0)
    cells = control_cells()
    with torch.no_grad():
        mean = model(cells, [])["mean"]
    assert torch.allclose(mean, cells + model.modulation.turn_on_offset(), atol=1e-6)


def test_g3_order_does_not_change_the_prediction():
    model = build()
    train_like(model)
    cells = control_cells()
    with torch.no_grad():
        a = model(cells, [1, 4, 5])["mean"]
        b = model(cells, [5, 1, 4])["mean"]
    assert torch.equal(a, b)


def test_g4_no_parameter_is_indexed_by_a_pair_of_perturbations():
    for name, parameter in build().named_parameters():
        assert N_PERTURBATIONS ** 2 not in tuple(parameter.shape), name


def test_g5_an_untrained_model_is_the_control_for_every_condition():
    model = build()
    cells = control_cells()
    with torch.no_grad():
        for perturbations in ([0], [2, 4]):
            mean = model(cells, perturbations)["mean"]
            assert torch.allclose(mean, cells + model.modulation.turn_on_offset(),
                                  atol=1e-6)


def test_g5_the_gradient_reaches_the_readout_at_step_zero():
    """Zero readouts with zero generators would be a model that never moves: dL/dW is
    proportional to dp. The generators therefore start small and random."""
    model = build().train()
    cells = control_cells()
    model(cells, [1, 3])["mean"].sum().backward()
    assert float(model.modulation.to_u.weight.grad.abs().max()) > 0.0


def test_g6_the_constant_coordinate_stays_one_under_the_flow():
    model = build()
    train_like(model)
    phi = model.observables(control_cells())
    with torch.no_grad():
        dp = model.modulation.displacement(phi, [0, 3])
    assert dp.shape[1] == K + 1
    assert float(dp[:, -1].abs().max()) == 0.0


# ------------------------------------------------------ the composition claims

def _commuting(model: PathwayModulation) -> None:
    """Generators A_a = Q diag(c_a) Q^T, which commute with one another."""
    response = model.modulation
    q, _ = torch.linalg.qr(torch.randn(response.dim, response.dim))
    with torch.no_grad():
        response.operators.basis_u.copy_(q)
        response.operators.basis_v.copy_(q.T)
        response.operators.coefficient.normal_(0.0, 0.3)


def test_g7_commuting_generators_compose_as_a_product():
    """exp(A + B) = exp(A) exp(B) when AB = BA: both at once is one after the other.

    This is NOT the multiplicative null. That needs the displacements to add, which takes
    AB = BA = 0; commuting generators still leave (exp(A)-I)(exp(B)-I) p between them.

    Without the constant coordinate, because masking its row breaks the orthogonal
    construction used here to make the generators commute.
    """
    model = build(["model.generator_constant=false", f"model.shared_rank={K}",
                   "model.private_rank=0"])
    torch.manual_seed(1)
    _commuting(model)
    response = model.modulation
    phi = model.observables(control_cells())
    with torch.no_grad():
        both = phi + response.displacement(phi, [1, 4])
        first = phi + response.displacement(phi, [1])
        then = first + response.displacement(first, [4])
    assert torch.allclose(both, then, atol=1e-4)


def test_the_interaction_begins_with_the_anticommutator():
    """exp(A+B) - I - [exp(A)-I] - [exp(B)-I] = (AB + BA)/2 + O(3).

    Checked on the code path the model uses, at a scale small enough for the third-order
    remainder to be visibly smaller than the term it is the remainder of.
    """
    model = build()
    response = model.modulation
    with torch.no_grad():
        for parameter in response.operators.parameters():
            parameter.normal_(0.0, 0.15)
        phi = model.observables(control_cells())
        p = response.augment(phi)
        a, b = (response.compose([i]) for i in (1, 4))
        gap = (response.displacement(phi, [1, 4]) - response.displacement(phi, [1])
               - response.displacement(phi, [4]))
        leading = p @ (0.5 * (a @ b + b @ a)).T
    remainder = float((gap - leading).norm() / leading.norm())
    assert float(leading.norm()) > 0.0
    assert remainder < 0.2, f"the anticommutator leaves {remainder:.3f} of the gap"


def test_the_linear_arm_is_the_first_order_model():
    model = build(["model.generator_linear=true"])
    train_like(model)
    response = model.modulation
    phi = model.observables(control_cells())
    with torch.no_grad():
        expected = response.augment(phi) @ response.compose([1, 4]).T
        assert torch.allclose(response.displacement(phi, [1, 4]), expected, atol=1e-6)


def test_the_explicit_interaction_arm_starts_as_the_plain_model():
    """kappa is zero at the start, so arm F is nested in the model rather than a rival
    initialisation of it."""
    plain = build()
    explicit = build(["model.generator_explicit_interaction=true"])
    assert float(explicit.modulation.kappa) == 0.0
    cells = control_cells()
    for model in (plain, explicit):
        with torch.no_grad():
            for layer in (model.modulation.to_u, model.modulation.to_v):
                torch.manual_seed(3)
                layer.weight.normal_(0.0, 1.0)
    with torch.no_grad():
        assert torch.allclose(plain(cells, [1, 4])["mean"], explicit(cells, [1, 4])["mean"],
                              atol=1e-6)


def test_the_flow_and_the_velocity_agree_with_the_displacement():
    """What the observable-space losses supervise is the same generator the prediction
    uses, restricted to the observables."""
    model = build()
    train_like(model)
    response = model.modulation
    phi = model.observables(control_cells())
    with torch.no_grad():
        dp = response.displacement(phi, [0, 2])
        assert torch.allclose(response.flow(phi, [0, 2]), phi + dp[:, :K], atol=1e-6)
        velocity = response.velocity(phi, [0, 2])
        expected = (response.augment(phi) @ response.compose([0, 2]).T)[:, :K]
        assert torch.allclose(velocity, expected, atol=1e-6)


# ------------------------------------------------------------- compatibility

def test_the_bounded_mean_holds_for_the_generator_model_too():
    cells = control_cells()
    ceiling = cells.max(dim=0).values.clamp(min=1e-3)
    model = build(["model.bound_mean=true"], ceiling=ceiling)
    train_like(model, scale=5.0)
    with torch.no_grad():
        mean = model(cells, [1, 3])["mean"]
    assert float(mean.min()) >= 0.0
    assert float((mean - ceiling).max()) <= 1e-5


def test_a_config_without_the_key_is_still_stage_one():
    """Every run trained before `response` existed must rebuild as the MLP it was."""
    config = config_module.load([])
    del config["model"]["response"]
    torch.manual_seed(0)
    model = PathwayModulation(config, FakeObservables(), N_PERTURBATIONS)
    assert isinstance(model.modulation, PerturbationModulation)
    assert not model.has_generators
