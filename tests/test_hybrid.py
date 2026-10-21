"""What the hybrid coordinates and the stochastic FM guarantee - docs/DESIGN_HYBRID.md 8.

    python -m pytest tests/test_hybrid.py -v
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
from src.train.loop import bridge_interpolant
from test_modulation import K, N_PERTURBATIONS, FakeObservables, control_cells

D_Z = 6


def build(*overrides: str) -> PathwayModulation:
    config = config_module.load(["model.response=generator", *overrides])
    torch.manual_seed(0)
    return PathwayModulation(config, FakeObservables(), N_PERTURBATIONS).eval()


def train_like(model: PathwayModulation) -> None:
    response = model.modulation
    with torch.no_grad():
        for layer in (response.to_u, response.to_v, response.to_q):
            layer.weight.normal_(0.0, 1.0)
        for parameter in response.operators.parameters():
            parameter.normal_(0.0, 0.3)


def test_y1_no_latent_is_the_model_as_it_was():
    plain, explicit = build(), build("model.latent_dim=0")
    assert plain.latent is None and explicit.latent is None
    x = control_cells()
    with torch.no_grad():
        assert torch.equal(plain.coordinates(x), plain.observables(x))
        assert torch.equal(plain(x, [1, 3])["mean"], explicit(x, [1, 3])["mean"])


def test_y1_the_generator_spans_both_blocks_and_the_constant():
    model = build(f"model.latent_dim={D_Z}")
    assert model.coordinate_dim == K + D_Z
    assert model.modulation.dim == K + D_Z + 1
    assert model.coordinates(control_cells()).shape[1] == K + D_Z


def test_y2_sigma_zero_is_the_straight_path_drawn_as_before():
    p0, p1 = torch.randn(32, 5), torch.randn(32, 5)
    torch.manual_seed(4)
    p_t, target = bridge_interpolant(p0, p1, 0.0)
    torch.manual_seed(4)
    t = torch.rand(32, 1)
    assert torch.equal(p_t, (1.0 - t) * p0 + t * p1)
    assert torch.equal(target, p1 - p0)


def test_y2_the_bridge_noise_vanishes_on_average():
    """E[u_t | p0, p1] = p1 - p0: the bridge adds noise to the target, not a bias."""
    p0, p1 = torch.zeros(200_000, 2), torch.ones(200_000, 2)
    torch.manual_seed(5)
    p_t, target = bridge_interpolant(p0, p1, 0.5)
    assert torch.allclose(target.mean(dim=0), torch.ones(2), atol=0.02)
    assert float((p_t - p_t.mean(0)).std()) > 0.0


def test_y3_the_latent_cannot_collapse():
    model = build(f"model.latent_dim={D_Z}")
    with torch.no_grad():
        for parameter in model.latent.parameters():
            parameter.mul_(1e-4)                 # as if training shrank the encoder
        z = model.latent(control_cells())
    assert torch.allclose(z.mean(dim=1), torch.zeros(z.shape[0]), atol=1e-4)
    assert torch.allclose(z.var(dim=1, unbiased=False), torch.ones(z.shape[0]), atol=1e-2)


def test_y3_a_cell_s_latent_does_not_depend_on_its_batch():
    model = build(f"model.latent_dim={D_Z}").train()
    x = control_cells()
    with torch.no_grad():
        alone = model.latent(x[:5])
        together = model.latent(x)[:5]
    assert torch.allclose(alone, together, atol=1e-6)


def test_y4_the_pairing_coordinates_ignore_the_latent():
    model = build(f"model.latent_dim={D_Z}")
    x = control_cells()
    with torch.no_grad():
        before = model.source_observables(x, [1]).clone()
        for parameter in model.latent.parameters():
            parameter.normal_(0.0, 3.0)
        after = model.source_observables(x, [1])
    assert torch.equal(before, after)


def test_y5_stage_twos_guarantees_hold_with_the_latent():
    model = build(f"model.latent_dim={D_Z}")
    train_like(model)
    with torch.no_grad():
        for parameter in model.latent.parameters():
            parameter.normal_(0.0, 1.0)
    x = control_cells()
    with torch.no_grad():
        control = model(x, [])["mean"]
        treated = model(x, [2, 4])["mean"]
        swapped = model(x, [4, 2])["mean"]
    assert torch.allclose(control, x + model.modulation.turn_on_offset(), atol=1e-6)
    assert float(treated.min()) >= 0.0
    assert torch.equal(treated, swapped)


def test_y6_evaluation_is_deterministic():
    model = build(f"model.latent_dim={D_Z}")
    train_like(model)
    x = control_cells()
    with torch.no_grad():
        assert torch.equal(model(x, [1])["mean"], model(x, [1])["mean"])


def test_fm_on_the_coordinates_reaches_the_encoder():
    """The point of putting FM on [Phi; z]: it trains z, not only the generators."""
    model = build(f"model.latent_dim={D_Z}").train()
    train_like(model)
    x0, x1 = control_cells(seed=1), control_cells(seed=2)
    p_t, target = bridge_interpolant(model.coordinates(x0), model.coordinates(x1), 0.3)
    loss = torch.nn.functional.mse_loss(model.modulation.velocity(p_t, [1]), target)
    loss.backward()
    grads = [p.grad for p in model.latent.parameters() if p.grad is not None]
    assert grads and max(float(g.abs().max()) for g in grads) > 0.0


def test_a_latent_with_a_graph_mask_is_refused():
    with pytest.raises(ValueError, match="mask"):
        build(f"model.latent_dim={D_Z}", "model.generator_mask=kegg")


def test_a_latent_without_generators_is_refused():
    config = config_module.load([f"model.latent_dim={D_Z}"])
    with pytest.raises(ValueError, match="latent_dim"):
        PathwayModulation(config, FakeObservables(), N_PERTURBATIONS)
