"""What the soft prior guarantees - docs/DESIGN_PRIOR.md section 6.

    python -m pytest tests/test_prior.py -v

A = G * A_K + A_W, the dense branch paying lambda ||A_W||^2. Built on the small
KEGG-like space of tests/test_graph.py, whose every edge is named there.
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
from test_graph import K, N_PERTURBATIONS, FakeKeggObservables, build, cells, train_like

RESIDUAL = "model.generator_residual=true"


def zero(module) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()


def test_p1_with_the_residual_at_zero_the_model_is_the_masked_model():
    with_residual = build("kegg", RESIDUAL)
    masked = build("kegg")
    train_like(with_residual)
    zero(with_residual.modulation.residual)
    shared = {k: v for k, v in with_residual.state_dict().items()
              if ".residual." not in k}
    masked.load_state_dict(shared)
    x = cells()
    with torch.no_grad():
        assert torch.equal(with_residual(x, [0, 3])["mean"], masked(x, [0, 3])["mean"])


def test_p2_the_residual_really_does_reach_outside_the_graph():
    model = build("kegg", RESIDUAL)
    train_like(model)
    response = model.modulation
    with torch.no_grad():
        for parameter in response.residual.parameters():
            parameter.normal_(0.0, 0.3)
    outside = (response.graph_mask[:K, :K] == 0)
    a = response.generator(1)[:K, :K]
    assert float(a[outside].abs().max()) > 0.0


def test_p3_the_penalty_reaches_the_residual_and_never_the_backbone():
    model = build("kegg", RESIDUAL).train()
    train_like(model)
    with torch.no_grad():
        for parameter in model.modulation.residual.parameters():
            parameter.normal_(0.0, 0.3)
    penalty = model.modulation.residual_penalty([0, 2])
    penalty.backward()
    assert float(penalty) > 0.0
    for parameter in model.modulation.residual.parameters():
        assert parameter.grad is not None and float(parameter.grad.abs().max()) > 0.0
    for parameter in model.modulation.operators.parameters():
        assert parameter.grad is None or float(parameter.grad.abs().max()) == 0.0


def test_p3_no_residual_no_penalty():
    model = build("kegg")
    assert float(model.modulation.residual_penalty([0, 2])) == 0.0


def test_p4_stage_twos_guarantees_hold_with_the_residual_on():
    model = build("kegg", RESIDUAL)
    train_like(model)
    with torch.no_grad():
        for parameter in model.modulation.residual.parameters():
            parameter.normal_(0.0, 0.3)
    x = cells()
    with torch.no_grad():
        control = model(x, [])["mean"]
        treated = model(x, [1, 4])["mean"]
        swapped = model(x, [4, 1])["mean"]
        constant_row = model.modulation.generator(2)[K, :]
    assert torch.allclose(control, x + model.modulation.turn_on_offset(), atol=1e-6)
    assert float(treated.min()) >= 0.0
    assert torch.equal(treated, swapped)
    assert float(constant_row.abs().max()) == 0.0


def test_p5_the_offgraph_share_is_zero_for_the_backbone_alone():
    model = build("kegg", RESIDUAL)
    train_like(model)
    zero(model.modulation.residual)
    assert model.modulation.offgraph_share(list(range(N_PERTURBATIONS))) == 0.0


def test_p5_the_offgraph_share_is_the_energy_outside_the_graph():
    model = build("kegg", RESIDUAL)
    response = model.modulation
    zero(response.operators)
    with torch.no_grad():
        for parameter in response.residual.parameters():
            parameter.normal_(0.0, 0.3)
    outside = (1.0 - response.graph_mask)[:K, :K]
    off = total = 0.0
    for pert in range(N_PERTURBATIONS):
        a = (response.residual.matrix(pert) * response.row_mask)[:K, :K]
        off += float((a * outside).square().sum())
        total += float(a.square().sum())
    share = response.offgraph_share(list(range(N_PERTURBATIONS)))
    assert share == pytest.approx(off / total, rel=1e-6)
    assert 0.0 < share < 1.0


def test_p6_a_residual_without_a_backbone_is_refused():
    config = config_module.load(["model.response=generator", RESIDUAL])
    with pytest.raises(ValueError, match="needs model.generator_mask"):
        PathwayModulation(config, FakeKeggObservables(), N_PERTURBATIONS)


def test_p7_the_residual_starts_at_exactly_zero_and_still_learns():
    """LoRA's initialisation: the model starts as the hard mask, the off-graph share
    starts at zero, and the prediction's gradient still reaches the residual."""
    model = build("kegg", RESIDUAL).train()
    train_like(model)
    zero_residual = all(float(model.modulation.residual.matrix(p).abs().max()) == 0.0
                        for p in range(N_PERTURBATIONS))
    assert zero_residual
    assert model.modulation.offgraph_share(list(range(N_PERTURBATIONS))) == 0.0
    model(cells(), [1, 3])["mean"].sum().backward()
    grads = [p.grad for p in model.modulation.residual.parameters() if p.grad is not None]
    assert grads and max(float(g.abs().max()) for g in grads) > 0.0
