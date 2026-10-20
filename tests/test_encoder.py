"""What the knowledge encoder guarantees - docs/DESIGN_ENCODER.md section 7.

    python -m pytest tests/test_encoder.py -v

c_a = c_bar + E k_a + delta_a, and A_a = U diag(c_a) V + P_a Q_a with the private parts
(delta_a, P_a Q_a) left out for a perturbation training never saw.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config as config_module
from src.data.knowledge import shuffle_rows
from src.models.model import PathwayModulation
from test_modulation import K, N_PERTURBATIONS, FakeObservables, control_cells

D_KNOWLEDGE = 5


def knowledge(seed: int = 0, zero_rows=(4,)) -> np.ndarray:
    matrix = np.random.default_rng(seed).normal(size=(N_PERTURBATIONS, D_KNOWLEDGE))
    matrix[list(zero_rows)] = 0.0
    return matrix.astype(np.float32)


def build(k=None, *overrides: str) -> PathwayModulation:
    config = config_module.load(["model.response=generator", *overrides])
    torch.manual_seed(0)
    return PathwayModulation(config, FakeObservables(), N_PERTURBATIONS,
                             knowledge=k).eval()


def train_like(model: PathwayModulation) -> None:
    response = model.modulation
    with torch.no_grad():
        for layer in (response.to_u, response.to_v, response.to_q):
            layer.weight.normal_(0.0, 1.0)
        for parameter in response.operators.parameters():
            parameter.normal_(0.0, 0.3)


def test_e1_at_the_start_the_encoder_is_the_lookup_table():
    """E = 0, delta = 0, c_bar = 1: every perturbation starts at c_a = 1, as before."""
    table, encoder = build(), build(knowledge())
    for pert in range(N_PERTURBATIONS):
        with torch.no_grad():
            assert torch.equal(table.modulation.generator(pert),
                               encoder.modulation.generator(pert))


def test_e2_an_unseen_perturbation_reads_no_private_parameter():
    model = build(knowledge())
    train_like(model)
    operators = model.modulation.operators
    operators.seen[2] = False
    with torch.no_grad():
        before = model.modulation.generator(2).clone()
        operators.coefficient[2].normal_(0.0, 5.0)
        operators.private_u[2].normal_(0.0, 5.0)
        after = model.modulation.generator(2)
        expected = operators.matrix(2, private=False) * model.modulation.row_mask
    assert torch.equal(before, after)
    assert torch.allclose(after, expected)


def test_e2_an_unseen_perturbation_does_use_its_knowledge():
    model = build(knowledge())
    train_like(model)
    operators = model.modulation.operators
    operators.seen[2] = False
    with torch.no_grad():
        before = model.modulation.generator(2).clone()
        operators.knowledge[2] += 1.0
        assert not torch.equal(before, model.modulation.generator(2))


def test_e3_no_knowledge_and_unseen_is_the_average_perturbation():
    model = build(knowledge(zero_rows=(4,)))
    train_like(model)
    operators = model.modulation.operators
    operators.seen[4] = False
    with torch.no_grad():
        expected = ((operators.basis_u * operators.coefficient_mean) @ operators.basis_v
                    * model.modulation.row_mask)
        assert torch.allclose(model.modulation.generator(4), expected, atol=1e-6)


def test_e4_dropout_happens_only_in_training():
    model = build(knowledge(), "model.perturbation_dropout=1.0")
    train_like(model)
    operators = model.modulation.operators
    with torch.no_grad():
        full = operators.matrix(1, private=True) * model.modulation.row_mask
        alone = operators.matrix(1, private=False) * model.modulation.row_mask
        model.eval()
        assert torch.allclose(model.modulation.generator(1), full)
        model.train()
        assert torch.allclose(model.modulation.generator(1), alone)
    assert not torch.allclose(full, alone)


def test_e4_dropout_zero_never_drops():
    model = build(knowledge(), "model.perturbation_dropout=0.0").train()
    train_like(model)
    operators = model.modulation.operators
    with torch.no_grad():
        full = operators.matrix(1) * model.modulation.row_mask
        for _ in range(20):
            assert torch.allclose(model.modulation.generator(1), full)


def test_e5_the_shuffle_moves_vectors_not_coverage():
    matrix = knowledge(seed=3, zero_rows=(0, 4))
    shuffled = shuffle_rows(matrix, seed=7)
    assert np.array_equal(shuffled[[0, 4]], matrix[[0, 4]])          # zeros stay
    assert sorted(map(tuple, shuffled.tolist())) == sorted(map(tuple, matrix.tolist()))
    assert not np.array_equal(shuffled, matrix)
    assert np.array_equal(shuffle_rows(matrix, seed=7), shuffled)    # deterministic


def test_e6_stage_twos_guarantees_hold_with_the_encoder():
    model = build(knowledge())
    train_like(model)
    model.modulation.operators.seen[3] = False
    x = control_cells()
    with torch.no_grad():
        control = model(x, [])["mean"]
        treated = model(x, [1, 3])["mean"]
        swapped = model(x, [3, 1])["mean"]
    assert torch.allclose(control, x + model.modulation.turn_on_offset(), atol=1e-6)
    assert float(treated.min()) >= 0.0
    assert torch.equal(treated, swapped)


def test_the_knowledge_map_learns_from_a_dropped_perturbation():
    model = build(knowledge(), "model.perturbation_dropout=1.0").train()
    train_like(model)
    model(control_cells(), [1])["mean"].sum().backward()
    operators = model.modulation.operators
    assert float(operators.knowledge_map.grad.abs().max()) > 0.0
    assert operators.coefficient.grad is None or float(operators.coefficient.grad.abs().max()) == 0.0


def test_a_checkpoint_without_the_encoder_loads_as_before():
    """No knowledge registers nothing new, so every run trained before still loads."""
    model = build()
    assert "modulation.operators.knowledge" not in model.state_dict()
    assert "modulation.operators.seen" not in model.state_dict()


def test_an_encoder_without_shared_modes_is_refused():
    with pytest.raises(ValueError, match="shared modes"):
        build(knowledge(), "model.shared_rank=0", "model.private_rank=4")


def test_the_mlp_cannot_be_given_knowledge():
    config = config_module.load(["model.perturbation_knowledge=string"])
    with pytest.raises(ValueError, match="builds generators"):
        PathwayModulation(config, FakeObservables(), N_PERTURBATIONS)


def test_private_false_without_an_encoder_is_refused():
    with pytest.raises(ValueError, match="needs a knowledge encoder"):
        build().modulation.operators.matrix(0, private=False)
