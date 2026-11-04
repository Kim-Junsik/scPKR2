"""What the shrink term guarantees - src/config.py train.shrink_weight.

    python -m pytest tests/test_shrink.py -v

penalty(S) = sum_{a in S} ||A_a - A_bar||_F^2 / n_a, A_bar the detached average generator.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import config as config_module
from src.models.model import PathwayModulation
from test_modulation import N_PERTURBATIONS, FakeObservables

CONDITIONS = [[0, 1], [1, 2], [2], [3, 0]]          # perturbation indices per condition


def build() -> PathwayModulation:
    config = config_module.load(["model.response=generator"])
    torch.manual_seed(0)
    model = PathwayModulation(config, FakeObservables(), N_PERTURBATIONS).train()
    with torch.no_grad():
        for parameter in model.modulation.operators.parameters():
            parameter.normal_(0.0, 0.3)
    return model


def counts_of(conditions):
    counts = {}
    for condition in conditions:
        for pert in set(condition):
            counts[pert] = counts.get(pert, 0) + 1
    return counts


def test_the_default_is_off():
    assert config_module.load([])["train"]["shrink_weight"] == 0.0


def test_identical_generators_pay_nothing():
    model = build()
    operators = model.modulation.operators
    with torch.no_grad():
        operators.coefficient[:] = operators.coefficient[0]
        operators.private_u[:] = operators.private_u[0]
        operators.private_v[:] = operators.private_v[0]
    counts = counts_of(CONDITIONS)
    centre = model.modulation.mean_generator(sorted(counts))
    for condition in CONDITIONS:
        assert float(model.modulation.shrink_penalty(condition, centre, counts)) < 1e-10


def test_an_epoch_charges_each_perturbation_once():
    """Summed over the conditions, the 1 / n_a weights leave each drug counted once."""
    model = build()
    counts = counts_of(CONDITIONS)
    members = sorted(counts)
    centre = model.modulation.mean_generator(members)
    with torch.no_grad():
        epoch = sum(model.modulation.shrink_penalty(c, centre, counts) for c in CONDITIONS)
        once = sum((model.modulation._masked(p) - centre).square().sum() for p in members)
    assert torch.allclose(epoch, once, rtol=1e-5)


def test_the_basis_scale_cannot_pay_the_penalty_off():
    """U diag(c) V is unchanged by U -> kU, c -> c/k, and so is the penalty."""
    model = build()
    counts = counts_of(CONDITIONS)
    centre = model.modulation.mean_generator(sorted(counts))
    before = float(model.modulation.shrink_penalty([0, 1], centre, counts))
    with torch.no_grad():
        model.modulation.operators.basis_u.mul_(4.0)
        model.modulation.operators.coefficient.div_(4.0)
    centre = model.modulation.mean_generator(sorted(counts))
    after = float(model.modulation.shrink_penalty([0, 1], centre, counts))
    assert abs(before - after) <= 1e-5 * before


def test_only_the_conditions_perturbations_are_pushed():
    model = build()
    counts = counts_of(CONDITIONS)
    centre = model.modulation.mean_generator(sorted(counts))
    assert not centre.requires_grad
    model.modulation.shrink_penalty([0, 1], centre, counts).backward()
    grad = model.modulation.operators.private_u.grad
    moved = {p for p in range(N_PERTURBATIONS) if float(grad[p].abs().max()) > 0}
    assert moved == {0, 1}


def test_the_mean_generator_is_the_eval_generator_average():
    model = build().eval()
    members = [0, 2, 3]
    with torch.no_grad():
        expected = sum(model.modulation.generator(p) for p in members) / len(members)
    assert torch.allclose(model.modulation.mean_generator(members), expected, atol=1e-6)
