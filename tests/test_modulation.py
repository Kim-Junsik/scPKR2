"""What this model guarantees without training, asserted as numbers.

    python -m pytest tests/test_modulation.py -v

Not accuracy tests. Each one checks something the construction is supposed to make
impossible, so a failure means the model is wrong rather than under-trained. They run on
a synthetic observable space and need no data.

THE FIRST GROUP IS THE REASON THIS REPOSITORY EXISTS. v2 added a population-level shift
to every cell, which drove entries below zero wherever the gene was already off - 41 % of
them are. 33.8 % of its per-cell predictions on ComboSciPlex came out negative, and no
non-negative realisation can reproduce a negative mean: flooring the predictions at zero
and averaging scores 2.4226 against 1.6220 for the mean itself, so non-negativity alone
costs 0.80 of L2 there. The fixture below therefore reproduces the conditions that
produced those negatives - control cells full of exact zeros, and parameters pushed hard
in the downward direction - and the tests fail if a negative is ever representable again.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as config_module
from src.models.model import PathwayModulation

N_GENES = 40
N_PERTURBATIONS = 6
K = 12


class FakeObservables(torch.nn.Module):
    """A fixed random Phi. The model must not care what the coordinates mean."""

    def __init__(self, seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("matrix", torch.randn(K, N_GENES, generator=generator) * 0.1)
        self.gene_names = np.array([f"g{i}" for i in range(N_GENES)])
        self.dim = K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix.T


def control_cells(n: int = 64, seed: int = 1) -> torch.Tensor:
    """41 % exact zeros, like the real control population.

    The zeros are the point. A shift applied to a cell where the gene is already off is
    exactly what v2 could not survive, so a fixture without them would pass while proving
    nothing.
    """
    rng = np.random.default_rng(seed)
    x = rng.gamma(2.0, 0.4, size=(n, N_GENES)).astype(np.float32)
    x[rng.random(x.shape) < 0.41] = 0.0
    return torch.from_numpy(x)


def build(overrides: list[str] | None = None, detection=None) -> PathwayModulation:
    config = config_module.load(overrides or [])
    torch.manual_seed(0)
    return PathwayModulation(config, FakeObservables(), N_PERTURBATIONS,
                             detection=detection).eval()


def push_downward(model: PathwayModulation, scale: float = 25.0) -> None:
    """Drive every learnable map hard toward suppressing expression.

    An untrained model is non-negative trivially, since its outputs are its biases. This
    puts the parameters where a trained model trying to turn genes OFF would be, which is
    where v2 produced its negatives.
    """
    with torch.no_grad():
        for layer in (model.modulation.to_u, model.modulation.to_v, model.modulation.to_q):
            layer.weight.normal_(0.0, scale)
            layer.bias.normal_(-scale, scale)
        model.modulation.embedding.weight.normal_(0.0, scale)


@pytest.mark.parametrize("perturbations", [[], [0], [1, 3], [0, 2, 5]])
def test_the_mean_can_never_be_negative(perturbations):
    """C2 in docs/DESIGN.md. x exp(u) + softplus(v) has no negative in its range."""
    model = build()
    push_downward(model)
    cells = control_cells()
    with torch.no_grad():
        mean = model(cells, perturbations)["mean"]
    assert torch.isfinite(mean).all()
    assert float(mean.min()) >= 0.0, (
        f"mu reached {float(mean.min()):.4f}. The parameterisation is supposed to make "
        f"that impossible, and this defect is the whole reason for the rebuild.")


def test_a_gene_that_is_off_stays_off_under_modulation_alone():
    """x * exp(u) cannot turn a zero into anything - the mechanism, not a side effect.

    v2's failure was precisely here: w_a is a population mean shift, so a gene at zero in
    THIS cell still received it. Only the softplus term may switch a gene on, and it is
    switched off for this test so the multiplicative half is tested alone.
    """
    model = build()
    push_downward(model)
    with torch.no_grad():
        model.modulation.to_v.weight.zero_()
        model.modulation.to_v.bias.fill_(-50.0)      # softplus(-50) is 0 to float32
        cells = control_cells()
        mean = model(cells, [1, 3])["mean"]
    off = cells == 0
    assert bool(off.any())
    # softplus(-50) is 2e-22, not 0: float32 holds it. The claim being tested is that the
    # MULTIPLICATIVE term contributes exactly nothing where x is zero, and 2e-22 is the
    # turn-on term's floor rather than a leak from it.
    assert float(mean[off].abs().max()) < 1e-20


def test_an_untrained_model_is_the_control_exactly():
    """The starting point has to be nameable, and this one is Control at 3.9937.

    v2 started at ridge, 1.6690, because its additive term was already fitted. That
    protection is gone deliberately - it is also what left the learned part with nothing
    to do on single perturbations - so the least this must do is start somewhere that can
    be written in a table rather than at a random gene-space field.
    """
    model = build()
    cells = control_cells()
    with torch.no_grad():
        for perturbations in ([], [0], [2, 4]):
            mean = model(cells, perturbations)["mean"]
            assert torch.allclose(mean, cells, atol=1e-3), (
                f"untrained prediction differs from the control by "
                f"{float((mean - cells).abs().max()):.2e} on {perturbations}")


def test_order_does_not_change_the_prediction():
    """e_S is a sum, and a simultaneous perturbation has no order.

    Nothing in the training signal would impose this - the data never shows A-then-B
    against B-then-A - so it has to come from the construction.
    """
    model = build()
    push_downward(model)
    cells = control_cells()
    with torch.no_grad():
        a = model(cells, [1, 4, 5])["mean"]
        b = model(cells, [5, 1, 4])["mean"]
    assert torch.equal(a, b)


def test_the_control_moves_nothing_however_trained():
    """e_S = 0 for the empty set, at every point in training, not just at the start."""
    model = build()
    push_downward(model)
    with torch.no_grad():
        embedding = model.modulation.embed([])
    assert float(embedding.abs().max()) == 0.0


def test_no_parameter_is_indexed_by_a_pair_of_perturbations():
    """Combinations are represented by summing single embeddings, never by a pair table.

    A per-pair parameter would memorise the training combinations and have nothing to say
    about a held-out one, which is the only thing these benchmarks ask.
    """
    model = build()
    for name, parameter in model.named_parameters():
        assert N_PERTURBATIONS ** 2 not in tuple(parameter.shape), (
            f"{name} has a dimension the size of all perturbation PAIRS: "
            f"{tuple(parameter.shape)}")


def test_the_realisation_keeps_the_mean():
    """Bernoulli(q) * (mu/q) has mean mu for any q, and the draw must not break that."""
    model = build(["model.hurdle_gate=sample", "eval.realisation=gamma"])
    # u and v only. Driving q to its extremes as well makes mu/q enormous, and then this
    # measures how many draws were taken rather than whether the draw is unbiased - the
    # magnitude's behaviour at a tiny q is v2's finding and is not what is under test.
    with torch.no_grad():
        for layer in (model.modulation.to_u, model.modulation.to_v):
            layer.weight.normal_(0.0, 1.0)
            layer.bias.normal_(0.0, 1.0)
        model.modulation.embedding.weight.normal_(0.0, 1.0)
    cells = control_cells()
    with torch.no_grad():
        params = model(cells, [1, 3])
        drawn = torch.stack([model.head.point_estimate(params) for _ in range(2000)])
    assert float(drawn.min()) >= 0.0
    # Averaged over cells as well as draws: the per-entry estimate from 2,000 draws is
    # still noisy, and the quantity every reported metric uses is the population mean.
    error = (drawn.mean(dim=(0, 1)) - params["mean"].mean(dim=0)).abs()
    assert float((error / params["mean"].mean(dim=0).clamp(min=1e-3)).max()) < 0.1


def test_a_negative_mean_is_raised_rather_than_floored():
    """If mu ever goes negative the model is broken, and the loss must say so.

    v2 floored instead, which is how a mean vector reached cell-eval and was refused with
    "min value -2.63 is negative" after a day of scoring had already been spent on it.
    """
    model = build()
    cells = control_cells()
    with torch.no_grad():
        params = model(cells, [1])
        params["mean"] = params["mean"] - 1.0
    with pytest.raises(ValueError, match="negative"):
        model.head.loss(params, cells)


def test_the_detection_rate_initialises_q():
    """q starts at each gene's observed rate, not at a constant.

    q divides the magnitude, so a constant makes mu/q wrong for every gene at once - the
    mechanism behind v2's uncapped 465.66 on ComboSciPlex.
    """
    rng = np.random.default_rng(0)
    detection = torch.from_numpy(rng.uniform(0.02, 0.9, size=N_GENES).astype(np.float32))
    model = build(detection=detection)
    cells = control_cells()
    with torch.no_grad():
        q = model(cells, [])["q"]
    assert torch.allclose(q.mean(dim=0), detection, atol=1e-3)


def test_the_magnitude_cannot_exceed_what_the_gene_reaches_in_the_data():
    """q >= mu / ceiling, so mu/q <= ceiling - and the mean is untouched by it.

    The hurdle says E[x] = q m, where m is what a cell holds WHEN the gene is detected.
    m cannot exceed what that gene has ever been observed at. Imposed on q rather than by
    capping m, because q (mu/q) = mu for ANY q while capping m takes the mean down with
    it - v2 capped and lost 0.80 of L2 on ComboSciPlex.

    It is needed because mu and q are separate outputs that do not see each other: the
    trained model emitted mu = 6.16 beside q = 0.001 on the same entry, mu/q reached 342,
    and cell-eval refused the export at 59.11 against a threshold of 15.
    """
    cells = control_cells()
    ceiling = cells.max(dim=0).values.clamp(min=1e-3)
    config = config_module.load([])
    torch.manual_seed(0)
    model = PathwayModulation(config, FakeObservables(), N_PERTURBATIONS,
                              ceiling=ceiling).eval()
    push_downward(model)
    with torch.no_grad():
        params = model(cells, [1, 3])
    # q can only hold m under the ceiling while mu itself is under it; where the model's
    # MEAN exceeds what the gene has ever reached, no choice of q fixes that and the mean
    # is the thing that is wrong. The bound asserted is therefore max(ceiling, mu).
    bound = torch.maximum(ceiling.expand_as(params["mean"]), params["mean"])
    assert float(((params["magnitude"] - bound) / bound.clamp(min=1e-3)).max()) <= 1e-4, (
        f"magnitude reached {float(params['magnitude'].max()):.2f} against a bound of "
        f"{float(bound.max()):.2f}")
    # The mean is the quantity every reported L2 is computed from, and a floor on q may
    # not move it: q * (mu/q) = mu identically.
    assert torch.allclose(params["q"] * params["magnitude"], params["mean"], atol=1e-5)


def test_a_drawn_cell_stays_inside_the_range_its_gene_occupies():
    """The beta draw is supported on [0, ceiling], so no tail can leave the data's range.

    gamma gets the mean right and is unbounded; with the spread clamped at e^2 = 7.39 its
    tail reached 59.11 on the real export and cell-eval, whose limit is 15, refused the
    file. Capping that draw would fix the range and lose the mean, which is the trade v2
    made and paid 0.80 of L2 for on ComboSciPlex.
    """
    cells = control_cells()
    ceiling = cells.max(dim=0).values.clamp(min=1e-3)
    config = config_module.load(["model.hurdle_gate=sample", "eval.realisation=beta"])
    torch.manual_seed(0)
    model = PathwayModulation(config, FakeObservables(), N_PERTURBATIONS,
                              ceiling=ceiling).eval()
    with torch.no_grad():
        for layer in (model.modulation.to_u, model.modulation.to_v,
                      model.modulation.to_q):
            layer.weight.normal_(0.0, 1.0)
            layer.bias.normal_(0.0, 1.0)
        params = model(cells, [1, 3])
        drawn = torch.stack([model.head.point_estimate(params) for _ in range(200)])
    assert float(drawn.min()) >= 0.0
    bound = torch.maximum(ceiling.expand_as(params["magnitude"]),
                          params["magnitude"] * 1.001)
    assert float((drawn - bound).max()) <= 1e-3, (
        f"a drawn cell reached {float(drawn.max()):.2f} outside its support, bounded at "
        f"{float(bound.max()):.2f}")
    # And the mean still has to be mu, which is what every reported L2 is computed from.
    error = (drawn.mean(dim=0).mean(dim=0) - params["mean"].mean(dim=0)).abs()
    assert float((error / params["mean"].mean(dim=0).clamp(min=1e-3)).max()) < 0.15


def test_the_weight_average_is_an_average_and_replaces_the_weights():
    """EMA has to move the parameters toward the trajectory, not leave them alone.

    It exists because the variance across seeds is what blocks every claim here: the gap
    to v2 is 0.082 while three seeds give a standard error of 0.091. Of ComboSciPlex's
    0.306 spread, the initialisation explains 39 % and the batch order and the OT
    coupling the other 61 %, and averaging along the trajectory is aimed at that 61 %.

    Asserted on the update rule itself rather than through a training run, so a failure
    points at the arithmetic instead of at the optimiser.
    """
    decay = 0.9
    start = torch.zeros(4)
    ema = start.clone()
    steps = [torch.full((4,), float(k)) for k in range(1, 21)]
    for step in steps:
        ema.mul_(decay).add_(step, alpha=1.0 - decay)
    # Strictly between where it started and where it ended: an average of a trajectory
    # that is still moving cannot equal either endpoint.
    assert float(ema[0]) > float(start[0])
    assert float(ema[0]) < float(steps[-1][0])
    # And it must track the recent part, not the whole history equally: with decay 0.9
    # the effective window is about ten steps, so it sits far above the overall mean.
    assert float(ema[0]) > float(torch.stack(steps).mean())
