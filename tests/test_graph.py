"""What the generator's graph guarantees - docs/DESIGN_GRAPH.md section 4.

    python -m pytest tests/test_graph.py -v

A small KEGG-like observable space with known structure, so every edge the mask should
or should not have can be named:

    p0 genes 0-5      p1 genes 4-9      p2 genes 9-13     p0-p1 share 2, p1-p2 share 1
    p3 genes 14-17    p4 genes 15-19                      p3-p4 share 3
    anchors on genes 2 (in p0), 9 (in p1, p2), 18 (in p4), 20 (in nothing)

At m = 1 that is three components: {p0, p1, p2, a2, a9}, {p3, p4, a18}, {a20}.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as config_module
from src.models.graph import (coordinate_graph, correlation_graph, generator_mask,
                              permute_within_blocks)
from src.models.model import PathwayModulation

N_GENES = 22
N_PERTURBATIONS = 5
PATHWAYS = [range(0, 6), range(4, 10), range(9, 14), range(14, 18), range(15, 20)]
ANCHOR_GENES = [2, 9, 18, 20]
P, A = len(PATHWAYS), len(ANCHOR_GENES)
K = P + A
FIRST = [0, 1, 2, P + 0, P + 1]          # component of p0..p2, a2, a9
SECOND = [3, 4, P + 2]                    # component of p3, p4, a18
ISOLATED = [P + 3]                        # a20


class FakeKeggObservables(torch.nn.Module):
    """Pathway rows mean-pooled, anchor rows reading one gene - the real layout."""

    def __init__(self):
        super().__init__()
        matrix = torch.zeros(K, N_GENES)
        for row, genes in enumerate(PATHWAYS):
            matrix[row, list(genes)] = 1.0 / len(genes)
        for i, gene in enumerate(ANCHOR_GENES):
            matrix[P + i, gene] = 1.0
        self.register_buffer("matrix", matrix)
        self.gene_names = np.array([f"g{i}" for i in range(N_GENES)])
        self.n_pathways, self.n_anchors, self.dim = P, A, K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix.T


def cells(n: int = 64, seed: int = 1) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    x = rng.gamma(2.0, 0.4, size=(n, N_GENES)).astype(np.float32)
    x[rng.random(x.shape) < 0.41] = 0.0
    return torch.from_numpy(x)


def config_for(kind: str, *extra: str) -> dict:
    return config_module.load(["model.response=generator", f"model.generator_mask={kind}",
                               "model.generator_mask_min_shared=1", *extra])


def build(kind: str = "kegg", *extra: str) -> PathwayModulation:
    config = config_for(kind, *extra)
    observables = FakeKeggObservables()
    control_phi = observables(cells(200, seed=7)).numpy()
    mask, summary = generator_mask(config, observables, control_phi,
                                   bool(config["model"]["generator_constant"]))
    torch.manual_seed(0)
    return PathwayModulation(config, observables, N_PERTURBATIONS,
                             graph_mask=mask, graph_summary=summary).eval()


def train_like(model: PathwayModulation) -> None:
    response = model.modulation
    with torch.no_grad():
        for layer in (response.to_u, response.to_v, response.to_q):
            layer.weight.normal_(0.0, 1.0)
        for parameter in response.operators.parameters():
            parameter.normal_(0.0, 0.3)


def kegg_graph(min_shared: int = 1) -> np.ndarray:
    return coordinate_graph(FakeKeggObservables().matrix.numpy() != 0, P, min_shared)


# ------------------------------------------------------------- the graph itself

def test_the_graph_has_exactly_the_edges_the_design_names():
    g = kegg_graph(1)
    assert np.array_equal(g, g.T)
    assert g.diagonal().all()
    assert g[0, 1] and g[1, 2] and g[3, 4]             # shared genes
    assert not g[0, 2] and not g[0, 3]                 # nothing shared
    assert g[P + 0, 0] and not g[P + 0, 1]             # gene 2 is in p0 only
    assert g[P + 1, 1] and g[P + 1, 2]                 # gene 9 is in p1 and p2
    assert not g[P + 3, :P].any()                      # gene 20 is in nothing
    anchors = g[P:, P:].copy()
    np.fill_diagonal(anchors, False)
    assert not anchors.any()                           # no anchor-anchor edge


def test_the_threshold_removes_weak_overlaps():
    g = kegg_graph(2)
    assert g[0, 1] and not g[1, 2] and g[3, 4]         # shares 2, 1, 3 genes


# ---------------------------------------------------------------- M1 .. M5

def test_m1_no_generator_entry_outside_the_mask_is_ever_nonzero():
    model = build()
    train_like(model)
    response = model.modulation
    outside = (response.graph_mask == 0)
    for pert in range(N_PERTURBATIONS):
        a = response.compose([pert])
        assert float(a[outside].abs().max()) == 0.0
        assert float(a[outside == 0].abs().max()) > 0.0


def test_m2_coordinates_the_graph_does_not_connect_never_exchange_anything():
    model = build()
    train_like(model)
    with torch.no_grad():
        flow = torch.matrix_exp(model.modulation.compose([0, 3]))[:K, :K]
    for left, right in ((FIRST, SECOND), (FIRST, ISOLATED), (SECOND, ISOLATED)):
        assert float(flow[np.ix_(left, right)].abs().max()) == 0.0
        assert float(flow[np.ix_(right, left)].abs().max()) == 0.0
    assert float(flow[np.ix_(FIRST, FIRST)].abs().max()) > 0.0


def test_m2_but_every_coordinate_can_still_be_pushed_directly():
    """The constant's column is open: where a perturbation first pushes is free."""
    model = build()
    train_like(model)
    a = model.modulation.compose([2])
    assert float(a[:K, K].abs().min()) > 0.0
    assert float(a[K, :].abs().max()) == 0.0           # and the constant stays 1


def test_m3_the_permuted_graph_has_the_same_shape_and_different_wiring():
    kegg = kegg_graph(1)
    rewired = 0
    for seed in range(10):
        shuffled = permute_within_blocks(kegg, P, seed)
        assert shuffled.sum() == kegg.sum()
        assert shuffled.diagonal().all()
        for block in (np.s_[:P, :P], np.s_[P:, :P]):
            assert sorted(shuffled[block].sum(axis=1)) == sorted(kegg[block].sum(axis=1))
        rewired += int(not np.array_equal(shuffled, kegg))
    assert rewired > 0


def test_m3_the_permuted_arm_uses_its_own_seed():
    a = build("permuted", "model.generator_mask_seed=1").modulation.graph_mask
    b = build("permuted", "model.generator_mask_seed=2").modulation.graph_mask
    c = build("permuted", "model.generator_mask_seed=1").modulation.graph_mask
    assert torch.equal(a, c)
    assert a.sum() == b.sum()


def test_m4_the_correlation_graph_has_the_kegg_edge_count_in_every_block():
    kegg = kegg_graph(1)
    phi = FakeKeggObservables()(cells(200, seed=7)).numpy()
    corr = correlation_graph(phi, kegg, P)
    assert np.array_equal(corr, corr.T)
    assert corr.diagonal().all()
    for block in (np.s_[:P, :P], np.s_[P:, :P]):
        assert corr[block].sum() == kegg[block].sum()
    anchors = corr[P:, P:].copy()
    np.fill_diagonal(anchors, False)
    assert not anchors.any()


def test_m5_no_mask_is_stage_two_to_the_bit():
    config = config_module.load(["model.response=generator"])
    observables = FakeKeggObservables()
    mask, _ = generator_mask(config, observables, None, True)
    assert torch.equal(mask, torch.ones(K + 1, K + 1))
    torch.manual_seed(0)
    plain = PathwayModulation(config, observables, N_PERTURBATIONS).eval()
    torch.manual_seed(0)
    explicit = PathwayModulation(config, observables, N_PERTURBATIONS,
                                 graph_mask=mask).eval()
    train_like(plain)
    explicit.load_state_dict(plain.state_dict())
    x = cells()
    with torch.no_grad():
        assert torch.equal(plain(x, [1, 3])["mean"], explicit(x, [1, 3])["mean"])


def test_m5_a_checkpoint_from_before_the_graph_still_loads():
    config = config_module.load(["model.response=generator"])
    torch.manual_seed(0)
    model = PathwayModulation(config, FakeKeggObservables(), N_PERTURBATIONS).eval()
    train_like(model)
    old = {k: v for k, v in model.state_dict().items() if not k.endswith("graph_mask")}
    torch.manual_seed(5)
    fresh = PathwayModulation(config, FakeKeggObservables(), N_PERTURBATIONS).eval()
    fresh.load_state_dict(old)
    x = cells()
    with torch.no_grad():
        assert torch.equal(model(x, [0, 2])["mean"], fresh(x, [0, 2])["mean"])


# ----------------------------------------------- stage 2's guarantees survive

def test_the_control_is_still_exact_and_the_mean_still_non_negative():
    model = build()
    train_like(model)
    x = cells()
    with torch.no_grad():
        control = model(x, [])["mean"]
        treated = model(x, [0, 4])["mean"]
        swapped = model(x, [4, 0])["mean"]
    assert torch.allclose(control, x + model.modulation.turn_on_offset(), atol=1e-6)
    assert float(treated.min()) >= 0.0
    assert torch.equal(treated, swapped)


# ------------------------------------------------------------------- guards

def test_a_mask_on_whitened_coordinates_is_refused():
    config = config_for("kegg", "model.observable_whiten=zca")
    with pytest.raises(ValueError, match="observable_whiten=none"):
        generator_mask(config, FakeKeggObservables(), None, True)


def test_a_mask_on_non_kegg_observables_is_refused():
    config = config_for("kegg", "model.observables=random")
    with pytest.raises(ValueError, match="observables=kegg"):
        generator_mask(config, FakeKeggObservables(), None, True)


def test_asking_for_a_graph_without_building_one_is_refused():
    config = config_for("kegg")
    with pytest.raises(ValueError, match="no graph was built"):
        PathwayModulation(config, FakeKeggObservables(), N_PERTURBATIONS)


def test_the_mlp_cannot_be_given_a_graph():
    config = config_module.load(["model.generator_mask=kegg"])
    with pytest.raises(ValueError, match="restricts generators"):
        PathwayModulation(config, FakeKeggObservables(), N_PERTURBATIONS)
