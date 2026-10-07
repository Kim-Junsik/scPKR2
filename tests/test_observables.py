"""The observable map's own guarantees. Carried over from v2 unchanged.

    python -m pytest tests/test_observables.py -v

Phi is the one part of v2 this repository did not rebuild, so its tests come across as
they were. The whitening result they protect is v2's only improvement that replicated
across three observable families, and it is the reason the KEGG dictionary is usable at
all: pathways share genes heavily, and unwhitened coordinates reach 0.117 off-diagonal
correlation on synthetic data alone.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config as config_module
from src.models.observables import Observables


def _observables(whiten: str, n_genes: int = 60, n_cells: int = 400, seed: int = 0):
    from src.models.observables import Observables

    rng = np.random.default_rng(seed)
    x = rng.gamma(2.0, 0.6, size=(n_cells, n_genes)).astype(np.float32)
    names = np.array([f"g{i}" for i in range(n_genes)])
    config = config_module.load([f"model.observable_whiten={whiten}",
                                 "model.observables=random",
                                 "model.anchor_genes=variance", "model.n_anchor_genes=8"])
    return Observables(config, names, x, np.arange(n_cells), ["g0", "g1"]), x


def test_whitening_decorrelates_the_coordinates():
    """zca must make the coordinate correlation the identity on the rows it was fitted on.

    Measured here rather than asserted: the unwhitened coordinates of a random dictionary
    already reach 0.117 off-diagonal correlation on synthetic data, and real KEGG pathways
    share genes far more heavily than that.
    """
    for whiten, bound in (("none", None), ("zca", 1e-4)):
        observables, x = _observables(whiten)
        p = observables(torch.from_numpy(x)).detach().numpy()
        correlation = np.corrcoef(p.T)
        off = np.abs(correlation - np.eye(len(correlation))).max()
        if bound is None:
            assert off > 1e-3, "the unwhitened case must not already be decorrelated"
        else:
            assert off < bound, f"zca left {off:.2e} of off-diagonal correlation"


def test_whitening_leaves_the_span_untouched():
    """The row space of the effective map must be identical, or this tests the wrong thing.

    Whitening is only defensible as a test of the BASIS if it cannot change which subspace
    of gene space the observables can see. The effective map is W diag(1/std) M, and
    multiplying on the left by an invertible W cannot change its row space - asserted
    numerically, because an eigenvalue floor is applied to the inverse square root and a
    floor that bites would silently drop a direction.
    """
    plain, _ = _observables("none")
    white, _ = _observables("zca")
    rows = []
    for observables in (plain, white):
        effective = (observables.whitener
                     @ torch.diag(1.0 / observables.std)
                     @ observables.matrix).detach().numpy()
        rows.append(effective)
    a, b = rows
    assert a.shape == b.shape
    # Project each row of b onto the row space of a; nothing may be left over.
    basis = np.linalg.svd(a, full_matrices=False)[2]
    residual = b - (b @ basis.T) @ basis
    assert np.linalg.norm(residual) / np.linalg.norm(b) < 1e-5
    assert np.linalg.matrix_rank(a, tol=1e-5) == np.linalg.matrix_rank(b, tol=1e-5)


def test_a_checkpoint_written_before_whitening_still_loads():
    """78 runs were trained before `whitener` existed and must stay scoreable.

    A new buffer makes load_state_dict fail under its default strict=True, which turned
    every finished run unscoreable the moment the buffer landed - discovered when a sweep
    over four experiments died on the first checkpoint. The identity is not a fallback but
    the exact model those runs had, so this asserts the loaded module reproduces the saved
    one rather than merely surviving the load.
    """
    observables, x = _observables("none")
    saved = {k: v for k, v in observables.state_dict().items() if k != "whitener"}
    assert "whitener" not in saved

    fresh, _ = _observables("none")
    with torch.no_grad():
        fresh.whitener.normal_()          # make the identity impossible to get by accident
    fresh.load_state_dict(saved)
    assert torch.allclose(fresh.whitener, torch.eye(fresh.matrix.shape[0]), atol=0)
    cells = torch.from_numpy(x)
    assert torch.allclose(fresh(cells), observables(cells), atol=1e-6)
