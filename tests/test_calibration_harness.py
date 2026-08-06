"""Tests for the calibration harness (procedure validation on synthetic data)."""
import numpy as np
import pytest
from co2dash.calibration_harness import (make_linear_synthetic, ConstStdSurrogate,
                                         calibrate_and_evaluate, split_indices,
                                         join_labeled)

SIGMA = 0.05


def _run(reported_std, seed=1):
    X, y, _ = make_linear_synthetic(n=600, d=4, sigma_true=SIGMA, seed=seed)
    fac = lambda Xt, yt: ConstStdSurrogate(reported_std).fit(Xt, yt)
    return calibrate_and_evaluate(X, y, surrogate_factory=fac, alpha=0.1, seed=seed)


def test_splits_are_disjoint_and_complete():
    itr, ical, ite = split_indices(100, 0.5, 0.25, seed=0)
    all_idx = np.concatenate([itr, ical, ite])
    assert len(np.unique(all_idx)) == 100
    assert len(itr) == 50 and len(ical) == 25 and len(ite) == 25


def test_overconfident_is_corrected():
    rep = _run(SIGMA / 1.8)
    assert rep.miscal_before > 0.10            # raw model is badly miscalibrated
    assert rep.miscal_after < 0.05             # calibration fixes it
    assert rep.temperature_s == pytest.approx(1.8, abs=0.25)   # recovers noise scale
    assert 0.84 <= rep.conformal_coverage <= 0.96              # ~90% conformal coverage


def test_underconfident_is_corrected():
    rep = _run(SIGMA * 1.8)
    assert rep.miscal_after < 0.05
    assert rep.temperature_s == pytest.approx(1 / 1.8, abs=0.2)  # shrinks the std
    assert rep.improved


def test_well_specified_left_alone():
    rep = _run(SIGMA)
    assert rep.temperature_s == pytest.approx(1.0, abs=0.15)
    assert rep.miscal_after < 0.06


def test_join_labeled_drops_nonoverlapping():
    descriptors = {"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]}
    targets = {"a": 0.9, "b": 0.8}                 # 'c' has no target
    X, y, ids = join_labeled(descriptors, targets, keys=[0, 1])
    assert set(ids) == {"a", "b"}
    assert X.shape == (2, 2) and y.shape == (2,)


def test_join_labeled_raises_on_no_overlap():
    with pytest.raises(ValueError):
        join_labeled({"a": [1.0]}, {"z": 0.5}, keys=[0])


def test_literature_seed_dataset_wellformed():
    """Guard the shipped real CO2->CO literature seed (Osorio-Tejada 2024 Table 2)."""
    from co2dash.intake import read_csv
    from conftest import LITERATURE_CO_CSV, example
    path = example(LITERATURE_CO_CSV)
    with open(path) as fh:
        rows = read_csv(fh.read())
    assert len(rows) == 7                                  # 7 real cited studies
    fes = [float(r["faradaic_efficiency"]) for r in rows]
    assert all(50 <= f <= 100 for f in fes)                # stored as percent, plausible
    # case-insensitive: row_to_scenario resolves the product via REACTION_ALIASES,
    # which lowercases, so 'CO' and 'co' are equivalent inputs
    assert all(str(r["product"]).strip().lower() == "co" for r in rows)
