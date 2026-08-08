"""Tests for the recommendation synthesis."""
import dataclasses
import math

import pytest

from co2dash import load_scenario, recommend
from conftest import SCENARIO_CO, example

YAML = example(SCENARIO_CO)


def _cases():
    import dataclasses as dc
    base, reg = load_scenario(YAML)
    return reg, {
        "baseline": base,
        "great_fe": dc.replace(base, faradaic_efficiency=0.95),
        "cheap_power": dc.replace(base, c_elec=0.01),
        "capex_x4": dc.replace(base, capex_total=base.capex_total * 4),
        "low_voltage": dc.replace(base, cell_voltage=1.8),
    }


def test_the_verdict_names_its_subject():
    """The app renders a verdict in three places — the headline strip, the
    recommendation panel, and one per uploaded row. Without a subject they are
    word-for-word identical while describing different scenarios."""
    base, reg = load_scenario(YAML)
    plain = recommend(base, 0.30, registry=reg, n_mc=5000)
    named = recommend(base, 0.30, registry=reg, n_mc=5000, subject="Ag-MEA (your row 0)")
    assert plain.steps[0].startswith("Verdict:")
    assert named.steps[0].startswith("Verdict for Ag-MEA (your row 0):")
    assert named.subject == "Ag-MEA (your row 0)"


def test_subject_appears_even_when_not_climate_positive():
    import dataclasses as dc
    base, reg = load_scenario(YAML)
    dirty = dc.replace(base, grid_intensity=0.9)
    r = recommend(dirty, 0.30, registry=reg, n_mc=5000, subject="the sidebar sliders")
    assert r.verdict == "Not climate-positive"
    assert "for the sidebar sliders" in r.steps[0]


def test_recommendation_text_actually_varies_with_the_scenario():
    """Regression: every bullet used to be identical across scenarios except the
    first two -- the 'next candidate' line was static and the closing note
    hard-coded 'uncertainty is dominated by CAPEX' regardless of the analysis."""
    reg, cases = _cases()
    recs = {k: recommend(sc, 0.30, registry=reg, n_mc=6000)
            for k, sc in cases.items()}
    n = min(len(r.steps) for r in recs.values())
    identical = [i for i in range(n)
                 if len({r.steps[i] for r in recs.values()}) == 1]
    assert not identical, f"bullet(s) {[i + 1 for i in identical]} never change"


def test_closing_note_is_derived_not_asserted():
    reg, cases = _cases()
    for sc in cases.values():
        r = recommend(sc, 0.30, registry=reg, n_mc=6000)
        last = r.steps[-1]
        if r.top_uncertainty is None:
            assert "no claim is made" in last
        else:
            # it must name the input the analysis actually found, using the
            # module's own label map -- so a newly uncertain input cannot be
            # reported under a raw field name without this test noticing
            from co2dash.recommend import _LABEL
            assert r.top_uncertainty in _LABEL, (
                f"{r.top_uncertainty!r} has no human-readable label")
            label = _LABEL[r.top_uncertainty]
            assert label[1:] in last          # skip the first char: it may be capitalised


def test_sobol_ranges_come_from_the_mc_distributions():
    """Regression: hand-written ranges gave grid_intensity a span 3x wider than
    every other input, making it 'dominant' almost by construction."""
    from co2dash.recommend import _bounds_from_distributions
    base, _ = load_scenario(YAML)
    dists = {"faradaic_efficiency": ("normal", 0.6, 0.05),
             "cell_voltage": ("normal", 3.0, 0.3),
             "capex_total": ("lognormal", 5e7, 1.4),
             "grid_intensity": ("uniform", 0.02, 0.08)}
    b = _bounds_from_distributions(dists, base)
    assert b["grid_intensity"] == (0.02, 0.08)                  # uniform passes through
    assert b["faradaic_efficiency"][0] > 0 and b["faradaic_efficiency"][1] <= 1.0
    assert b["capex_total"][0] > 0 and b["capex_total"][1] > b["capex_total"][0]
    # symmetric normal -> range centred on the mean
    lo, hi = b["cell_voltage"]
    assert (lo + hi) / 2 == pytest.approx(3.0, abs=1e-9)


def test_degenerate_distribution_is_dropped_not_inverted():
    from co2dash.recommend import _bounds_from_distributions
    base, _ = load_scenario(YAML)
    b = _bounds_from_distributions({"grid_intensity": ("uniform", 0.05, 0.05)}, base)
    assert "grid_intensity" not in b


def test_controllable_lever_and_uncertainty_driver_are_reported_separately():
    reg, cases = _cases()
    r = recommend(cases["baseline"], 0.30, registry=reg, n_mc=6000)
    from co2dash.recommend import _LEVERS
    if r.dominant_lever is not None:
        assert r.dominant_lever in _LEVERS      # must be something you can change


def test_recommend_renewable_is_actionable():
    base, reg = load_scenario(YAML)                  # renewable grid
    r = recommend(base, carbon_price_usd_per_kg=0.60, registry=reg, n_mc=15000)
    assert r.verdict in ("Feasible", "Marginal")
    assert math.isfinite(r.mac_median)
    assert r.p_net_positive > 0.5
    assert r.dominant_lever is not None
    assert len(r.steps) >= 3


def test_recommend_dirty_grid_flags_electricity():
    base, reg = load_scenario(YAML)
    dirty = dataclasses.replace(base, grid_intensity=0.66)   # NSW grid
    r = recommend(dirty, carbon_price_usd_per_kg=0.60, registry=reg, n_mc=15000)
    assert r.verdict == "Not climate-positive"
    assert r.grid_ok is False
    # the guidance must lead with the climate problem, not catalyst tweaks
    assert any("climate-positive" in s or "grid" in s.lower() for s in r.steps)


def test_recommend_next_candidate_included():
    base, reg = load_scenario(YAML)
    r = recommend(base, 0.60, registry=reg, next_candidate="mat_007", n_mc=8000)
    assert r.next_candidate == "mat_007"
    assert any("mat_007" in s for s in r.steps)
