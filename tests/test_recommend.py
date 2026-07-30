"""Tests for the recommendation synthesis."""
import dataclasses
import math
from co2dash import load_scenario, recommend

YAML = "examples/scenario_co_real.yaml"


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
