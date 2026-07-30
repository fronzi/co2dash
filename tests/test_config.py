"""Piece 1 tests: YAML loader + provenance registry."""
import os
import numpy as np
from co2dash import load_scenario, DataTier, RXN_METHANOL, propagate_mc

YAML = os.path.join(os.path.dirname(__file__), "..", "examples", "scenario_methanol.yaml")


def test_loader_builds_scenario_with_reaction_stoichiometry():
    sc, reg = load_scenario(YAML)
    # reaction stoichiometry was injected, not read from YAML
    assert sc.n_electrons == RXN_METHANOL.n_electrons
    assert abs(sc.m_co2 - RXN_METHANOL.kg_co2_per_kg_prod) < 1e-12
    # a scalar field round-trips
    assert sc.disc_rate == 0.08
    # the scenario evaluates without error
    out = sc.evaluate()
    assert np.isfinite(out["lcop_usd_per_kg"])


def test_provenance_registry_records_tiers():
    _, reg = load_scenario(YAML)
    assert reg.entries["faradaic_efficiency"].tier == DataTier.LAB_VALIDATED
    assert reg.entries["capex_total"].tier == DataTier.ESTIMATED
    # effective_std respects the tier floor (>0 even if YAML std small)
    assert reg.entries["grid_intensity"].effective_std() > 0


def test_mc_distributions_auto_derived_from_tiers():
    sc, reg = load_scenario(YAML)
    fields = ["faradaic_efficiency", "capex_total", "grid_intensity", "disc_rate"]
    dists = reg.mc_distributions(fields)
    # scalar-only field is excluded
    assert "disc_rate" not in dists
    # ESTIMATED positive cost -> lognormal; LAB_VALIDATED -> normal
    assert dists["capex_total"][0] == "lognormal"
    assert dists["faradaic_efficiency"][0] == "normal"
    # the derived distributions actually drive a propagation
    res = propagate_mc(sc, dists, carbon_price_usd_per_kg=2.0, n=3000, seed=1)
    assert 0.0 <= res["p_net_positive"] <= 1.0
