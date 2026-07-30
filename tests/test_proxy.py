"""Tests for the CHE physics activity proxy.

Energies below are arbitrary numbers chosen to exercise the equations -- NOT
chemistry data. They validate the limiting-potential logic, not any catalyst.
"""
import math
from co2dash.proxy import (limiting_potential, build_activity_targets,
                           proxy_cell_voltage, PATHWAYS)


def test_limiting_potential_picks_potential_determining_step():
    # CO pathway steps: CO2->COOH (dG=0.6), COOH->CO (dG = 0.2-0.6 = -0.4)
    e = {"COOH": 0.6, "CO": 0.2}
    lp = limiting_potential(e, "CO")
    assert lp["pds"] == "CO2->COOH"          # most endergonic step
    assert abs(lp["U_L"] - (-0.6)) < 1e-9    # U_L = -max(dG) = -0.6
    # overpotential = U_eq - U_L = -0.10 - (-0.6) = 0.50
    assert abs(lp["overpotential"] - 0.50) < 1e-9


def test_missing_intermediate_returns_none():
    assert limiting_potential({"COOH": 0.6}, "CO") is None   # *CO missing


def test_stronger_binding_changes_pds_and_lowers_overpotential():
    weak = limiting_potential({"COOH": 0.8, "CO": 0.3}, "CO")
    strong = limiting_potential({"COOH": 0.3, "CO": 0.1}, "CO")
    assert strong["overpotential"] < weak["overpotential"]   # more active


def test_proxy_cell_voltage_monotonic_in_overpotential():
    assert proxy_cell_voltage(0.2) < proxy_cell_voltage(0.6)
    assert proxy_cell_voltage(0.0) >= 0.2                    # floored positive


def test_build_activity_targets_skips_incomplete_and_maps_voltage():
    table = [
        {"surface": "Cu", "facet": "111", "dE_CO": 0.2, "dE_COOH": 0.6},
        {"surface": "Ag", "facet": "100", "dE_CO": None, "dE_COOH": 0.5},  # incomplete
    ]
    rows = build_activity_targets(table, product="CO", descriptor_keys=["dE_CO", "dE_COOH"])
    assert len(rows) == 1                       # Ag skipped
    r = rows[0]
    assert "Cu" in r["material_id"]
    assert set(r["descriptors"]) == {"dE_CO", "dE_COOH"}
    assert r["v_cell"] == proxy_cell_voltage(r["overpotential"])
    assert math.isfinite(r["v_cell"])


def test_known_products_have_pathways():
    for p in ("CO", "HCOOH", "CH3OH"):
        assert p in PATHWAYS and len(PATHWAYS[p]) >= 1
