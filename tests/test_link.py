"""Tests for the FE<->descriptor link module."""
import numpy as np
import pytest
from co2dash.link import (canonical_material, availability_report,
                          descriptor_request_list, link_fe_to_descriptors)


def test_canonical_material():
    assert canonical_material("Cu") == ("Cu", "public")
    assert canonical_material("Cu/C") == ("Cu", "public")
    assert canonical_material("Cu-M")[1] == "bespoke"
    assert canonical_material("CuOx")[1] == "bespoke"
    assert canonical_material("Cu-MOF") == ("Cu-MOF", "none")


def test_availability_and_request_list():
    rows = ([{"material": "Cu", "faradaic_efficiency": 0.9}] * 3
            + [{"material": "Cu-M", "faradaic_efficiency": 0.5}] * 2
            + [{"material": "Cu-MOF", "faradaic_efficiency": 0.4}])
    rep = availability_report(rows)
    assert rep["n"] == 6
    assert rep["tiers"]["public"] == 3
    assert rep["public_frac"] == pytest.approx(0.5)
    reqs = descriptor_request_list(rows)
    assert reqs[0]["surface"] == "Cu" and reqs[0]["fe_records"] == 3


def test_link_joins_and_reports_unmatched():
    rows = [{"material": "Cu", "faradaic_efficiency": 0.9},
            {"material": "Cu", "faradaic_efficiency": 0.8},
            {"material": "Cu-MOF", "faradaic_efficiency": 0.4}]   # no descriptor -> unmatched
    table = {"Cu": {"dE_CO": -0.5, "dE_H": -0.3}}
    X, y, rep = link_fe_to_descriptors(rows, table, keys=["dE_CO", "dE_H"])
    assert X.shape == (2, 2) and y.shape == (2,)
    assert rep["matched"] == 2 and rep["unmatched"] == 1
    assert rep["unmatched_by_key"].get("Cu-MOF") == 1


def test_oxide_canonicalisation_and_aggregation():
    from co2dash.link import canonical_material, descriptors_to_canonical
    assert canonical_material("Cu2O")[0] == "CuOx"
    assert canonical_material("CuO")[0] == "CuOx"
    assert canonical_material("Cu(111)")[0] == "Cu"
    surf = {"Cu(111)": {"dE_CO": -0.59}, "Cu(100)": {"dE_CO": -0.62},
            "Cu2O": {"dE_CO": -0.20}}
    agg = descriptors_to_canonical(surf, reduce="min")
    assert set(agg) == {"Cu", "CuOx"}
    assert agg["Cu"]["dE_CO"] == -0.62      # strongest binding over Cu facets
    assert agg["CuOx"]["dE_CO"] == -0.20
