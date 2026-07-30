"""Tests for the user data-intake layer."""
import math
import pytest
from co2dash import map_columns, row_to_scenario, read_csv, ingest_table
from co2dash.intake import normalise_units, resolve_reaction


def test_map_columns_aliases_and_unknown():
    m, unknown = map_columns(["Catalyst", "FE (%)", "Cell Voltage", "notes"])
    assert m["Catalyst"] == "material_id"
    assert m["FE (%)"] == "faradaic_efficiency"
    assert m["Cell Voltage"] == "cell_voltage"
    assert "notes" in unknown


def test_unit_normalisation():
    assert normalise_units("faradaic_efficiency", 92.0, "92")[0] == pytest.approx(0.92)
    assert normalise_units("cell_voltage", 3200.0, "3200")[0] == pytest.approx(3.2)
    assert normalise_units("grid_intensity", 620.0, "620")[0] == pytest.approx(0.62)
    # already-canonical values are untouched
    assert normalise_units("faradaic_efficiency", 0.9, "0.9")[0] == pytest.approx(0.9)


def test_resolve_reaction():
    assert resolve_reaction({"product": "CO"}) == "co"
    assert resolve_reaction({"product": "methanol"}) == "methanol"
    assert resolve_reaction({"product": "formic acid"}) == "formate"
    assert resolve_reaction({}, default_key="co") == "co"


def test_row_to_scenario_fills_defaults_with_provenance():
    res = row_to_scenario({"faradaic_efficiency": "0.9", "cell_voltage": "3.2",
                           "product": "co"})
    assert res.ok
    assert res.provenance["faradaic_efficiency"] == "user"
    # unmeasured economics come from sourced defaults, flagged as such
    assert res.provenance["c_co2"].startswith("default:")
    assert res.provenance["capex_total"].startswith("default:")
    assert res.scenario.faradaic_efficiency == pytest.approx(0.9)


def test_row_to_scenario_flags_bad_fe():
    res = row_to_scenario({"faradaic_efficiency": "150", "product": "co"})
    assert not res.ok
    assert any("faradaic_efficiency" in e for e in res.errors)


def test_row_to_scenario_missing_fe_warns_not_errors():
    res = row_to_scenario({"cell_voltage": "3.2", "product": "co"})
    assert res.ok                                  # usable (default FE) but flagged
    assert any("faradaic_efficiency" in w for w in res.warnings)


def test_read_csv_and_ingest_table():
    csv_text = ("catalyst,product,FE (%),Cell Voltage\n"
                "Ag-foam,CO,92,3.2\nAg-NP,CO,88,3.1\n")
    rows = read_csv(csv_text)
    assert len(rows) == 2 and rows[0]["faradaic_efficiency"] == "92"
    results = ingest_table(csv_text)
    assert len(results) == 2 and all(r.ok for r in results)
    assert results[0].scenario.faradaic_efficiency == pytest.approx(0.92)
