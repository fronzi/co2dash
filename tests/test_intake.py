"""Tests for the user data-intake layer."""
import math
import pytest
from co2dash import map_columns, row_to_scenario, read_csv, ingest_table
from co2dash.intake import normalise_units, resolve_reaction

_CSV = "material,product,FE (%),cell voltage\nAg-MEA,CO,93,2.2\n"


def _loaded_scenario():
    """The favourable scenario, chosen because the base one happens to carry the
    same values as GENERIC_DEFAULTS — so it could not distinguish 'filled from
    the YAML' from 'filled from the defaults'."""
    from conftest import SCENARIO_CO_FAVOURABLE, example
    from co2dash import load_scenario
    return load_scenario(example(SCENARIO_CO_FAVOURABLE))[0]


def test_loaded_scenario_fills_unmeasured_fields_instead_of_generic_defaults():
    """Regression: the 'Your data' tab evaluated rows against generic defaults
    even when a YAML scenario was loaded, so the plant and grid behind each row
    differed from the ones driving the verdict on screen — silently."""
    base = _loaded_scenario()
    plain = ingest_table(_CSV, "co")[0].scenario
    linked = ingest_table(_CSV, "co", base=base)[0].scenario

    assert linked.grid_intensity == pytest.approx(base.grid_intensity)
    assert linked.release_fraction == pytest.approx(base.release_fraction)
    assert linked.annual_production_kg == pytest.approx(base.annual_production_kg)
    assert plain.grid_intensity != pytest.approx(base.grid_intensity)


def test_measured_columns_still_win_over_the_loaded_scenario():
    base = _loaded_scenario()
    r = row_to_scenario({"faradaic_efficiency": 93.0, "cell_voltage": 2.2,
                         "product": "co"}, "co", base=base)
    assert r.scenario.faradaic_efficiency == pytest.approx(0.93)
    assert r.scenario.cell_voltage == pytest.approx(2.2)
    assert r.provenance["faradaic_efficiency"] == "user"


def test_provenance_names_the_scenario_as_the_source():
    base = _loaded_scenario()
    r = ingest_table(_CSV, "co", base=base)[0]
    assert r.provenance["grid_intensity"].startswith("scenario:")
    assert r.provenance["cell_voltage"] == "user"
    r2 = ingest_table(_CSV, "co")[0]
    assert r2.provenance["grid_intensity"].startswith("default:")


def test_csv_columns_that_contradict_the_scenario_are_reported():
    """A CSV column beating a declared scenario is fine for the catalyst's own
    KPIs, but electricity price and grid intensity describe the plant and the
    site. Overriding those silently yields a hybrid matching nothing."""
    base = _loaded_scenario()          # c_elec 0.03, grid 0.02
    csv = ("material,product,FE (%),cell voltage,electricity price,grid intensity\n"
           "Ag-foam,CO,92,3.2,0.04,0.05\n")
    r = ingest_table(csv, "co", base=base)[0]
    joined = " | ".join(r.warnings)
    assert "c_elec" in joined and "context, not a catalyst property" in joined
    assert "grid_intensity" in joined
    # a genuine measurement is flagged too, but labelled differently
    assert "cell_voltage" in joined and "— measurement" in joined


def test_no_clash_warning_when_the_csv_agrees_with_the_scenario():
    base = _loaded_scenario()
    csv = (f"material,product,FE (%),cell voltage,grid intensity\n"
           f"X,CO,95,2.2,{base.grid_intensity}\n")
    r = ingest_table(csv, "co", base=base)[0]
    assert not any("overrides the loaded scenario" in w for w in r.warnings)


def test_no_clash_warning_without_a_loaded_scenario():
    csv = "material,product,FE (%),cell voltage,grid intensity\nX,CO,92,3.2,0.05\n"
    r = ingest_table(csv, "co")[0]
    assert not any("overrides" in w for w in r.warnings)


def test_current_density_now_drives_the_capital():
    """Regression: current_density was parsed, range-checked and then dropped —
    Scenario had no such field. It decides how much electrode you must buy, so a
    fast cell and a slow one cannot cost the same."""
    csv = ("material,product,FE (%),cell voltage,current density\n"
           "fast,CO,90,3.0,600\nslow,CO,90,3.0,100\n")
    fast, slow = ingest_table(csv, "co")
    assert fast.scenario.capex_total < slow.scenario.capex_total
    # area, and therefore capital, scales as 1/j
    assert slow.scenario.capex_total / fast.scenario.capex_total == pytest.approx(6.0, rel=0.02)
    assert fast.provenance["capex_total"].startswith("from current density")
    assert "electrode_area_m2" in fast.provenance


def test_area_method_matches_the_validated_jouny_reference():
    from co2dash.techno_economic import electrode_area_m2
    from co2dash.validation import PRODUCTS, ref_electrolyzer_area_and_capex
    p = PRODUCTS["co"]
    ref, _ = ref_electrolyzer_area_and_capex(100, p.n, p.molar_mass, 0.90, 200, 2830.0)
    got = electrode_area_m2(100 * 1000 * 365, p.n, p.molar_mass, 0.90, 200,
                            operating_days_per_year=365)
    assert got == pytest.approx(ref, rel=1e-12)


def test_user_capex_is_kept_but_a_disagreement_is_reported():
    csv = ("material,product,FE (%),cell voltage,current density,capex\n"
           "X,CO,90,3.0,600,90000000\n")
    r = ingest_table(csv, "co")[0]
    assert r.scenario.capex_total == pytest.approx(9.0e7)      # theirs wins
    assert any("disagree" in w and "area method" in w for w in r.warnings)


def test_consistent_capex_and_current_density_raise_no_complaint():
    from co2dash.techno_economic import capex_from_current_density
    from co2dash.schema import RXN_CO
    cap = capex_from_current_density(2.0e7, RXN_CO.n_electrons,
                                     RXN_CO.molar_mass_prod, 0.90, 300)
    csv = ("material,product,FE (%),cell voltage,current density,capex\n"
           f"X,CO,90,3.0,300,{cap['total_capex_usd']:.0f}\n")
    r = ingest_table(csv, "co")[0]
    assert not any("disagree" in w for w in r.warnings)


def test_without_current_density_the_capital_is_untouched():
    csv = "material,product,FE (%),cell voltage\nX,CO,90,3.0\n"
    r = ingest_table(csv, "co")[0]
    assert r.provenance["capex_total"].startswith("default:")


def test_behaviour_without_a_base_is_unchanged():
    a = ingest_table(_CSV, "co")[0].scenario
    b = ingest_table(_CSV, "co", base=None)[0].scenario
    assert a == b


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
