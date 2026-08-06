"""Tests for the TEA validation anchor (Jouny, Luc & Jiao 2018)."""
import pytest
from co2dash.validation import (JOUNY_BASE, JOUNY_OPT, PRODUCTS, validate_energy,
                                ref_co2_cost, ref_electricity_cost,
                                ref_electrolyzer_area_and_capex, energetic_efficiency)


def test_energy_engine_reproduces_reference_exactly():
    # the co2dash energy term must equal the independent first-principles balance
    for case in (JOUNY_BASE, JOUNY_OPT):
        for r in validate_energy(case):
            assert r["rel_err"] < 1e-9


def test_specific_energy_values_are_physical():
    # CO at V=2.3, FE=0.9 -> ~4.9 kWh/kg (well-known order of magnitude)
    rows = {r["product"]: r for r in validate_energy(JOUNY_BASE)}
    assert rows["co"]["E_ref_kWh_kg"] == pytest.approx(4.89, abs=0.05)
    # higher-electron products cost far more energy per kg
    assert rows["methanol"]["E_ref_kWh_kg"] > 2 * rows["co"]["E_ref_kWh_kg"]


def test_co2_feedstock_cost_stoichiometric():
    p = PRODUCTS["co"]
    # 44.01/28.01 * $0.10/kg = 0.157 $/kg CO
    assert ref_co2_cost(p.molar_mass, p.co2_per_prod_mol, 100.0) == pytest.approx(0.157, abs=0.005)


def test_economic_ordering_matches_paper():
    # Jouny: CO & formic acid favourable; methanol/ethylene electricity-dominated
    def elec(k):
        p = PRODUCTS[k]
        return ref_electricity_cost(p.n, p.molar_mass, JOUNY_BASE["cell_voltage"],
                                    JOUNY_BASE["faradaic"], JOUNY_BASE["elec_price"])
    # methanol electricity alone exceeds its market price at base case
    assert elec("methanol") > PRODUCTS["methanol"].market_price
    # CO electricity is a small fraction of its market price
    assert elec("co") < 0.5 * PRODUCTS["co"].market_price


def test_electrolyzer_area_and_capex_reasonable():
    p = PRODUCTS["co"]
    area, capex = ref_electrolyzer_area_and_capex(
        JOUNY_BASE["prod_ton_day"], p.n, p.molar_mass, JOUNY_BASE["faradaic"],
        JOUNY_BASE["current_density_mA"], JOUNY_BASE["electrolyzer_cost_per_m2"])
    assert area == pytest.approx(4430, rel=0.02)
    assert 8e6 < capex < 2e7


def test_energetic_efficiency_below_60pct():
    p = PRODUCTS["co"]
    eff = energetic_efficiency(p.n, p.molar_mass, JOUNY_BASE["cell_voltage"],
                               JOUNY_BASE["faradaic"], 1.34)
    assert 0.45 < eff < 0.60


# --- second anchor: Osorio-Tejada et al. 2024 (EES) --------------------------
def test_anchor_osorio_physics_within_tolerance():
    from co2dash.validation import validate_anchor_osorio
    r = validate_anchor_osorio()
    # specific energy (cell and 80% plug-to-power) and electrode area: < 2%
    for k in ("cell_kwh_kg", "total_kwh_kg", "cell_area_m2"):
        got, ref = r[k]
        assert abs(got - ref) / ref < 0.02
    # variable OPEX components (electricity, feedstock): < 5%
    for k in ("elec_per_t", "feedstock_per_t"):
        got, ref = r[k]
        assert abs(got - ref) / ref < 0.05


def test_anchor_osorio_lcop_in_reported_range():
    from co2dash import load_scenario
    from co2dash.validation import co2dash_lcop_in_reported_range
    from conftest import SCENARIO_CO, example
    base, _ = load_scenario(example(SCENARIO_CO))
    chk = co2dash_lcop_in_reported_range(base)
    assert chk["in_range"]           # co2dash CO LCOP within paper's $570-1392/t band


def test_lcop_brackets_literature_band():
    from co2dash.validation import validate_lcop_band
    b = validate_lcop_band()
    # co2dash must span the published CO LCOP band as assumptions move
    assert b["brackets_literature"]
    assert b["co2dash_favourable"] < b["co2dash_conservative"]
    # conservative endpoint should be a plausibly high (but finite) $/kg
    assert 0.8 < b["co2dash_conservative"] < 2.0


def test_lcop_favourable_is_competitive():
    from co2dash.validation import co2dash_lcop_co, FAVOURABLE_CO, PRODUCTS
    lcop = co2dash_lcop_co(**FAVOURABLE_CO)["lcop"]
    # under favourable assumptions CO LCOP should rival the conventional price
    assert lcop < PRODUCTS["co"].market_price
