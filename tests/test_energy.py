"""Tests for the energy/grid region module (offline; no live calls)."""
import math
import pytest
from co2dash import (load_scenario, COUNTRY_PROFILES, list_regions,
                     get_energy, apply_to_scenario)
from co2dash.schema import DataTier
from conftest import SCENARIO_CO, example

YAML = example(SCENARIO_CO)


def test_profiles_present_and_sourced():
    # key regions for the Australia-Japan context exist and carry sources
    for code in ("AU", "JP", "US", "EU", "RENEWABLE"):
        assert code in COUNTRY_PROFILES
        q = COUNTRY_PROFILES[code].grid_intensity
        assert q.source and 0.0 < q.value < 1.5      # plausible kgCO2/kWh
        assert q.tier in (DataTier.LIT_EXTRACTED, DataTier.ESTIMATED)


def test_known_values():
    # values match the sourced 2024 figures used to build the table
    assert COUNTRY_PROFILES["JP"].grid_intensity.value == pytest.approx(0.48)
    assert COUNTRY_PROFILES["US"].grid_intensity.value == pytest.approx(0.384)
    assert COUNTRY_PROFILES["RENEWABLE"].grid_intensity.value < 0.1


def test_list_regions_roundtrip():
    regions = list_regions()
    assert regions["AU"].startswith("Australia")
    assert set(regions) == set(COUNTRY_PROFILES)


def test_get_energy_static_no_network():
    e = get_energy("JP")                      # live=False default -> no network
    assert e["live"] is False
    assert e["grid_intensity"].value == pytest.approx(0.48)
    assert e["electricity_price"] is None     # prices left unset (not fabricated)


def test_get_energy_unknown_code_raises():
    with pytest.raises(KeyError):
        get_energy("ATLANTIS")


def test_apply_to_scenario_sets_grid():
    base, _ = load_scenario(YAML)
    jp = apply_to_scenario(base, "JP")
    assert jp.grid_intensity == pytest.approx(0.48)
    assert base.grid_intensity != jp.grid_intensity        # base untouched (copy)
    # on a high-carbon grid the route stops being climate-positive
    assert jp.evaluate()["net_abatement_kg_per_kg"] < 0
    assert math.isinf(jp.evaluate()["mac_usd_per_tonne_co2"])


def test_apply_renewable_is_climate_positive():
    base, _ = load_scenario(YAML)
    ren = apply_to_scenario(base, "RENEWABLE")
    ev = ren.evaluate()
    assert ev["net_abatement_kg_per_kg"] > 0
    assert math.isfinite(ev["mac_usd_per_tonne_co2"])
