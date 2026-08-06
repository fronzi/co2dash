"""Tests for the HEA multi-sheet DFT workbook loader and the per-configuration
intermediate join.

The workbook here is SYNTHETIC: it reproduces the published file's column
layout and element encoding, but the energies are arbitrary numbers chosen to
exercise the join and the CHE hand-off. They are not chemistry data, and no test
depends on the proprietary supplementary file.
"""
import io

import numpy as np
import pandas as pd
import pytest

from co2dash.hea import (ELEMENT_BY_DESCRIPTOR, SPECIES_COMPOSITION, JoinReport,
                         assert_no_leakage, che_reference_shift,
                         check_energy_reference, convert_rows_to_che,
                         decode_site, join_intermediates, load_sheet,
                         load_workbook, pathway_coverage,
                         to_activity_table, to_che_formation_energies)
from co2dash.proxy import (PATHWAYS, build_activity_targets, limiting_potential,
                           proxy_cell_voltage)

N_SITES = 3
ELEMENTS = {sym: desc for desc, sym in ELEMENT_BY_DESCRIPTOR.items()}


def _columns():
    cols = []
    for k in range(1, N_SITES + 1):
        cols += [f"Site {k} Group", f"Site {k} Period", f"Site {k} EN", f"Site {k} Nied"]
    return cols


def _config_row(symbols):
    """['Fe','Cu','Ni'] -> flat descriptor row in the published column order."""
    row = []
    for s in symbols:
        row += list(ELEMENTS[s])
    return row


def _sheet(configs, energies, duplicate_labels=True):
    """configs: list of element-symbol tuples; energies: matching Eads values."""
    data = [_config_row(c) for c in configs]
    df = pd.DataFrame(data, columns=_columns())
    df.insert(0, "Labels", energies if duplicate_labels else list(range(len(energies))))
    df["Eads (eV)"] = energies
    return df


# three configurations, deliberately overlapping only in part
CFG_A = ("Fe", "Co", "Ni")
CFG_B = ("Cu", "Cu", "Mo")
CFG_C = ("Ni", "Mo", "Fe")
CFG_D = ("Co", "Co", "Cu")


def _workbook_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
    return buf.getvalue()


@pytest.fixture
def workbook():
    """CO has {A,B,C}; COOH has {A,B,D}; CHO has {A,D}. Common to CO+COOH: {A,B}.
    Common to all three: {A}."""
    return _workbook_bytes({
        "CO":   _sheet([CFG_A, CFG_B, CFG_C], [-2.0, -1.6, -1.9]),
        "CHO":  _sheet([CFG_A, CFG_D], [-1.5, -1.1]),
        "COOH": _sheet([CFG_A, CFG_B, CFG_D], [-0.8, -0.3, -0.6]),
    })


# --------------------------------------------------------------------- decoding
def test_decode_site_round_trips_every_known_element():
    for desc, sym in ELEMENT_BY_DESCRIPTOR.items():
        assert decode_site(*desc) == sym


def test_decode_site_unknown_descriptor_is_flagged_not_guessed():
    assert decode_site(7, 4, 1.55, 5) == "?"


# --------------------------------------------------------------------- leakage
def test_labels_column_duplicating_target_is_detected():
    df = _sheet([CFG_A], [-2.0], duplicate_labels=True)
    assert "leakage" in assert_no_leakage(df)


def test_labels_column_not_duplicating_target_is_not_flagged():
    df = _sheet([CFG_A], [-2.0], duplicate_labels=False)
    assert assert_no_leakage(df) is None


def test_labels_and_target_never_enter_the_feature_matrix():
    sd = load_sheet(_sheet([CFG_A, CFG_B], [-2.0, -1.6]), "CO")
    assert "Labels" not in sd.feature_names
    assert "Eads (eV)" not in sd.feature_names
    assert sd.X.shape == (2, 4 * N_SITES)
    # the target must not be reconstructible as a feature column
    assert not np.any(np.all(np.isclose(sd.X, sd.y[:, None]), axis=0))


# --------------------------------------------------------------------- loading
def test_load_workbook_reads_every_recognised_sheet(workbook):
    sheets = load_workbook(workbook)
    assert set(sheets) == {"CO", "CHO", "COOH"}
    assert len(sheets["CO"]) == 3 and len(sheets["COOH"]) == 3 and len(sheets["CHO"]) == 2
    assert sheets["CO"].site1 == ["Fe", "Cu", "Ni"]


def test_load_workbook_rejects_a_file_with_no_intermediate_sheets():
    bad = _workbook_bytes({"Notes": pd.DataFrame({"a": [1]})})
    with pytest.raises(ValueError, match="no recognised intermediate sheets"):
        load_workbook(bad)


def test_missing_target_column_raises_rather_than_guessing():
    df = _sheet([CFG_A], [-2.0]).drop(columns=["Eads (eV)"])
    with pytest.raises(KeyError, match="Eads"):
        load_sheet(df, "CO")


def test_rows_with_missing_values_are_dropped_and_counted():
    df = _sheet([CFG_A, CFG_B], [-2.0, -1.6])
    df.loc[1, "Site 2 EN"] = np.nan
    sd = load_sheet(df, "CO")
    assert len(sd) == 1
    assert any("dropped 1 row" in w for w in sd.warnings)


# --------------------------------------------------------------------- the join
def test_join_keeps_only_configurations_present_in_all_requested_sheets(workbook):
    sheets = load_workbook(workbook)
    rows, rep = join_intermediates(sheets, ["CO", "COOH"])
    assert rep.n_joined == 2                      # CFG_A and CFG_B
    assert isinstance(rep, JoinReport)
    assert rep.n_per_species == {"CO": 3, "COOH": 3}
    assert all(set(r["energies"]) == {"CO", "COOH"} for r in rows)


def test_join_across_all_three_is_stricter_than_any_pair(workbook):
    sheets = load_workbook(workbook)
    _, pair = join_intermediates(sheets, ["CO", "COOH"])
    _, triple = join_intermediates(sheets, ["CO", "CHO", "COOH"])
    assert triple.n_joined == 1                   # only CFG_A
    assert triple.n_joined <= pair.n_joined


def test_join_never_imputes_a_missing_intermediate(workbook):
    sheets = load_workbook(workbook)
    rows, _ = join_intermediates(sheets, ["CO", "CHO", "COOH"])
    # CFG_C is in CO only; it must be absent, not filled with a mean
    assert all(r["site1"] != "Ni" for r in rows)
    for r in rows:
        assert all(np.isfinite(v) for v in r["energies"].values())


def test_join_pairs_the_right_energies_with_the_right_configuration(workbook):
    sheets = load_workbook(workbook)
    rows, _ = join_intermediates(sheets, ["CO", "COOH"])
    by_site1 = {r["site1"]: r["energies"] for r in rows}
    assert by_site1["Fe"] == {"CO": -2.0, "COOH": -0.8}     # CFG_A
    assert by_site1["Cu"] == {"CO": -1.6, "COOH": -0.3}     # CFG_B


def test_unknown_species_requested_raises(workbook):
    with pytest.raises(KeyError, match="OCH3"):
        join_intermediates(load_workbook(workbook), ["CO", "OCH3"])


def test_degenerate_descriptor_vectors_are_reported(workbook):
    # same configuration twice with different energies -> irreducible spread
    df = _sheet([CFG_A, CFG_A], [-2.0, -1.7])
    sd = load_sheet(df, "CO")
    deg = sd.degeneracy()
    assert deg["n_degenerate"] == 1
    assert deg["max_spread_eV"] == pytest.approx(0.3)


def test_degenerate_rows_are_reduced_not_duplicated_in_the_join():
    wb = _workbook_bytes({
        "CO":   _sheet([CFG_A, CFG_A], [-2.0, -1.8]),
        "COOH": _sheet([CFG_A], [-0.8]),
    })
    sheets = load_workbook(wb)
    rows, rep = join_intermediates(sheets, ["CO", "COOH"], reduce="mean")
    assert rep.n_joined == 1
    assert rows[0]["energies"]["CO"] == pytest.approx(-1.9)
    assert rep.duplicate_keys["CO"] == 1

    rows_min, _ = join_intermediates(sheets, ["CO", "COOH"], reduce="min")
    assert rows_min[0]["energies"]["CO"] == pytest.approx(-2.0)   # strongest binding


# --------------------------------------------------------- hand-off to the CHE proxy
def test_joined_rows_feed_the_activity_proxy_end_to_end(workbook):
    sheets = load_workbook(workbook)
    rows, _ = join_intermediates(sheets, ["CO", "COOH"])
    table = to_activity_table(rows)
    assert all({"dE_CO", "dE_COOH"} <= set(t) for t in table)

    targets = build_activity_targets(table, product="CO",
                                     descriptor_keys=["dE_CO", "dE_COOH"])
    assert len(targets) == len(rows)              # every joined config is computable
    for t in targets:
        assert np.isfinite(t["U_L"]) and np.isfinite(t["v_cell"])
        assert t["v_cell"] == proxy_cell_voltage(t["overpotential"])
        assert t["pds"] in ("CO2->COOH", "COOH->CO")


def test_adsorption_energies_are_rejected_as_a_wrong_reference_frame(workbook):
    """Adsorption energies (all negative, |E| large) give U_L > 0 for CO2
    reduction, which is unphysical. The guard must catch it rather than let the
    proxy emit a constant cell voltage."""
    rows, _ = join_intermediates(load_workbook(workbook), ["CO", "COOH"])
    chk = check_energy_reference(rows, product="CO")
    assert chk["ok"] is False
    assert chk["n_positive_U_L"] == chk["n"] > 0
    assert "ADSORPTION energies" in chk["reason"]


def test_proper_che_formation_energies_pass_the_guard():
    rows = [{"key": (0.0,), "descriptors": {}, "site1": "Cu",
             "energies": {"COOH": 0.55, "CO": 0.15}}]      # endergonic first step
    chk = check_energy_reference(rows, product="CO")
    assert chk["ok"] is True and chk["n_positive_U_L"] == 0
    assert chk["U_L_range"][1] <= 0.0


# ------------------------------------------------- CHE reference-frame conversion
# The gas energies below are ARBITRARY numbers that exercise the stoichiometry,
# not DFT results. Only their algebraic role is under test.
GAS = {"CO2": -20.0, "H2": -6.0, "H2O": -14.0}


@pytest.mark.parametrize("species,n_c,n_h,n_o", [
    ("CO", 1, 0, 1), ("COOH", 1, 1, 2), ("CHO", 1, 1, 1),
    ("CH2O", 1, 2, 1), ("OCH3", 1, 3, 1), ("H", 0, 1, 0), ("OH", 0, 1, 1),
])
def test_reference_shift_matches_the_balanced_half_reaction(species, n_c, n_h, n_o):
    """shift = w E(H2O) - n_C E(CO2) - ((n_H + 2w)/2) E(H2),  w = 2 n_C - n_O."""
    assert SPECIES_COMPOSITION[species] == (n_c, n_h, n_o)
    w = 2 * n_c - n_o
    expected = (w * GAS["H2O"] - n_c * GAS["CO2"] - 0.5 * (n_h + 2 * w) * GAS["H2"])
    assert che_reference_shift(species, GAS) == pytest.approx(expected)


def test_reference_shift_reproduces_the_textbook_cases():
    # *COOH: CO2 + 1/2 H2 -> *COOH   (no water)
    assert che_reference_shift("COOH", GAS) == pytest.approx(
        -GAS["CO2"] - 0.5 * GAS["H2"])
    # *CO: CO2 + H2 -> *CO + H2O
    assert che_reference_shift("CO", GAS) == pytest.approx(
        GAS["H2O"] - GAS["CO2"] - GAS["H2"])
    # *H: 1/2 H2 -> *H
    assert che_reference_shift("H", GAS) == pytest.approx(-0.5 * GAS["H2"])
    # *OH: H2O -> *OH + 1/2 H2   (water on the LEFT: w = -1)
    assert che_reference_shift("OH", GAS) == pytest.approx(
        -GAS["H2O"] + 0.5 * GAS["H2"])


def test_missing_gas_reference_raises_and_names_what_is_needed():
    with pytest.raises(KeyError, match="H2O"):
        che_reference_shift("CO", {"CO2": -20.0, "H2": -6.0})
    with pytest.raises(KeyError, match="unknown adsorbate"):
        che_reference_shift("CH4", GAS)


def test_species_needing_no_water_does_not_demand_a_water_reference():
    # *COOH has w = 0, so E(H2O) is genuinely unnecessary and must not be required
    assert che_reference_shift("COOH", {"CO2": -20.0, "H2": -6.0}) == pytest.approx(23.0)


def test_conversion_is_a_per_species_constant_offset():
    a = to_che_formation_energies({"CO": -2.0, "COOH": -0.9}, GAS)
    b = to_che_formation_energies({"CO": -1.5, "COOH": -0.4}, GAS)
    assert (a["CO"] - b["CO"]) == pytest.approx(-0.5)
    assert (a["COOH"] - b["COOH"]) == pytest.approx(-0.5)


def test_adsorbate_gas_reference_is_added_when_supplied():
    plain = to_che_formation_energies({"CO": -2.0}, GAS)
    shifted = to_che_formation_energies({"CO": -2.0}, GAS,
                                        adsorbate_gas_energies={"CO": -13.0})
    assert (shifted["CO"] - plain["CO"]) == pytest.approx(-13.0)


def test_thermochemical_corrections_are_off_by_default_and_additive_when_on():
    plain = to_che_formation_energies({"CO": -2.0}, GAS)
    corrected = to_che_formation_energies({"CO": -2.0}, GAS,
                                          corrections={"CO": 0.10})
    assert (corrected["CO"] - plain["CO"]) == pytest.approx(0.10)


def test_conversion_keeps_the_raw_energies_auditable(workbook):
    rows, _ = join_intermediates(load_workbook(workbook), ["CO", "COOH"])
    conv = convert_rows_to_che(rows, GAS)
    assert len(conv) == len(rows)
    for original, new in zip(rows, conv):
        assert new["energies_ads"] == original["energies"]     # provenance kept
        assert new["energies"] != original["energies"]         # and actually converted
        assert new["key"] == original["key"]


def test_conversion_makes_the_pathway_physical_with_consistent_references():
    """With gas references chosen so the first PCET step is endergonic, U_L
    becomes negative and the guard passes -- the whole point of the conversion."""
    rows = [{"key": (0.0,), "descriptors": {}, "site1": "Fe",
             "energies": {"CO": -2.0, "COOH": -0.9}}]
    assert check_energy_reference(rows, "CO")["ok"] is False   # raw E_ads: unphysical

    # references that place *COOH above the CO2 datum
    gas = {"CO2": -1.0, "H2": 0.0, "H2O": -1.0}
    conv = convert_rows_to_che(rows, gas)
    lp = limiting_potential(conv[0]["energies"], "CO")
    assert lp is not None and lp["U_L"] < 0
    assert check_energy_reference(conv, "CO")["ok"] is True


def test_pathway_coverage_reports_ch3oh_as_not_computable(workbook):
    cov = pathway_coverage(load_workbook(workbook), PATHWAYS)
    assert cov["CO"]["computable"] is True
    assert cov["CO"]["n_configurations"] == 2
    assert set(cov["CO"]["required"]) == {"COOH", "CO"}
    # CH3OH additionally needs *CH2O and *OCH3, absent from this workbook
    assert cov["CH3OH"]["computable"] is False
    assert set(cov["CH3OH"]["missing"]) >= {"CH2O", "OCH3"}
    # HCOOH needs *OCHO, also absent
    assert cov["HCOOH"]["computable"] is False
    assert "OCHO" in cov["HCOOH"]["missing"]
