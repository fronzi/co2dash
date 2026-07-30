"""Tests for the adaptive Figshare CO2RR loader (plumbing; not real-data results)."""
import numpy as np
import pytest
from co2dash.loaders import species_of, target_of, resolve_columns, to_descriptor_activity


def test_species_detection():
    assert species_of("E_CO") == "CO"
    assert species_of("dE_COOH") == "COOH"
    assert species_of("*CHO") == "CHO"
    assert species_of("Eads_CO (eV)") == "CO"
    assert species_of("ECO2") is None          # CO2 is not an intermediate here
    assert species_of("random_col") is None


def test_target_detection_and_resolve():
    desc, target = resolve_columns(["E_CO", "E_COOH", "U_L", "note"])
    assert desc == {"E_CO": "CO", "E_COOH": "COOH"}
    assert target == "U_L"


def test_che_target_path():
    # only CO/COOH -> U_L(CO) computed via CHE proxy
    rows = [{"E_CO": -0.6, "E_COOH": 0.1}, {"E_CO": -0.3, "E_COOH": 0.4},
            {"E_CO": -0.9, "E_COOH": -0.2}]
    X, y, rep = to_descriptor_activity(rows, product="CO")
    assert X.shape == (3, 2) and y.shape == (3,)
    assert rep["target"].startswith("CHE U_L")
    assert set(rep["descriptors"]) == {"CO", "COOH"}


def test_explicit_target_path_and_skip():
    rows = [{"E_CO": -0.6, "E_COOH": 0.1, "U_L": -0.4},
            {"E_CO": -0.3, "E_COOH": 0.4, "U_L": -0.7},
            {"E_CO": "", "E_COOH": 0.0, "U_L": -0.5}]     # missing descriptor -> skipped
    X, y, rep = to_descriptor_activity(rows)
    assert rep["target"] == "U_L" and rep["matched"] == 2 and rep["skipped"] == 1
    assert list(y) == [-0.4, -0.7]


def test_no_descriptor_columns_raises():
    with pytest.raises(ValueError):
        to_descriptor_activity([{"foo": 1, "bar": 2}])
