"""Tests for the CO2RR corpus featurizer (plumbing; not real-data results)."""
import numpy as np
import pytest
from co2dash.corpus import featurize_co2rr, map_corpus_columns, PRODUCT_CLASSES


def test_map_corpus_columns():
    m = map_corpus_columns(["material", "product", "faradaic efficiency",
                            "current density", "voltage"])
    assert m["material"] == "material_id"
    assert m["faradaic efficiency"] == "faradaic_efficiency"
    assert m["current density"] == "current_density"
    assert m["voltage"] == "cell_voltage"


def test_featurize_normalises_fe_and_shapes():
    rows = [
        {"material": "Cu2O", "product": "C2H4", "faradaic efficiency": 45.0,
         "current density": 200, "voltage": 3.4},
        {"material": "Ag", "product": "CO", "faradaic efficiency": 92.0,
         "current density": 250, "voltage": 3.1},
        {"material": "Sn", "product": "HCOOH", "faradaic efficiency": 80.0,
         "current density": 150, "voltage": 3.3},
    ]
    X, y, names = featurize_co2rr(rows)
    assert X.shape[0] == 3 and y.shape == (3,)
    assert y.max() <= 1.0                      # % auto-normalised to fraction
    assert y[1] == pytest.approx(0.92)
    assert "current_density" in names and any(n.startswith("prod::") for n in names)


def test_featurize_drops_rows_without_fe():
    rows = [
        {"material": "Cu", "product": "CO", "faradaic efficiency": 90,
         "current density": 200, "voltage": 3.0},
        {"material": "Cu", "product": "CO", "faradaic efficiency": "",
         "current density": 200, "voltage": 3.0},   # no FE -> dropped
    ]
    X, y, _ = featurize_co2rr(rows)
    assert len(y) == 1


def test_featurize_raises_when_no_valid_rows():
    with pytest.raises(ValueError):
        featurize_co2rr([{"material": "Cu", "product": "CO"}])
