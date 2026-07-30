"""Validation tests for the translation layer. Run: PYTHONPATH=src python -m pytest tests/
These are the kind of sanity checks that catch sign errors and unit slips."""
import math
import numpy as np
from co2dash.techno_economic import (capital_recovery_factor, specific_electricity_kwh_per_kg,
                                     net_abatement_kg_per_kg, breakeven_grid_intensity,
                                     marginal_abatement_cost)
from co2dash.schema import RXN_METHANOL


def test_crf_known_value():
    # CRF(8%, 20y) ~ 0.10185 (standard finance table value)
    assert abs(capital_recovery_factor(0.08, 20) - 0.101852) < 1e-4

def test_crf_zero_rate():
    assert abs(capital_recovery_factor(0.0, 10) - 0.1) < 1e-12

def test_electricity_scales_inverse_with_FE():
    e_low = specific_electricity_kwh_per_kg(6, 3.0, 0.03204, 0.30)
    e_high = specific_electricity_kwh_per_kg(6, 3.0, 0.03204, 0.90)
    assert e_low > e_high                          # lower FE -> more energy
    assert abs(e_low / e_high - 3.0) < 1e-6        # exact 1/FE scaling

def test_net_abatement_fuel_vs_durable():
    m = RXN_METHANOL.kg_co2_per_kg_prod
    durable = net_abatement_kg_per_kg(m, 0.02, 20, 0.1, 0.05, release_fraction=0.0)
    fuel    = net_abatement_kg_per_kg(m, 0.02, 20, 0.1, 0.05, release_fraction=1.0)
    assert durable > fuel                          # storage credit lost for fuels

def test_breakeven_threshold_consistency():
    m, e = 1.37, 20.0
    istar = breakeven_grid_intensity(m, e, 0.1, 0.05, 0.1)
    at_star = net_abatement_kg_per_kg(m, istar, e, 0.1, 0.05, 0.1)
    assert abs(at_star) < 1e-9                      # net abatement is exactly 0 at I*

def test_mac_infinite_when_no_abatement():
    assert not math.isfinite(marginal_abatement_cost(2.0, 0.4, -0.1))
    assert math.isfinite(marginal_abatement_cost(2.0, 0.4, 0.5))
