"""Piece 2 tests: vectorised path must agree with the scalar path."""
import numpy as np
from co2dash import RXN_METHANOL, Scenario, evaluate_array, propagate_mc


def _base():
    r = RXN_METHANOL
    return Scenario(
        n_electrons=r.n_electrons, molar_mass_prod=r.molar_mass_prod,
        m_co2=r.kg_co2_per_kg_prod, m_h2=0.0,
        faradaic_efficiency=0.60, cell_voltage=3.0,
        capex_total=5.0e7, annual_production_kg=2.0e7, opex_fix_per_yr=3.0e6,
        disc_rate=0.08, lifetime_yr=20, c_co2=0.05, c_elec=0.06, c_h2=0.0,
        lcop_conventional=0.40, grid_intensity=0.02, e_capture=0.1,
        e_process=0.05, release_fraction=0.10)


def test_vectorised_matches_scalar():
    base = _base()
    fes = np.array([0.3, 0.5, 0.7, 0.9])
    out = evaluate_array(base, {"faradaic_efficiency": fes})
    for i, fe in enumerate(fes):
        sc = _base(); sc.faradaic_efficiency = float(fe)
        ref = sc.evaluate()
        assert abs(out["lcop_usd_per_kg"][i] - ref["lcop_usd_per_kg"]) < 1e-9
        assert abs(out["net_abatement_kg_per_kg"][i]
                   - ref["net_abatement_kg_per_kg"]) < 1e-9
        # MAC: both finite here, compare; semantics for net<=0 (inf) preserved
        assert abs(out["mac_usd_per_kg_co2"][i] - ref["mac_usd_per_kg_co2"]) < 1e-6


def test_mac_inf_preserved_vectorised():
    base = _base(); base.release_fraction = 1.0   # fuel -> never climate-positive
    out = evaluate_array(base, {"faradaic_efficiency": np.array([0.5, 0.9])})
    assert np.all(~np.isfinite(out["mac_usd_per_kg_co2"]))


def test_propagate_mc_runs_and_bounds_probabilities():
    base = _base()
    res = propagate_mc(base, {"faradaic_efficiency": ("normal", 0.6, 0.1)},
                       carbon_price_usd_per_kg=2.0, n=5000, seed=0)
    assert 0.0 <= res["p_mac_below_carbon_price"] <= 1.0
    assert 0.0 <= res["p_net_positive"] <= 1.0
    assert res["mac"].shape == (5000,)
