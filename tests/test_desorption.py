"""Tests for the desorption limit — the non-electrochemical bottleneck.

Energies here are arbitrary numbers chosen to exercise the equations, not
chemistry data. The invariant under test: a surface whose proton-coupled steps
are all downhill must NOT be reported as an excellent catalyst if the product
never leaves, and no cell voltage may be derived in that regime.
"""
import numpy as np
import pytest

from co2dash.proxy import (DESORBING_STATE, desorption_free_energy,
                           equilibrium_coverage, limiting_analysis,
                           limiting_potential)


# --------------------------------------------------------------- desorption dG
def test_desorption_energy_is_gas_minus_adsorbed():
    e = {"COOH": 0.5, "CO": -1.2}
    assert desorption_free_energy(e, "CO", gas_formation_energy=0.7) == pytest.approx(1.9)


def test_stronger_binding_costs_more_to_desorb():
    weak = desorption_free_energy({"CO": -0.2, "COOH": 0.4}, "CO", 0.7)
    strong = desorption_free_energy({"CO": -2.0, "COOH": 0.4}, "CO", 0.7)
    assert strong > weak


def test_corrections_shift_the_adsorbed_state_only():
    plain = desorption_free_energy({"CO": -1.0, "COOH": 0.4}, "CO", 0.7)
    corr = desorption_free_energy({"CO": -1.0, "COOH": 0.4}, "CO", 0.7,
                                  corrections={"CO": 0.10})
    assert (plain - corr) == pytest.approx(0.10)


def test_missing_desorbing_species_returns_none():
    assert desorption_free_energy({"COOH": 0.4}, "CO", 0.7) is None
    assert DESORBING_STATE["CO"] == "CO"


# ------------------------------------------------------------------- coverage
def test_coverage_saturates_for_strong_binding_and_vanishes_for_weak():
    assert equilibrium_coverage(1.5) > 0.999
    assert equilibrium_coverage(-1.5) < 0.001
    assert equilibrium_coverage(0.0) == pytest.approx(0.5)


def test_coverage_is_monotonic_and_temperature_dependent():
    assert equilibrium_coverage(0.3) < equilibrium_coverage(0.6)
    # hotter surface holds the product less tightly
    assert equilibrium_coverage(0.3, 800.0) < equilibrium_coverage(0.3, 298.15)


def test_coverage_does_not_overflow_at_extremes():
    assert equilibrium_coverage(1e4) == 1.0
    assert equilibrium_coverage(-1e4) == 0.0


# ----------------------------------------------------------- limiting analysis
def test_electrochemically_limited_surface_is_labelled_as_such():
    # endergonic first step -> a potential is genuinely required
    la = limiting_analysis({"COOH": 0.6, "CO": 0.2}, "CO", gas_formation_energy=0.7)
    assert la["U_L"] < 0
    assert la["limitation"] == "electrochemical"


def test_the_two_limits_are_independent_and_can_both_bind():
    """A surface can need a potential AND hold onto its product. `limitation`
    names the step that sets U_L; `poisoned` is a separate fact, not its
    complement. Reporting only the first would hide half the problem."""
    la = limiting_analysis({"COOH": 0.6, "CO": 0.2}, "CO", gas_formation_energy=0.7)
    assert la["limitation"] == "electrochemical"
    assert la["poisoned"] is True                 # dG_des = 0.5 eV -> saturated
    # a genuinely clean surface: potential needed, product leaves easily
    clean = limiting_analysis({"COOH": 0.6, "CO": 0.68}, "CO",
                              gas_formation_energy=0.7)
    assert clean["limitation"] == "electrochemical" and clean["poisoned"] is False


def test_downhill_but_strongly_bound_surface_is_desorption_limited():
    """The case the CHE ladder alone gets wrong: U_L > 0 looks perfect."""
    la = limiting_analysis({"COOH": -0.5, "CO": -2.0}, "CO", gas_formation_energy=0.7)
    assert la["U_L"] > 0                       # every PCET step downhill
    assert la["limitation"] == "desorption"
    assert la["poisoned"] is True
    assert la["coverage"] > 0.99
    assert la["dG_desorption"] == pytest.approx(2.7)


def test_without_a_gas_reference_no_desorption_claim_is_made():
    la = limiting_analysis({"COOH": -0.5, "CO": -2.0}, "CO")
    assert la["dG_desorption"] is None and la["coverage"] is None
    assert la["limitation"] == "none"          # not silently called 'good'
    assert la["poisoned"] is False


def test_downhill_pcet_steps_imply_strong_binding_for_the_CO_pathway():
    """A structural consequence worth pinning down, found while writing these
    tests: for CO2 -> CO you cannot have U_L > 0 and a weakly bound product.

    U_L > 0 requires both dG(CO2->COOH) = G(COOH) <= 0 and
    dG(COOH->CO) = G(CO) - G(COOH) <= 0, hence G(CO) <= G(COOH) <= 0. Desorption
    costs dG_des = G_f(CO(g)) - G(CO) >= G_f(CO(g)) > 0, which at 298 K already
    saturates the coverage. So 'no limitation at all' is unreachable here --
    an apparently perfect CHE ladder means a poisoned surface, not a good
    catalyst.
    """
    for e_cooh, e_co in ((-0.1, -0.2), (-0.5, -1.0), (-0.05, -0.05)):
        la = limiting_analysis({"COOH": e_cooh, "CO": e_co}, "CO",
                               gas_formation_energy=0.7)
        assert la["U_L"] >= 0
        assert la["limitation"] == "desorption"

    # 'none' is only reachable if the desorbed-product reference itself is
    # negative, i.e. the overall reaction is downhill -- not the CO2RR case
    la = limiting_analysis({"COOH": -0.1, "CO": -0.2}, "CO",
                           gas_formation_energy=-0.5)
    assert la["limitation"] == "none"


def test_analysis_preserves_the_limiting_potential_fields():
    e = {"COOH": 0.6, "CO": 0.2}
    lp = limiting_potential(e, "CO")
    la = limiting_analysis(e, "CO", gas_formation_energy=0.7)
    for k in ("U_L", "overpotential", "pds", "U_eq"):
        assert la[k] == lp[k]


def test_incomplete_pathway_returns_none():
    assert limiting_analysis({"COOH": 0.6}, "CO", gas_formation_energy=0.7) is None


def test_poisoning_threshold_is_adjustable():
    # weak binding: dG_des = 0.7 - 0.65 = 0.05 eV -> coverage ~0.88, so the
    # verdict genuinely depends on where the threshold is placed
    e = {"COOH": -0.5, "CO": 0.65}
    lenient = limiting_analysis(e, "CO", gas_formation_energy=0.7,
                                poisoning_coverage=0.99)
    strict = limiting_analysis(e, "CO", gas_formation_energy=0.7,
                               poisoning_coverage=0.50)
    assert 0.5 < lenient["coverage"] < 0.99
    assert strict["poisoned"] and not lenient["poisoned"]


# --------------------------------------------------------------- chain wiring
def test_chain_refuses_a_voltage_when_desorption_limits():
    """Applied potential cannot remove a bound product, so no V_cell is derived."""
    import numpy as np
    from co2dash.chain import (ASSUMED, ReferenceFrame, run_chain,
                               train_intermediate_models)
    from co2dash.composition import (ELEMENTS, configurations_to_descriptors,
                                     feature_names, Composition)
    from co2dash.hea import SheetData
    from co2dash.techno_economic import Scenario

    rng = np.random.default_rng(0)
    sheets = {}
    for sp, centre in (("CO", -2.6), ("COOH", -0.9)):
        cfg = rng.choice(ELEMENTS, size=(80, 10))
        X = configurations_to_descriptors(cfg)
        y = centre + rng.normal(0, 0.02, 80)
        sheets[sp] = SheetData(species=sp, X=X, y=y, feature_names=feature_names(),
                               keys=[tuple(r) for r in X],
                               site1=[c for c in cfg[:, 0]])
    models = train_intermediate_models(sheets)
    base = Scenario(n_electrons=2, molar_mass_prod=0.028, m_co2=1.57, m_h2=0.0,
                    faradaic_efficiency=0.9, cell_voltage=3.0, capex_total=5e7,
                    annual_production_kg=2e7, opex_fix_per_yr=3e6, disc_rate=0.08,
                    lifetime_yr=20, c_co2=0.05, c_elec=0.04, c_h2=0.0,
                    lcop_conventional=0.4, grid_intensity=0.02, e_capture=0.1,
                    e_process=0.05, release_fraction=0.0)
    # anchor placed so every PCET step comes out downhill — the regime where the
    # CHE ladder alone would report an excellent catalyst
    frame = ReferenceFrame(mode="anchored", anchor_energies={"COOH": -0.9},
                           anchor_U_L=0.30, gas_formation_energy=0.7)

    r = run_chain(Composition.equimolar(), models, base, frame, n_samples=50)
    assert r.limitation == "desorption"
    assert r.v_cell is None                                   # refused, not fudged
    assert r.scenario.cell_voltage == base.cell_voltage
    assert r.provenance.origins["cell_voltage"] == ASSUMED
    assert any("DESORPTION-LIMITED" in w for w in r.warnings)
    assert r.coverage > 0.99
