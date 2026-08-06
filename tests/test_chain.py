"""Tests for composition -> descriptors and the discovery->decision chain.

The invariant under test throughout: the chain must never present an assumed
input as a predicted one. Faradaic efficiency in particular is carried through
untouched and must always be labelled assumed.
"""
import numpy as np
import pytest

from co2dash.chain import (ASSUMED, DFT, ChainProvenance, ReferenceFrame,
                           apply_reference, pds_uniform, predict_composition,
                           rank_compositions, run_chain,
                           train_intermediate_models)
from co2dash.composition import (Composition, DESCRIPTOR_BY_ELEMENT, ELEMENTS,
                                 align_to_training_columns,
                                 configurations_to_descriptors,
                                 descriptors_for_composition, feature_names,
                                 sample_configurations)
from co2dash.hea import SheetData
from co2dash.proxy import PATHWAYS
from co2dash.techno_economic import Scenario


# --------------------------------------------------------------------- fixtures
def _sheet(species, n=120, seed=0):
    """Synthetic sheet whose target depends on the site-1 electronegativity, so a
    linear surrogate can learn something. Numbers exercise the plumbing only."""
    rng = np.random.default_rng(seed)
    names = feature_names()
    configs = rng.choice(ELEMENTS, size=(n, 10))
    X = configurations_to_descriptors(configs)
    y = -2.0 + 0.8 * (X[:, 2] - 1.9) + rng.normal(0, 0.02, n)
    return SheetData(species=species, X=X, y=y, feature_names=names,
                     keys=[tuple(r) for r in X],
                     site1=[c for c in configs[:, 0]])


@pytest.fixture
def models():
    return train_intermediate_models({"CO": _sheet("CO", seed=1),
                                      "COOH": _sheet("COOH", seed=2)})


@pytest.fixture
def base():
    return Scenario(n_electrons=2, molar_mass_prod=0.028, m_co2=1.57, m_h2=0.0,
                    faradaic_efficiency=0.90, cell_voltage=3.0,
                    capex_total=5e7, annual_production_kg=2e7,
                    opex_fix_per_yr=3e6, disc_rate=0.08, lifetime_yr=20,
                    c_co2=0.05, c_elec=0.04, c_h2=0.0, lcop_conventional=0.40,
                    grid_intensity=0.05, e_capture=0.1, e_process=0.05,
                    release_fraction=0.0)


# --------------------------------------------------------------------- parsing
def test_equimolar_and_string_parsing_agree():
    a = Composition.equimolar()
    b = Composition.from_string("FeCoNiCuMo")
    assert a.fractions == pytest.approx(b.fractions)


def test_explicit_fractions_are_parsed():
    c = Composition.from_string("Fe0.4Co0.3Ni0.3")
    assert c.fractions["Fe"] == pytest.approx(0.4)
    assert sum(c.fractions.values()) == pytest.approx(1.0)


def test_unnormalised_amounts_are_normalised():
    c = Composition.from_string("Fe2Co2Ni1")
    assert c.fractions["Fe"] == pytest.approx(0.4)
    assert c.fractions["Ni"] == pytest.approx(0.2)


def test_element_outside_the_descriptor_table_is_refused():
    with pytest.raises(ValueError, match="own DFT"):
        Composition({"Ag": 1.0})
    with pytest.raises(ValueError, match="own DFT"):
        Composition.from_string("Ag0.5Sn0.5")


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        Composition({"Fe": 0.5, "Co": 0.2})


# --------------------------------------------------------------------- sampling
def test_sampled_occupation_frequencies_match_the_composition():
    comp = Composition({"Fe": 0.5, "Cu": 0.5})
    configs = sample_configurations(comp, n_samples=4000, seed=0)
    frac_fe = float(np.mean(configs == "Fe"))
    assert frac_fe == pytest.approx(0.5, abs=0.02)
    assert set(np.unique(configs)) == {"Fe", "Cu"}


def test_sampling_is_reproducible_and_seed_sensitive():
    c = Composition.equimolar()
    assert np.array_equal(sample_configurations(c, 50, seed=3),
                          sample_configurations(c, 50, seed=3))
    assert not np.array_equal(sample_configurations(c, 50, seed=3),
                              sample_configurations(c, 50, seed=4))


def test_fixed_site1_pins_the_adsorption_site_only():
    comp = Composition.equimolar()
    configs = sample_configurations(comp, n_samples=200, fixed_site1="Cu", seed=0)
    assert set(np.unique(configs[:, 0])) == {"Cu"}
    assert len(set(np.unique(configs[:, 1:]))) > 1        # environment still varies


def test_descriptor_columns_follow_the_workbook_order():
    names = feature_names()
    assert names[:4] == ["Site 1 Group", "Site 1 Period", "Site 1 EN", "Site 1 Nied"]
    assert len(names) == 40
    X = configurations_to_descriptors(np.array([["Cu"] * 10]))
    assert tuple(X[0, :4]) == DESCRIPTOR_BY_ELEMENT["Cu"]


def test_misaligned_training_columns_raise_instead_of_predicting():
    X, _ = descriptors_for_composition(Composition.equimolar(), n_samples=5)
    with pytest.raises(KeyError, match="Refusing to predict"):
        align_to_training_columns(X, ["Site 1 Group", "Bogus Column"])


def test_alignment_reorders_rather_than_reshuffling_silently():
    X, _ = descriptors_for_composition(Composition.equimolar(), n_samples=4, seed=0)
    reordered = align_to_training_columns(X, ["Site 1 EN", "Site 1 Group"])
    assert np.allclose(reordered[:, 0], X[:, 2])
    assert np.allclose(reordered[:, 1], X[:, 0])


# --------------------------------------------------------------------- prediction
def test_prediction_returns_an_ensemble_not_a_point(models):
    preds = predict_composition(Composition.equimolar(), models, n_samples=300)
    assert set(preds) == {"CO", "COOH"}
    p = preds["CO"]
    assert p.samples.shape == (300,)
    assert p.configurational_sd > 0            # a composition is not one surface
    assert p.model_sd > 0
    assert p.total_sd >= max(p.configurational_sd, p.model_sd)


def test_pure_element_has_no_configurational_spread(models):
    preds = predict_composition(Composition({"Cu": 1.0}), models, n_samples=200)
    assert preds["CO"].configurational_sd == pytest.approx(0.0, abs=1e-9)


def test_composition_changes_the_prediction(models):
    cu = predict_composition(Composition({"Cu": 1.0}), models)["CO"].mean
    fe = predict_composition(Composition({"Fe": 1.0}), models)["CO"].mean
    assert abs(cu - fe) > 0.05                 # the catalyst actually matters


# --------------------------------------------------------------------- reference frames
def test_relative_mode_claims_no_absolute_potential():
    assert ReferenceFrame().mode == "relative"
    assert ReferenceFrame().gives_absolute_U_L() is False
    assert apply_reference({"CO": -2.0}, ReferenceFrame()) is None


def test_absolute_mode_requires_your_own_gas_energies():
    with pytest.raises(ValueError, match="your own calculations"):
        ReferenceFrame(mode="absolute")


def test_anchored_mode_requires_both_anchor_pieces():
    with pytest.raises(ValueError, match="anchor_energies and anchor_U_L"):
        ReferenceFrame(mode="anchored", anchor_U_L=-0.5)


def test_anchor_reproduces_its_own_known_potential():
    frame = ReferenceFrame(mode="anchored", anchor_energies={"COOH": -0.90},
                           anchor_U_L=-0.45, anchor_source="test")
    che = apply_reference({"COOH": -0.90, "CO": -2.0}, frame, "CO")
    assert -che["COOH"] == pytest.approx(-0.45)


def test_unknown_reference_mode_is_rejected():
    with pytest.raises(ValueError, match="mode must be one of"):
        ReferenceFrame(mode="magic")


# --------------------------------------------------------------------- the chain
def test_relative_chain_does_not_touch_cell_voltage(models, base):
    r = run_chain(Composition.equimolar(), models, base)
    assert r.v_cell is None and r.U_L is None
    assert r.scenario.cell_voltage == base.cell_voltage
    assert r.provenance.origins["cell_voltage"] == ASSUMED
    assert r.relative_score is not None          # still ranks


def test_anchored_chain_drives_cell_voltage_from_dft(models, base):
    frame = ReferenceFrame(mode="anchored", anchor_energies={"COOH": -0.90},
                           anchor_U_L=-0.45, anchor_source="unit test")
    r = run_chain(Composition.equimolar(), models, base, frame)
    assert r.U_L is not None and r.v_cell is not None
    assert r.provenance.origins["cell_voltage"] == DFT
    assert r.scenario.cell_voltage == pytest.approx(r.v_cell)
    assert r.scenario.cell_voltage != base.cell_voltage
    assert r.v_cell_sd is not None and r.v_cell_sd > 0


def test_faradaic_efficiency_is_never_predicted(models, base):
    for frame in (None, ReferenceFrame(mode="anchored",
                                       anchor_energies={"COOH": -0.9},
                                       anchor_U_L=-0.45)):
        r = run_chain(Composition.equimolar(), models, base, frame)
        assert r.scenario.faradaic_efficiency == base.faradaic_efficiency
        assert r.provenance.origins["faradaic_efficiency"] == ASSUMED
        assert "faradaic_efficiency" in r.provenance.assumed_fields()
        assert any("selectivity" in n for n in r.provenance.notes)


def test_provenance_headline_names_both_sides(models, base):
    frame = ReferenceFrame(mode="anchored", anchor_energies={"COOH": -0.9},
                           anchor_U_L=-0.45)
    r = run_chain(Composition.equimolar(), models, base, frame)
    head = r.provenance.headline()
    assert "cell_voltage" in head and "faradaic_efficiency" in head


def test_provenance_headline_is_explicit_when_nothing_is_dft_driven():
    assert "No KPI is DFT-driven" in ChainProvenance().headline()


def test_configurational_caveat_is_always_reported(models, base):
    r = run_chain(Composition.equimolar(), models, base)
    assert any("short-range order" in n.lower() for n in r.provenance.notes)


def test_ranking_works_without_any_reference_frame(models, base):
    comps = [Composition({"Cu": 1.0}), Composition({"Fe": 1.0}),
             Composition.equimolar()]
    ranked = rank_compositions(comps, models, base)
    assert len(ranked) == 3
    scores = [r.relative_score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_reference_free_ranking_matches_absolute_only_when_the_pds_is_common(models, base):
    """The reference shift is common to all configurations, so it cannot reorder
    them WHILE the same step limits everywhere. If the potential-determining step
    differs, the second step's free energy depends on the difference of two
    unknown constants and the reference-free ordering is no longer valid. The
    code must not claim otherwise."""
    comps = [Composition({"Cu": 1.0}), Composition({"Fe": 1.0}),
             Composition({"Ni": 1.0}), Composition.equimolar()]
    frame = ReferenceFrame(mode="anchored", anchor_energies={"COOH": -0.9},
                           anchor_U_L=-0.45)

    rel = rank_compositions(comps, models, base)
    absolute = rank_compositions(comps, models, base, frame)

    # relative_score uses the FIRST PCET step, so the orderings coincide only
    # when that step is the one limiting everywhere. If the second step limits,
    # the shift cancels differently and the reference-free score is the wrong
    # quantity -- which is precisely why relative mode records the assumption.
    first_step = f"CO2->{PATHWAYS['CO'][0][1]}"
    if pds_uniform(absolute) and absolute[0].pds == first_step:
        assert ([r.composition.label() for r in rel]
                == [r.composition.label() for r in absolute])
    else:
        assert all(any("potential-determining" in n for n in r.provenance.notes)
                   for r in rel)


def test_relative_mode_states_its_pds_assumption(models, base):
    r = run_chain(Composition.equimolar(), models, base)
    assert any("potential-determining" in n for n in r.provenance.notes)


def test_pds_uniform_detects_a_mixed_set():
    from co2dash.chain import ChainResult
    mk = lambda p: ChainResult(composition=Composition.equimolar(),   # noqa: E731
                               predictions={}, provenance=ChainProvenance(), pds=p)
    assert pds_uniform([mk("CO2->COOH"), mk("CO2->COOH")]) is True
    assert pds_uniform([mk("CO2->COOH"), mk("COOH->CO")]) is False
    assert pds_uniform([mk(None), mk("CO2->COOH")]) is True
