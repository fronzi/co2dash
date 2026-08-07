"""Tests for the end-to-end pathway validation and the Sobol non-finite fix.

Synthetic data throughout: the numbers exercise the machinery, not any chemistry.
"""
import numpy as np
import pytest

from co2dash.chain import (PathwayValidation, ReferenceFrame, validate_pathway,
                           _sheet_excluding)
from co2dash.composition import (ELEMENTS, configurations_to_descriptors,
                                 feature_names)
from co2dash.hea import SheetData
from co2dash.techno_economic import Scenario
from co2dash.uncertainty import SOBOL_TARGETS, sobol_indices


# --------------------------------------------------------------------- fixtures
def _shared_sheets(n_shared=25, n_extra=90, seed=0):
    """Two sheets sharing `n_shared` configurations, each with its own extras."""
    rng = np.random.default_rng(seed)

    def block(m, s):
        configs = np.random.default_rng(s).choice(ELEMENTS, size=(m, 10))
        return configs, configurations_to_descriptors(configs)

    shared_cfg, shared_X = block(n_shared, 100)
    out = {}
    for i, sp in enumerate(("CO", "COOH")):
        extra_cfg, extra_X = block(n_extra, 200 + i)
        X = np.vstack([shared_X, extra_X])
        configs = np.vstack([shared_cfg, extra_cfg])
        # target depends on site-1 electronegativity, plus a species offset
        y = (-2.0 + i * 1.1) + 0.9 * (X[:, 2] - 1.9) + rng.normal(0, 0.02, len(X))
        out[sp] = SheetData(species=sp, X=X, y=y, feature_names=feature_names(),
                            keys=[tuple(r) for r in np.round(X, 6)],
                            site1=[c for c in configs[:, 0]])
    return out, n_shared


@pytest.fixture
def sheets():
    return _shared_sheets()[0]


@pytest.fixture
def frame():
    return ReferenceFrame(mode="anchored", anchor_energies={"COOH": -0.90},
                          anchor_U_L=-0.45, anchor_source="unit test")


@pytest.fixture
def base():
    return Scenario(n_electrons=2, molar_mass_prod=0.028, m_co2=1.57, m_h2=0.0,
                    faradaic_efficiency=0.90, cell_voltage=3.0, capex_total=5e7,
                    annual_production_kg=2e7, opex_fix_per_yr=3e6, disc_rate=0.08,
                    lifetime_yr=20, c_co2=0.05, c_elec=0.04, c_h2=0.0,
                    lcop_conventional=0.40, grid_intensity=0.02, e_capture=0.1,
                    e_process=0.05, release_fraction=0.0)


# ------------------------------------------------------------------- hold-out
def test_sheet_excluding_removes_exactly_the_named_configurations(sheets):
    sd = sheets["CO"]
    drop = sd.keys[:5]
    trimmed = _sheet_excluding(sd, drop)
    assert len(trimmed) == len(sd) - 5
    assert not (set(trimmed.keys) & set(drop))
    assert trimmed.feature_names == sd.feature_names
    assert len(trimmed.site1) == len(trimmed)


def test_validation_retrains_without_the_held_out_configurations(sheets):
    """The joined configurations are in every sheet, so evaluating without
    removing them would measure memorisation."""
    sheets_, n_shared = _shared_sheets()
    v = validate_pathway(sheets_, ReferenceFrame(
        mode="anchored", anchor_energies={"COOH": -0.9}, anchor_U_L=-0.45), "CO")
    assert v.n == n_shared
    for sp in v.species:
        assert v.n_train_after_holdout[sp] == len(sheets_[sp]) - n_shared


def test_validation_reports_the_full_metric_set(sheets, frame):
    v = validate_pathway(sheets, frame, "CO")
    assert isinstance(v, PathwayValidation)
    assert set(v.species) == {"CO", "COOH"}
    assert v.u_l_true.shape == v.u_l_pred.shape == (v.n,)
    assert np.isfinite(v.u_l_rmse) and v.u_l_rmse >= 0
    assert 0.0 <= v.pds_agreement <= 1.0
    assert set(v.amplification) == set(v.species)
    assert "U_L RMSE" in v.summary()


def test_validation_recovers_a_learnable_signal(sheets, frame):
    v = validate_pathway(sheets, frame, "CO")
    # the synthetic target is a clean function of site-1 EN, so the surrogate
    # should track U_L closely
    assert v.u_l_rmse < 0.15
    assert abs(v.u_l_bias) < 0.10


def test_range_compression_is_measured_and_explained(sheets, frame):
    """The slope of predicted-on-true U_L quantifies regression toward the mean.
    On the real workbook it is ~0.62 for every model class tried, which is why
    the note says the ceiling is the descriptor set rather than the regressor."""
    v = validate_pathway(sheets, frame, "CO")
    assert np.isfinite(v.u_l_slope) and v.u_l_slope > 0
    assert f"{v.u_l_slope:.2f}" in v.summary()

    faithful = PathwayValidation(
        n=5, species=["CO"], e_ads_rmse={"CO": 0.1},
        u_l_true=np.zeros(5), u_l_pred=np.zeros(5), u_l_rmse=0.0, u_l_bias=0.0,
        u_l_rank_corr=1.0, u_l_slope=0.98, pds_agreement=1.0,
        amplification={"CO": 1.0}, n_train_after_holdout={"CO": 10})
    assert faithful.compression_note() is None

    compressed = dc_replace(faithful, u_l_slope=0.62)
    note = compressed.compression_note()
    assert note is not None and "0.62" in note and "1.6x" in note


def dc_replace(obj, **kw):
    import dataclasses
    return dataclasses.replace(obj, **kw)


def test_amplification_is_reported_per_species(sheets, frame):
    v = validate_pathway(sheets, frame, "CO")
    for sp, amp in v.amplification.items():
        assert amp >= 0
        assert np.isclose(amp, v.u_l_rmse / v.e_ads_rmse[sp])


def test_relative_mode_cannot_be_validated(sheets):
    with pytest.raises(ValueError, match="absolute U_L"):
        validate_pathway(sheets, ReferenceFrame(), "CO")


def test_validation_needs_every_pathway_species(sheets, frame):
    with pytest.raises(KeyError, match="missing"):
        validate_pathway({"CO": sheets["CO"]}, frame, "CO")


def test_validation_raises_when_no_configuration_is_shared(frame):
    a, _ = _shared_sheets(n_shared=0, n_extra=40, seed=1)
    with pytest.raises(ValueError, match="no configuration carries"):
        validate_pathway(a, frame, "CO")


# --------------------------------------------------------- Sobol non-finite fix
def _bounds():
    return {"faradaic_efficiency": (0.3, 0.95), "cell_voltage": (2.2, 4.0),
            "c_elec": (0.02, 0.15)}


def test_total_order_indices_stay_below_one(base):
    idx, diag = sobol_indices(base, _bounds(), n=256, return_diagnostics=True)
    assert all(v["ST"] <= 1.0 + 0.2 for v in idx.values())
    assert diag["target"] == "mac"
    assert 0.0 <= diag["nonfinite_fraction"] <= 1.0


def test_partly_non_finite_sample_is_winsorised_and_flagged(base):
    """A moderately dirty grid drives net abatement negative over PART of the
    sample. The old code replaced those draws with 10x the maximum, inflating the
    variance until ST could exceed 1. They are now capped at a finite percentile
    and the substitution is reported."""
    import dataclasses as dc
    dirty = dc.replace(base, grid_intensity=0.15)
    idx, diag = sobol_indices(dirty, _bounds(), n=256, return_diagnostics=True)
    assert 0.0 < diag["nonfinite_fraction"] < 1.0
    assert diag["replaced_fraction"] == pytest.approx(diag["nonfinite_fraction"])
    assert np.isfinite(diag["output_variance"])
    assert diag["reliable"] is False and diag["reason"]      # >5% replaced


def test_wholly_non_finite_sample_refuses_rather_than_inventing_a_value(base):
    """If nothing is climate-positive there is no MAC distribution at all. The
    old penalty produced a confident-looking answer from pure substitute values."""
    import dataclasses as dc
    hopeless = dc.replace(base, grid_intensity=0.9)
    with pytest.raises(ValueError, match="every MAC draw is non-finite"):
        sobol_indices(hopeless, _bounds(), n=128)
    # the bounded target still works there, which is the documented fallback
    idx = sobol_indices(hopeless, _bounds(), n=128, target="feasible",
                        carbon_price_usd_per_kg=0.30)
    assert all(np.isfinite(v["ST"]) for v in idx.values())


def test_feasibility_target_is_bounded_and_needs_no_penalty(base):
    idx, diag = sobol_indices(base, _bounds(), n=256, target="feasible",
                              carbon_price_usd_per_kg=0.30,
                              return_diagnostics=True)
    assert diag["replaced_fraction"] == 0.0
    assert diag["target"] == "feasible"


def test_unknown_target_is_rejected(base):
    with pytest.raises(ValueError, match="target must be one of"):
        sobol_indices(base, _bounds(), n=64, target="profit")
    assert "mac" in SOBOL_TARGETS and "feasible" in SOBOL_TARGETS


def test_default_call_signature_is_unchanged(base):
    """Existing callers pass no target and expect a plain dict."""
    idx = sobol_indices(base, _bounds(), n=128)
    assert isinstance(idx, dict)
    assert set(idx) == set(_bounds())
    assert all({"S1", "ST"} <= set(v) for v in idx.values())
