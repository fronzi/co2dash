"""
The discovery -> decision chain: composition -> E_ads -> U_L -> V_cell -> Scenario.

This module closes the loop the platform was missing: until now DFT descriptors
reached only the active-learning and calibration tabs, and the headline verdict
was a pure function of the sliders. Here the catalyst finally enters the
economics -- and, just as importantly, every field says where it came from.

THE REFERENCE PROBLEM, AND WHY THERE ARE THREE MODES
----------------------------------------------------
The CHE limiting potential needs formation free energies referenced to
CO2(g) + n_H*(1/2 H2). A workbook of adsorption energies is in a different
reference frame, and converting needs gas-phase total energies from the SAME
calculation setup (code, functional, pseudopotentials, cutoff). Those are not
portable between studies and are essentially never published, so 'just use
literature values for a standard functional' is not available: borrowing another
group's totals shifts every U_L by an unknown constant.

What IS rigorous is the observation that the unknown reference enters as a
CONSTANT per species. So:

  'relative'  (default) -- no absolute U_L is claimed. Configurations are ranked
                by dU_L relative to a chosen reference configuration, which is
                exactly determined by the data whenever the potential-determining
                step is common to both. Needs nothing extra. V_cell is NOT
                derived; it stays user-set and is labelled as such.
  'anchored'  -- you supply ONE configuration whose U_L is known (from
                experiment, or a published calculation of a surface also present
                in your set). The species shift is solved from that anchor and
                absolute U_L follows. One number instead of three.
  'absolute'  -- you supply E(CO2), E(H2), E(H2O) from YOUR OWN calculations.
                Fully rigorous; see hea.convert_rows_to_che.

Default is 'relative' because it is the only mode that is correct with no extra
input. The others activate when you provide the information they need.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .composition import (Composition, align_to_training_columns,
                          descriptors_for_composition, sro_note)
from .hea import SheetData, join_intermediates, to_che_formation_energies
from .proxy import PATHWAYS, limiting_potential, proxy_cell_voltage
from .surrogate import BayesianLinearSurrogate
from .techno_economic import Scenario

REFERENCE_MODES = ("relative", "anchored", "absolute")


# ---------------------------------------------------------------------------
# published activity band for this system -- a consistency check, NOT an anchor
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PublishedBand:
    """A limiting-potential range reported in the literature for a system."""
    system: str
    u_l_min_abs: float          # |U_L| lower edge, V
    u_l_max_abs: float          # |U_L| upper edge, V
    citation: str
    doi: str
    note: str = ""

    @property
    def width(self) -> float:
        return self.u_l_max_abs - self.u_l_min_abs


# The workbook's own paper. Reported |U_L| = 0.29-0.51 V for the designed
# FeCoNiCuMo surface. Tempting as an anchor -- same system, same level of theory
# -- but see check_against_published_band(): the |U_L| spread implied by the
# supplementary adsorption energies is several times wider than this band, and
# the discrepancy is INDEPENDENT of any shift, so no anchor choice reconciles
# them. Kept as a falsification test rather than a calibration source.
HEA_CO2RR_BAND = PublishedBand(
    system="FeCoNiCuMo HEA, CO2RR to CO",
    u_l_min_abs=0.29, u_l_max_abs=0.51,
    citation="Chen et al., ACS Catal. 12, 14864-14871 (2022)",
    doi="10.1021/acscatal.2c03675",
    note="Value taken from the abstract; whether it covers all sampled sites or "
         "only the designed surface is not established from the abstract alone.")


def check_against_published_band(u_l: Sequence[float],
                                 band: PublishedBand = HEA_CO2RR_BAND) -> Dict:
    """Compare a set of computed limiting potentials with a published band.

    Reports the fraction inside the band AND -- the decisive quantity -- the
    ratio of widths. The reference shift is a constant, so it moves the whole
    distribution without changing its width: if the computed spread is much
    wider than the published one, the two cannot be reconciled by ANY anchor,
    and the disagreement is about the model or the data selection rather than
    the reference frame.
    """
    u = np.abs(np.asarray(list(u_l), float))
    u = u[np.isfinite(u)]
    if u.size == 0:
        raise ValueError("no finite limiting potentials to compare")
    width = float(u.max() - u.min())
    ratio = width / band.width if band.width > 0 else float("inf")
    inside = float(np.mean((u >= band.u_l_min_abs) & (u <= band.u_l_max_abs)))
    reconcilable = ratio <= 1.5
    return {
        "band": f"{band.u_l_min_abs}-{band.u_l_max_abs} V ({band.citation})",
        "computed_width": width, "band_width": band.width,
        "width_ratio": ratio, "fraction_inside": inside,
        "shift_can_reconcile": reconcilable,
        "reason": "" if reconcilable else (
            f"the computed |U_L| spread is {ratio:.1f}x the published band. A "
            f"reference shift translates the distribution without narrowing it, "
            f"so no anchor can bring these into agreement. Likely causes: the "
            f"published band describes selected surfaces rather than the full "
            f"sampled set, or the potential-determining step varies between "
            f"configurations so U_L is not a single linear function of one "
            f"adsorption energy."),
    }


# ---------------------------------------------------------------------------
# provenance: every Scenario field says where it came from
# ---------------------------------------------------------------------------
DFT = "DFT-derived"
ASSUMED = "user-set (not modelled)"
DEFAULT = "sourced default"


@dataclass
class ChainProvenance:
    """Origin of every performance field, so the UI can never imply that an
    assumed input was predicted."""
    origins: Dict[str, str] = field(default_factory=dict)
    reference_mode: str = "relative"
    notes: List[str] = field(default_factory=list)

    def dft_fields(self) -> List[str]:
        return sorted(k for k, v in self.origins.items() if v == DFT)

    def assumed_fields(self) -> List[str]:
        return sorted(k for k, v in self.origins.items() if v != DFT)

    def headline(self) -> str:
        dft = self.dft_fields()
        if not dft:
            return "No KPI is DFT-driven; this result depends only on your inputs."
        return (f"DFT-driven: {', '.join(dft)}. "
                f"Assumed: {', '.join(self.assumed_fields())}.")


# ---------------------------------------------------------------------------
# per-intermediate surrogates trained on the workbook
# ---------------------------------------------------------------------------
@dataclass
class IntermediateModel:
    species: str
    model: BayesianLinearSurrogate
    feature_names: List[str]
    n_train: int
    sigma_eV: float
    site1_support: frozenset = frozenset()   # adsorption-site elements seen in training
    n_by_site1: Dict[str, int] = field(default_factory=dict)
    train_sd_eV: float = 0.0                 # spread of the training targets

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.model.predict(X)

    def unsupported(self, elements) -> List[str]:
        """Which of `elements` never occupied the adsorption site in training."""
        return sorted(set(elements) - set(self.site1_support))


def train_intermediate_models(sheets: Dict[str, SheetData],
                              species: Optional[Sequence[str]] = None
                              ) -> Dict[str, IntermediateModel]:
    """One surrogate per intermediate, each trained on its own sheet.

    Training per sheet rather than on the sheets' intersection is deliberate:
    the sheets share few configurations, so intersecting would discard most of
    the data. The intersection is better spent as a validation set.

    The site-1 support is recorded because the sheets do NOT cover the same
    adsorption-site elements (in the published workbook the *CO sheet contains
    no Cu-terminated sites at all). Predicting there is extrapolation, and it is
    invisible to the model's own uncertainty -- see applicability() below.
    """
    want = list(species) if species is not None else sorted(sheets)
    out: Dict[str, IntermediateModel] = {}
    for sp in want:
        sd = sheets[sp]
        model = BayesianLinearSurrogate().fit_evidence(sd.X, sd.y)
        counts: Dict[str, int] = {}
        for e in sd.site1:
            counts[e] = counts.get(e, 0) + 1
        out[sp] = IntermediateModel(
            species=sp, model=model, feature_names=list(sd.feature_names),
            n_train=len(sd), sigma_eV=float(1.0 / np.sqrt(model.beta)),
            site1_support=frozenset(counts) - {"?"}, n_by_site1=counts,
            train_sd_eV=float(np.std(sd.y)))
    return out


# ---------------------------------------------------------------------------
# composition -> predicted adsorption energies
# ---------------------------------------------------------------------------
# A configurational spread this many times the training spread is a symptom of
# extrapolation rather than physical diversity.
SPREAD_ALARM_RATIO = 1.5


@dataclass
class EnsemblePrediction:
    """Predicted E_ads over the configurational ensemble of one composition."""
    species: str
    mean: float                  # ensemble mean of the predictive means
    configurational_sd: float    # spread across configurations
    model_sd: float              # mean predictive (model) uncertainty
    samples: np.ndarray          # per-configuration predictive means
    # --- applicability domain ---
    out_of_domain_fraction: float = 0.0
    unsupported_site1: List[str] = field(default_factory=list)
    in_domain_mean: Optional[float] = None
    in_domain_sd: Optional[float] = None
    spread_ratio: Optional[float] = None    # configurational_sd / training sd

    @property
    def total_sd(self) -> float:
        """Configurational and model uncertainty added in quadrature. They are
        different things and are kept separate above; this is for propagation."""
        return float(np.sqrt(self.configurational_sd ** 2 + self.model_sd ** 2))

    @property
    def in_domain(self) -> bool:
        return self.out_of_domain_fraction == 0.0

    def warning(self) -> Optional[str]:
        """Human-readable reason this prediction should not be trusted, or None.

        Two independent symptoms are reported because the model's own predictive
        std does NOT catch this failure: an element absent from the adsorption
        site in training still has in-range descriptor values (it appears in the
        environment columns), so phi^T S phi stays small for a linear model. The
        novelty is in the JOINT position, which a linear model cannot see.
        """
        parts = []
        if self.unsupported_site1:
            parts.append(
                f"{self.out_of_domain_fraction:.0%} of sampled configurations put "
                f"{', '.join(self.unsupported_site1)} at the adsorption site, which "
                f"never occurs in the *{self.species} training data")
        if self.spread_ratio is not None and self.spread_ratio > SPREAD_ALARM_RATIO:
            parts.append(
                f"configurational spread is {self.spread_ratio:.1f}x the spread of "
                f"the training targets, a symptom of extrapolation rather than "
                f"physical diversity")
        if not parts:
            return None
        return ("EXTRAPOLATION — " + "; ".join(parts) +
                ". The model's own uncertainty does not detect this. Treat this "
                "value as unsupported.")


def predict_composition(comp: Composition,
                        models: Dict[str, IntermediateModel],
                        n_samples: int = 500,
                        fixed_site1: Optional[str] = None,
                        seed: int = 0) -> Dict[str, EnsemblePrediction]:
    """Predict every trained intermediate for one composition.

    Also evaluates the applicability domain per species and, where part of the
    ensemble is out of domain, reports in-domain-only statistics alongside the
    full-ensemble ones so there is a usable fallback rather than just a warning.
    """
    X_gen, configs = descriptors_for_composition(comp, n_samples=n_samples,
                                                 fixed_site1=fixed_site1, seed=seed)
    site1 = np.asarray([str(s) for s in configs[:, 0]])

    out: Dict[str, EnsemblePrediction] = {}
    for sp, im in models.items():
        X = align_to_training_columns(X_gen, im.feature_names)
        mu, sd = im.predict(X)

        unsupported = im.unsupported(np.unique(site1))
        ok = ~np.isin(site1, list(unsupported)) if unsupported else np.ones(len(mu), bool)
        ood = float(1.0 - ok.mean())
        cfg_sd = float(np.std(mu))

        out[sp] = EnsemblePrediction(
            species=sp, mean=float(np.mean(mu)),
            configurational_sd=cfg_sd,
            model_sd=float(np.mean(sd)), samples=np.asarray(mu, float),
            out_of_domain_fraction=ood, unsupported_site1=unsupported,
            in_domain_mean=float(np.mean(mu[ok])) if ok.any() else None,
            in_domain_sd=float(np.std(mu[ok])) if ok.any() else None,
            spread_ratio=(cfg_sd / im.train_sd_eV) if im.train_sd_eV > 1e-9 else None)
    return out


# ---------------------------------------------------------------------------
# reference frames
# ---------------------------------------------------------------------------
@dataclass
class ReferenceFrame:
    """How adsorption energies are converted to CHE formation energies."""
    mode: str = "relative"
    gas_energies: Optional[Dict[str, float]] = None       # 'absolute'
    anchor_energies: Optional[Dict[str, float]] = None    # 'anchored': E_ads of anchor
    anchor_U_L: Optional[float] = None                    # 'anchored': its known U_L
    anchor_source: str = ""
    corrections: Optional[Dict[str, float]] = None

    def __post_init__(self):
        if self.mode not in REFERENCE_MODES:
            raise ValueError(f"mode must be one of {REFERENCE_MODES}, got '{self.mode}'")
        if self.mode == "absolute" and not self.gas_energies:
            raise ValueError("mode='absolute' requires gas_energies "
                             "{'CO2':..,'H2':..,'H2O':..} from your own calculations")
        if self.mode == "anchored" and (self.anchor_energies is None
                                        or self.anchor_U_L is None):
            raise ValueError("mode='anchored' requires anchor_energies and anchor_U_L")

    def gives_absolute_U_L(self) -> bool:
        return self.mode in ("anchored", "absolute")


def _anchor_shift(frame: ReferenceFrame, product: str) -> float:
    """Solve the constant that makes the anchor reproduce its known U_L.

    IMPORTANT LIMITATION. Each species has its OWN reference shift -- see
    hea.che_reference_shift, where shift(*CO) and shift(*COOH) differ because the
    balanced half-reactions differ. A single anchor determines only ONE of them.

    This function therefore applies the anchored species' shift to every species,
    which is correct ONLY while the potential-determining step is the first PCET
    (CO2 -> X), because then U_L = -(E_ads(X) + shift(X)) and no other shift
    enters. If the second step limits, its free energy depends on the DIFFERENCE
    of two independent shifts, which one anchor cannot supply -- and the result
    would be wrong without being obviously so. `run_chain` checks the resulting
    PDS and flags this; `validate_pathway` inherits the same caveat.

    Use mode='absolute' with your own gas-phase energies whenever the PDS is not
    reliably the first step.
    """
    first_product = PATHWAYS[product][0][1]
    e = frame.anchor_energies.get(first_product)
    if e is None:
        raise KeyError(f"anchor_energies must contain '{first_product}' for the "
                       f"{product} pathway")
    # U_L = -(E_ads + s)  =>  s = -U_L - E_ads
    return -float(frame.anchor_U_L) - float(e)


def apply_reference(energies: Dict[str, float], frame: ReferenceFrame,
                    product: str = "CO") -> Optional[Dict[str, float]]:
    """Adsorption energies -> CHE formation energies under `frame`.
    Returns None in 'relative' mode, where no absolute frame is claimed."""
    if frame.mode == "relative":
        return None
    if frame.mode == "absolute":
        return to_che_formation_energies(energies, frame.gas_energies,
                                         corrections=frame.corrections)
    s = _anchor_shift(frame, product)
    corr = frame.corrections or {}
    return {sp: float(v) + s + float(corr.get(sp, 0.0)) for sp, v in energies.items()}


# ---------------------------------------------------------------------------
# the chain
# ---------------------------------------------------------------------------
@dataclass
class ChainResult:
    composition: Composition
    predictions: Dict[str, EnsemblePrediction]
    provenance: ChainProvenance
    scenario: Optional[Scenario] = None
    U_L: Optional[float] = None
    overpotential: Optional[float] = None
    pds: Optional[str] = None
    v_cell: Optional[float] = None
    v_cell_sd: Optional[float] = None
    relative_score: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def extrapolating(self) -> bool:
        """True if any intermediate used here fell outside its training domain."""
        return any(p.warning() is not None for p in self.predictions.values())

    def unsupported_species(self) -> List[str]:
        return sorted(sp for sp, p in self.predictions.items()
                      if p.warning() is not None)


def run_chain(comp: Composition,
              models: Dict[str, IntermediateModel],
              base: Scenario,
              frame: Optional[ReferenceFrame] = None,
              product: str = "CO",
              v_baseline: float = 2.0,
              n_samples: int = 500,
              fixed_site1: Optional[str] = None,
              seed: int = 0) -> ChainResult:
    """composition -> predicted E_ads -> (reference frame) -> V_cell -> Scenario.

    Faradaic efficiency is NEVER derived here: no descriptor->selectivity model
    exists, so `base.faradaic_efficiency` is carried through unchanged and
    recorded in the provenance as assumed. That is the single most important
    honesty property of this function.
    """
    frame = frame or ReferenceFrame()
    preds = predict_composition(comp, models, n_samples=n_samples,
                                fixed_site1=fixed_site1, seed=seed)

    prov = ChainProvenance(reference_mode=frame.mode)
    prov.origins["faradaic_efficiency"] = ASSUMED
    prov.notes.append(sro_note())
    prov.notes.append("Faradaic efficiency is not predicted: no descriptor->"
                      "selectivity model exists. It remains your input.")

    res = ChainResult(composition=comp, predictions=preds, provenance=prov)

    # applicability domain: surface every extrapolating intermediate before any
    # number derived from it is shown
    for sp in sorted(preds):
        w = preds[sp].warning()
        if w:
            res.warnings.append(f"*{sp}: {w}")
            prov.notes.append(f"*{sp} is out of the training domain.")

    e_ads = {sp: p.mean for sp, p in preds.items()}

    if not frame.gives_absolute_U_L():
        prov.origins["cell_voltage"] = ASSUMED
        prov.notes.append(
            "Reference frame is 'relative': no absolute limiting potential is "
            "claimed, so cell voltage is not derived from DFT. Configurations "
            "are still ranked correctly against each other.")
        prov.notes.append(
            "Relative ranking assumes the first PCET step is potential-"
            "determining for every composition compared. That assumption cannot "
            "be checked without a reference frame: the second step's free energy "
            "depends on the DIFFERENCE of two unknown constants, which the data "
            "does not determine. Use pds_uniform() once you have a frame.")
        first_product = PATHWAYS[product][0][1]
        if first_product in e_ads:
            # lower (more negative) E_ads of the first intermediate = more active,
            # exactly determined by the data up to the common unknown constant
            res.relative_score = -e_ads[first_product]
            if preds[first_product].warning():
                res.warnings.append(
                    f"the relative activity score rests on *{first_product}, "
                    f"which is extrapolating — the ranking is unreliable")
        res.scenario = base
        return res

    che = apply_reference(e_ads, frame, product)
    lp = limiting_potential(che, product, corrections=None)
    if lp is None:
        prov.origins["cell_voltage"] = ASSUMED
        prov.notes.append(f"the {product} pathway is not computable from the "
                          f"available intermediates {sorted(e_ads)}")
        res.scenario = base
        return res

    v = proxy_cell_voltage(lp["overpotential"], v_baseline)
    # propagate the dominant intermediate's uncertainty into the voltage
    pds_species = lp["pds"].split("->")[1]
    sd = preds[pds_species].total_sd if pds_species in preds else 0.0

    res.U_L, res.overpotential, res.pds = lp["U_L"], lp["overpotential"], lp["pds"]
    res.v_cell, res.v_cell_sd = v, float(sd)
    prov.origins["cell_voltage"] = DFT

    # a single anchor determines only the anchored species' reference shift, so
    # it is only sufficient while the FIRST PCET step limits (see _anchor_shift)
    first_step = f"CO2->{PATHWAYS[product][0][1]}"
    if frame.mode == "anchored" and lp["pds"] != first_step:
        res.warnings.append(
            f"the potential-determining step here is {lp['pds']}, not {first_step}. "
            f"A single anchor fixes only one species' reference shift, and the "
            f"second step depends on the difference of two independent shifts — "
            f"so this U_L is not determined by the anchor. Use mode='absolute' "
            f"with your own gas-phase energies.")
        prov.notes.append("anchored mode is under-determined for this configuration")

    # a voltage derived from an extrapolating intermediate must say so, loudly:
    # it propagates straight into the headline MAC
    ood = [sp for sp in lp["pds"].split("->") if sp in preds and preds[sp].warning()]
    if ood:
        res.warnings.append(
            f"cell voltage is derived from the potential-determining step "
            f"{lp['pds']}, which uses extrapolating intermediate(s) "
            f"{', '.join(ood)} — the resulting MAC is not supported by the data")
        prov.origins["cell_voltage"] = DFT + " (EXTRAPOLATING)"
    if frame.mode == "anchored":
        prov.notes.append(
            f"Absolute U_L rests on a single anchor ({frame.anchor_source or 'unspecified'}); "
            f"an error in the anchor shifts every configuration equally.")

    res.scenario = _with_cell_voltage(base, v)
    return res


def _with_cell_voltage(base: Scenario, v_cell: float) -> Scenario:
    import dataclasses
    return dataclasses.replace(base, cell_voltage=float(v_cell))


# ---------------------------------------------------------------------------
# end-to-end validation on configurations where every intermediate was computed
# ---------------------------------------------------------------------------
def _sheet_excluding(sd: SheetData, drop_keys) -> SheetData:
    """Copy of a sheet with the given configuration keys removed."""
    drop = set(drop_keys)
    keep = [i for i, k in enumerate(sd.keys) if k not in drop]
    idx = np.asarray(keep, int)
    return SheetData(species=sd.species, X=sd.X[idx], y=sd.y[idx],
                     feature_names=list(sd.feature_names),
                     keys=[sd.keys[i] for i in keep],
                     site1=[sd.site1[i] for i in keep],
                     warnings=list(sd.warnings))


@dataclass
class PathwayValidation:
    """How well a surrogate-driven U_L reproduces the DFT-computed one."""
    n: int
    species: List[str]
    e_ads_rmse: Dict[str, float]
    u_l_true: np.ndarray
    u_l_pred: np.ndarray
    u_l_rmse: float
    u_l_bias: float
    u_l_rank_corr: float
    u_l_slope: float          # regression of predicted on true; < 1 = compressed
    pds_agreement: float
    amplification: Dict[str, float]      # U_L error / that species' E_ads error
    n_train_after_holdout: Dict[str, int]

    def summary(self) -> str:
        eads = ", ".join(f"*{k} {v:.3f}" for k, v in sorted(self.e_ads_rmse.items()))
        return (f"n={self.n} held-out configurations | E_ads RMSE (eV): {eads} | "
                f"U_L RMSE {self.u_l_rmse:.3f} V, bias {self.u_l_bias:+.3f} V, "
                f"slope {self.u_l_slope:.2f}, rank corr {self.u_l_rank_corr:.3f}, "
                f"PDS agreement {self.pds_agreement:.0%}")

    def compression_note(self) -> Optional[str]:
        """Warning about range compression, or None if predictions are faithful.

        A slope below 1 means predicted U_L spans a narrower range than the true
        one, so good catalysts look worse than they are and bad ones look better.
        This is regression toward the mean and is set by how much of the variance
        the descriptors can explain -- on the published workbook every model class
        tried (Bayesian linear, random forest, gradient boosting, GP with ARD)
        gave slope 0.61-0.69 and RMSE ~0.10 eV. That agreement is the point: the
        ceiling is the DESCRIPTOR SET, not the regressor, so switching model is
        not the fix. Richer features (geometry, d-band centre, coordination) are.
        """
        if not np.isfinite(self.u_l_slope) or self.u_l_slope >= 0.9:
            return None
        return (f"Predicted U_L spans only {self.u_l_slope:.0%} of the true range "
                f"(regression toward the mean). Divide by {self.u_l_slope:.2f} to "
                f"de-attenuate a spread, but do NOT do so for a single prediction: "
                f"it would inflate the error. Rankings are unaffected; the "
                f"magnitude of any predicted difference is understated by about "
                f"{1 / self.u_l_slope:.1f}x.")


def validate_pathway(sheets: Dict[str, SheetData],
                     frame: "ReferenceFrame",
                     product: str = "CO",
                     seed: int = 0) -> PathwayValidation:
    """Hold out every configuration for which ALL required intermediates were
    computed, retrain on the remainder, and check whether the surrogate-driven
    limiting potential reproduces the DFT one.

    This is the only test that exercises the WHOLE chain on real data --
    including the max() over PCET steps, which is where per-species errors can
    amplify: an error in whichever step is limiting passes straight into U_L,
    and an error large enough to swap which step limits produces a
    discontinuous jump.

    Retraining after the hold-out is essential. The joined configurations are
    part of every sheet, so evaluating without removing them would measure
    memorisation, not generalisation.
    """
    if not frame.gives_absolute_U_L():
        raise ValueError(
            "validation needs a reference frame that yields an absolute U_L "
            "('anchored' or 'absolute'); in 'relative' mode there is no U_L to "
            "compare against.")

    needed = sorted({s for step in PATHWAYS[product] for s in step} - {"CO2"})
    missing = [s for s in needed if s not in sheets]
    if missing:
        raise KeyError(f"the {product} pathway needs {needed}; missing {missing}")

    rows, _ = join_intermediates(sheets, needed)
    if not rows:
        raise ValueError(f"no configuration carries all of {needed}")
    hold = [r["key"] for r in rows]

    trimmed = {sp: _sheet_excluding(sheets[sp], hold) for sp in needed}
    models = train_intermediate_models(trimmed)

    # predict each intermediate on the held-out descriptor vectors
    pred: Dict[str, np.ndarray] = {}
    true: Dict[str, np.ndarray] = {}
    X_hold = np.asarray([[r["descriptors"][c] for c in sheets[needed[0]].feature_names]
                         for r in rows], float)
    for sp in needed:
        im = models[sp]
        mu, _ = im.predict(align_to_training_columns(X_hold, im.feature_names))
        pred[sp] = np.asarray(mu, float)
        true[sp] = np.asarray([r["energies"][sp] for r in rows], float)

    u_true, u_pred, pds_true, pds_pred = [], [], [], []
    for i in range(len(rows)):
        lp_t = limiting_potential(apply_reference({s: float(true[s][i]) for s in needed},
                                                  frame, product), product)
        lp_p = limiting_potential(apply_reference({s: float(pred[s][i]) for s in needed},
                                                  frame, product), product)
        if lp_t is None or lp_p is None:
            continue
        u_true.append(lp_t["U_L"]); u_pred.append(lp_p["U_L"])
        pds_true.append(lp_t["pds"]); pds_pred.append(lp_p["pds"])

    u_true = np.asarray(u_true, float)
    u_pred = np.asarray(u_pred, float)
    err = u_pred - u_true

    e_rmse = {sp: float(np.sqrt(np.mean((pred[sp] - true[sp]) ** 2))) for sp in needed}
    u_rmse = float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan")

    if len(u_true) > 2 and np.std(u_true) > 0 and np.std(u_pred) > 0:
        rt = np.argsort(np.argsort(u_true))
        rp = np.argsort(np.argsort(u_pred))
        rank_corr = float(np.corrcoef(rt, rp)[0, 1])
        slope = float(np.polyfit(u_true, u_pred, 1)[0])
    else:
        rank_corr, slope = float("nan"), float("nan")

    return PathwayValidation(
        n=len(u_true), species=needed, e_ads_rmse=e_rmse,
        u_l_true=u_true, u_l_pred=u_pred, u_l_rmse=u_rmse,
        u_l_bias=float(np.mean(err)) if len(err) else float("nan"),
        u_l_rank_corr=rank_corr, u_l_slope=slope,
        pds_agreement=float(np.mean([a == b for a, b in zip(pds_true, pds_pred)]))
        if pds_true else float("nan"),
        amplification={sp: (u_rmse / e_rmse[sp]) if e_rmse[sp] > 1e-12 else float("inf")
                       for sp in needed},
        n_train_after_holdout={sp: len(trimmed[sp]) for sp in needed})


def applicability_report(models: Dict[str, IntermediateModel]) -> Dict[str, Dict]:
    """Which adsorption-site elements each trained intermediate actually covers.

    Answers 'what can this workbook support?' before you predict anything. On the
    published file the *CO sheet has no Cu-terminated sites, so every Cu-bearing
    composition extrapolates for that intermediate.
    """
    all_elements = set()
    for im in models.values():
        all_elements |= set(im.site1_support)
    return {sp: {"n_train": im.n_train,
                 "site1_support": sorted(im.site1_support),
                 "missing_site1": sorted(all_elements - set(im.site1_support)),
                 "n_by_site1": dict(sorted(im.n_by_site1.items())),
                 "train_sd_eV": im.train_sd_eV}
            for sp, im in sorted(models.items())}


def pds_uniform(results: Sequence[ChainResult]) -> bool:
    """True if every result that has a potential-determining step shares the same
    one. When this holds, the relative (reference-free) ranking and the absolute
    ranking agree; when it does not, they can differ and only the absolute one is
    meaningful."""
    seen = {r.pds for r in results if r.pds is not None}
    return len(seen) <= 1


def rank_compositions(comps: Sequence[Composition],
                      models: Dict[str, IntermediateModel],
                      base: Scenario,
                      frame: Optional[ReferenceFrame] = None,
                      product: str = "CO",
                      **kw) -> List[ChainResult]:
    """Run the chain over several compositions and sort best-first.

    Sorting uses absolute U_L when the frame provides it, otherwise the
    reference-free relative score. The two orderings coincide only while the
    potential-determining step is common to all compositions -- if it is not,
    every result carries a note saying so, because the relative ordering is then
    no longer trustworthy.
    """
    out = [run_chain(c, models, base, frame, product, **kw) for c in comps]
    if not pds_uniform(out):
        for r in out:
            r.provenance.notes.append(
                "The potential-determining step is NOT the same for every "
                "composition compared, so a reference-free ranking would be "
                "misleading here; this ordering uses absolute U_L.")
    def key(r: ChainResult):
        if r.U_L is not None:
            return -r.U_L                      # less negative U_L = more active
        return -(r.relative_score if r.relative_score is not None else -np.inf)
    return sorted(out, key=key)
