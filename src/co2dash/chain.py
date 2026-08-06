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
from .hea import SheetData, to_che_formation_energies
from .proxy import PATHWAYS, limiting_potential, proxy_cell_voltage
from .surrogate import BayesianLinearSurrogate
from .techno_economic import Scenario

REFERENCE_MODES = ("relative", "anchored", "absolute")


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

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.model.predict(X)


def train_intermediate_models(sheets: Dict[str, SheetData],
                              species: Optional[Sequence[str]] = None
                              ) -> Dict[str, IntermediateModel]:
    """One surrogate per intermediate, each trained on its own sheet.

    Training per sheet rather than on the sheets' intersection is deliberate:
    the sheets share few configurations, so intersecting would discard most of
    the data. The intersection is better spent as a validation set.
    """
    want = list(species) if species is not None else sorted(sheets)
    out: Dict[str, IntermediateModel] = {}
    for sp in want:
        sd = sheets[sp]
        model = BayesianLinearSurrogate().fit_evidence(sd.X, sd.y)
        out[sp] = IntermediateModel(
            species=sp, model=model, feature_names=list(sd.feature_names),
            n_train=len(sd), sigma_eV=float(1.0 / np.sqrt(model.beta)))
    return out


# ---------------------------------------------------------------------------
# composition -> predicted adsorption energies
# ---------------------------------------------------------------------------
@dataclass
class EnsemblePrediction:
    """Predicted E_ads over the configurational ensemble of one composition."""
    species: str
    mean: float                  # ensemble mean of the predictive means
    configurational_sd: float    # spread across configurations
    model_sd: float              # mean predictive (model) uncertainty
    samples: np.ndarray          # per-configuration predictive means

    @property
    def total_sd(self) -> float:
        """Configurational and model uncertainty added in quadrature. They are
        different things and are kept separate above; this is for propagation."""
        return float(np.sqrt(self.configurational_sd ** 2 + self.model_sd ** 2))


def predict_composition(comp: Composition,
                        models: Dict[str, IntermediateModel],
                        n_samples: int = 500,
                        fixed_site1: Optional[str] = None,
                        seed: int = 0) -> Dict[str, EnsemblePrediction]:
    """Predict every trained intermediate for one composition."""
    X_gen, _ = descriptors_for_composition(comp, n_samples=n_samples,
                                           fixed_site1=fixed_site1, seed=seed)
    out: Dict[str, EnsemblePrediction] = {}
    for sp, im in models.items():
        X = align_to_training_columns(X_gen, im.feature_names)
        mu, sd = im.predict(X)
        out[sp] = EnsemblePrediction(
            species=sp, mean=float(np.mean(mu)),
            configurational_sd=float(np.std(mu)),
            model_sd=float(np.mean(sd)), samples=np.asarray(mu, float))
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
    """Solve the single constant that makes the anchor reproduce its known U_L.

    Valid while the potential-determining step is the first PCET (CO2->X), which
    holds across the plausible shift range for the CO pathway; `apply_reference`
    re-checks the PDS after shifting and flags if it moved.
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
    if frame.mode == "anchored":
        prov.notes.append(
            f"Absolute U_L rests on a single anchor ({frame.anchor_source or 'unspecified'}); "
            f"an error in the anchor shifts every configuration equally.")

    res.scenario = _with_cell_voltage(base, v)
    return res


def _with_cell_voltage(base: Scenario, v_cell: float) -> Scenario:
    import dataclasses
    return dataclasses.replace(base, cell_voltage=float(v_cell))


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
