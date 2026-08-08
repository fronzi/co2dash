"""
User data intake.

Turn a user's own spreadsheet of measurements/calculations into a validated
Scenario, filling unmeasured fields from sourced defaults (see defaults.py).

Design goals for a tool used by non-authors of the code:
  * accept messy column names (aliases, case/space/percent-insensitive)
  * normalise obvious unit slips (FE as %, voltage in mV, grid in g/kWh)
  * validate ranges and report clear warnings/errors (never silently accept junk)
  * always show provenance: which fields came from the user vs a sourced default

No pandas dependency: a table is a list[dict] (use the stdlib csv reader, or
pass rows directly from the GUI's uploaded file).
"""
from __future__ import annotations
import csv
import io
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .schema import Reaction, RXN_CO, RXN_METHANOL, RXN_FORMATE
from .techno_economic import Scenario, capex_from_current_density
from .defaults import defaults_for

# ------------------------------------------------------------------ aliases
# canonical field  ->  accepted header spellings (compared lowercased, stripped)
COLUMN_ALIASES: Dict[str, List[str]] = {
    "material_id":      ["material", "material_id", "catalyst", "id", "name", "sample", "system"],
    "product":          ["product", "reaction", "target", "route"],
    "faradaic_efficiency": ["fe", "faradaic_efficiency", "faradaic efficiency",
                            "fe_co", "feco", "fe co", "selectivity", "fe%", "fe (%)"],
    "cell_voltage":     ["cell_voltage", "voltage", "v_cell", "vcell", "ecell",
                         "cell voltage", "potential", "v"],
    "current_density":  ["current_density", "current density", "j", "cd",
                         "ma/cm2", "ma cm-2", "ma cm2", "j (ma/cm2)"],
    "c_elec":           ["c_elec", "electricity_price", "electricity price",
                         "electricity cost", "power price"],
    "grid_intensity":   ["grid_intensity", "grid", "carbon_intensity",
                         "grid intensity", "ci"],
    "c_co2":            ["c_co2", "co2_cost", "capture_cost", "co2 cost", "capture cost"],
    "capex_total":      ["capex", "capex_total", "capital", "capex ($)", "capex_m"],
    "lcop_conventional": ["conventional_price", "market_price", "product_price",
                          "conventional price", "market price"],
    "release_fraction": ["release_fraction", "phi", "release", "eol_release"],
}

_HEADER_LOOKUP = {alias: canon for canon, al in COLUMN_ALIASES.items() for alias in al}

# Fields that describe the PLANT or the SITE rather than the catalyst. A CSV may
# legitimately carry them, but when they contradict a loaded scenario the result
# is a hybrid, so the clash is reported rather than resolved silently.
_CONTEXT_FIELDS = {"c_elec", "grid_intensity", "c_co2", "capex_total",
                   "lcop_conventional", "release_fraction"}

REACTION_ALIASES: Dict[str, Reaction] = {
    "co": RXN_CO, "carbon monoxide": RXN_CO,
    "methanol": RXN_METHANOL, "ch3oh": RXN_METHANOL, "meoh": RXN_METHANOL,
    "formate": RXN_FORMATE, "formic": RXN_FORMATE, "formic acid": RXN_FORMATE,
    "hcooh": RXN_FORMATE,
}
_REACTION_KEY = {id(RXN_CO): "co", id(RXN_METHANOL): "methanol", id(RXN_FORMATE): "formate"}

# plausibility ranges for validation (canonical units)
_RANGES = {
    "faradaic_efficiency": (0.0, 1.0),
    "cell_voltage": (1.2, 6.0),
    "current_density": (0.0, 3000.0),
    "c_elec": (0.0, 0.5),
    "grid_intensity": (0.0, 2.0),
    "c_co2": (0.0, 1.0),
    "capex_total": (1.0e5, 1.0e10),
    "lcop_conventional": (0.0, 10.0),
    "release_fraction": (0.0, 1.0),
}


@dataclass
class IntakeResult:
    scenario: Scenario
    reaction_key: str
    provenance: Dict[str, str]              # field -> 'user' or 'default: <source>'
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    material_id: str = ""                   # so a per-row verdict can name its subject

    @property
    def ok(self) -> bool:
        return not self.errors


# ------------------------------------------------------------------ mapping
def map_columns(headers: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """Map raw headers to canonical field names.
    Returns (mapping {raw_header: canonical}, list of unrecognised headers)."""
    mapping, unknown = {}, []
    for h in headers:
        key = _HEADER_LOOKUP.get(str(h).strip().lower())
        if key:
            mapping[h] = key
        else:
            unknown.append(h)
    return mapping, unknown


def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().replace("%", "").replace(",", "")
    if s == "" or s.lower() in ("na", "nan", "none", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def normalise_units(field_name: str, value: float, raw: str) -> Tuple[float, Optional[str]]:
    """Fix obvious unit slips. Returns (value, warning|None)."""
    w = None
    if field_name == "faradaic_efficiency":
        if value > 1.5:                       # entered as a percentage
            value /= 100.0
            w = f"FE {raw!r} read as percent -> {value:.3f}"
    elif field_name == "cell_voltage":
        if value > 20:                        # entered in mV
            value /= 1000.0
            w = f"cell_voltage {raw!r} read as mV -> {value:.3f} V"
    elif field_name == "grid_intensity":
        if value > 3.0:                       # entered in g/kWh
            value /= 1000.0
            w = f"grid_intensity {raw!r} read as g/kWh -> {value:.3f} kg/kWh"
    elif field_name == "current_density":
        if value < 0:
            value = abs(value)
    return value, w


def resolve_reaction(row: Dict[str, object], default_key: str = "co") -> str:
    raw = row.get("product")
    if raw is not None:
        key = REACTION_ALIASES.get(str(raw).strip().lower())
        if key is not None:
            return _REACTION_KEY[id(key)]
    return default_key


# ------------------------------------------------------------------ build
def row_to_scenario(row: Dict[str, object], default_reaction: str = "co",
                    base: Optional[Scenario] = None) -> IntakeResult:
    """Build a validated Scenario from a single mapped row (keys already
    canonical).

    Fill order for anything the CSV does not supply:
        1. the user's row               -> provenance "user"
        2. `base`, if given             -> provenance "scenario: <label>"
        3. the generic sourced defaults -> provenance "default: <source>"

    `base` is normally the Scenario loaded from a tier-tagged YAML. Without it,
    a user who had carefully declared a plant and a grid in YAML would still see
    their CSV rows evaluated against the generic defaults — a different plant
    from the one on screen, with no indication that the two disagreed.
    """
    warnings: List[str] = []
    errors: List[str] = []

    rkey = resolve_reaction(row, default_reaction)
    rxn = REACTION_ALIASES[rkey]
    defaults = defaults_for(rkey)

    provenance: Dict[str, str] = {}
    vals: Dict[str, float] = {}

    # user-provided numeric fields
    user_fields = ["faradaic_efficiency", "cell_voltage", "current_density",
                   "c_elec", "grid_intensity", "c_co2", "capex_total",
                   "lcop_conventional", "release_fraction"]
    for f in user_fields:
        if f in row and row[f] not in (None, ""):
            v = _to_float(row[f])
            if v is None:
                errors.append(f"{f}: could not parse {row[f]!r} as a number")
                continue
            v, w = normalise_units(f, v, str(row[f]))
            if w:
                warnings.append(w)
            lo, hi = _RANGES.get(f, (-math.inf, math.inf))
            if not (lo <= v <= hi):
                (errors if f in ("faradaic_efficiency", "release_fraction")
                 else warnings).append(
                    f"{f}={v:g} outside plausible range [{lo:g}, {hi:g}]")
            vals[f] = v
            provenance[f] = "user"
            # A CSV column silently beating a declared scenario is fine for the
            # catalyst's own KPIs, but c_elec / grid_intensity / capex and the
            # like describe the PLANT and the SITE, not the material. Overriding
            # those without saying so produces a hybrid that matches neither the
            # scenario on screen nor anything the user intended.
            if base is not None and hasattr(base, f):
                b = float(getattr(base, f))
                if not math.isclose(b, v, rel_tol=1e-9, abs_tol=1e-12):
                    tag = ("context, not a catalyst property"
                           if f in _CONTEXT_FIELDS else "measurement")
                    warnings.append(
                        f"{f}: your CSV column ({v:g}) overrides the loaded "
                        f"scenario ({b:g}) — {tag}")

    # fields the engine needs; fill from defaults where the user didn't give them
    needed = ["faradaic_efficiency", "cell_voltage", "capex_total",
              "annual_production_kg", "opex_fix_per_yr", "disc_rate", "lifetime_yr",
              "c_co2", "c_elec", "lcop_conventional", "grid_intensity",
              "e_capture", "e_process", "release_fraction", "rectifier_eff"]
    for f in needed:
        if f in vals:
            continue
        if base is not None and hasattr(base, f):
            vals[f] = float(getattr(base, f))
            provenance[f] = "scenario: from the loaded YAML"
            continue
        q = defaults.get(f)
        if q is None:
            errors.append(f"no default available for required field {f}")
            continue
        vals[f] = q.value
        provenance[f] = f"default: {q.source}"

    # Current density decides how much electrode you must buy, so it belongs in
    # the capital, not just in the metadata. It used to be parsed, validated and
    # then dropped -- a user supplying it reasonably expected it to matter.
    if "current_density" in vals and vals["current_density"] > 0:
        cap = capex_from_current_density(
            vals["annual_production_kg"], rxn.n_electrons, rxn.molar_mass_prod,
            vals["faradaic_efficiency"], vals["current_density"])
        if provenance.get("capex_total") == "user":
            ratio = cap["total_capex_usd"] / max(vals["capex_total"], 1.0)
            if ratio > 2.0 or ratio < 0.5:
                warnings.append(
                    f"your CAPEX ({vals['capex_total']:.3g} $) and your current "
                    f"density ({vals['current_density']:g} mA/cm²) disagree: the "
                    f"area method implies {cap['total_capex_usd']:.3g} $ "
                    f"({cap['area_m2']:.0f} m² of electrode). Yours is kept.")
        else:
            vals["capex_total"] = cap["total_capex_usd"]
            provenance["capex_total"] = (
                f"from current density: {cap['area_m2']:.0f} m² × "
                f"{cap['cost_per_m2']:.0f} $/m² × {cap['bop_multiple']:g} "
                f"(Jouny 2018 area method)")
            provenance["electrode_area_m2"] = f"{cap['area_m2']:.0f}"

    if "faradaic_efficiency" not in provenance or provenance["faradaic_efficiency"] != "user":
        warnings.append("faradaic_efficiency not provided by user — using a default; "
                        "results are illustrative until you enter your measured FE")

    scen = Scenario(
        n_electrons=rxn.n_electrons, molar_mass_prod=rxn.molar_mass_prod,
        m_co2=rxn.kg_co2_per_kg_prod, m_h2=rxn.kg_h2_per_kg_prod,
        faradaic_efficiency=vals["faradaic_efficiency"], cell_voltage=vals["cell_voltage"],
        capex_total=vals["capex_total"], annual_production_kg=vals["annual_production_kg"],
        opex_fix_per_yr=vals["opex_fix_per_yr"], disc_rate=vals["disc_rate"],
        lifetime_yr=vals["lifetime_yr"], c_co2=vals["c_co2"], c_elec=vals["c_elec"],
        c_h2=defaults["c_h2"].value if rxn.kg_h2_per_kg_prod > 0 else 0.0,
        lcop_conventional=vals["lcop_conventional"], grid_intensity=vals["grid_intensity"],
        e_capture=vals["e_capture"], e_process=vals["e_process"],
        release_fraction=vals["release_fraction"], rectifier_eff=vals["rectifier_eff"])

    return IntakeResult(scenario=scen, reaction_key=rkey,
                        provenance=provenance, warnings=warnings, errors=errors,
                        material_id=str(row.get("material_id", "") or ""))


def read_csv(text_or_buffer) -> List[Dict[str, object]]:
    """Read a CSV (string or file-like) into canonical-keyed rows.
    Unrecognised columns are dropped (and available via map_columns for display)."""
    if hasattr(text_or_buffer, "read"):
        raw = text_or_buffer.read()
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    else:
        text = text_or_buffer
    reader = csv.DictReader(io.StringIO(text))
    mapping, _ = map_columns(reader.fieldnames or [])
    rows = []
    for r in reader:
        rows.append({mapping[h]: v for h, v in r.items() if h in mapping})
    return rows


def ingest_table(text_or_buffer, default_reaction: str = "co",
                 base: Optional[Scenario] = None) -> List[IntakeResult]:
    """Full path: CSV -> list of validated IntakeResults (one per row).

    Pass `base` (e.g. the Scenario from a loaded YAML) so that economic and
    environmental context comes from that scenario rather than the generic
    defaults; the CSV then supplies only what it actually measures.
    """
    return [row_to_scenario(r, default_reaction, base) for r in read_csv(text_or_buffer)]
