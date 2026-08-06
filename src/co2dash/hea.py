"""
High-entropy-alloy DFT descriptor workbook loader (CoCuFeMoNi).

Reads the multi-sheet .xlsx supplement in which each sheet is one adsorbed
intermediate (*CO, *CHO, *COOH) and each row is one alloy configuration
described by 10 surface sites x 4 elemental descriptors, plus the DFT
adsorption energy.

    columns: Labels | Site k Group | Site k Period | Site k EN | Site k Nied
             (k = 1..10)                                       | Eads (eV)

WHY THIS MODULE EXISTS
----------------------
The app previously read ONE sheet at a time, so it could never assemble the
{*CO, *COOH} pair that `proxy.limiting_potential` needs for the CO pathway.
Consequently the DFT data could never reach the techno-economic verdict. This
module loads all sheets and joins them per configuration, producing exactly the
`{species: energy}` mapping the CHE proxy consumes.

HONESTY NOTES
-------------
* `Labels` duplicates `Eads (eV)` in the published file. It is NOT a material
  name and is dropped from the feature matrix -- using it would be target
  leakage. `assert_no_leakage()` checks this explicitly.
* Sheets do NOT share a common configuration set. The join is partial and its
  size is reported, never imputed. On the published file the CO n COOH overlap
  is small (order tens of configurations), so `join_intermediates` is intended
  as a VALIDATION set; the production route is to predict each intermediate with
  its own surrogate and combine the predictions.
* Site 1 is treated as the adsorption site and sites 2..10 as its environment.
  This follows the column ordering of the published file; verify against your
  own generation script before relying on the per-element summaries.
* `Nied` is interpreted as the number of unpaired d electrons purely for
  human-readable element decoding. Nothing in the numerical pipeline depends on
  that interpretation -- decoding is cosmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# element decoding (cosmetic: labels only, never used as a feature)
# --------------------------------------------------------------------------
# (Group, Period, electronegativity, unpaired d electrons) -> symbol
ELEMENT_BY_DESCRIPTOR: Dict[Tuple[int, int, float, int], str] = {
    (8, 4, 1.83, 4): "Fe",
    (9, 4, 1.88, 3): "Co",
    (10, 4, 1.91, 2): "Ni",
    (11, 4, 1.90, 0): "Cu",
    (6, 5, 2.16, 5): "Mo",
}

TARGET_COL = "Eads (eV)"
LABEL_COL = "Labels"

# sheet name -> CHE species symbol used by proxy.PATHWAYS
SHEET_TO_SPECIES = {"CO": "CO", "CHO": "CHO", "COOH": "COOH",
                    "OCHO": "OCHO", "CH2O": "CH2O", "OCH3": "OCH3",
                    "H": "H", "OH": "OH"}


def decode_site(group, period, en, nied) -> str:
    """(Group, Period, EN, Nied) -> element symbol, or '?' if unrecognised."""
    key = (int(group), int(period), round(float(en), 2), int(nied))
    return ELEMENT_BY_DESCRIPTOR.get(key, "?")


def _feature_columns(columns: Sequence[str]) -> List[str]:
    """Site descriptor columns, in file order. Excludes Labels and the target."""
    return [c for c in columns if c not in (LABEL_COL, TARGET_COL)]


def assert_no_leakage(df, tol: float = 1e-9) -> Optional[str]:
    """Return a warning string if `Labels` duplicates the target (it does in the
    published file), else None. The caller drops `Labels` either way; this makes
    the reason explicit rather than tacit."""
    if LABEL_COL not in df.columns or TARGET_COL not in df.columns:
        return None
    a = df[LABEL_COL].to_numpy(dtype=float, copy=False)
    b = df[TARGET_COL].to_numpy(dtype=float, copy=False)
    if a.shape == b.shape and np.allclose(a, b, atol=tol, equal_nan=True):
        return (f"'{LABEL_COL}' duplicates '{TARGET_COL}' -- dropped from the "
                f"feature matrix to avoid target leakage.")
    return None


# --------------------------------------------------------------------------
# per-sheet payload
# --------------------------------------------------------------------------
@dataclass
class SheetData:
    """One intermediate: feature matrix, target, and per-row configuration keys."""
    species: str
    X: np.ndarray                      # (n, d) site descriptors
    y: np.ndarray                      # (n,) Eads in eV
    feature_names: List[str]
    keys: List[Tuple[float, ...]]      # hashable configuration fingerprints
    site1: List[str]                   # decoded adsorption-site element
    warnings: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    def degeneracy(self) -> Dict[str, float]:
        """Same descriptor vector, different Eads -> an irreducible (aleatoric)
        error floor for any model built on these features alone."""
        buckets: Dict[Tuple[float, ...], List[float]] = {}
        for k, v in zip(self.keys, self.y):
            buckets.setdefault(k, []).append(float(v))
        spreads = [max(v) - min(v) for v in buckets.values() if len(v) > 1]
        return {"n_degenerate": len(spreads),
                "mean_spread_eV": float(np.mean(spreads)) if spreads else 0.0,
                "max_spread_eV": float(max(spreads)) if spreads else 0.0}


def _row_keys(X: np.ndarray, decimals: int = 6) -> List[Tuple[float, ...]]:
    """Hashable fingerprint per configuration. Rounded so that float noise in the
    spreadsheet does not split configurations that are physically identical."""
    return [tuple(r) for r in np.round(X, decimals)]


def load_sheet(df, species: str) -> SheetData:
    """DataFrame for one intermediate -> SheetData. Rows with any non-finite
    feature or target are dropped and counted (no imputation)."""
    import pandas as pd  # local import: pandas is only needed for .xlsx paths

    warnings: List[str] = []
    leak = assert_no_leakage(df)
    if leak:
        warnings.append(leak)

    if TARGET_COL not in df.columns:
        raise KeyError(f"sheet '{species}' has no '{TARGET_COL}' column; "
                       f"found {list(df.columns)[:6]}...")

    feats = _feature_columns(df.columns)
    X = df[feats].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").to_numpy(float)

    ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if not ok.all():
        warnings.append(f"dropped {int((~ok).sum())} row(s) with missing values")
    X, y = X[ok], y[ok]

    site_cols = ["Site 1 Group", "Site 1 Period", "Site 1 EN", "Site 1 Nied"]
    if all(c in df.columns for c in site_cols):
        sub = df.loc[ok, site_cols].to_numpy()
        site1 = [decode_site(*row) for row in sub]
        if "?" in site1:
            warnings.append(f"{site1.count('?')} site-1 descriptor(s) did not "
                            f"match a known element")
    else:
        site1 = ["?"] * len(y)

    return SheetData(species=species, X=X, y=y, feature_names=feats,
                     keys=_row_keys(X), site1=site1, warnings=warnings)


def load_workbook(source, sheets: Optional[Sequence[str]] = None) -> Dict[str, SheetData]:
    """Load every intermediate sheet from an HEA workbook.

    `source` may be a path, a file-like object, or raw bytes (Streamlit upload).
    Returns {species: SheetData}. Sheets whose name is not a recognised species
    are skipped.
    """
    import io
    import pandas as pd

    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    xl = pd.ExcelFile(source)
    names = list(sheets) if sheets is not None else xl.sheet_names

    out: Dict[str, SheetData] = {}
    for name in names:
        species = SHEET_TO_SPECIES.get(str(name).strip())
        if species is None:
            continue
        out[species] = load_sheet(xl.parse(name), species)
    if not out:
        raise ValueError(f"no recognised intermediate sheets in {list(xl.sheet_names)}; "
                         f"expected any of {sorted(SHEET_TO_SPECIES)}")
    return out


# --------------------------------------------------------------------------
# the join: per-configuration intermediate energies
# --------------------------------------------------------------------------
@dataclass
class JoinReport:
    n_per_species: Dict[str, int]
    n_joined: int
    species: List[str]
    duplicate_keys: Dict[str, int]

    def summary(self) -> str:
        per = ", ".join(f"{k}={v}" for k, v in sorted(self.n_per_species.items()))
        return (f"joined {self.n_joined} configuration(s) carrying all of "
                f"{'+'.join(self.species)} (per-sheet: {per})")


def join_intermediates(sheets: Dict[str, SheetData],
                       species: Optional[Sequence[str]] = None,
                       reduce: str = "mean") -> Tuple[List[Dict], JoinReport]:
    """Inner-join configurations present in ALL requested species sheets.

    Returns (rows, report) where each row is

        {"key", "descriptors": {name: value}, "site1",
         "energies": {species: Eads_eV}}

    `energies` is exactly the mapping `proxy.limiting_potential` consumes.

    Configurations appearing more than once within a sheet (the descriptor
    vector is degenerate) are reduced by `reduce` ('mean' or 'min'; 'min' =
    strongest binding, the usual CHE convention). No imputation: a configuration
    missing from any requested sheet is dropped.
    """
    want = list(species) if species is not None else sorted(sheets)
    missing = [s for s in want if s not in sheets]
    if missing:
        raise KeyError(f"species {missing} not present; have {sorted(sheets)}")

    agg = np.mean if reduce == "mean" else np.min

    per_key: Dict[str, Dict[Tuple[float, ...], List[float]]] = {}
    dup_counts: Dict[str, int] = {}
    for s in want:
        sd = sheets[s]
        d: Dict[Tuple[float, ...], List[float]] = {}
        for k, v in zip(sd.keys, sd.y):
            d.setdefault(k, []).append(float(v))
        per_key[s] = d
        dup_counts[s] = sum(1 for v in d.values() if len(v) > 1)

    common = set(per_key[want[0]])
    for s in want[1:]:
        common &= set(per_key[s])

    ref = sheets[want[0]]
    key_to_row = {k: i for i, k in enumerate(ref.keys)}
    names = ref.feature_names

    rows: List[Dict] = []
    for k in sorted(common):
        i = key_to_row[k]
        rows.append({
            "key": k,
            "descriptors": {n: float(v) for n, v in zip(names, ref.X[i])},
            "site1": ref.site1[i],
            "energies": {s: float(agg(per_key[s][k])) for s in want},
        })

    report = JoinReport(n_per_species={s: len(sheets[s]) for s in want},
                        n_joined=len(rows), species=want,
                        duplicate_keys=dup_counts)
    return rows, report


def pathway_coverage(sheets: Dict[str, SheetData], pathways: Dict[str, List]) -> Dict[str, Dict]:
    """For each product pathway, report whether every state it needs is present
    in the workbook and how many configurations carry all of them.

    This is the honest answer to 'which products can this file support?'.

    Every state in a pathway except the 'CO2' reference needs a free energy,
    INCLUDING the terminal one -- `proxy.limiting_potential` returns None if any
    is absent. For the CO pathway the terminal state is the adsorbed *CO, which
    is a sheet; for HCOOH and CH3OH the terminal states are desorbed molecules
    with no corresponding sheet, so those pathways are reported as not
    computable from an adsorption-energy workbook alone. That is a real
    limitation, not a bookkeeping artefact: it is surfaced rather than hidden.
    """
    out: Dict[str, Dict] = {}
    for product, steps in pathways.items():
        needed = sorted({s for step in steps for s in step} - {"CO2"})
        have = [s for s in needed if s in sheets]
        absent = [s for s in needed if s not in sheets]
        if absent:
            out[product] = {"computable": False, "required": needed,
                            "missing": absent, "n_configurations": 0}
            continue
        _, rep = join_intermediates(sheets, have)
        out[product] = {"computable": rep.n_joined > 0, "required": needed,
                        "missing": [], "n_configurations": rep.n_joined}
    return out


# --------------------------------------------------------------------------
# reference-frame conversion: adsorption energies -> CHE formation free energies
# --------------------------------------------------------------------------
# adsorbate -> (n_C, n_H, n_O)
SPECIES_COMPOSITION: Dict[str, Tuple[int, int, int]] = {
    "CO": (1, 0, 1), "COOH": (1, 1, 2), "OCHO": (1, 1, 2), "CHO": (1, 1, 1),
    "CH2O": (1, 2, 1), "OCH3": (1, 3, 1), "H": (0, 1, 0), "OH": (0, 1, 1),
}


def che_reference_shift(species: str, gas_energies: Dict[str, float]) -> float:
    """Constant shift converting a slab-referenced binding energy of *X into a
    CHE formation energy referenced to CO2(g) + (H+ + e-).

    Balancing n_C CO2 + ((n_H + 2w)/2) H2  ->  *X + w H2O   with  w = 2 n_C - n_O
    (w may be negative, e.g. *OH, meaning water appears on the left), the shift is

        shift(X) = w E(H2O) - n_C E(CO2) - ((n_H + 2w)/2) E(H2)

    so that   dE_f(*X) = [E(*X) - E(*)] + shift(X).

    `gas_energies` must contain the DFT TOTAL energies of CO2, H2 and H2O at the
    SAME level of theory as the surface calculations. Missing keys raise.
    """
    if species not in SPECIES_COMPOSITION:
        raise KeyError(f"unknown adsorbate '{species}'; known: "
                       f"{sorted(SPECIES_COMPOSITION)}")
    n_c, n_h, n_o = SPECIES_COMPOSITION[species]
    w = 2 * n_c - n_o
    needed = [k for k, used in (("CO2", n_c != 0), ("H2O", w != 0),
                                ("H2", (n_h + 2 * w) != 0)) if used]
    missing = [k for k in needed if k not in gas_energies]
    if missing:
        raise KeyError(
            f"gas-phase reference energies {missing} required for *{species} but "
            f"not supplied. These must be DFT total energies at the same level of "
            f"theory as the slab calculations; they cannot be defaulted.")
    e = gas_energies
    return (w * e.get("H2O", 0.0)
            - n_c * e.get("CO2", 0.0)
            - 0.5 * (n_h + 2 * w) * e.get("H2", 0.0))


def to_che_formation_energies(energies: Dict[str, float],
                              gas_energies: Dict[str, float],
                              adsorbate_gas_energies: Optional[Dict[str, float]] = None,
                              corrections: Optional[Dict[str, float]] = None
                              ) -> Dict[str, float]:
    """Convert one configuration's adsorption energies into the CHE formation
    free energies that `proxy.limiting_potential` expects.

    energies
        {species: E_ads} as published in the workbook.
    gas_energies
        DFT total energies of CO2, H2, H2O (same level of theory as the slabs).
    adsorbate_gas_energies
        {species: E_gas(X)} -- required IF the published E_ads is defined as
        E(*X) - E(*) - E_gas(X), i.e. referenced to the isolated adsorbate. Pass
        None (or omit a species) only when E_ads is already E(*X) - E(*).
        NOTE: for radical fragments such as *COOH there is no stable gas-phase
        molecule, so papers reference them to a combination instead; check the
        source before choosing. Getting this wrong shifts a whole species by a
        constant, which moves U_L rigidly -- silent and large.
    corrections
        Optional d(ZPE - T dS) per species, added last. Defaults to none, matching
        `proxy`'s behaviour of leaving thermochemistry off unless opted into.

    Returns {species: dG_f} suitable for `proxy.limiting_potential`.
    """
    ads_gas = adsorbate_gas_energies or {}
    corr = corrections or {}
    out: Dict[str, float] = {}
    for sp, e_ads in energies.items():
        if sp not in SPECIES_COMPOSITION:
            continue
        binding = float(e_ads) + float(ads_gas.get(sp, 0.0))   # -> E(*X) - E(*)
        out[sp] = binding + che_reference_shift(sp, gas_energies) + float(corr.get(sp, 0.0))
    return out


def convert_rows_to_che(rows: List[Dict],
                        gas_energies: Dict[str, float],
                        adsorbate_gas_energies: Optional[Dict[str, float]] = None,
                        corrections: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Apply `to_che_formation_energies` to every joined configuration, returning
    new rows with converted `energies` (the originals are kept under
    `energies_ads` so the transformation stays auditable)."""
    out: List[Dict] = []
    for r in rows:
        new = dict(r)
        new["energies_ads"] = dict(r["energies"])
        new["energies"] = to_che_formation_energies(
            r["energies"], gas_energies, adsorbate_gas_energies, corrections)
        out.append(new)
    return out


def check_energy_reference(rows: List[Dict], product: str = "CO") -> Dict:
    """Guard: are these energies in the reference frame the CHE proxy assumes?

    `proxy.limiting_potential` expects CHE FORMATION free energies of *X relative
    to CO2(g) + n_H*(1/2 H2) + *. This workbook reports ADSORPTION energies
    relative to the gas-phase adsorbate + clean slab -- a DIFFERENT reference.
    Feeding them in directly is a units-style error that produces a physically
    impossible result rather than an obviously broken one:

        U_L > 0 for a CO2 reduction pathway (reduction should require U_L < 0)

    which then clips to zero overpotential and yields a CONSTANT cell voltage for
    every configuration -- i.e. silently no catalyst dependence at all. This
    function detects that and says so, instead of letting it through.

    Converting properly needs gas-phase total energies at the SAME level of
    theory (CO2, H2, H2O), which the workbook does not contain. Once you have
    them, `convert_rows_to_che()` performs the conversion and this guard should
    then pass.

    Returns {'ok', 'reason', 'n_positive_U_L', 'n', 'U_L_range'}.
    """
    from .proxy import limiting_potential

    uls = []
    for r in rows:
        lp = limiting_potential(r["energies"], product)
        if lp is not None:
            uls.append(lp["U_L"])
    if not uls:
        return {"ok": False, "reason": f"no configuration supports the {product} pathway",
                "n_positive_U_L": 0, "n": 0, "U_L_range": None}

    uls = np.asarray(uls, float)
    n_pos = int((uls > 0).sum())
    ok = n_pos == 0
    reason = "" if ok else (
        f"{n_pos}/{len(uls)} configurations give U_L > 0 for the {product} pathway, "
        f"which is unphysical for CO2 reduction. These energies are almost "
        f"certainly ADSORPTION energies, not CHE formation free energies "
        f"referenced to CO2(g) + n_H*(1/2 H2). Supply gas-phase reference "
        f"energies at the same level of theory before using the CHE proxy.")
    return {"ok": ok, "reason": reason, "n_positive_U_L": n_pos, "n": len(uls),
            "U_L_range": (float(uls.min()), float(uls.max()))}


def to_activity_table(rows: List[Dict]) -> List[Dict]:
    """Joined rows -> the flat `descriptor_table` shape that
    `proxy.build_activity_targets` expects: dE_<species> keys per surface."""
    table: List[Dict] = []
    for i, r in enumerate(rows):
        item = {"surface": f"HEA-{r['site1']}", "facet": "", "config_index": i}
        item.update({f"dE_{s}": e for s, e in r["energies"].items()})
        table.append(item)
    return table
