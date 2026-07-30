"""
Featurize a public CO2RR *experimental* corpus into (X, y) for the calibration
gate, where y = measured faradaic efficiency.

Target corpora (open access; download on your machine — see examples/
calibrate_co2rr_corpus.py for the exact command):
  * Scientific Data 2023, "A corpus of CO2 electrocatalytic reduction process
    extracted from the scientific literature", doi:10.1038/s41597-023-02089-z
    (benchmark: 6,086 records; fields: material, product, faradaic efficiency,
    cell setup, electrolyte, synthesis method, current density, voltage).
  * Scientific Data 2024, "Large language model enhanced corpus ...",
    doi:10.1038/s41597-024-03180-9.

Because these corpora are text-mined, the features here are operating
conditions + catalyst/product identity (NOT DFT descriptors). This calibrates a
FE(conditions, catalyst) surrogate on real experimental FE — a legitimate
literature substitute for the discovery→FE task, with that scope stated.
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np
from .intake import _HEADER_LOOKUP, _to_float, normalise_units

# catalyst element families (substring match on the material string)
CATALYST_FAMILIES = ["Cu", "Ag", "Au", "Sn", "Bi", "Zn", "Pd", "Ni", "Co",
                     "Fe", "In", "Pb", "Sb", "Ga", "Mo", "C"]
# product canonicalisation
_PRODUCT_MAP = {
    "co": "CO", "carbon monoxide": "CO",
    "hcooh": "formate", "formate": "formate", "formic": "formic acid",
    "formic acid": "formate",
    "ch3oh": "methanol", "methanol": "methanol",
    "ch4": "methane", "methane": "methane",
    "c2h4": "ethylene", "ethylene": "ethylene",
    "c2h5oh": "ethanol", "ethanol": "ethanol",
    "h2": "H2", "hydrogen": "H2",
}
PRODUCT_CLASSES = ["CO", "formate", "methanol", "methane", "ethylene", "ethanol", "H2", "other"]

# corpus-specific header aliases layered on top of intake's aliases
_CORPUS_ALIASES = {
    "material": "material_id", "catalyst": "material_id",
    "product": "product",
    "faradaic efficiency": "faradaic_efficiency", "fe": "faradaic_efficiency",
    "faradaicefficiency": "faradaic_efficiency",
    "current density": "current_density", "current_density": "current_density",
    "voltage": "cell_voltage", "potential": "cell_voltage",
    "electrolyte": "electrolyte",
}


def _canon_key(h: str):
    h = str(h).strip().lower()
    return _CORPUS_ALIASES.get(h) or _HEADER_LOOKUP.get(h)


def _product_class(s) -> str:
    return _PRODUCT_MAP.get(str(s).strip().lower(), "other") if s is not None else "other"


def _catalyst_onehot(material: str) -> List[float]:
    m = str(material or "")
    return [1.0 if fam in m else 0.0 for fam in CATALYST_FAMILIES]


def map_corpus_columns(headers: Sequence[str]) -> Dict[str, str]:
    out = {}
    for h in headers:
        k = _canon_key(h)
        if k:
            out[h] = k
    return out


def featurize_co2rr(rows: List[Dict[str, object]],
                    require_conditions: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """rows: list of dicts (canonical or raw corpus keys). Returns (X, y, names).
    y = faradaic efficiency as a fraction. Rows without a valid FE are dropped;
    if require_conditions, rows missing both current density and voltage are dropped."""
    # normalise keys
    norm = []
    for r in rows:
        d = {}
        for h, v in r.items():
            k = _canon_key(h) or h
            d[k] = v
        norm.append(d)

    names = (["current_density", "cell_voltage"]
             + [f"prod::{p}" for p in PRODUCT_CLASSES]
             + [f"cat::{c}" for c in CATALYST_FAMILIES])
    X, y = [], []
    for d in norm:
        fe = _to_float(d.get("faradaic_efficiency"))
        if fe is None:
            continue
        fe, _ = normalise_units("faradaic_efficiency", fe, str(d.get("faradaic_efficiency")))
        if not (0.0 <= fe <= 1.0):
            continue
        j = _to_float(d.get("current_density"))
        v = _to_float(d.get("cell_voltage"))
        if require_conditions and j is None and v is None:
            continue
        row = [j if j is not None else 0.0, v if v is not None else 0.0]
        pc = _product_class(d.get("product"))
        row += [1.0 if pc == p else 0.0 for p in PRODUCT_CLASSES]
        row += _catalyst_onehot(d.get("material_id"))
        X.append(row); y.append(fe)
    if not X:
        raise ValueError("no usable rows (need a valid faradaic efficiency per row)")
    X = np.asarray(X, float)
    keep = X.std(axis=0) > 1e-9
    return X[:, keep], np.asarray(y, float), [n for n, k in zip(names, keep) if k]
