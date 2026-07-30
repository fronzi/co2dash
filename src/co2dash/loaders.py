"""
Adaptive loader for public CO2RR DFT descriptor datasets (e.g. the Figshare
sets: Chen et al. HEA CoCuFeMoNi; ACS Catalysis FeCoNiCuMo, U_L 0.29-0.51 V).

Because these files' exact column layout is not known here (Figshare is not
reachable from the build sandbox), the loader is schema-tolerant and transparent:
it auto-detects adsorption-energy columns for *CO/*COOH/*CHO/*OCHO/*H/*OH by
fuzzy header matching and reports what it matched. The activity target y is
either a limiting-potential/overpotential column if the file has one, or U_L
computed from the intermediate energies via co2dash's CHE proxy.

Run it on your machine (which can reach Figshare):
    python examples/load_figshare_dataset.py <downloaded_file>.csv
If a column is missed, the printed mapping tells you; pass --map or extend
SPECIES_PREFIXES here.
"""
from __future__ import annotations
import csv, io, json, os
from typing import Dict, List, Optional, Tuple
import numpy as np
from .intake import _to_float
from .proxy import limiting_potential

SPECIES = {"co": "CO", "cooh": "COOH", "cho": "CHO", "ocho": "OCHO",
           "ch2o": "CH2O", "och3": "OCH3", "ch3oh": "CH3OH",
           "hcooh": "HCOOH", "oh": "OH", "h": "H"}
# energy prefixes/suffixes stripped before matching the species token (longest first)
_PREFIXES = ["adsorptionenergy", "bindingenergy", "deltag", "deltae", "eads",
             "ebind", "dg", "de", "g", "e"]
_SUFFIXES = ["star", "ads", "ev"]
TARGET_ALIASES = {"ul", "limitingpotential", "u_l", "overpotential", "eta",
                  "activity", "ulimiting"}


def _norm(h: str) -> str:
    return "".join(c for c in str(h).lower() if c.isalnum())


def species_of(header: str) -> Optional[str]:
    """Return canonical species (CO, COOH, ...) if the header is an adsorption
    energy for one, else None. 'CO2' and unrelated columns return None."""
    n = _norm(header)
    for suf in _SUFFIXES:
        if n.endswith(suf) and len(n) > len(suf):
            n = n[: -len(suf)]
    if n in SPECIES:
        return SPECIES[n]
    for p in sorted(_PREFIXES, key=len, reverse=True):
        if n.startswith(p) and n[len(p):] in SPECIES:
            return SPECIES[n[len(p):]]
    return None


def target_of(header: str) -> bool:
    return _norm(header) in TARGET_ALIASES


def load_table(path: str) -> List[Dict[str, object]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        d = json.load(open(path)); return d if isinstance(d, list) else d.get("records", d)
    if ext in (".xlsx", ".xls"):
        import pandas as pd
        return pd.read_excel(path).to_dict("records")
    text = open(path, encoding="utf-8", errors="replace").read()
    return list(csv.DictReader(io.StringIO(text)))


def resolve_columns(headers: List[str]) -> Tuple[Dict[str, str], Optional[str]]:
    """Return ({header: species}, target_header|None)."""
    desc, target = {}, None
    for h in headers:
        sp = species_of(h)
        if sp and sp not in desc.values():
            desc[h] = sp
        elif target is None and target_of(h):
            target = h
    return desc, target


def to_descriptor_activity(rows: List[Dict], product: str = "CO",
                           co2_reference: float = 0.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Build (X, y). X = matched adsorption-energy descriptors. y = a limiting-
    potential/overpotential column if present, else U_L from the CHE proxy for
    `product` (needs the pathway's species; rows missing them are dropped)."""
    if not rows:
        raise ValueError("empty table")
    desc_map, target_h = resolve_columns(list(rows[0].keys()))
    if not desc_map:
        raise ValueError("no adsorption-energy columns recognised; check headers")
    species_cols = list(desc_map.items())           # [(header, species)]
    X, y, skipped = [], [], 0
    for r in rows:
        vals = {sp: _to_float(r[h]) for h, sp in species_cols}
        if any(v is None for v in vals.values()):
            skipped += 1; continue
        xrow = [vals[sp] for _, sp in species_cols]
        if target_h is not None:
            t = _to_float(r[target_h])
            if t is None:
                skipped += 1; continue
            yv = t
        else:
            energies = {"CO2": co2_reference, **vals}
            res = limiting_potential(energies, product)
            if res is None:
                skipped += 1; continue
            yv = res["U_L"]
        X.append(xrow); y.append(yv)
    if not X:
        raise ValueError("no usable rows after matching/CHE (check species/target)")
    report = {"descriptors": [sp for _, sp in species_cols],
              "target": target_h or f"CHE U_L({product})",
              "matched": len(y), "skipped": skipped}
    return np.asarray(X, float), np.asarray(y, float), report
