"""
Link experimental FE to DFT descriptors — the discovery↔decision join.

To train a descriptor→FE surrogate you need, for the SAME material, both a
measured FE and DFT descriptors (ΔE of *CO, *COOH, *H, ...). The honest
obstacle: literature catalysts are messy synthesized materials (alloys, oxide-
derived Cu, MOFs, composites) while public DFT descriptors exist for well-
defined surfaces (Cu(111), Cu(100), simple oxides). The two live in different
material spaces, so the join is partial and must be quantified, not forced.

This module:
  * canonicalises a messy catalyst string to a surface key + an availability tag
    ('public'  = descriptors obtainable from Catalysis-Hub/OC20 for a defined
                 surface; 'bespoke' = needs your own DFT; 'none' = ill-defined),
  * joins FE rows to a descriptor table on that key (no imputation),
  * builds a descriptor-request list: which surfaces to compute/fetch, ranked by
    how many FE records each would unlock.
"""
from __future__ import annotations
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np

# canonical surface key + descriptor availability, from the material-type label.
_MATERIAL_MAP = {
    "Cu": ("Cu", "public"),            # pure Cu -> Cu(111)/(100)/(211) in Catalysis-Hub
    "Cu/C": ("Cu", "public"),          # Cu on carbon -> Cu facets (support effect ignored)
    "CuOx": ("CuOx", "bespoke"),       # oxide-derived Cu, structure not well defined
    "Cu(Ox)-MOx": ("CuOx-MOx", "bespoke"),
    "Cu-MOx": ("Cu-MOx", "bespoke"),
    "Cu-M": ("Cu-alloy", "bespoke"),   # composition-specific alloy -> needs its own DFT
    "alloy": ("Cu-alloy", "bespoke"),
    "composite": ("composite", "none"),
    "Cu-MOF": ("Cu-MOF", "none"),
    "CuSx": ("CuSx", "bespoke"),
    "other": ("other", "none"),
}


def canonical_material(material: str) -> Tuple[str, str]:
    """messy catalyst/type string -> (canonical surface key, availability tag)."""
    m = str(material).strip()
    if m in _MATERIAL_MAP:
        return _MATERIAL_MAP[m]
    ml = m.lower()
    if "mof" in ml:
        return ("Cu-MOF", "none")
    if any(t in ml for t in ("alloy", "-m", "bimetal")):
        return ("Cu-alloy", "bespoke")
    import re as _re
    if "oxide" in ml or "ox" in ml or _re.search(r"cu\d*o", ml):
        return ("CuOx", "bespoke")
    if ml.startswith("cu"):
        return ("Cu", "public")
    return (m or "other", "none")


def availability_report(fe_rows: List[Dict]) -> Dict:
    """How many FE records fall into each availability tier."""
    tiers = Counter()
    by_key = Counter()
    for r in fe_rows:
        key, avail = canonical_material(r.get("material", "other"))
        tiers[avail] += 1
        by_key[(key, avail)] += 1
    n = len(fe_rows)
    return {"n": n, "tiers": dict(tiers),
            "public_frac": tiers.get("public", 0) / n if n else 0.0,
            "by_key": dict(by_key)}


def descriptor_request_list(fe_rows: List[Dict]) -> List[Dict]:
    """Surfaces to obtain descriptors for, ranked by FE records unlocked."""
    by_key = availability_report(fe_rows)["by_key"]
    out = [{"surface": k, "availability": a, "fe_records": c}
           for (k, a), c in sorted(by_key.items(), key=lambda kv: -kv[1])]
    return out


def link_fe_to_descriptors(fe_rows: List[Dict],
                           descriptor_table: Dict[str, Dict[str, float]],
                           keys: List[str]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Inner-join FE rows to a descriptor table on the canonical surface key.
    descriptor_table: {canonical_key: {descriptor_name: value}}.
    keys: descriptor names to use as X columns (fixed order).
    Returns (X, y, report). No imputation: unmatched rows are dropped and counted.
    """
    X, y, matched, unmatched = [], [], 0, Counter()
    for r in fe_rows:
        fe = r.get("faradaic_efficiency")
        if fe is None:
            continue
        key, _ = canonical_material(r.get("material", "other"))
        d = descriptor_table.get(key)
        if d is None or any(k not in d for k in keys):
            unmatched[key] += 1
            continue
        X.append([float(d[k]) for k in keys]); y.append(float(fe)); matched += 1
    report = {"matched": matched, "unmatched": sum(unmatched.values()),
              "unmatched_by_key": dict(unmatched), "keys": keys}
    return (np.asarray(X, float), np.asarray(y, float), report)


def descriptors_to_canonical(surface_table: Dict[str, Dict[str, float]],
                             reduce: str = "min") -> Dict[str, Dict[str, float]]:
    """Aggregate a per-surface descriptor table {surface_string: {dE_x: val}} into
    canonical-key descriptors {canonical_key: {dE_x: val}}. Multiple surfaces
    mapping to the same key (e.g. Cu(111), Cu(100) -> 'Cu') are reduced per
    descriptor by `reduce` ('min' = strongest binding, the usual CHE convention;
    'mean' also allowed). This is what feeds link_fe_to_descriptors()."""
    import numpy as _np
    buckets: Dict[str, Dict[str, list]] = {}
    for surf, d in surface_table.items():
        key, _ = canonical_material(surf)
        b = buckets.setdefault(key, {})
        for name, val in d.items():
            if val is None:
                continue
            b.setdefault(name, []).append(float(val))
    agg = _np.min if reduce == "min" else _np.mean
    return {k: {name: float(agg(vals)) for name, vals in b.items()}
            for k, b in buckets.items()}
