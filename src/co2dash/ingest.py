"""
Ingestion layer: raw public-DFT records -> surrogate-ready descriptor table.

The discovery layer gives adsorption/reaction energies for specific intermediates
on specific surfaces. The surrogate needs, per material, a feature VECTOR of the
key CO2RR intermediate energies (the scaling-relation descriptors). This module
assembles that table from the canonical records produced by connectors.py.

It produces X (descriptors). The TARGET y (faradaic efficiency, or a stated
theoretical activity proxy) must come from your experiments / validation set or a
clearly-documented physics proxy -- it is NOT invented here.
"""
from __future__ import annotations
from typing import List, Dict, Iterable, Optional
import math

from .schema import Candidate, DataTier


# Default CO2RR intermediates -> the species filter used on Catalysis-Hub.
DEFAULT_INTERMEDIATES = {
    "dE_CO":   "COstar",
    "dE_COOH": "COOHstar",
    "dE_OCHO": "OCHOstar",
    "dE_H":    "Hstar",
    "dE_OH":   "OHstar",
}


def _normalize_species(name: str) -> str:
    """Map species labels to a canonical adsorbate token so 'COstar', 'CO*',
    'COgas' all compare on the same basis. Returns '' for the bare surface."""
    s = str(name).strip().lower()
    for suff in ("star", "gas", "(g)", "*"):
        if s.endswith(suff):
            s = s[: -len(suff)]
    return s.strip("_* ")


def _products_contain(products: dict, target_token: str) -> bool:
    return any(_normalize_species(k) == target_token for k in (products or {}))


def _surface_key(rec: dict) -> str:
    return f"{rec.get('surface','?')}|{rec.get('facet','')}"


def build_descriptor_table(records: List[Dict],
                           intermediates: Optional[Dict[str, str]] = None,
                           reduce: str = "min") -> List[Dict]:
    """
    Group records by surface and assemble one descriptor row per surface:
      {surface, facet, dE_CO, dE_COOH, ..., n_intermediates, source, tier}
    Missing intermediates are left as None. `reduce` picks the value when a
    surface has several records for the same intermediate ('min' = most stable
    adsorption, the usual descriptor choice; 'mean' also available).
    """
    inter = intermediates or DEFAULT_INTERMEDIATES
    targets = {key: _normalize_species(species) for key, species in inter.items()}

    by_surface: Dict[str, dict] = {}
    for rec in records:
        e = rec.get("reaction_energy")
        if e is None:
            continue
        for key, tok in targets.items():
            if _products_contain(rec.get("products", {}), tok):
                sk = _surface_key(rec)
                row = by_surface.setdefault(sk, {
                    "surface": rec.get("surface", ""), "facet": rec.get("facet", ""),
                    "_vals": {}, "sources": set()})
                row["_vals"].setdefault(key, []).append(float(e))
                row["sources"].add(rec.get("source", ""))

    out: List[Dict] = []
    for sk, row in by_surface.items():
        entry = {"surface": row["surface"], "facet": row["facet"]}
        n = 0
        for key in inter:
            vals = row["_vals"].get(key)
            if vals:
                entry[key] = min(vals) if reduce == "min" else sum(vals) / len(vals)
                n += 1
            else:
                entry[key] = None
        entry["n_intermediates"] = n
        entry["tier"] = DataTier.COMPUTED.name
        entry["source"] = ";".join(sorted(s for s in row["sources"] if s))[:200]
        out.append(entry)
    # most-complete surfaces first
    return sorted(out, key=lambda r: r["n_intermediates"], reverse=True)


def descriptor_coverage(table: List[Dict], keys: Iterable[str]) -> Dict[str, float]:
    """Fraction of surfaces that have each descriptor (data-completeness audit)."""
    keys = list(keys)
    n = max(len(table), 1)
    return {k: sum(1 for r in table if r.get(k) is not None) / n for k in keys}


def assemble_candidates(table: List[Dict], descriptor_keys: List[str]) -> List[Candidate]:
    """Build Candidate objects from surfaces that have ALL required descriptors
    (the surrogate needs complete feature vectors). Incomplete surfaces are
    skipped, not imputed."""
    cands = []
    for i, row in enumerate(table):
        if all(row.get(k) is not None and not math.isnan(row[k]) for k in descriptor_keys):
            cands.append(Candidate(
                material_id=f"{row.get('surface','?')}({row.get('facet','')})#{i}",
                descriptors={k: float(row[k]) for k in descriptor_keys},
                source_db="catalysis-hub"))
    return cands


def ingest_co2rr(intermediates: Optional[Dict[str, str]] = None,
                 max_records_per_intermediate: int = 300,
                 cache_dir: str = "data") -> List[Dict]:
    """
    Orchestrate a real fetch: pull each CO2RR intermediate from Catalysis-Hub,
    cache each pull, and return the assembled descriptor table. Runs on an open
    network (imports the live connector lazily).
    """
    import os
    from .connectors import fetch_catalysis_hub_reactions
    inter = intermediates or DEFAULT_INTERMEDIATES
    os.makedirs(cache_dir, exist_ok=True)
    all_records: List[Dict] = []
    for key, species in inter.items():
        cache = os.path.join(cache_dir, f"chub_{key}.json")
        recs = fetch_catalysis_hub_reactions(products=species,
                                             max_records=max_records_per_intermediate,
                                             cache_path=cache)
        all_records.extend(recs)
    return build_descriptor_table(all_records, inter)
