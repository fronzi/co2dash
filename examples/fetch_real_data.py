"""
Live ingestion of REAL public DFT data from Catalysis-Hub.

Run THIS ON YOUR MACHINE (open network required; it will not run inside a
restricted sandbox). It fetches CO2RR intermediate adsorption energies, builds
the per-surface descriptor table, caches everything under ./data, and reports
how complete the data is.

    python examples/fetch_real_data.py

Notes
-----
* This produces X (descriptors) from real DFT. The TARGET y (faradaic efficiency
  or a stated activity proxy) must come from your experiments / validation set;
  it is not invented here. Once you have y, train the surrogate and feed the
  candidates into rank_candidates exactly as in examples/operational.py.
* If the descriptor table comes back empty, the API schema may have drifted:
  run probe_schema() (printed below) and adjust connectors.REACTION_FIELDS.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from co2dash import (probe_schema, ingest_co2rr, DEFAULT_INTERMEDIATES,
                     descriptor_coverage, assemble_candidates)

print("=== 1. Schema check (run if parsing returns empty) ===")
try:
    fields = probe_schema("Reaction")
    print(f"  live Reaction fields ({len(fields)}): {', '.join(fields[:12])} ...")
except Exception as exc:
    print(f"  probe skipped: {exc}")

print("\n=== 2. Fetch + build descriptor table (cached under ./data) ===")
table = ingest_co2rr(DEFAULT_INTERMEDIATES,
                     max_records_per_intermediate=300, cache_dir="data")
print(f"  surfaces assembled: {len(table)}")

keys = list(DEFAULT_INTERMEDIATES.keys())
cov = descriptor_coverage(table, keys)
print("\n=== 3. Descriptor coverage (fraction of surfaces with each) ===")
for k, v in cov.items():
    print(f"  {k:8s} {v:5.0%}")

print("\n=== 4. Most-complete surfaces (top 8) ===")
for row in table[:8]:
    vals = " ".join(f"{k}={row[k]:+.2f}" if row[k] is not None else f"{k}=NA"
                    for k in keys)
    print(f"  {row['surface']:>10s}({row['facet']:>3s})  {vals}")

# require a complete pair of descriptors for the surrogate's feature vector
need = ["dE_CO", "dE_COOH"]
cands = assemble_candidates(table, need)
print(f"\n=== 5. Candidates with complete {need}: {len(cands)} ===")
for c in cands[:5]:
    print(f"  {c.material_id}: {c.descriptors}")

print("\nDONE. Real descriptors fetched & cached. Provide y (FE/activity) to train"
      " the surrogate, then reuse the operational.py pipeline.")
