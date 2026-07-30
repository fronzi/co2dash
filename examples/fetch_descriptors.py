"""
Fetch DFT descriptors from Catalysis-Hub and export descriptors.json keyed by
canonical surface — the input to build_descriptor_fe_dataset.py.

    python examples/fetch_descriptors.py            # default Cu-facet CO2RR set
    python examples/fetch_descriptors.py Cu Ag Sn   # specify surfaces/elements

Runs on YOUR machine (needs network to api.catalysis-hub.org). It queries CO2RR
intermediate adsorption energies (*CO, *COOH, *OCHO, *H, *OH), builds a per-
surface descriptor table, aggregates to canonical surface keys (Cu(111)/Cu(100)
-> 'Cu', etc.), and writes descriptors.json = {surface_key: {dE_CO:.., ...}}.

Scope note: Catalysis-Hub covers well-defined surfaces, so this fills the
'public' tier (pure Cu). 'bespoke' surfaces (specific alloys, oxide-derived Cu)
must come from your own DFT (Setonix/Gadi), written into the same JSON schema.
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from co2dash.connectors import fetch_catalysis_hub_reactions
from co2dash.ingest import build_descriptor_table, DEFAULT_INTERMEDIATES
from co2dash.link import descriptors_to_canonical


def fetch_surface_descriptors(products=("CO", "H2"), max_records=2000):
    """Fetch CO2RR-relevant reactions and build a per-surface descriptor table.
    Returns {surface_string: {dE_x: value}} using DEFAULT_INTERMEDIATES."""
    records = []
    for p in products:
        records += fetch_catalysis_hub_reactions(products=p, max_records=max_records)
    table = build_descriptor_table(records, intermediates=DEFAULT_INTERMEDIATES, reduce="min")
    # build_descriptor_table returns rows keyed by surface; normalise to a dict
    surf = {}
    for row in table:
        key = row.get("surface") or row.get("material_id") or row.get("chemicalComposition")
        surf[str(key)] = {k: v for k, v in row.items()
                          if k.startswith("dE_") and v is not None}
    return surf


def main(elements):
    print(f"Fetching CO2RR descriptors from Catalysis-Hub (surfaces: {elements or 'default'}) ...")
    try:
        surf = fetch_surface_descriptors()
    except Exception as e:
        print(f"ERROR fetching (needs network to api.catalysis-hub.org): {e}")
        print("On a restricted network this must run on your own machine.")
        return
    if elements:
        surf = {s: d for s, d in surf.items() if any(el in s for el in elements)}
    canon = descriptors_to_canonical(surf, reduce="min")
    out = os.path.join(os.getcwd(), "descriptors.json")
    json.dump(canon, open(out, "w"), indent=2)
    print(f"Wrote {out} with {len(canon)} canonical surfaces: {list(canon)}")
    print("Next: python examples/build_descriptor_fe_dataset.py <corpus>.xls descriptors.json")


if __name__ == "__main__":
    main(sys.argv[1:])
