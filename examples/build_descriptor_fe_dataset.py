"""
Build the descriptor->FE dataset: join experimental FE to DFT descriptors.

    # 1. see the join gap + which surfaces to compute (no descriptors needed):
    python examples/build_descriptor_fe_dataset.py <path>/merge_data_final.xls

    # 2. once you have a descriptor table (JSON: {surface_key: {dE_CO:..,dE_H:..}}),
    #    join and run the calibration gate on descriptor->FE:
    python examples/build_descriptor_fe_dataset.py <path>/merge_data_final.xls descriptors.json

Honest scope: literature catalysts and public DFT surfaces live in different
material spaces, so the join is partial. This script quantifies exactly how
partial (per your data) and lists the surfaces to obtain descriptors for
(Catalysis-Hub for 'public'; your HPC DFT for 'bespoke'). Descriptor keys must
match canonical_material() surface keys (Cu, Cu-alloy, CuOx, ...).
"""
import os, sys, json, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from calibrate_corpus_scidata import reconstruct
from co2dash.link import availability_report, descriptor_request_list, link_fe_to_descriptors
from co2dash.calibration_harness import calibrate_and_evaluate
from co2dash.surrogate import BayesianLinearSurrogate


def main(xls_path, desc_json=None):
    rows = reconstruct(xls_path)
    rep = availability_report(rows)
    print(f"FE records: {rep['n']}")
    print("Descriptor availability:")
    for tier, c in sorted(rep["tiers"].items(), key=lambda kv: -kv[1]):
        print(f"  {tier:8s} {c:4d}  ({c/rep['n']:.0%})")
    print(f"  -> joinable to PUBLIC DFT now: {rep['public_frac']:.0%}\n")

    print("Descriptor-request list (obtain these surfaces, by FE records unlocked):")
    reqs = descriptor_request_list(rows)
    for r in reqs:
        print(f"  {r['surface']:12s} [{r['availability']:7s}]  unlocks {r['fe_records']} FE records")

    out_csv = os.path.join(os.path.dirname(xls_path), "descriptor_request_list.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["surface", "availability", "fe_records"])
        w.writeheader(); w.writerows(reqs)
    print(f"\nWrote {out_csv}")

    if desc_json is None:
        print("\nNo descriptor table supplied — stopping before the join (nothing fabricated).")
        print("Provide descriptors.json = {surface_key: {dE_CO:.., dE_H:.., dE_OCHO:..}} to join.")
        return

    table = json.load(open(desc_json))
    keys = sorted({k for d in table.values() for k in d})
    X, y, jr = link_fe_to_descriptors(rows, table, keys=keys)
    print(f"\nJoined {jr['matched']} FE records to descriptors "
          f"({jr['unmatched']} unmatched). Descriptor keys: {keys}")
    if len(y) < 40:
        print("Too few matched records for a meaningful calibration split — "
              "add more surfaces to the descriptor table.")
        return
    r = calibrate_and_evaluate(
        X, y, surrogate_factory=lambda Xt, yt: BayesianLinearSurrogate().fit(Xt, yt),
        alpha=0.1, seed=0)
    print("\n=== calibration on descriptor->FE (matched pairs) ===")
    print(f"  raw 90% coverage {r.coverage_before[0.9]:.0%}; temperature s={r.temperature_s:.2f}; "
          f"miscalibration {r.miscal_before:.3f}->{r.miscal_after:.3f}; "
          f"after 90% {r.coverage_after[0.9]:.0%}, conformal {r.conformal_coverage:.0%}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); print("ERROR: pass merge_data_final.xls [descriptors.json]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
