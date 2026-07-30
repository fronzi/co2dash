"""
Load a public Figshare CO2RR DFT dataset -> descriptor→activity + calibration.

    # you download the file on your machine (Figshare is reachable there), then:
    python examples/load_figshare_dataset.py <file>.csv            # CHE U_L target
    python examples/load_figshare_dataset.py <file>.xlsx --product CH3OH

KNOWN SOURCES (open access):
  * ACS Catalysis 2023, FeCoNiCuMo HEA — adsorption energies of COOH*/CO*/CHO*,
    limiting potential U_L 0.29-0.51 V.  Figshare article 21606332
    (acs.figshare.com/.../21606332).
  * Chen et al. 2022, CoCuFeMoNi HEA — 691 pts, *CO/*CHO/*COOH adsorption
    energies (used by the AGRA paper). Locate its Figshare/SI and download.

The loader auto-detects the adsorption-energy columns and prints the mapping. If
the file has a limiting-potential/overpotential column it is used as the target;
otherwise U_L is computed from the intermediate energies via the CHE proxy.

NOT tested here against the live files (Figshare is unreachable from the build
sandbox). If a column is not matched, the printed mapping shows it — paste the
header back and the aliases in loaders.py can be extended in one line.
"""
import os, sys, json, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from co2dash.loaders import load_table, resolve_columns, to_descriptor_activity
from co2dash.calibration_harness import calibrate_and_evaluate
from co2dash.surrogate import BayesianLinearSurrogate


def download_figshare(article_id: int, dest_dir: str = ".") -> list:
    """Download all files of a public Figshare article (run on your machine).
    Returns local file paths. Uses the public Figshare API."""
    api = f"https://api.figshare.com/v2/articles/{article_id}/files"
    files = json.loads(urllib.request.urlopen(api, timeout=60).read())
    out = []
    for f in files:
        p = os.path.join(dest_dir, f["name"])
        urllib.request.urlretrieve(f["download_url"], p)
        out.append(p); print("downloaded", p)
    return out


def main(path, product="CO"):
    rows = load_table(path)
    desc, target = resolve_columns(list(rows[0].keys()) if rows else [])
    print(f"Loaded {len(rows)} rows from {os.path.basename(path)}")
    print(f"Recognised descriptor columns: {desc}")
    print(f"Target column: {target or f'(none found -> CHE U_L for {product})'}\n")
    X, y, rep = to_descriptor_activity(rows, product=product)
    print(f"Built dataset: descriptors={rep['descriptors']}, target={rep['target']}, "
          f"matched={rep['matched']}, skipped={rep['skipped']}")
    print(f"activity y range [{y.min():.2f}, {y.max():.2f}]\n")
    if len(y) < 40:
        print("Few matched rows — calibration will be coarse.")
    r = calibrate_and_evaluate(
        X, y, surrogate_factory=lambda Xt, yt: BayesianLinearSurrogate().fit(Xt, yt),
        alpha=0.1, seed=0)
    print("=== descriptor→activity calibration ===")
    print(f"  raw 90% coverage {r.coverage_before[0.9]:.0%}; temperature s={r.temperature_s:.2f}; "
          f"miscalibration {r.miscal_before:.3f}->{r.miscal_after:.3f}; "
          f"after 90% {r.coverage_after[0.9]:.0%}, conformal {r.conformal_coverage:.0%}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    product = "CO"
    if "--product" in sys.argv:
        product = sys.argv[sys.argv.index("--product") + 1]
    if not args:
        print(__doc__); print("ERROR: pass the downloaded dataset file (csv/xlsx/json).")
        sys.exit(1)
    main(args[0], product=product)
