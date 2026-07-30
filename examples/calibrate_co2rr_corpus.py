"""
Calibrate the surrogate on a public CO2RR EXPERIMENTAL corpus (measured FE).

    python examples/calibrate_co2rr_corpus.py /path/to/corpus.csv
    python examples/calibrate_co2rr_corpus.py /path/to/corpus.json

WHERE TO GET THE DATA (open access; download on your machine):
  Scientific Data 2023 — "A corpus of CO2 electrocatalytic reduction process
  extracted from the scientific literature", doi:10.1038/s41597-023-02089-z.
  Open the paper's "Data availability" / "Code availability" section and
  download the benchmark corpus (6,086 records). Fields include: material,
  product, faradaic efficiency, current density, voltage, electrolyte.
  (Newer: doi:10.1038/s41597-024-03180-9.)

  Save it as CSV (columns e.g. material, product, faradaic efficiency, current
  density, voltage) or JSON (list of records with those keys), then pass the
  path. Column names are matched by alias; FE as a percentage is auto-detected.

This calibrates FE(conditions, catalyst) on REAL measured FE. Honest scope: the
features are operating conditions + catalyst/product identity, not DFT
descriptors — a real-experimental-FE substitute for your own lab data. The
CO2RR *CO/*H descriptor path uses the identical calibrate_and_evaluate.
"""
import os, sys, csv, io, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from co2dash.corpus import featurize_co2rr, map_corpus_columns
from co2dash.calibration_harness import calibrate_and_evaluate
from co2dash.surrogate import BayesianLinearSurrogate


def _read(path):
    if path.lower().endswith(".json"):
        data = json.load(open(path))
        return data if isinstance(data, list) else data.get("records", [])
    text = open(path, encoding="utf-8").read()
    return list(csv.DictReader(io.StringIO(text)))


def main(path):
    rows = _read(path)
    print(f"Loaded {len(rows)} records from {os.path.basename(path)}")
    if rows:
        print("Recognised columns:", map_corpus_columns(list(rows[0].keys())))
    X, y, names = featurize_co2rr(rows)
    print(f"Usable rows with valid FE: {len(y)};  features: {len(names)}")
    print(f"FE range: [{y.min():.2f}, {y.max():.2f}]  (fraction)")

    fac = lambda Xt, yt: BayesianLinearSurrogate().fit(Xt, yt)
    r = calibrate_and_evaluate(X, y, surrogate_factory=fac, alpha=0.1, seed=0)
    print("\n=== calibration on REAL experimental FE ===")
    print(f"  raw 90% interval covers {r.coverage_before[0.9]:.0%} of held-out points")
    print(f"  temperature s = {r.temperature_s:.2f}; miscalibration "
          f"{r.miscal_before:.3f} -> {r.miscal_after:.3f}")
    print(f"  after: 90% coverage {r.coverage_after[0.9]:.0%}, conformal {r.conformal_coverage:.0%}")
    print("  " + r.summary())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("ERROR: pass the path to the downloaded corpus (CSV or JSON).")
        sys.exit(1)
    main(sys.argv[1])
