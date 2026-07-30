"""
Calibrate the surrogate on a LITERATURE faradaic-efficiency dataset (a stand-in
for your own experimental data).

    python examples/calibrate_literature.py [path/to/fe_dataset.csv]

Default dataset: examples/literature_fe_co.csv — 7 real, cited CO2->CO studies
(Osorio-Tejada 2024, Table 2). Features used: cell_voltage, current_density;
target: measured FE.

IMPORTANT (honesty): calibration statistics (temperature scaling, conformal
coverage) only mean something with enough points. Below MIN_FOR_CALIBRATION we
do NOT report a calibration result — we run a leave-one-out fit sanity check and
tell you a credible calibration needs a larger dataset. For a real result, drop
in one of the open corpora listed at the bottom.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from co2dash.intake import read_csv
from co2dash.surrogate import BayesianLinearSurrogate
from co2dash.calibration_harness import calibrate_and_evaluate

MIN_FOR_CALIBRATION = 40


def load_features(path):
    with open(path) as fh:
        rows = read_csv(fh.read())            # canonical keys; FE %->fraction handled below
    X, y, ids = [], [], []
    for i, r in enumerate(rows):
        # intake.read_csv maps columns but does not unit-normalise; do FE %->fraction here
        try:
            v = float(r["cell_voltage"]); j = float(r["current_density"])
            fe = float(r["faradaic_efficiency"])
        except (KeyError, ValueError):
            continue
        if fe > 1.5:
            fe /= 100.0
        X.append([v, j]); y.append(fe); ids.append(r.get("material_id", f"row{i}"))
    return np.array(X, float), np.array(y, float), ids


def loo_rmse(X, y):
    """Leave-one-out RMSE of the surrogate mean — a fit sanity check (not calibration)."""
    n = len(y); errs = []
    for k in range(n):
        m = np.ones(n, bool); m[k] = False
        s = BayesianLinearSurrogate().fit(X[m], y[m])
        pred, _ = s.predict(X[k:k + 1])
        errs.append((pred[0] - y[k]) ** 2)
    return float(np.sqrt(np.mean(errs)))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "literature_fe_co.csv")
    X, y, ids = load_features(path)
    print(f"Loaded {len(y)} real records from {os.path.basename(path)}")
    print(f"  features: [cell_voltage, current_density]; target: FE "
          f"(range {y.min():.2f}-{y.max():.2f})")

    if len(y) >= MIN_FOR_CALIBRATION:
        rep = calibrate_and_evaluate(
            X, y, surrogate_factory=lambda Xt, yt: BayesianLinearSurrogate().fit(Xt, yt),
            alpha=0.1, seed=0)
        print("\nCALIBRATION REPORT (real literature data):")
        print("  " + rep.summary())
        print("  coverage after:", {k: round(v, 2) for k, v in rep.coverage_after.items()})
    else:
        rmse = loo_rmse(X, y)
        print(f"\n  n = {len(y)} < {MIN_FOR_CALIBRATION}: too few points for meaningful")
        print("  calibration statistics (a 25% test split would be ~2 points, whose")
        print("  coverage estimate carries ~±35% binomial error). NOT reporting a")
        print("  calibration result on this — that would be theatre, not validation.")
        print(f"  Fit sanity check instead: leave-one-out FE RMSE = {rmse:.3f}")
        print("  (the surrogate does learn the V/j -> FE trend; use it as a seed only.)")

    print("\n  For a CREDIBLE literature calibration, drop in an open corpus (run on")
    print("  your machine; these live on Figshare/Zenodo, reachable there not here):")
    print("   - Scientific Data 2023, 6086 records, DOI 10.1038/s41597-023-02089-z")
    print("   - Scientific Data 2024 (LLM-enhanced), 6985 records, DOI 10.1038/s41597-024-03180-9")
    print("   Map their (material/voltage/current density) -> features and FE -> target,")
    print("   then: calibrate_and_evaluate(X, y). The DFT-descriptor version needs the")
    print("   adsorption-energy features (Catalysis-Hub) joined per material via join_labeled.")


if __name__ == "__main__":
    main()
