"""
End-to-end descriptor→activity/FE pipeline with a DATA-QUALITY GATE.

    python examples/run_pipeline.py data.csv --target U_L
    python examples/run_pipeline.py data.csv            # last column is the target

Steps: load -> data_quality_report (gate) -> calibrated surrogate -> honest
verdict. Designed for possibly LOW-QUALITY data: the quality tier decides how
much to trust the output. 'poor' -> results are exploratory only; the pipeline
still runs but says so loudly rather than pretending.
"""
import os, sys, csv, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from co2dash.quality import data_quality_report
from co2dash.calibration_harness import calibrate_and_evaluate
from co2dash.surrogate import BayesianLinearSurrogate


def load_xy(path, target=None):
    rows = list(csv.DictReader(io.StringIO(open(path, encoding="utf-8").read())))
    cols = list(rows[0].keys())
    tcol = target or cols[-1]
    xcols = [c for c in cols if c != tcol]
    def f(v):
        try: return float(v)
        except: return np.nan
    X = np.array([[f(r[c]) for c in xcols] for r in rows], float)
    y = np.array([f(r[tcol]) for r in rows], float)
    m = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    return X[m], y[m], xcols, tcol


def main(path, target=None):
    X, y, names, tcol = load_xy(path, target)
    print(f"Loaded {len(y)} complete rows; target = '{tcol}'\n")

    q = data_quality_report(X, y, feature_names=names)
    print(q.summary(), "\n")
    if not q.usable:
        print(">> QUALITY POOR: point predictions are NOT trustworthy. Treat any")
        print("   output below as exploratory only, and collect more/cleaner data.")
    # drop constant columns the report flagged
    keep = X.std(axis=0) > 1e-9
    X = X[:, keep]

    if len(y) < 12:
        print("Too few rows to calibrate. Stopping (nothing fabricated).")
        return
    r = calibrate_and_evaluate(
        X, y, surrogate_factory=lambda Xt, yt: BayesianLinearSurrogate().fit(Xt, yt),
        alpha=0.1, seed=0)
    print("=== calibrated surrogate ===")
    print(f"  raw 90% coverage {r.coverage_before[0.9]:.0%} -> after {r.coverage_after[0.9]:.0%} "
          f"(conformal {r.conformal_coverage:.0%}); temperature s={r.temperature_s:.2f}")
    print(f"  miscalibration {r.miscal_before:.3f} -> {r.miscal_after:.3f}")

    trust = {"ok": "trust the calibrated intervals",
             "marginal": "use the calibrated intervals but treat point values cautiously",
             "poor": "exploratory only — do not act on point values"}[q.tier]
    print(f"\nVERDICT: data quality {q.tier.upper()} -> {trust}.")
    print("Downstream, these calibrated predictions feed the activity/MAC loop; the")
    print("wider the calibrated uncertainty, the wider (honestly) the MAC distribution.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tgt = sys.argv[sys.argv.index("--target") + 1] if "--target" in sys.argv else None
    if not args:
        print(__doc__); print("ERROR: pass a CSV (features + a target column).")
        sys.exit(1)
    main(args[0], tgt)
