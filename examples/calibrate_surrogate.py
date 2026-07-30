"""
Calibration gate for the co2dash surrogate.

    python examples/calibrate_surrogate.py

Part A validates the calibration PROCEDURE on synthetic ground truth (you need
known noise to verify that empirical coverage matches the nominal level).
Part B is the REAL path: it calibrates the surrogate on your real DFT
descriptors + measured faradaic efficiencies, and runs on your machine.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from co2dash.calibration_harness import (make_linear_synthetic, ConstStdSurrogate,
                                         calibrate_and_evaluate, join_labeled)
from co2dash.surrogate import BayesianLinearSurrogate


def _hr(t): print("\n" + t + "\n" + "-" * len(t))


def part_a_validate_procedure():
    _hr("A) Validate the calibration PROCEDURE on synthetic ground truth")
    sigma_true = 0.05
    X, y, _ = make_linear_synthetic(n=600, d=4, sigma_true=sigma_true, seed=1)

    for label, reported in (("over-confident (std 1.8x too small)", sigma_true / 1.8),
                            ("under-confident (std 1.8x too large)", sigma_true * 1.8),
                            ("well-specified (reports true std)",     sigma_true)):
        fac = (lambda r: (lambda Xt, yt: ConstStdSurrogate(r).fit(Xt, yt)))(reported)
        rep = calibrate_and_evaluate(X, y, surrogate_factory=fac, alpha=0.1, seed=1)
        print(f"\n  {label}")
        print(f"    temperature s = {rep.temperature_s:.3f}")
        print(f"    miscalibration {rep.miscal_before:.3f} -> {rep.miscal_after:.3f}")
        print(f"    90% coverage on test: before {rep.coverage_before[0.9]:.0%}, "
              f"after {rep.coverage_after[0.9]:.0%}, conformal {rep.conformal_coverage:.0%}")
    print("\n  -> the procedure recovers honest coverage regardless of the surrogate's")
    print("     initial over/under-confidence; well-specified models are left alone.")


def part_b_real_path():
    _hr("B) Calibrate the REAL surrogate on your experimental FE (runs on your machine)")
    print("  Steps:")
    print("   1. python examples/fetch_real_data.py         # cache Catalysis-Hub descriptors")
    print("   2. build_descriptor_table(...)                # -> {material_id: descriptor vector}")
    print("   3. read your measured FE as a CSV (material, FE) via co2dash.intake.read_csv")
    print("   4. join_labeled(descriptors, fe_targets, keys) -> X, y")
    print("   5. calibrate_and_evaluate(X, y, surrogate_factory=<KAN/BNN/BayesianLinear>)")
    print("   6. if miscalibration is high, the fitted temperature/conformal correct the")
    print("      surrogate's uncertainty BEFORE it is propagated into the MAC distribution.")

    # If a real labelled dataset is present, run it; otherwise stop (no fabrication).
    path = os.path.join(os.path.dirname(__file__), "labeled_fe_dataset.npz")
    if not os.path.exists(path):
        print(f"\n  (no real dataset at {os.path.basename(path)} — provide X (descriptors) and")
        print("   y (measured FE) to run the gate on real data; nothing is fabricated here.)")
        return
    d = np.load(path)
    rep = calibrate_and_evaluate(d["X"], d["y"],
                                 surrogate_factory=lambda Xt, yt: BayesianLinearSurrogate().fit(Xt, yt))
    print("\n  REAL calibration report:")
    print("   " + rep.summary())


if __name__ == "__main__":
    part_a_validate_procedure()
    part_b_real_path()
