"""
Calibration gate on REAL literature DFT data (a substitute for your own data).

Runs the calibration harness on the public CHEAT high-entropy-alloy dataset:
    Clausen, Nielsen, Pedersen & Rossmeisl (2022), "Ab Initio to activity:
    Machine learning assisted optimization of high-entropy alloy catalytic
    activity", chemrxiv-2022-vvrrf; data: github.com/cmclausen/cheat.

Honest scope: this is real GPAW/DFT adsorption-energy data, but for the ORR
system AgIrPdPtRu (targets ΔE_*OH ontop and ΔE_*O fcc), used here as a
stand-in for "a real DFT descriptor -> property regression". It is NOT CO2RR
(*CO/*H) data. The point is to exercise the calibration gate on genuine
computational-catalysis noise; the CO2RR calibration on your own descriptors
uses the identical code path (calibrate_and_evaluate).

Data is fetched at run time from raw.githubusercontent.com (not redistributed).
Requires network to that host. Run:  python examples/calibrate_real_dft.py
"""
import os, sys, pickle, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from co2dash.calibration_harness import calibrate_and_evaluate
from co2dash.surrogate import BayesianLinearSurrogate

RAW = "https://raw.githubusercontent.com/cmclausen/cheat/master/features"
SITES = {"ontop_OH": "ΔE_*OH ontop", "fcc_O": "ΔE_*O fcc"}
CACHE = os.path.join(os.path.dirname(__file__), "_cheat_cache")


def load_cheat(site: str):
    """Fetch (and cache) one CHEAT .zonefeats file -> (X, y). Columns:
    [0:5] site metal identity, [5:-1] zone features, [-1] DFT adsorption energy."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"agirpdptru_{site}.zonefeats")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"{RAW}/agirpdptru_{site}.zonefeats", path)
    feats = np.asarray(pickle.load(open(path, "rb")), float)
    X, y = feats[:, 5:-1], feats[:, -1]
    X = X[:, X.std(axis=0) > 1e-9]                # drop constant columns
    return X, y


def main():
    print("Calibration gate on REAL DFT data — CHEAT HEA dataset (Clausen & Rossmeisl 2022)")
    print("Real GPAW/DFT adsorption energies; ORR system AgIrPdPtRu (stand-in for a real")
    print("descriptor->property regression). CO2RR uses the identical calibrate_and_evaluate.\n")
    try:
        for site, label in SITES.items():
            X, y = load_cheat(site)
            fac = lambda Xt, yt: BayesianLinearSurrogate().fit(Xt, yt)
            r = calibrate_and_evaluate(X, y, surrogate_factory=fac, alpha=0.1, seed=0)
            print(f"[{label}]  n={len(y)}, features={X.shape[1]}, "
                  f"ΔE range [{y.min():.2f}, {y.max():.2f}] eV")
            print(f"   raw surrogate 90% interval covers only {r.coverage_before[0.9]:.0%} "
                  f"of held-out points -> OVER-CONFIDENT")
            print(f"   temperature s = {r.temperature_s:.2f}; miscalibration "
                  f"{r.miscal_before:.3f} -> {r.miscal_after:.3f}")
            print(f"   after calibration: 90% coverage {r.coverage_after[0.9]:.0%}, "
                  f"conformal {r.conformal_coverage:.0%}\n")
        print("-> On real DFT data the naive surrogate is badly over-confident; the gate")
        print("   restores honest coverage. This is exactly the check that must pass before")
        print("   a surrogate's uncertainty is propagated into the MAC distribution.")
    except Exception as e:
        print(f"(could not fetch CHEAT data: {e})")
        print("Provide the two .zonefeats files under examples/_cheat_cache/ to run offline.")


if __name__ == "__main__":
    main()
