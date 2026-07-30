"""
Descriptor→activity on PUBLIC same-source CO2RR DFT data (no user data).

    python examples/descriptor_activity_public.py

Dataset: Wu, Zhang, Cheng, Lu & Zhang, "Machine Learning Investigation of
Supplementary Adsorbate Influence on Copper for Enhanced Electrochemical CO2
Reduction Performance", J. Phys. Chem. C 2021, 125, 15363; data at
github.com/LuGroup/CO2RR-Adsorbates (fetched at run time from raw.github, cited,
not redistributed). Real VASP DFT on Cu(100) with supplementary adsorbates.

Task: predict the DFT-computed **C–C coupling free energy** (G_C2O2 − G_CO) — a
CO2RR activity descriptor for the C2 pathway — from cheap elemental features of
the adsorbate pair, and run the calibration gate. This is 'descriptor→activity'
(not →FE): activity ties to descriptors through solid physics, FE does not.

Honest scope: same-source and public, but SMALL (≈65 fully-complete rows), so the
calibration numbers are illustrative. The large CO2RR descriptor→limiting-
potential datasets (Chen et al. HEA, 691 pts; ACS Catalysis FeCoNiCuMo, U_L
0.29–0.51 V) live on Figshare and drop into this same pipeline on a machine that
can reach Figshare — swap the loader, keep calibrate_and_evaluate.
"""
import os, sys, csv, io, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from co2dash.calibration_harness import calibrate_and_evaluate
from co2dash.surrogate import BayesianLinearSurrogate

URL = ("https://raw.githubusercontent.com/LuGroup/CO2RR-Adsorbates/master/"
       "Data/CO%20Dimerization%20Full%20Data.csv")


def load():
    text = urllib.request.urlopen(URL, timeout=60).read().decode("utf-8")
    rows = list(csv.reader(io.StringIO(text)))
    hdr, data = rows[0], rows[1:]
    xcols = list(range(2, len(hdr) - 1))          # skip adsorbate names; target = last col
    X, y = [], []
    for r in data:
        try:
            xs = [float(r[c]) for c in xcols]; yy = float(r[-1])
        except ValueError:
            continue                               # skip incomplete rows (no imputation)
        X.append(xs); y.append(yy)
    X = np.asarray(X, float); y = np.asarray(y, float)
    return X[:, X.std(0) > 1e-9], y, hdr[-1]


def main():
    try:
        X, y, target = load()
    except Exception as e:
        print(f"(could not fetch LuGroup data: {e}) — run on a network reaching raw.github")
        return
    print(f"Public CO2RR DFT (Wu et al. 2021): n={len(y)} complete rows, "
          f"features={X.shape[1]}")
    print(f"target = {target} (eV), range [{y.min():.2f}, {y.max():.2f}]\n")
    r = calibrate_and_evaluate(
        X, y, surrogate_factory=lambda Xt, yt: BayesianLinearSurrogate().fit(Xt, yt),
        alpha=0.1, seed=1)
    print("=== descriptor→activity calibration (public same-source CO2RR DFT) ===")
    print(f"  raw 90% interval covers {r.coverage_before[0.9]:.0%} of held-out points")
    print(f"  temperature s = {r.temperature_s:.2f}; "
          f"miscalibration {r.miscal_before:.3f} -> {r.miscal_after:.3f}")
    print(f"  after: 90% coverage {r.coverage_after[0.9]:.0%}, "
          f"conformal {r.conformal_coverage:.0%}")
    print("  (small n -> illustrative; the gate detects and reduces the miscalibration)")


if __name__ == "__main__":
    main()
