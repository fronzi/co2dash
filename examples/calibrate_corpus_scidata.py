"""
Calibrate on REAL experimental FE from the Scientific Data 2023 CO2RR corpus.

    python examples/calibrate_corpus_scidata.py /path/to/merge_data_final.xls

Source: Zhang et al., "A corpus of CO2 electrocatalytic reduction process
extracted from the scientific literature", Scientific Data 2023,
doi:10.1038/s41597-023-02089-z  (github: electrocatalytic_db, gold_corpus/
merge_data_final.xls). Requires: pandas, xlrd.

The corpus is entity-level (one row per extracted mention) with a Chinese label
scheme that pairs a product with its faradaic efficiency by ordinality:
  第一产物 / 第一产物法拉第  = first product / its FE   (likewise 第二/第三)
  电流密度 = current density,  法拉第效率电压 = potential (V vs RHE)
  material-type labels: Cu, CuOx, Cu/C, alloy, composite, ...

Reconstruction (honest scope): product↔FE are paired by the corpus's own
ordinality labels; current density and potential are linked at the PAPER level
(their per-measurement linkage is not annotated), so they are coarse features.
This yields a real experimental-FE dataset with some condition-linkage noise —
a genuine stress test for the calibration gate, not a clean structure-property
model. y = measured faradaic efficiency (fraction).
"""
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from co2dash.calibration_harness import calibrate_and_evaluate
from co2dash.surrogate import BayesianLinearSurrogate

PROD = {"第一产物": "第一产物法拉第", "第二产物": "第二产物法拉第", "第三产物": "第三产物法拉第"}
MAT_TYPES = ["Cu-M", "Cu", "CuOx", "Cu/C", "composite", "alloy", "CuSx",
             "Cu-MOF", "Cu(Ox)-MOx", "Cu-MOx"]
CD, VOLT = "电流密度", "法拉第效率电压"


def _num(s):
    m = re.search(r"[-−]?\d+\.?\d*", str(s).replace("−", "-"))
    return float(m.group()) if m else None


def reconstruct(xls_path):
    import pandas as pd
    x = pd.read_excel(xls_path)
    x["entity"] = x["entity"].astype(str)
    rows = []
    for _, g in x.groupby("doi"):
        lab = g["entity_label"]
        mats = g[lab.isin(MAT_TYPES)]["entity_label"].tolist()
        mat = max(set(mats), key=mats.count) if mats else "other"
        cds = [abs(_num(v)) for v in g[lab == CD]["entity"] if _num(v) is not None]
        cd_med = float(np.median(cds)) if cds else np.nan
        volts = [_num(v) for v in g[lab == VOLT]["entity"] if _num(v) is not None]
        for p_lab, fe_lab in PROD.items():
            prods = g[lab == p_lab]["entity"].tolist()
            fes = [_num(v) for v in g[lab == fe_lab]["entity"]]
            for i, (p, fe) in enumerate(zip(prods, fes)):
                if fe is None:
                    continue
                fe = fe / 100.0 if fe > 1.5 else fe
                if not (0.0 <= fe <= 1.0):
                    continue
                v = volts[i] if i < len(volts) else (float(np.median(volts)) if volts else np.nan)
                rows.append({"material": mat, "product": p.strip(),
                             "faradaic_efficiency": fe, "current_density": cd_med,
                             "cell_voltage": v})
    return rows


def featurize(rows):
    prods = sorted({r["product"] for r in rows})
    mats = sorted({r["material"] for r in rows})
    cd = np.array([r["current_density"] for r in rows], float)
    vv = np.array([r["cell_voltage"] for r in rows], float)
    cd_med = np.nanmedian(cd); v_med = np.nanmedian(vv)
    oh = lambda v, cats: [1.0 if v == c else 0.0 for c in cats]
    X, y = [], []
    for r in rows:
        c = r["current_density"] if np.isfinite(r["current_density"]) else cd_med
        v = r["cell_voltage"] if np.isfinite(r["cell_voltage"]) else v_med
        X.append([c, v] + oh(r["product"], prods) + oh(r["material"], mats))
        y.append(r["faradaic_efficiency"])
    X = np.asarray(X, float); y = np.asarray(y, float)
    return X[:, X.std(0) > 1e-9], y


def main(xls_path):
    rows = reconstruct(xls_path)
    X, y = featurize(rows)
    print(f"Reconstructed {len(rows)} experimental records; features {X.shape[1]}")
    print(f"FE (fraction): min {y.min():.2f}, mean {y.mean():.2f}, max {y.max():.2f}\n")
    r = calibrate_and_evaluate(
        X, y, surrogate_factory=lambda Xt, yt: BayesianLinearSurrogate().fit(Xt, yt),
        alpha=0.1, seed=0)
    print("=== calibration gate on REAL experimental FE (Scientific Data 2023) ===")
    print(f"  raw 90% interval covers {r.coverage_before[0.9]:.0%} of held-out points"
          f" -> {'OVER-CONFIDENT' if r.coverage_before[0.9] < 0.88 else 'ok'}")
    print(f"  temperature s = {r.temperature_s:.2f}; "
          f"miscalibration {r.miscal_before:.3f} -> {r.miscal_after:.3f}")
    print(f"  after: 90% coverage {r.coverage_after[0.9]:.0%}, "
          f"conformal {r.conformal_coverage:.0%}")
    print("  " + r.summary())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); print("ERROR: pass the path to gold_corpus/merge_data_final.xls")
        sys.exit(1)
    main(sys.argv[1])
