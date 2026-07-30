"""
Train + calibrate a surrogate on REAL descriptors using the physics activity
proxy as the target -- the full loop with no experimental FE required.

Flow:
  1. load cached Catalysis-Hub descriptors (./data) or, if absent, generate a
     CLEARLY-LABELLED synthetic table so the code path runs offline;
  2. compute the CHE activity target y = proxy cell voltage per surface;
  3. train the Bayesian surrogate X(descriptors) -> y(cell voltage);
  4. calibrate its uncertainty (coverage -> temperature scaling);
  5. rank surfaces by EVOI toward MAC feasibility, with the surrogate predicting
     cell_voltage (NOT faradaic efficiency -- activity != selectivity).

Run:
  python examples/fetch_real_data.py        # first, to populate ./data (real)
  python examples/train_real_surrogate.py

Replace the proxy target with experimental FE when available; only steps 2 and
the target_field in step 5 change.
"""
import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np

from co2dash import (load_dataset, build_descriptor_table, build_activity_targets,
                     BayesianLinearSurrogate, coverage_report, miscalibration_area,
                     TemperatureScaler, SplitConformal, CalibratedSurrogate,
                     Candidate, rank_candidates, RXN_CO, Scenario)

PRODUCT = "CO"
KEYS = ["dE_CO", "dE_COOH"]
DATA_DIR = "data"


def load_real_descriptor_table():
    files = glob.glob(os.path.join(DATA_DIR, "chub_*.json"))
    if not files:
        return None
    records = []
    for f in files:
        records.extend(load_dataset(f)["records"])
    return build_descriptor_table(records)


def synthetic_table(n=80, seed=0):
    """CLEARLY SYNTHETIC fallback so the script runs offline. NOT chemistry data:
    arbitrary descriptor values only to exercise the pipeline."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        dCO = float(rng.uniform(-0.6, 0.8))
        dCOOH = float(rng.uniform(-0.2, 1.2))
        rows.append({"surface": f"Synth{i}", "facet": "111",
                     "dE_CO": dCO, "dE_COOH": dCOOH})
    return rows


# ---- 1. data ------------------------------------------------------------------
table = load_real_descriptor_table()
if table is None:
    print("!! No cached Catalysis-Hub data in ./data -- using SYNTHETIC descriptors "
          "(run examples/fetch_real_data.py first for real data).")
    table = synthetic_table()
else:
    print(f"Loaded {len(table)} real surfaces from {DATA_DIR}/")

# ---- 2. physics activity target ----------------------------------------------
targets = build_activity_targets(table, product=PRODUCT, descriptor_keys=KEYS)
print(f"Surfaces with computable {PRODUCT} pathway: {len(targets)}")
if len(targets) < 8:
    print("Too few complete surfaces to train meaningfully -- fetch more data.")
X = np.array([[t["descriptors"][k] for k in KEYS] for t in targets])
y = np.array([t["v_cell"] for t in targets])               # target = proxy cell voltage
print(f"Target (proxy cell voltage) range: {y.min():.2f}-{y.max():.2f} V")

# ---- 3. train surrogate -------------------------------------------------------
cut = max(8, int(0.6 * len(X)))
surr = BayesianLinearSurrogate(degree=2).fit(X[:cut], y[:cut])

# ---- 4. calibrate -------------------------------------------------------------
mcal, scal = surr.predict(X[cut:])
print("\nCalibration on held-out surfaces:")
print("  coverage BEFORE:", {k: round(v, 2) for k, v in coverage_report(mcal, scal, y[cut:]).items()})
ts = TemperatureScaler().fit(mcal, scal, y[cut:])
m2, s2 = ts.transform(mcal, scal)
print(f"  temperature s = {ts.s:.2f}  (miscalibration "
      f"{miscalibration_area(mcal, scal, y[cut:]):.3f} -> {miscalibration_area(m2, s2, y[cut:]):.3f})")
conf = SplitConformal().fit(m2, s2, y[cut:])
calibrated = CalibratedSurrogate(surr, scaler=ts, conformal=conf, alpha=0.1)

# ---- 5. EVOI ranking with the calibrated VOLTAGE surrogate --------------------
r = RXN_CO
base = Scenario(
    n_electrons=r.n_electrons, molar_mass_prod=r.molar_mass_prod,
    m_co2=r.kg_co2_per_kg_prod, m_h2=0.0,
    faradaic_efficiency=0.85, cell_voltage=3.0,          # FE held fixed (separate input)
    capex_total=5.0e7, annual_production_kg=2.0e7, opex_fix_per_yr=3.0e6,
    disc_rate=0.08, lifetime_yr=20, c_co2=0.05, c_elec=0.06, c_h2=0.0,
    lcop_conventional=0.40, grid_intensity=0.02, e_capture=0.1, e_process=0.05,
    release_fraction=0.10)
carbon_price = 0.40   # $/kg CO2 (=$400/t), illustrative -- placed near the MAC
# spread so the feasibility decision (and thus EVOI) is non-degenerate.

cands = [Candidate(material_id=t["material_id"], descriptors=t["descriptors"])
         for t in targets]
ranked = rank_candidates(cands, calibrated, KEYS, base, carbon_price,
                         target_field="cell_voltage", clip=(0.5, 8.0), seed=1)
print("\nTop 8 surfaces to compute next (EVOI; surrogate predicts CELL VOLTAGE):")
print(f"  {'material_id':28s} {'V_pred':>7s} {'V_std':>6s} {'MAC_med':>8s} {'p_feas':>7s} {'acq':>6s}")
for row in ranked[:8]:
    print(f"  {row['material_id'][:28]:28s} {row['pred_mean']:7.2f} {row['pred_std']:6.2f} "
          f"{row['mac_median']:8.2f} {row['p_feas']:7.2f} {row['acquisition']:6.3f}")

print("\nDONE. Whole loop ran on the physics activity proxy. Swap y for experimental"
      " FE (target_field='faradaic_efficiency') when available.")
