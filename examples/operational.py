"""
Operational example: the three new pieces working together.

  piece 1  load a tier-tagged scenario from YAML + provenance registry
  piece 2  vectorised Monte-Carlo, with distributions AUTO-DERIVED from tiers
  piece 3  calibrate the surrogate (temperature + conformal) before active learning

Run:  python examples/operational.py
All numeric values remain illustrative placeholders.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np

from co2dash import (load_scenario, propagate_mc, BayesianLinearSurrogate,
                     coverage_report, miscalibration_area, TemperatureScaler,
                     SplitConformal, CalibratedSurrogate, rank_candidates, Candidate)

YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_SCENARIO_methanol.yaml")
CARBON_PRICE = 2.0  # $/kg CO2, illustrative (see demo.py note)

# ---- piece 1: load scenario + provenance --------------------------------------
base, reg = load_scenario(YAML)
print("=== Provenance registry (every number has a source) ===")
print(f"  {'field':22s} {'value':>12s} {'eff_std':>10s} {'tier':>14s}")
for row in reg.table():
    print(f"  {row['field']:22s} {row['value']:12.4g} {row['eff_std']:10.4g} "
          f"{row['tier']:>14s}")

# ---- piece 2: MC with distributions derived from the tiers --------------------
uncertain_fields = ["faradaic_efficiency", "cell_voltage", "c_elec",
                    "capex_total", "grid_intensity"]
dists = reg.mc_distributions(uncertain_fields)
print("\n=== MC distributions auto-derived from provenance tiers ===")
for k, d in dists.items():
    print(f"  {k:22s} {d}")
res = propagate_mc(base, dists, CARBON_PRICE, n=100_000, seed=0)
print(f"\n  MAC median {res['mac_median']:.3f}  "
      f"[P05 {res['mac_p05']:.3f}, P95 {res['mac_p95']:.3f}]  $/kg CO2")
print(f"  P(MAC < carbon price) = {res['p_mac_below_carbon_price']:.1%}")
print(f"  P(net abatement > 0)  = {res['p_net_positive']:.1%}")

# ---- piece 3: calibrate the surrogate before using it for active learning -----
# ILLUSTRATIVE synthetic descriptor->FE data, deliberately OVER-CONFIDENT model
# (small beta noise) so calibration has something to fix.
rng = np.random.default_rng(0)
keys = ["dE_CO", "dE_COOH"]
Xtr = rng.uniform([-1.5, -1.0], [0.5, 1.5], size=(60, 2))
ytr = 0.9 * np.exp(-((Xtr[:, 0] + 0.6) ** 2 + (Xtr[:, 1] - 0.2) ** 2)) \
      + rng.normal(0, 0.06, 60)
surr = BayesianLinearSurrogate(beta=400.0, degree=2).fit(Xtr, ytr)  # beta high -> overconfident

# hold-out calibration set
Xcal = rng.uniform([-1.5, -1.0], [0.5, 1.5], size=(40, 2))
ycal = 0.9 * np.exp(-((Xcal[:, 0] + 0.6) ** 2 + (Xcal[:, 1] - 0.2) ** 2)) \
       + rng.normal(0, 0.06, 40)
mcal, scal = surr.predict(Xcal)

print("\n=== Calibration (piece 3) ===")
print("  coverage BEFORE:", {k: round(v, 2)
      for k, v in coverage_report(mcal, scal, ycal).items()})
ts = TemperatureScaler().fit(mcal, scal, ycal)
mc2, sc2 = ts.transform(mcal, scal)
print(f"  temperature scale s = {ts.s:.2f}  "
      f"(miscalibration {miscalibration_area(mcal, scal, ycal):.3f} "
      f"-> {miscalibration_area(mc2, sc2, ycal):.3f})")
conf = SplitConformal().fit(mc2, sc2, ycal)
calibrated = CalibratedSurrogate(surr, scaler=ts, conformal=conf, alpha=0.1)

# active learning WITH the calibrated surrogate
cands = [Candidate(material_id=f"cand_{i}",
                   descriptors={"dE_CO": float(x[0]), "dE_COOH": float(x[1])})
         for i, x in enumerate(rng.uniform([-1.5, -1.0], [0.5, 1.5], size=(12, 2)))]
ranked = rank_candidates(cands, calibrated, keys, base, CARBON_PRICE, seed=2)
print("\n=== Active learning with CALIBRATED surrogate (top 5) ===")
print(f"  {'id':9s} {'FE_pred':>8s} {'FE_std':>7s} {'p_feas':>7s} {'acq':>7s}")
for r in ranked[:5]:
    print(f"  {r['material_id']:9s} {r['pred_mean']:8.3f} {r['pred_std']:7.3f} "
          f"{r['p_feas']:7.2f} {r['acquisition']:7.4f}")

print("\nDONE. Pieces 1+2+3 wired together. Values are illustrative placeholders.")
