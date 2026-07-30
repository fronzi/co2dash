"""
End-to-end demo. Runs with NO new DFT and NO network.

IMPORTANT: every numeric input below is an ILLUSTRATIVE PLACEHOLDER chosen only
to exercise the pipeline. They are NOT empirical claims. Replace each with a
sourced, tier-tagged value before drawing any conclusion. The point of the demo
is to show the machinery runs end to end and produces decision-relevant outputs.
"""
import os, sys
# Make the package importable whether or not it was pip-installed: add ../src.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
from co2dash import (RXN_METHANOL, Scenario, BayesianLinearSurrogate,
                     propagate_mc, sobol_indices, rank_candidates, Candidate)

rxn = RXN_METHANOL

# ----- nominal scenario (ILLUSTRATIVE placeholders) ------------------------
base = Scenario(
    n_electrons=rxn.n_electrons, molar_mass_prod=rxn.molar_mass_prod,
    m_co2=rxn.kg_co2_per_kg_prod, m_h2=0.0,
    faradaic_efficiency=0.60, cell_voltage=3.0,          # placeholder performance
    capex_total=5.0e7, annual_production_kg=2.0e7,        # placeholder plant
    opex_fix_per_yr=3.0e6, disc_rate=0.08, lifetime_yr=20,
    c_co2=0.05, c_elec=0.06, c_h2=0.0,                   # placeholder prices
    lcop_conventional=0.40,                               # placeholder fossil MeOH cost
    grid_intensity=0.02, e_capture=0.1, e_process=0.05,  # placeholder LCA terms (low-C grid)
    release_fraction=0.10)                                # durable-utilisation framing
# NOTE: with release_fraction=1.0 (methanol burned as fuel) the storage credit
# (1-phi)*m_co2 -> 0 and net abatement is *always* negative -> MAC=inf. That is
# the utilisation!=sequestration lesson, not a bug. We use phi=0.1 (durable
# product) here so the rest of the machinery has a feasible regime to act on.

print("=== Nominal point evaluation ===")
for k, v in base.evaluate().items():
    print(f"  {k:30s} {v:.4g}")

# ----- uncertainty propagation -> MAC distribution -------------------------
CARBON_PRICE = 2.00  # $/kg CO2 -- ILLUSTRATIVE, set high to exercise the loop.
# (At a realistic ~$0.1-0.3/kg, electrochem methanol is far from feasible with
#  these placeholders -- itself an honest finding. Here we raise it so the
#  feasibility decision is non-degenerate and the active-learning ranking has
#  signal to act on.)
unc = {
    "faradaic_efficiency": ("normal", 0.60, 0.12),   # surrogate-class spread
    "cell_voltage":        ("normal", 3.0, 0.3),
    "c_elec":              ("lognormal", 0.06, 1.4),  # energy price volatility
    "capex_total":         ("lognormal", 5.0e7, 1.5), # ESTIMATED tier -> wide
    "grid_intensity":      ("uniform", 0.005, 0.05),
}
res = propagate_mc(base, unc, CARBON_PRICE, n=40_000, seed=1)
print("\n=== Monte-Carlo MAC distribution ($/kg CO2) ===")
print(f"  median {res['mac_median']:.3f}  [P05 {res['mac_p05']:.3f}, "
      f"P95 {res['mac_p95']:.3f}]")
print(f"  P(MAC < carbon price) = {res['p_mac_below_carbon_price']:.2%}")
print(f"  P(net abatement > 0)  = {res['p_net_positive']:.2%}")

# ----- global sensitivity: which lever matters most? -----------------------
bounds = {
    "faradaic_efficiency": (0.30, 0.95),
    "cell_voltage":        (2.2, 4.0),
    "c_elec":              (0.02, 0.15),
    "capex_total":         (2.0e7, 1.2e8),
    "grid_intensity":      (0.005, 0.05),
}
print("\n=== Sobol sensitivity of MAC (S1=direct, ST=total) ===")
sob = sobol_indices(base, bounds, n=512)
for name, idx in sorted(sob.items(), key=lambda kv: kv[1]["ST"], reverse=True):
    print(f"  {name:22s} S1={idx['S1']:+.3f}  ST={idx['ST']:+.3f}")

# ----- surrogate + active learning: which candidate to compute next? -------
# ILLUSTRATIVE: synthetic descriptor->FE training data standing in for public
# Catalysis-Hub / OC20 adsorption energies. Replace with real fetched data.
rng = np.random.default_rng(0)
keys = ["dE_CO", "dE_COOH"]
Xtr = rng.uniform([-1.5, -1.0], [0.5, 1.5], size=(40, 2))
# synthetic Sabatier-like response (placeholder ground truth):
ytr = 0.9 * np.exp(-((Xtr[:, 0] + 0.6) ** 2 + (Xtr[:, 1] - 0.2) ** 2)) \
      + rng.normal(0, 0.03, 40)
surr = BayesianLinearSurrogate(degree=2).fit(Xtr, ytr)

cands = [Candidate(material_id=f"cand_{i}",
                   descriptors={"dE_CO": float(x[0]), "dE_COOH": float(x[1])})
         for i, x in enumerate(rng.uniform([-1.5, -1.0], [0.5, 1.5], size=(12, 2)))]

ranked = rank_candidates(cands, surr, keys, base, CARBON_PRICE, seed=2)
print("\n=== Active-learning ranking (top 5 to compute next, by EVOI) ===")
print(f"  {'id':9s} {'FE_pred':>8s} {'FE_std':>7s} {'MAC_med':>9s} "
      f"{'p_feas':>7s} {'acq':>7s}")
for r in ranked[:5]:
    print(f"  {r['material_id']:9s} {r['pred_mean']:8.3f} {r['pred_std']:7.3f} "
          f"{r['mac_median']:9.3f} {r['p_feas']:7.2f} {r['acquisition']:7.4f}")

print("\nDONE. All numbers above are illustrative placeholders.")
