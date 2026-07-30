# co2dash — zero-new-DFT techno-economic / environmental platform for CO₂ utilisation

A reference scaffold for a CCU decision-support platform that **computes no new
DFT**. It federates public data and concentrates all value in the parts no
existing platform does well: an **uncertainty-aware translation layer**
(technical KPI → LCOP / net-abatement / MAC) and an **active-learning loop** that
ranks which candidate is worth a DFT run *next*, targeting the techno-economic
feasibility decision rather than abstract model accuracy.

## Why this is the defensible contribution
The discovery layer already exists (Materials Project, Catalysis-Hub, Open
Catalyst). Rebuilding it is not novel. The gap is the **translation layer with
propagated uncertainty** and its **coupling to active learning**. That is what
this scaffold implements.

## Architecture
```
connectors.py        public DFT / energy / grid data, each record tier-tagged
schema.py            canonical data model: value + uncertainty + provenance TIER
surrogate.py         descriptors -> KPI (mean, std)   [swap-in slot for KAN/BNN]
techno_economic.py   KPI -> LCOP / net-abatement / MAC   (pure, testable functions)
uncertainty.py       Monte-Carlo propagation + Sobol global sensitivity
active_learning.py   EVOI acquisition: which candidate to compute next
app/streamlit_app.py thin UI over the core (no business logic in the GUI)
```

## Key design decisions
- **Data tier = noise model.** Each `Quantity` carries a provenance tier
  (COMPUTED / LAB_VALIDATED / LIT_EXTRACTED / ESTIMATED). The tier sets a floor on
  relative std (`schema.TIER_REL_STD_FLOOR`), which propagates to the MAC
  distribution. A scraped FE enters with larger variance than a lab-validated one.
- **Monte-Carlo, not linear error propagation.** The TEA/LCA map is strongly
  nonlinear (1/FE, division by net-abatement); a Taylor expansion would misstate
  the tails.
- **Acquisition on the bounded feasibility decision.** The active-learning score
  is defined on `p_feas = P(MAC < carbon_price)` (bounded), not on raw MAC
  (unbounded, heavy-tailed). This avoids rewarding candidates whose only merit is
  enormous predictive variance.
- **Closed-form Bayesian surrogate** as a placeholder; the only contract is
  `predict(X) -> (mean, std)`, so a KAN/BNN drops in unchanged.

## Honesty boundaries (read before using outputs)
- Every numeric default in `examples/demo.py` is an **illustrative placeholder**,
  not an empirical claim. Replace with sourced, tier-tagged values.
- The connectors are **structurally-correct templates**; verify each API's
  current endpoint / schema / licence before relying on it. Nothing fabricates
  data — failed calls raise.
- The platform **can** prioritise, explore scenarios, run sensitivity, and rank
  experiments. It **cannot** produce a predictive plant cost without lab-grade
  stability and CAPEX data — that bottleneck is not on the web.

## Run
```bash
pip install -r requirements.txt
PYTHONPATH=src python examples/demo.py        # end-to-end, no network, no DFT
PYTHONPATH=src python -m pytest tests/         # validation checks
streamlit run app/streamlit_app.py             # interactive explorer
```

## Suggested MVP scope
One product, one route (e.g. CO₂→formate, electrochemical). Pull discovery data
from Catalysis-Hub/OC20 + grid intensity from your TSO (AEMO NEM for AU); curate a
small CAPEX/price set by hand; build the rigorous TEA/LCA+UQ core on that narrow
slice. A narrow, honest slice is a citable preliminary result.

## Operational additions (pieces 1–3)
- **`config.py` (piece 1).** `load_scenario("scenario.yaml") -> (Scenario, ProvenanceRegistry)`.
  Every field is a tagged record `{value, std, tier, source}`. The registry renders a
  provenance table and **auto-derives the Monte-Carlo distributions from the tiers**
  (`reg.mc_distributions(fields)`) — provenance becomes uncertainty, specified once.
- **Vectorised path (piece 2).** `techno_economic.evaluate_array` + a vectorised
  `propagate_mc`/`sobol_indices`: a 200k-sample MC runs in ~0.03 s (no Python loop).
  The scalar `Scenario.evaluate` is unchanged; vectorised == scalar element-wise (tested).
- **`calibration.py` (piece 3).** `coverage_report`, `TemperatureScaler` (Gaussian-MLE
  global std scaling), `SplitConformal` (distribution-free finite-sample intervals), and
  `CalibratedSurrogate` which wraps any `predict(X)->(mean,std)` model — so a miscalibrated
  KAN/BNN is corrected *before* its uncertainty is propagated. An over-confident model
  that is fixed here is the difference between feasibility probabilities that are right
  and ones that are confidently wrong.

Run the integrated example: `python examples/operational.py` (loader → tier-derived MC →
calibrated surrogate → active learning). Tests: `pytest` (16 total — 6 original + 10 new).

## Real data ingestion (Catalysis-Hub)
The discovery layer pulls real CO2RR adsorption energies from Catalysis-Hub
(`connectors.py`, GraphQL with pagination + caching + schema introspection) and
assembles a per-surface descriptor table (`ingest.py`). Run on an open network:

```bash
python examples/fetch_real_data.py     # fetches, caches to ./data, reports coverage
```

This yields **X** (real DFT descriptors: ΔE of *CO, *COOH, *OCHO, *H, *OH per
surface). The **target y** (faradaic efficiency or a stated activity proxy) must
come from your experiments / validation set — it is never invented. With y in
hand, train the surrogate and reuse the `operational.py` pipeline unchanged.

If parsing returns empty, the API schema has drifted: run `probe_schema()` (the
script prints live field names) and adjust `connectors.REACTION_FIELDS`.

## Physics activity proxy + real-data training (no experimental FE needed)
`proxy.py` turns the fetched intermediate energies into a theoretical **activity**
target via the Computational Hydrogen Electrode / limiting-potential framework
(`U_L = -max ΔG_step`, overpotential `η = U_eq - U_L`, potential-determining step).
Because activity ≠ selectivity, the proxy predicts a **cell voltage**
(`V_cell ≈ V_baseline + η`), which the TEA already consumes — not a faradaic
efficiency. Thermochemical Δ(ZPE-TΔS) corrections are OFF by default and provided
as documented, opt-in literature values.

`examples/train_real_surrogate.py` runs the whole loop on this proxy:
descriptors → CHE activity target → Bayesian surrogate → calibration → EVOI ranking
(with the surrogate predicting `cell_voltage`). It uses cached real data from
`./data` if present, else a clearly-labelled synthetic fallback so it runs offline.
`rank_candidates` now accepts `target_field` so the same machinery serves a voltage
(activity) surrogate or an FE (selectivity) surrogate. Swap the proxy target for
experimental FE when available — only the target and `target_field` change.

## For users: check your own data & get next-step ideas

Two additions make co2dash usable by people who did not build it:

- **Your data intake** (`co2dash.intake`): drop a CSV of your own measurements
  (FE, cell voltage, current density, …). Column names are matched by alias,
  obvious unit slips are fixed (FE as %, voltage in mV, grid in g/kWh), values
  are range-checked, and any input you did not provide is filled from **sourced
  defaults** (`co2dash.defaults`) — every filled field shows its provenance.
  In the GUI: the **"Your data"** tab. Programmatically: `ingest_table(csv_text)`.

- **Recommendations** (`co2dash.recommend`): `recommend(scenario, carbon_price)`
  returns a plain-language "what to do next" — the verdict, whether electricity
  carbon is the binding constraint, the single dominant lever (Sobol) and the
  **target value** it must reach to become viable, and which candidate to compute
  next. In the GUI: the **"Recommended next steps"** panel.

- **Grid region** (`co2dash.energy`): pick a country/region to set grid carbon
  intensity from sourced 2024 data (Ember/IEA/NGA), with live connector templates
  (OpenNEM/AEMO, ElectricityMaps) for current values.

Example measurement template: `examples/user_measurements_example.csv`.

## Validation

The engine is validated against a published TEA (Jouny, Luc & Jiao 2018). It
reproduces the model-independent physics — specific energy, electricity/CO₂
operating cost, electrolyzer area/capital, energetic efficiency — to machine
precision, and reproduces the paper's economic ordering (CO/formic acid
favourable; methanol/ethylene electricity-dominated). See `docs/VALIDATION.md`
and run `python examples/validate_tea_jouny2018.py`.
