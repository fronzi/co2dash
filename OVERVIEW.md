# co2dash — project overview

**What it is.** A zero-new-DFT decision-support platform for CO₂ utilisation that
bridges the *discovery* world (DFT/ML catalyst descriptors) and the *decision*
world (techno-economics, life-cycle, marginal abatement cost) with **calibrated
uncertainty** and an **active-learning loop**. The contribution is the honest
translation layer, not another catalyst database.

## Pipeline

```
descriptors (DFT)  ->  surrogate (+calibration gate)  ->  activity (CHE U_L)
                                                              |
experimental FE  ------------------------------------------->|
                                                              v
        TEA / LCA engine  ->  LCOP, net abatement, MAC distribution (Monte-Carlo)
                                                              v
        Sobol sensitivity + feasibility envelope + recommendation ("what next")
```

## Modules (`src/co2dash/`)

- `schema`, `techno_economic` — data tiers + TEA/LCA/MAC engine (vectorised).
- `uncertainty` — Monte-Carlo MAC distribution, Sobol, feasibility envelope.
- `surrogate`, `calibration`, `calibration_harness` — ML surrogate + the
  calibration **gate** (temperature scaling + split conformal, train/cal/test).
- `quality` — data-quality guards (honest behaviour on low-quality inputs).
- `proxy` — CHE physics activity proxy (descriptors → limiting potential).
- `energy` — sourced per-country grid intensities + live connectors.
- `intake`, `corpus`, `loaders`, `link`, `defaults` — get real data in:
  user CSV, CO₂RR experimental corpora, Figshare DFT sets, and the
  descriptor↔FE join (with an availability/gap report).
- `recommend` — plain-language "what to improve, by how much, what to compute next".
- `connectors`, `ingest` — Catalysis-Hub fetch + descriptor tables.
- `app/` — Streamlit dashboard.

## What is validated (see `docs/VALIDATION.md`)

- **Physics** reproduced against Jouny et al. 2018 to machine precision.
- **Levelised cost** reproduced against Osorio-Tejada et al. 2024 (like-for-like,
  1–4%; LCOP inside the reported band).
- **Calibration gate** recovers honest coverage on synthetic ground truth, on
  real public DFT (CHEAT, ~2500 pts, strong), and on real experimental FE
  (Scientific Data 2023 corpus, 884 records).
- **Descriptor→activity** demonstrated on public CO₂RR DFT (LuGroup; small,
  illustrative). Large clean run (Chen HEA / ACS Catalysis) drops into the same
  pipeline via `loaders` on a network that reaches Figshare.

## Honest limitations

- Descriptor→**FE** (selectivity) is scientifically harder than descriptor→
  **activity**; treat →FE as aspirational until same-source data supports it.
- The corpus↔public-DFT join is partial (~30% public / 45% bespoke / 25% ill-
  defined) — a real material-space bottleneck, quantified, not hidden.
- CAPEX dominates MAC uncertainty; it is an ESTIMATED input, not predicted.
- Live web connectors run on your machine (network), not in the build sandbox.
- Low-quality data → the tool widens uncertainty and warns; it never fabricates.

## Command map

```
pip install -e ".[ui]"                      # install (add ,connectors for fetches)
pytest                                      # 80+ tests
streamlit run app/streamlit_app.py          # dashboard

python examples/validate_tea_jouny2018.py           # TEA validation anchors
python examples/calibrate_surrogate.py              # calibration on synthetic GT
python examples/calibrate_real_dft.py               # calibration on real DFT (CHEAT)
python examples/calibrate_corpus_scidata.py <xls>   # calibration on real experimental FE
python examples/descriptor_activity_public.py       # descriptor->activity, public CO2RR DFT
python examples/load_figshare_dataset.py <file>     # Chen HEA / ACS Catalysis -> activity
python examples/build_descriptor_fe_dataset.py <xls> [descriptors.json]  # the join
python examples/run_pipeline.py <data.csv>          # end-to-end with data-quality gate
```

## Deployment

See `DEPLOY.md`. Ready to *show* privately (Streamlit Cloud/HF Spaces); not yet
*published* publicly — needs a hosted instance, secrets config, and a USyd
data-governance / foreign-interference review first.
