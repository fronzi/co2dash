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

## The composition→MAC chain (`composition`, `chain`)

Enter an alloy composition, not 40 descriptor columns: the descriptors are
element properties, so they are derived from a lookup table. A composition
specifies a *distribution* over site occupations, so predictions are ensembles —
configurational and model uncertainty are reported separately.

```
composition -> sampled site occupations -> descriptors -> E_ads (+/- sigma)
            -> reference frame -> U_L -> V_cell -> Scenario -> MAC
```

Every performance field records its origin (`ChainProvenance`), and the verdict
strip states which KPIs are DFT-driven and which are assumed. Faradaic
efficiency is **never** derived.

Measured behaviour, limitations, and the negative results are in
[`docs/CHAIN_VALIDATION.md`](docs/CHAIN_VALIDATION.md). Headline numbers on the
21 held-out FeCoNiCuMo configurations: U_L RMSE 0.102 V, no error amplification
through `max()`, but predicted-on-true slope 0.62 — the surrogate compresses the
range, putting the true best configuration 3rd.

## Honest limitations

- Descriptor→**FE** (selectivity) is scientifically harder than descriptor→
  **activity**; treat →FE as aspirational until same-source data supports it.
  The chain never derives FE — it is carried through and labelled assumed.
- **The descriptor set, not the regressor, is the accuracy ceiling.** Bayesian
  linear, random forest, gradient boosting and a GP with ARD all give U_L RMSE
  ~0.10 eV and slope 0.61–0.69 on the same hold-out. More ML buys nothing;
  richer features (d-band centre, coordination, strain) would.
- **Absolute U_L needs your own gas-phase reference energies.** DFT total
  energies are not portable between setups, and the source paper's published
  0.29–0.51 V band cannot serve as an anchor — even with correct shifts the
  computed spread is 2.26× the band width, a disagreement no shift can fix
  (`check_against_published_band`).
- **Anchored mode is under-determined** unless the first PCET step limits: one
  anchor fixes one species' shift, and the second step needs the difference of
  two. Checked and warned at run time. Use `absolute` — with real references the
  PDS *does* vary (COOH→CO ×16, CO2→COOH ×5), which one anchor cannot represent.
- **Activity is not the only limit.** On all 21 sampled configurations every
  proton-coupled step is downhill (U_L > 0) and the surface is CO-poisoned
  instead (ΔG_des 1.77–2.19 eV, coverage 1.0000). Desorption transfers no
  electrons, so no cell voltage is derived in that regime — applied potential
  cannot remove a bound product. An apparently perfect CHE ladder means a
  poisoned surface, not a good catalyst.
- **Applicability gaps are real**: the \*CO sheet has no Cu-terminated sites, so
  Cu-bearing compositions extrapolate there — biasing ΔE(\*CO) by 0.32 eV on
  equimolar FeCoNiCuMo. The surrogate's own sigma does not detect this; the
  site-coverage guard does.
- The corpus↔public-DFT join is partial (~30% public / 45% bespoke / 25% ill-
  defined) — a real material-space bottleneck, quantified, not hidden.
- CAPEX is an ESTIMATED input, not predicted. Which input dominates MAC
  uncertainty is now *computed* per scenario, not asserted.
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

```python
# composition -> MAC, and the hold-out validation behind it
from co2dash.hea import load_workbook
from co2dash.chain import (train_intermediate_models, run_chain, ReferenceFrame,
                           validate_pathway, applicability_report)
```

## Deployment

See `DEPLOY.md`. Ready to *show* privately (Streamlit Cloud/HF Spaces); not yet
*published* publicly — needs a hosted instance, secrets config, and a USyd
data-governance / foreign-interference review first.
