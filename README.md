# co2dash — uncertainty-aware techno-economic platform for CO₂ utilisation

Decision-support for CCU that connects **catalyst descriptors** to **cost per
tonne of CO₂ avoided**, propagating uncertainty and provenance the whole way.
The value is not another catalyst database — those exist (Materials Project,
Catalysis-Hub, Open Catalyst). It is the **translation layer** between the
discovery world and the decision world, and the discipline of never presenting
an assumption as a result.

## What it actually does

Three ways in, one economic engine:

| entry point | you supply | the model supplies |
|---|---|---|
| **Composition** | an alloy formula + a DFT workbook | adsorption energies → limiting potential → cell voltage |
| **Your data** | a CSV of measured FE / voltage / current | nothing — you already have the KPIs |
| **Sliders / YAML** | assumptions, tier-tagged | nothing — they are your hypotheses |

All three converge on the same engine: specific energy → LCOP → net abatement →
marginal abatement cost, as a Monte-Carlo **distribution**, plus a global
sensitivity analysis and a plain-language "what to do next".

Every performance field records its origin, and the verdict strip states which
KPIs are model-driven and which are assumed. With nothing loaded it says
*"no KPI is DFT-driven"* rather than implying otherwise.

## Architecture

```
schema.py            value + uncertainty + provenance TIER (the tier IS the noise model)
config.py            YAML -> (Scenario, ProvenanceRegistry); tiers -> MC distributions
techno_economic.py   KPI -> LCOP / net-abatement / MAC   (pure functions, scalar + vectorised)
uncertainty.py       Monte-Carlo propagation + Sobol global sensitivity
surrogate.py         descriptors -> KPI (mean, std); closed-form Bayesian linear
calibration*.py      temperature scaling + split conformal, with a train/cal/test gate
hea.py               multi-sheet DFT workbook loader + CHE reference-frame conversion
composition.py       alloy composition -> sampled site occupations -> descriptors
chain.py             composition -> E_ads -> U_L -> V_cell -> Scenario, with provenance
proxy.py             CHE limiting potential AND the desorption limit
intake.py            user CSV -> validated Scenario, with sourced defaults flagged
recommend.py         verdict + dominant lever + what to measure next
app/streamlit_app.py thin UI over the core (no business logic in the GUI)
```

## Key design decisions

**Data tier = noise model.** Each `Quantity` carries a tier (COMPUTED /
LAB_VALIDATED / LIT_EXTRACTED / ESTIMATED) which sets a *floor* on relative std
(`schema.TIER_REL_STD_FLOOR`). You cannot declare more confidence than the tier
allows: a std of 0.05 on a LIT_EXTRACTED value is used as 0.18. ESTIMATED
positive quantities become lognormal, so cost draws are never negative.

**Monte-Carlo, not linear error propagation.** The map contains 1/FE and a
division by net abatement, so a Taylor expansion would misstate the tails. On
the shipped CO scenario the MAC distribution has median 601 \$/t but mean
2097 \$/t, and 9.4% of draws are non-finite (net abatement ≤ 0). A point
estimate cannot express that.

**Sensitivity ranges come from the MC distributions**, not hand-written
multipliers. Earlier they were ±30% for everything except grid intensity, which
got ±100% — and so came out "dominant" in 5 of 7 unrelated scenarios. Sobol
indices are defined relative to the input ranges you choose; choosing them
inconsistently makes the answer an artefact.

**Two limits, reported separately.** Activity (CHE limiting potential) and
product desorption have different units, different physics and different
remedies. Desorption transfers no electrons, so applied potential cannot fix it
— the chain therefore derives **no cell voltage** when desorption is the
bottleneck rather than producing a flattering one.

**Applicability domain is checked explicitly.** The surrogate's own σ does not
detect extrapolation when an element is absent from the adsorption site but
present in the environment columns: every feature is marginally in range and
only the joint position is novel. A separate site-coverage guard catches it, and
reports in-domain-only statistics as a fallback.

**Closed-form Bayesian surrogate.** The contract is `predict(X) -> (mean, std)`,
so a KAN/BNN drops in unchanged. It is kept deliberately, not by default: on the
validation set, Bayesian linear, random forest, gradient boosting and a GP with
ARD all give U_L RMSE ~0.10 eV and a predicted-on-true slope of 0.61–0.69. The
ceiling is the **descriptor set**, not the regressor.

## Honesty boundaries — read before using outputs

* **Faradaic efficiency is never predicted.** No descriptor→selectivity model
  exists. FE passes through untouched and is labelled assumed. It enters the cost
  as 1/FE, so it dominates the TEA.
* **Absolute limiting potentials need YOUR gas-phase reference energies.** DFT
  total energies are not portable between codes or pseudopotential sets. The
  default `relative` mode claims no absolute U_L and ranks only.
* **This is no longer "zero new DFT".** It does not rebuild the discovery layer,
  but a handful of your own calculations (gas-phase references, and any surface
  outside the training domain) are required for quotable numbers.
* **Connectors are not called by the app.** `connectors.py` and the live
  endpoints in `energy.py` are run **manually**, via the scripts in `examples/`,
  and save a local file you then load. A silent network failure on a hosted app
  would produce a number of unknown provenance. The Grid-region selector reads
  static sourced profiles, not a live API.
* **The Active-learning tab ranks by predictive std**, which is not the same as
  the EVOI acquisition in `active_learning.py`. That module is available
  programmatically; the tab does not currently use it.
* **Defaults exist only on the CSV path.** A YAML scenario must be complete —
  omit a field and loading fails rather than silently defaulting. The CSV intake
  fills 15 fields from `defaults.py` and shows each one's provenance.
* It can prioritise, explore scenarios, run sensitivity and rank experiments. It
  cannot produce a predictive plant cost without lab-grade stability and CAPEX
  data — that bottleneck is not on the web.

## Run

```bash
pip install -e ".[ui]"                      # add ,connectors for the fetch scripts
pytest                                      # 207 tests
streamlit run app/streamlit_app.py          # the dashboard
```

The header shows the deployed commit SHA, so a stale deployment is visible
rather than inferred.

```bash
python examples/demo.py                              # end-to-end, no network, no DFT
python examples/operational.py                       # loader -> tier-derived MC -> calibration
python examples/validate_tea_jouny2018.py            # TEA validation against two published anchors
python examples/calibrate_surrogate.py               # calibration gate on synthetic ground truth
python examples/calibrate_real_dft.py                # calibration on real public DFT (CHEAT)
python examples/run_pipeline.py <data.csv>           # end-to-end with the data-quality gate
python examples/fetch_real_data.py                   # Catalysis-Hub fetch (needs network)
```

## Using a DFT workbook

Upload a multi-sheet `.xlsx` (one sheet per adsorbed intermediate) in the
sidebar. Then:

```python
from co2dash.hea import load_workbook
from co2dash.chain import (train_intermediate_models, run_chain, ReferenceFrame,
                           applicability_report, validate_pathway)

sheets = load_workbook("data/your_workbook.xlsx")
print(applicability_report(sheets))          # what the data can and cannot support
models = train_intermediate_models(sheets)   # one surrogate per intermediate

frame = ReferenceFrame(mode="absolute",
                       gas_energies={"CO2": ..., "H2": ..., "H2O": ...},
                       gas_formation_energy=...)   # enables the desorption test
```

Models are trained **at run time** — nothing pre-trained is shipped, so the model
always matches the file you loaded. Three sheets fit in ~11 ms.

## Validation

**TEA engine** — reproduces Jouny, Luc & Jiao (2018) physics to machine
precision and Osorio-Tejada et al. (2024) to 1–4%, with the shipped CO scenario
landing inside the paper's reported \$570–1392/t band. See
[`docs/VALIDATION.md`](docs/VALIDATION.md).

**Calibration gate** — validated on synthetic ground truth, real public DFT
(CHEAT, n≈2500) and real experimental FE (Scientific Data 2023, 884 records). In
every case the naive surrogate is over-confident and the gate restores honest
coverage.

**Discovery→decision chain** — held out every configuration carrying all pathway
intermediates, retrained, and compared surrogate-driven U_L against DFT.
Includes the negative results: the range compression, the model-class comparison
establishing the descriptor ceiling, and why a published activity band cannot
serve as a reference anchor. See
[`docs/CHAIN_VALIDATION.md`](docs/CHAIN_VALIDATION.md).

## Suggested scope for a citable result

One product, one route. Pull descriptors from Catalysis-Hub/OC20, grid intensity
from your TSO, curate a small CAPEX/price set by hand, and build the TEA/LCA+UQ
core on that narrow slice. A narrow, honest slice beats a broad, unvalidated one.
