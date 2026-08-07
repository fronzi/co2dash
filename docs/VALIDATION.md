# co2dash — TEA validation

## Anchor: Jouny, Luc & Jiao (2018)

*A General Techno-Economic Analysis of CO₂ Electrolysis Systems*, Ind. Eng. Chem.
Res. 2018 (manuscript: OSTI 1712664). The most cited, most transparent open TEA
for CO₂ electrolysis; its Table 2 (market prices) and Table 3 (process
assumptions) are fully specified, enabling an independent reproduction.

Reproduce with: `python examples/validate_tea_jouny2018.py`
(the reference physics in `co2dash/validation.py` is hand-coded independently of
the engine, so the comparison is a genuine cross-check, not a re-run.)

## What was validated

**1. Specific electricity consumption (model-independent physics).**
The engine's `specific_electricity_kwh_per_kg` reproduces the independent
first-principles mass–energy balance `E = nFV/(M·FE)` to **machine precision
(relative error < 1e-9)** for CO, formic acid, methanol, ethylene and ethanol,
in both the base and optimistic cases.

| Product (base case, V=2.3 V, FE=0.90) | Reference kWh/kg | co2dash kWh/kg |
|---|---|---|
| CO | 4.89 | 4.89 |
| Formic acid | 2.98 | 2.98 |
| Methanol | 12.83 | 12.83 |
| Ethylene | 29.30 | 29.30 |

**2. Operating cost per kg (electricity + CO₂ feedstock), base case.**
Reproduces the paper's central conclusion — CO and formic acid are favourable
while methanol and ethylene are electricity-dominated:

| Product | Electricity $/kg | CO₂ $/kg | Market $/kg |
|---|---|---|---|
| CO | 0.245 | 0.157 | 0.60 |
| Formic acid | 0.149 | 0.096 | 0.70 |
| Methanol | 0.641 | 0.137 | 0.60 |
| Ethylene | 1.465 | 0.314 | 1.30 |

Methanol's electricity cost alone ($0.641) exceeds its market price ($0.60),
consistent with Jouny's finding that methanol is unprofitable while CO/formic
acid are the profitable products at base-case conditions.

**3. Electrolyzer electrode area & stack capital** (100 t/day CO), from Jouny's
area-cost method: base **4,430 m² → $12.5 M**; optimistic 2,953 m² → $4.2 M.

**4. Energetic efficiency cross-check** (base case): CO 52%, formic acid 58%,
methanol 47% — all consistent with the paper's "CO₂R generally < 60%".

## What legitimately differs (scope, not error)

co2dash levelises capital with a capital-recovery factor (an **LCOP**), whereas
Jouny runs a full MACRS cash-flow with 40% income tax to an **end-of-life NPV**.
The absolute LCOP is therefore *not* expected to equal Jouny's NPV — this is a
deliberate scope choice. A like-for-like comparison is co2dash LCOP against
Jouny's cost-of-production breakdown (Fig. 5), not the NPV (Fig. 4).

## Second anchor: Osorio-Tejada et al. (2024)

*CO₂ conversion to CO via plasma and electrolysis*, Energy Environ. Sci. 17,
5833 (2024), DOI 10.1039/d4ee00164h (open access). Reports a **levelised unitary
cost of production** for CO₂→CO electrolysis and — importantly — uses an annual
capital charge ratio (ACCR, 10%/20 yr) that is **identical to co2dash's CRF**, so
its UCOP is a genuine like-for-like target (unlike Jouny's NPV). Independent
source, different operating point (V = 3.0 V, FE = 0.85), and it exercises the
plug-to-power / rectifier term.

Reproduce with the same script (section 6).

| Quantity (CO₂→CO) | co2dash | Osorio-Tejada 2024 | error |
|---|---|---|---|
| Cell energy (ideal rectifier) | 6.75 kWh/kg | 6.82 | 1.0% |
| Total conversion energy (80% plug-to-power) | 8.44 kWh/kg | 8.53 | 1.0% |
| Electrode area (100 t/day CO) | 3,752 m² | 3,791 | 1.0% |
| Electricity cost | $253/t | ~$260/t* | 2.5% |
| Feedstock (CO₂ + water) | $69/t | $71.9/t | 3.7% |

*The paper does not tabulate an electricity $/t directly; $260/t is derived from
its stated "electricity ≈ 27% of the $962/t UCOP". All other reference values are
stated verbatim in the paper (6.82 and 8.53 kWh/kg, 3791 m², $71.9/t, $962/t).

Non-circular corroboration: co2dash's real CO scenario
(`examples/scenario_co_real.yaml`) gives **LCOP $770/t**, inside the paper's
reported electrolysis band of **$570–1392/t**. A further literature datapoint —
an Energy & Fuels 2023 industrial-electrolysis TEA — reports CO $0.449/kg and
formic acid $0.468/kg, bracketing the lower end of the same range.

## Calibration gate

Separate from TEA validation: this checks that the surrogate's *uncertainty* is
trustworthy before it is propagated into the MAC distribution. The harness
(`co2dash.calibrate_and_evaluate`) splits data into train / calibration / test,
fits the surrogate on train, fits temperature scaling and split-conformal on the
held-out calibration set, and evaluates coverage on the untouched test set.

**Procedure validated on synthetic ground truth** (you need known noise to check
that empirical coverage matches the nominal level). With true noise σ and a
surrogate whose reported σ is deliberately wrong:

| Surrogate | temperature s | miscalibration (before → after) | 90% coverage (before → after → conformal) |
|---|---|---|---|
| Over-confident (σ 1.8× too small) | 1.807 | 0.246 → 0.014 | 69% → 90% → 90% |
| Under-confident (σ 1.8× too large) | 0.558 | 0.186 → 0.014 | 100% → 90% → 90% |
| Well-specified (reports true σ) | 1.004 | 0.013 → 0.014 | 90% → 90% → 90% |

The procedure recovers honest coverage regardless of the model's initial
over/under-confidence, and leaves a well-specified model alone (s ≈ 1). Reproduce
with `python examples/calibrate_surrogate.py`.

**Validated on real literature DFT data.** Beyond synthetic ground truth, the
gate was run on a real public dataset — the CHEAT high-entropy-alloy adsorption
energies (Clausen & Rossmeisl 2022, `github.com/cmclausen/cheat`; GPAW/DFT,
AgIrPdPtRu, targets ΔE_*OH and ΔE_*O). *Honest scope: this is an ORR system used
as a stand-in for a real descriptor→property regression, not CO₂RR data.* On this
real data a naive Bayesian-linear surrogate is badly **over-confident** — its 90%
intervals cover only ~30% of held-out points — and the gate restores honest
coverage:

| Target (real DFT, n≈2500) | raw 90% coverage | temperature s | miscalibration (before→after) | after: 90% / conformal |
|---|---|---|---|---|
| ΔE_*OH ontop | 30% | 3.27 | 0.526 → 0.025 | 91% / 90% |
| ΔE_*O fcc | 34% | 3.81 | 0.496 → 0.014 | 89% / 87% |

This is exactly the failure mode the gate exists to catch, shown on real data.
Reproduce with `python examples/calibrate_real_dft.py` (fetches the data at run
time; not redistributed).

**Run on REAL experimental FE.** Executed on the Scientific Data 2023 CO₂RR
corpus (Zhang et al., doi:10.1038/s41597-023-02089-z; `gold_corpus/
merge_data_final.xls`). 884 experiment records were reconstructed from 829
papers (product↔FE paired by the corpus's ordinality labels; current density and
potential linked at paper level — coarse features, so some condition-linkage
noise). Features: product + Cu-catalyst-type one-hot, current density, potential;
target = measured FE (mean 0.49). Result:

| | raw | after calibration |
|---|---|---|
| 90% interval coverage | **67%** (over-confident) | 90% |
| conformal 90% coverage | — | 89% |
| miscalibration | 0.220 | 0.018 (temperature s = 1.77) |

On genuinely real experimental FE the naive surrogate is over-confident and the
gate restores honest coverage — the same behaviour seen on synthetic and DFT
data. Reproduce: `python examples/calibrate_corpus_scidata.py <path>/merge_data_final.xls`.

## Descriptor→FE bridge (joining the two worlds)

Training a *descriptor→FE* surrogate needs, per material, both a measured FE and
DFT descriptors. Literature catalysts (alloys, oxide-derived Cu, MOFs,
composites) and public DFT (well-defined surfaces) live in different material
spaces, so the join is partial. Quantified on the 884-record corpus
(`co2dash.availability_report`, `link_fe_to_descriptors`):

| tier | FE records | meaning |
|---|---|---|
| public | 267 (30%) | pure Cu / Cu-on-C → Cu facets in Catalysis-Hub/OC20 |
| bespoke | ~400 (45%) | alloys, oxides, sulfides → need your own DFT |
| none | ~215 (25%) | composites/MOFs → no single well-defined surface |

So ~30% of the corpus can be joined to public DFT immediately; the rest needs
bespoke DFT (Setonix/Gadi) or is structurally ill-defined. `descriptor_request_
list()` emits the surfaces to compute, ranked by FE records unlocked (Cu 267,
Cu-alloy, CuOx, …). Recipe: obtain descriptors keyed by canonical surface →
`link_fe_to_descriptors(fe_rows, table, keys)` → `calibrate_and_evaluate`.
This makes the discovery↔decision bottleneck explicit and actionable rather than
papering over it. See `examples/build_descriptor_fe_dataset.py`.

## Descriptor→activity on public same-source data (no user data)

Without the user's own DFT, the defensible discovery-layer branch is
descriptor→**activity** (activity ties to descriptors through solid physics; FE
does not). Demonstrated on public, same-source CO₂RR DFT — Wu et al. (J. Phys.
Chem. C 2021; `github.com/LuGroup/CO2RR-Adsorbates`, real VASP on Cu(100)):
predict the DFT C–C coupling free energy (a C2-pathway activity descriptor) from
cheap elemental features, then calibrate. The set is small (~65 complete rows),
so the numbers are illustrative — the gate detects and reduces the
miscalibration (raw 90% interval mis-covers; temperature s≈0.18 corrects it;
miscalibration 0.258→0.064). Reproduce: `python examples/descriptor_activity_public.py`.

The large, clean CO₂RR descriptor→limiting-potential datasets — Chen et al. HEA
(691 pts, *CO/*CHO/*COOH) and ACS Catalysis FeCoNiCuMo (U_L 0.29–0.51 V) — are
Figshare-hosted (unreachable from the build sandbox but reachable from your
machine); they drop into the same pipeline via the adaptive loader
`co2dash.loaders` (auto-detects *CO/*COOH/*CHO columns, computes the CHE
limiting-potential target, or uses a U_L column if present). Run:
`python examples/load_figshare_dataset.py <downloaded_file>` — mapping is printed,
nothing is hardcoded to a guessed schema. The largest clean descriptor→property calibration to
date remains the CHEAT DFT run (≈2500 pts) documented above.

## Discovery→decision chain

The descriptor→activity→economics chain has its own validation document:
[`CHAIN_VALIDATION.md`](CHAIN_VALIDATION.md). It records the held-out test on the
21 FeCoNiCuMo configurations that carry both \*CO and \*COOH (U_L RMSE 0.102 V,
no amplification through `max()`, predicted-on-true slope 0.62), the model-class
comparison establishing that the **descriptor set rather than the regressor** is
the accuracy ceiling, the applicability gap (no Cu-terminated \*CO sites), and
why the source paper's published 0.29–0.51 V band **cannot** be used as a
reference anchor.

## Status and next validation steps

- [x] Physics engine reproduced against Jouny 2018 (machine precision).
- [x] Second, independent anchor (Osorio-Tejada 2024): physics within 1–4%,
      LCOP inside the reported band.
- [x] Calibration validated on synthetic ground truth, real public DFT (CHEAT),
      AND real experimental FE (Scientific Data 2023 corpus).
- [x] Descriptor↔FE join machinery built; join gap quantified on real data
      (30% public / 45% bespoke / 25% ill-defined) with a descriptor-request list.
- [x] Discovery→decision chain validated end-to-end on held-out DFT
      (`CHAIN_VALIDATION.md`): U_L RMSE 0.102 V, PDS agreement 100%, no error
      amplification through the `max()` over PCET steps.
- [x] Accuracy ceiling attributed to the descriptor set, not the regressor
      (four model classes agree to 0.007 eV RMSE and 0.08 in slope).
- [ ] **Obtain E(CO₂), E(H₂), E(H₂O) at the workbook's level of theory.** This
      is the single blocker on quotable absolute U_L and MAC; the published
      activity band cannot substitute (width ratio 3.8×, no shift reconciles it).
- [ ] Confirm the published E_ads convention: E(\*X)−E(\*)−E_gas(X) or
      E(\*X)−E(\*)? It shifts a whole species rigidly.
- [ ] Compute ~15–20 Cu-terminated \*CO configurations to close the
      applicability gap that currently biases every Cu-bearing composition.
- [ ] Obtain the 'public' Cu-facet descriptors (Catalysis-Hub) + 'bespoke' DFT
      for the ranked surfaces, complete the descriptor→FE dataset, and train the
      production surrogate (KAN/BNN) through the calibration gate into the MAC loop.
