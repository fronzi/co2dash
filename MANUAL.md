# co2dash — User Manual

A rigorous guide to using the co2dash dashboard: what each control does, how to
read every output, what the tool is genuinely useful for, and — with equal care —
what it cannot tell you. Read §1 and §9 before drawing conclusions from it.

---

## 1. What this is, and what it is not

co2dash is a **decision-support and prioritisation** tool for electrochemical
CO₂-utilisation routes. Given the properties of a catalyst/process and its
economic context, it estimates three things a chemist cannot read off directly:

- **cost** — the levelized product cost (LCOP) and the marginal abatement cost
  (MAC, $/tonne CO₂);
- **climate benefit** — net CO₂ abatement after subtracting the emissions of the
  energy used;
- **risk** — not a single number but a *probability distribution*, because the
  inputs are uncertain.

It is **not**: a detailed process-engineering TEA (it uses a levelized-cost model,
not a full cash-flow with tax/depreciation); a guarantee of an exact cost; or a
predictor you should act on without reading its uncertainty. Its purpose is to
tell you **which route/parameter/catalyst to pursue next**, with honest error
bars — not to certify a final price.

---

## 2. Opening the app

Open the URL (or run `streamlit run app/streamlit_app.py` locally). The screen has
three regions:

1. **Left sidebar** — the inputs (a scenario). You either move sliders or load a
   tier-tagged YAML scenario.
2. **Top of the main area** — the **verdict banner** and **KPI cards**: the
   headline answer.
3. **Tabs** — six analysis views (Economics & climate, Feasibility envelope,
   Sensitivity, Active learning, Calibration, Your data).

Everything recomputes live as you change an input (the Monte-Carlo runs ~40,000
samples in ~30 ms; heavier views take ~1 s).

---

## 3. The control panel (sidebar) — every input

| Control | Units | Range | Engine variable | What it does |
|---|---|---|---|---|
| **Product** | — | CO / formate / methanol | reaction stoichiometry | Sets electrons per molecule (n), molar mass, and CO₂ consumed per kg. Higher-n products need far more energy per kg. |
| **Faradaic efficiency** | fraction | 0.05–1.0 | `faradaic_efficiency` | Fraction of current going to the target product. Enters energy as **1/FE** — small FE inflates energy cost sharply. |
| **Cell voltage** | V | 1.5–5.0 | `cell_voltage` | Full-cell operating voltage. Energy per kg scales **linearly** with it. |
| **Electricity price** | $/kWh | 0.01–0.20 | `c_elec` | Price of electricity; multiplies the energy intensity in LCOP. |
| **Grid region** | — | Custom / country list | overrides `grid_intensity` | Pick a sourced 2024 grid carbon intensity (AU, JP, US, EU, …, or dedicated renewable). Overrides the slider and shows the source. |
| **Grid intensity** | kgCO₂/kWh | 0.0–0.8 | `grid_intensity` | Carbon intensity of the electricity. **Drives the climate result.** Disabled when a Grid region is selected. |
| **End-of-life release φ** | fraction | 0.0–1.0 | `release_fraction` | Fraction of the product's carbon re-released later (e.g. fuels ≈1; durable chemicals lower). Reduces the abatement credit. |
| **CAPEX total** | $M | 10–200 | `capex_total` | Total plant capital. The **dominant source of MAC uncertainty** (an estimate). |
| **Carbon price** | $/kg CO₂ | 0.0–3.0 | comparison threshold | The price against which feasibility is judged (MAC < carbon price → economically feasible). $0.30/kg = $300/t. |
| **Load tier-tagged scenario (YAML)** | file | — | full scenario | Upload a sourced scenario (every value with a data tier + citation). **Overrides the sliders** and enables tier-based uncertainty. |

**Sliders vs YAML.** Sliders are for "what-if" exploration and give a generic
uncertainty band. A YAML scenario (e.g. `examples/scenario_co_real.yaml`) is for a
defensible analysis: each field carries a *data tier* (computed / lab-validated /
literature / estimated) that sets the Monte-Carlo noise, and a source string.

---

## 4. Reading the headline (verdict banner + KPI cards)

### Verdict banner
One of four states, plus three statistics:

- **Feasible** — climate-positive with high probability *and* likely cheaper than
  the carbon price.
- **Marginal** — climate-positive, but not yet economic at the chosen carbon price.
- **Not feasible at this price** — climate-positive but MAC above the carbon price.
- **Not climate-positive** — emits more CO₂ than it stores; economics are moot.

The three statistics:
- **P(net > 0)** — probability the route actually removes CO₂. *Check this first.*
- **P(MAC < price)** — probability it's cheap enough at your carbon price.
- **MAC (median)** — the typical cost per tonne of CO₂ avoided.

**Decision rule:** trust the route only when **P(net>0) is high**; then judge
economics by **P(MAC<price)** and the MAC distribution (§5.1).

### KPI cards
| Card | Meaning | How to read |
|---|---|---|
| **LCOP ($/kg)** | Levelized product cost | Compare to the conventional product price; above it means the route costs more than fossil but may still abate. |
| **Energy intensity (kWh/kg)** | Electricity per kg product | Sanity check against literature (CO ≈ 5–7 kWh/kg). |
| **Net abatement (kg/kg)** | CO₂ removed per kg product | Must be green (+). Red (−) = emits more than it stores; nothing else matters. |
| **MAC ($/t CO₂)** | Marginal abatement cost | Compare directly to the carbon price. "∞" = not climate-positive. |
| **Breakeven grid (kg/kWh)** | Grid intensity above which the route stops abating | Your grid intensity must stay **below** this number. |

---

## 5. The tabs

### 5.1 Economics & climate
- **Cost waterfall** — where each dollar of LCOP comes from (CO₂ feedstock,
  electricity, capital, fixed O&M, H₂ if applicable). Identifies the cost driver.
- **MAC distribution** — the *full* Monte-Carlo histogram of the MAC, with the
  carbon-price line and the shaded feasible region. **Read the spread, not just the
  median:** a wide P05–P95 means the verdict is uncertain (usually CAPEX-driven).

### 5.2 Feasibility envelope
Pick two levers (X and Y axes: FE, cell voltage, electricity price, grid intensity,
CAPEX). The map shows, over that 2-D space, where the route becomes viable, with a
white boundary line. **Use it to read a target:** e.g. "at this grid intensity, FE
must reach ≈0.8 to cross into the viable region." Turns "make it better" into a
concrete number.

### 5.3 Sensitivity
A Sobol **tornado** ranks which input most changes the MAC (total-effect index ST).
**Use it to decide what to work on:** if FE tops the bar, invest in selectivity; if
grid intensity tops it, the bottleneck is your energy source, not the chemistry —
no catalyst work will fix it.

### 5.4 Active learning
Given a set of candidate materials (descriptors), the table ranks **which to
compute/test next** by an acquisition score that balances how promising and how
uncertain each is. The top row is the most informative next DFT run — it maximises
what you learn about viability per calculation. (In the demo, candidates are
generated; with real descriptors, load them via the data pipeline.)

### 5.5 Calibration
Checks whether the surrogate's **uncertainty is trustworthy**. The reliability
diagram plots nominal vs empirical coverage against the diagonal; points **below**
the line = over-confident. The **temperature s** rescales the model's error bars
(s>1 widens an over-confident model). *Why it matters:* only a calibrated model
produces honest probabilities downstream. The demo lets you slide a model's
reported spread to see the diagram bend away from the diagonal and the fix pull it
back.

### 5.6 Your data
Upload a CSV of your own measurements (columns like material, product, FE or %,
cell voltage in V or mV, current density). The tool:
- matches your column names by alias and shows what it recognised,
- fixes obvious unit slips (FE as %, voltage in mV),
- fills unmeasured economic inputs from **sourced defaults** (shown with
  provenance: "user" vs "default: <source>"),
- gives a per-row net abatement, MAC, and a recommendation.
It never silently accepts bad data — out-of-range values are flagged.

---

## 6. Recommended next steps panel

Below the KPIs, **"Generate recommendation"** runs the MC verdict, Sobol, breakeven
and a target search, then states in plain language: the verdict, whether
electricity carbon is the binding constraint, the single dominant lever and the
**value it must reach** to become viable, and which candidate to compute next. It
ends with an honesty note that CAPEX dominates the MAC uncertainty.

---

## 7. How to interpret the uncertainty (the core idea)

Every probability comes from a **Monte-Carlo** propagation: the inputs are sampled
from distributions (set by the YAML data tiers, or a generic spread on sliders) and
pushed through the TEA/LCA/MAC engine ~40,000 times. So:

- **P(net>0) = 0.91** means: across the plausible range of inputs, the route
  removes CO₂ in 91% of cases.
- A **wide MAC distribution** is information, not a defect — it says the answer is
  genuinely uncertain given what's known (usually because CAPEX is an estimate).
- **Calibration** (§5.5) is what makes these probabilities honest for a *learned*
  surrogate: an over-confident model would make P(...) look more certain than it is.

The guiding principle: the tool is designed to **widen its uncertainty and say so**
when data is poor, rather than produce a confident single number.

---

## 8. What the tool is good for (utility)

- **Screening routes** — comparing CO vs formate vs methanol on cost/climate/risk.
- **Finding the binding constraint** — is viability limited by the catalyst
  (FE/voltage), the energy (grid/price), or the plant (CAPEX)?
- **Setting research targets** — the feasibility envelope gives the numeric value a
  property must reach.
- **Prioritising computation/experiments** — active learning ranks the next
  catalyst to study.
- **Geographic/energy scenarios** — the grid-region selector shows where a route is
  climate-positive (e.g. renewable-powered vs a fossil grid).
- **Honest communication** — every output carries an uncertainty and, for YAML
  scenarios, a source.

---

## 9. Limits and cautions (read this)

1. **Decision-support, not exact costing.** Outputs are for prioritisation and
   comparison. Do not quote a single MAC as a project cost.
2. **Model scope (CRF, not NPV).** LCOP uses a capital-recovery factor, not a full
   cash-flow with tax/depreciation. It matches published *cost-of-production*
   figures, not end-of-life NPV. (Validated against Jouny 2018 and Osorio-Tejada
   2024 — see `VALIDATION.md`.)
3. **CAPEX dominates MAC uncertainty.** It is an *estimated* input, not predicted.
   Treat the MAC spread as real; tighten CAPEX with a costed design before quoting.
4. **Descriptor→activity is solid; descriptor→FE is aspirational.** Predicting
   *activity* from DFT descriptors rests on established physics; predicting
   *faradaic efficiency* (selectivity) from descriptors is much harder and is not a
   finished capability here.
5. **The discovery↔decision join is partial.** Real catalysts and public DFT
   surfaces live in different material spaces; only ~30% of a typical experimental
   corpus maps to public descriptors, the rest needs bespoke DFT. The tool
   quantifies this gap rather than hiding it.
6. **Low-quality data → wider uncertainty + warnings, never fabrication.** If you
   feed sparse or noisy data, the tool flags the quality tier and refuses to
   over-claim; small datasets give coarse, explicitly-caveated results.
7. **Validation status is mixed by branch.** Physics reproduction is machine-
   precision; the calibration machinery is proven on real data; but some model
   branches (e.g. descriptor→FE on real CO₂RR data) are still small/illustrative.
   Check `VALIDATION.md` for what is solid vs demonstrative.
8. **Live web connectors run separately.** Grid/price/descriptor fetches are one-off
   steps on a networked machine; the app itself reads cached values.
9. **Demo-data / privacy.** On the public Streamlit demo, use only public or
   synthetic inputs; do not upload confidential or partner data (see
   `STREAMLIT_DEPLOY.md`).

---

## 10. Validation summary

The engine reproduces Jouny et al. 2018 (physics, machine precision) and
Osorio-Tejada et al. 2024 (levelized cost, 1–4%); the calibration gate recovers
honest coverage on synthetic ground truth, real public DFT, and real experimental
FE. Full details, tables, and honest limitations are in **`docs/VALIDATION.md`**.

---

## 11. Reproducibility & provenance

- YAML scenarios carry a **data tier and source** per field — the audit trail for
  every number.
- The examples (`examples/…`) reproduce each validation and pipeline step from the
  command line.
- For a defensible analysis, load a YAML scenario rather than sliders, and record
  which scenario file + app version produced a figure.

---

## 12. Glossary

- **LCOP** — levelized cost of product ($/kg).
- **MAC** — marginal abatement cost ($/tonne CO₂ avoided).
- **FE (faradaic efficiency)** — fraction of current producing the target product.
- **Net abatement** — CO₂ stored minus CO₂ emitted by the energy used, per kg.
- **Breakeven grid intensity** — grid carbon above which the route stops abating.
- **Data tier** — provenance/quality label (computed / lab-validated / literature /
  estimated) that sets a value's Monte-Carlo noise floor.
- **Calibration (temperature scaling / conformal)** — making a model's stated
  uncertainty match reality.
- **Sobol ST** — total-effect sensitivity index; how much an input drives the output.

---

## 13. References

- Jouny, Luc & Jiao, *A General Techno-Economic Analysis of CO₂ Electrolysis
  Systems*, Ind. Eng. Chem. Res. 2018.
- Osorio-Tejada et al., *Techno-economic analysis of CO₂ conversion*, Energy
  Environ. Sci. 2024.
- Grid intensities: Ember / IEA 2024; NGA Factors 2025 (Australia).
- See `docs/VALIDATION.md` for the full dataset and method citations.
