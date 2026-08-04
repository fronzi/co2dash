# co2dash — Methods, equations, and references

Every equation the engine actually evaluates, with the source it is taken from
and the reason for the choice. Equations are transcribed from the source code
(`src/co2dash/techno_economic.py`, `proxy.py`, `calibration.py`,
`uncertainty.py`) so this document matches what the app computes, not an
idealised version.

> **Citation note.** The references below are the canonical/primary sources for
> each method. Exact bibliographic details (year, volume, DOI) should be
> re-verified against the original papers before use in a manuscript; they are
> given here to document provenance and rationale, not as camera-ready citations.

Conventions: all quantities are per kg of product unless stated. Constants:
Faraday F = 96 485 C·mol⁻¹; 1 kWh = 3.6×10⁶ J.

---

## 1. Specific electrical energy

**Equation (code: `specific_electricity_kwh_per_kg`)**

$$E_\text{el} = \frac{n\,F\,V_\text{cell}}{M\,\cdot\,\mathrm{FE}\,\cdot\,\eta_\text{rect}}\quad[\text{J·kg}^{-1}]\;\longrightarrow\;\text{kWh·kg}^{-1}$$

with n = electrons per molecule, V_cell = full-cell voltage, M = molar mass of
product, FE = faradaic efficiency, η_rect = rectifier (AC→DC) efficiency
(default 0.95).

**Source.** Faraday's law of electrolysis (standard electrochemistry, e.g. Bard &
Faulkner, *Electrochemical Methods*). The same expression is the energy core of
Jouny, Luc & Jiao, *Ind. Eng. Chem. Res.* 2018 — the study we validate against.

**Rationale.** This is the single term with the sharpest sensitivity to the two
levers research can actually move (FE enters as 1/FE; V_cell enters linearly).
Exposing it explicitly — rather than burying it inside an aggregate cost — is a
core design goal. η_rect is included because balance-of-plant AC→DC losses are
non-negligible and are what separate cell energy from plant energy.

---

## 2. Capital recovery factor (levelisation)

**Equation (code: `capital_recovery_factor`)**

$$\mathrm{CRF} = \frac{i(1+i)^{L}}{(1+i)^{L}-1}\,,\qquad \mathrm{CRF}=\tfrac{1}{L}\ \text{if } i=0$$

with i = discount rate, L = plant lifetime (years).

**Source.** Standard engineering economics (annualised capital charge). The same
device — an annualised capital charge rate (ACCR ≡ CRF) — is used by
Osorio-Tejada et al., *Energy Environ. Sci.* 2024, our second validation anchor.

**Rationale.** We deliberately levelise capital with a CRF rather than run a full
discounted cash-flow with tax and depreciation. This makes the output a
**cost of production** (comparable across routes at a glance) rather than an
after-tax NPV. It is a scope choice, stated openly: absolute LCOP is *not*
expected to equal a levered NPV such as Jouny 2018's, but *is* like-for-like with
Osorio-Tejada 2024 (which uses ACCR). See `VALIDATION.md`.

---

## 3. Levelised cost of product (LCOP)

**Equation (code: `lcop`)**

$$\mathrm{LCOP} = \underbrace{\frac{\mathrm{CAPEX}\cdot\mathrm{CRF} + \mathrm{OPEX}_\text{fix}}{\dot m_\text{annual}}}_{\text{fixed}} + \underbrace{c_{\mathrm{CO_2}}\,m_{\mathrm{CO_2}} + c_\text{el}\,E_\text{el} + c_{\mathrm{H_2}}\,m_{\mathrm{H_2}}}_{\text{variable}}\quad[\$\,\text{kg}^{-1}]$$

**Source.** Standard levelised-cost formulation for electrochemical production;
structure follows the TEA breakdowns in Jouny 2018 and Osorio-Tejada 2024
(feedstock + electricity + annualised capital + fixed O&M).

**Rationale.** The variable terms are physical and largely model-independent
(feedstock CO₂, electricity via E_el, optional H₂ co-feed), so they are the part
that reproduces published numbers to within a few percent. The fixed term carries
the model's largest uncertainty because CAPEX is an estimate (§6).

---

## 4. Net CO₂ abatement (life-cycle / climate)

**Equation (code: `net_abatement_kg_per_kg`)**

$$\text{net} = (1-\varphi)\,m_{\mathrm{CO_2}} \;-\; \big(I_\text{grid}\,E_\text{el} + e_\text{capture} + e_\text{process}\big)$$

with φ = end-of-life release fraction (0 = durable product, 1 = fuel re-oxidised),
I_grid = grid carbon intensity (kg CO₂·kWh⁻¹), e_capture, e_process = CO₂ penalties
of capture and other process steps.

**Source.** Life-cycle CCU accounting; the utilisation-vs-sequestration
distinction follows CCU LCA guidance (e.g. von der Assen, Bardow and co-workers on
CO₂ utilisation LCA) and IPCC-style grid-emission accounting.

**Rationale.** The (1−φ) term is the crux of *utilisation* vs *storage*: a fuel
that is later burned (φ→1) keeps almost none of the CO₂ credit. Making φ an
explicit slider prevents the common error of counting utilised CO₂ as permanently
abated. Grid intensity multiplies the (large) electrical energy, which is why the
climate verdict is dominated by the electricity source.

---

## 5. Break-even grid intensity

**Equation (code: `breakeven_grid_intensity`)** — net = 0 solved for I_grid:

$$I^{*} = \frac{(1-\varphi)\,m_{\mathrm{CO_2}} - e_\text{capture} - e_\text{process}}{E_\text{el}}$$

Above I\*, the conversion **increases** emissions.

**Source.** Algebraic rearrangement of §4 (no external source).

**Rationale.** Provided in closed form and surfaced as a first-class KPI because
"how clean must my electricity be for this to help at all?" is one of the most
decision-relevant questions, and it has an exact answer.

---

## 6. Marginal abatement cost (MAC)

**Equation (code: `marginal_abatement_cost`)**

$$\mathrm{MAC} = \frac{\mathrm{LCOP}_\text{CCU} - \mathrm{LCOP}_\text{conv}}{\text{net}}\quad[\$\,\text{kg}^{-1}\,\mathrm{CO_2}]\,,\qquad \mathrm{MAC}=+\infty\ \text{if net}\le 0$$

(×1000 for \$·tonne⁻¹).

**Source.** Standard definition of marginal abatement cost from climate/energy
economics (cost premium over the incumbent per unit CO₂ avoided).

**Rationale.** MAC is the single figure investors and regulators read, and it
couples cost and climate into one comparable number. Returning +∞ when net ≤ 0 is
deliberate: a route that is not climate-positive has no defined abatement cost, and
hiding that behind a finite number would mislead.

---

## 7. Activity proxy — Computational Hydrogen Electrode (CHE)

Used by the discovery layer to turn DFT intermediate energies into an activity
target when experimental FE is unavailable.

**Equations (code: `proxy.py`)**

Formation free energy of each adsorbed intermediate (CO₂ reference = 0):
$$G_f(^*X) = \Delta E_f(^*X) + \Delta(\mathrm{ZPE}-T\Delta S)_X$$

Per proton-coupled electron-transfer (PCET) step, and the limiting potential:
$$\Delta G_i(U) = \Delta G_i(0) - eU\,,\qquad U_L = -\max_i \Delta G_i(0)\quad(e=1\ \text{in eV/V})$$

Overpotential and the transparent activity→voltage map fed to the TEA:
$$\eta = U_\text{eq} - U_L\,,\qquad V_\text{cell} \approx V_\text{baseline} + \eta$$

Equilibrium potentials used (V vs RHE): CO −0.10, HCOOH −0.12, CH₃OH +0.02.

**Source.** Nørskov et al., *J. Phys. Chem. B* 2004 (computational hydrogen
electrode); CO₂RR pathway/limiting-potential analysis following Peterson &
Nørskov and the CO₂-reduction scaling-relation literature.

**Rationale.** CHE is the standard, transparent way to rank catalyst *activity*
from adsorption energies without kinetics. We map activity to **cell voltage**
(which the TEA already consumes), not to FE, because activity ≠ selectivity —
predicting FE from thermodynamic descriptors is not defensible, so we do not claim
it. **Honesty defaults:** Δ(ZPE−TΔS) corrections are **off** by default (electronic
energies only); equilibrium potentials are textbook values to be verified for
pH/reference convention; V_baseline is an explicit adjustable parameter, not a
first-principles cell model. All flagged in-code.

---

## 8. Uncertainty propagation (Monte-Carlo)

**Method (code: `uncertainty.py::propagate_mc`)**

Each uncertain input is sampled from its distribution and pushed through the
deterministic engine (§§1–6) N times (default ≈ 4×10⁴):

- Normal: $x\sim\mathcal N(\mu,\sigma)$, truncated to physical bounds.
- Lognormal: $x = a\,e^{\mathcal N(0,\ln b)}$ (median a, geometric std b) for
  strictly-positive, right-skewed quantities.

Outputs are read empirically from the sample:
$$P(\text{net}>0)=\tfrac1N\!\sum_k \mathbf 1[\text{net}_k>0]\,,\quad P(\mathrm{MAC}<c)=\tfrac1N\!\sum_k \mathbf 1[\mathrm{MAC}_k<c]$$
with P05/P95 percentiles reported for the MAC distribution.

The per-field **data tier** sets the noise floor σ: COMPUTED ≈ 5 %, LAB_VALIDATED
≈ 3 %, LIT_EXTRACTED ≈ 20 %, ESTIMATED ≈ 40 % (typically CAPEX).

**Source.** Monte-Carlo uncertainty propagation (GUM Supplement 1, JCGM 101:2008,
as the standard reference for propagating distributions).

**Rationale.** The engine is deterministic; all uncertainty is in the inputs, so
propagating input distributions is the correct, assumption-light way to obtain
honest output distributions. Probabilities are empirical frequencies over the
samples — not parametric approximations. Tiers tie the noise directly to data
provenance, so the width of the answer reflects how well the inputs are actually
known.

---

## 9. Uncertainty calibration

Applied to a learned surrogate's predictive std **before** it enters §8, so the
probabilities are trustworthy.

**Equations (code: `calibration.py`)**

Empirical coverage of central Gaussian intervals (diagnosis):
$$\widehat{\text{cov}}(\ell)=\tfrac1m\sum_i \mathbf 1\big[\,|y_i-\mu_i|\le z_\ell\,\sigma_i\,\big]\,,\quad z_\ell=\Phi^{-1}\!\big(\tfrac12+\tfrac{\ell}{2}\big)$$

Temperature scaling (global std correction), fitted by Gaussian MLE:
$$s^2=\frac1m\sum_i\frac{(y_i-\mu_i)^2}{\sigma_i^2}\,,\qquad \sigma_i' = s\,\sigma_i$$

Normalised split-conformal prediction (distribution-free, finite-sample):
$$r_i=\frac{|y_i-\mu_i|}{\sigma_i}\,,\quad q=\text{the }\big\lceil (m+1)(1-\alpha)\big\rceil/m\ \text{empirical quantile of }\{r_i\}\,,\quad \text{interval}=\mu\pm q\,\sigma$$

**Source.** Coverage/temperature scaling for regression: Kuleshov, Fenner & Ermon,
*ICML* 2018 (calibrated regression) and Levi et al. 2022 (calibrating regression
uncertainty); the scalar temperature idea generalises Guo et al., *ICML* 2017.
Split-conformal / normalised conformal: Vovk, Gammerman & Shafer 2005; Lei et al.,
*JASA* 2018; Papadopoulos et al. 2008 (normalised nonconformity).

**Rationale.** An over-confident surrogate yields feasibility probabilities that
are confident **and wrong** — the worst failure for a decision tool. Temperature
scaling is a one-parameter, robust fix for systematic over/under-confidence;
split-conformal adds a distribution-free finite-sample coverage guarantee. The
conformal half-width is mapped back to a Gaussian-equivalent std so the existing
Monte-Carlo sampler consumes calibrated uncertainty unchanged (a documented
width-matching approximation).

---

## 10. Global sensitivity (Sobol)

**Method (code: `uncertainty.py::sobol_indices`, via SALib)**

Variance-based total-effect indices S_T are computed with Saltelli sampling over
the input ranges, ranking each input by its share of MAC variance.

**Source.** Sobol, *Math. Comput. Simul.* 2001; Saltelli et al. 2010 (the
estimators SALib implements).

**Rationale.** Variance-based (rather than local/one-at-a-time) sensitivity is
appropriate because the model is nonlinear and inputs are uncertain over wide
ranges; total-effect indices capture interactions. The tornado of S_T tells the
user *what to work on* (catalyst vs energy vs plant), which is the tool's purpose.

---

## 11. Consolidated references

1. Bard, A. J.; Faulkner, L. R. *Electrochemical Methods: Fundamentals and
   Applications.* (Faraday's law; specific energy.)
2. Jouny, M.; Luc, W.; Jiao, F. *A General Techno-Economic Analysis of CO₂
   Electrolysis Systems.* Ind. Eng. Chem. Res., 2018. (TEA validation anchor.)
3. Osorio-Tejada, J. et al. *Techno-economic analysis of CO₂ conversion.*
   Energy Environ. Sci., 2024. (Like-for-like levelised-cost anchor; ACCR = CRF.)
4. von der Assen, N.; Bardow, A. and co-workers — CO₂-utilisation LCA methodology.
   (Utilisation-vs-sequestration accounting.)
5. JCGM 101:2008 — *Evaluation of measurement data — Supplement 1 to the GUM*
   (Monte-Carlo propagation of distributions).
6. Nørskov, J. K. et al. *Origin of the Overpotential...* / computational hydrogen
   electrode, J. Phys. Chem. B, 2004.
7. Peterson, A. A.; Nørskov, J. K. — CO₂ reduction limiting-potential / scaling
   relations.
8. Kuleshov, V.; Fenner, N.; Ermon, S. *Accurate Uncertainties for Deep Learning
   Using Calibrated Regression.* ICML, 2018.
9. Levi, D. et al. *Evaluating and Calibrating Uncertainty Prediction in
   Regression Tasks.* 2022.
10. Guo, C. et al. *On Calibration of Modern Neural Networks.* ICML, 2017.
11. Vovk, V.; Gammerman, A.; Shafer, G. *Algorithmic Learning in a Random World.*
    2005; Lei, J. et al. *Distribution-Free Predictive Inference for Regression.*
    JASA, 2018; Papadopoulos, H. et al. — normalised conformal, 2008.
12. Sobol, I. M. *Global sensitivity indices...* Math. Comput. Simul., 2001;
    Saltelli, A. et al. 2010 (variance-based estimators; SALib).

Grid carbon intensities used in scenarios: Ember / IEA 2024; Australian National
Greenhouse Accounts Factors 2025. Dataset citations for the ML/calibration work
are in `docs/VALIDATION.md`.
