# Validating the discovery→decision chain

How well does `composition → E_ads → U_L → V_cell → MAC` actually work on real
data? This document records the measurements, including the ones that came out
badly. Reproduce everything with the `co2dash.chain` API against the FeCoNiCuMo
supplementary workbook (Chen et al., ACS Catal. **12**, 14864–14871 (2022),
DOI [10.1021/acscatal.2c03675](https://doi.org/10.1021/acscatal.2c03675)).

## The dataset, and what it does not contain

Three sheets, one per adsorbed intermediate, 10 sites × 4 elemental descriptors
(group, period, electronegativity, unpaired d-count) + `Eads (eV)`. The
`Labels` column duplicates the target and is dropped — using it would be target
leakage (`hea.assert_no_leakage`).

| sheet | rows | sd(y) eV | site-1 elements covered |
|---|---|---|---|
| \*COOH | 280 | 0.284 | Co, Cu, Fe, Mo, Ni |
| \*CHO | 216 | 0.351 | Co, Cu, Fe, Mo, Ni |
| \*CO | 172 | 0.153 | Co, Fe, Mo, Ni — **no Cu** |

**The sheets are not a common configuration set.** Joining on the 40-dim
descriptor vector: CO ∩ COOH = **21**, CO ∩ CHO = 22, all three = 14. The CO
pathway is computable for 21 configurations; HCOOH and CH₃OH are not computable
at all, because \*OCHO, \*CH₂O and \*OCH₃ are absent (`hea.pathway_coverage`).

**Descriptor degeneracy.** Identical descriptor vectors with different targets:
\*CO 7 pairs (mean |Δy| 0.015 eV), \*CHO 1 pair (0.219 eV), \*COOH 2 pairs (mean
0.203 eV). For the polyatomic intermediates the 10-site composition does not
determine the adsorption geometry, which sets an irreducible error floor.

## Surrogate accuracy

5-fold CV, `BayesianLinearSurrogate` with α and β fitted by evidence
maximisation (`fit_evidence`):

| sheet | CV RMSE eV | R² | 95% coverage |
|---|---|---|---|
| \*COOH | 0.106 | 0.86 | 1.00 |
| \*CHO | 0.146 | 0.83 | 0.96 |
| \*CO | 0.112 | 0.47 | 0.99 |

\*CO's low R² is **low target variance**, not model failure: the sheet spans only
0.71 eV (sd 0.153), so there is little signal to capture.

Before the evidence fit, `beta` was hardcoded at 50, fixing the aleatoric floor
at 0.141 eV — larger than the measured CV RMSE on every sheet. The reported
uncertainty was a constructor default, and coverage sat at 0.99–1.00. Fitting
brings coverage to 0.93–0.95.

## End-to-end hold-out (`validate_pathway`)

All 21 CO ∩ COOH configurations held out, models **retrained** on the remainder
(CO 172→151, COOH 280→259), then surrogate-driven U_L compared with the U_L
computed from the DFT energies. Retraining is essential: those configurations
are in both sheets, so skipping it would measure memorisation.

```
E_ads RMSE      *CO 0.105   *COOH 0.102 eV
U_L RMSE        0.102 V     bias +0.017 V
amplification   0.98–1.00
PDS agreement   100%
rank corr       0.801
slope           0.62
```

**The max() operator does not amplify error** — amplification ≈ 1.0.

> **Superseded, and instructive.** The "PDS agreement 100%" in the block above
> was an **artefact of the single-shift anchor**, not a property of the system.
> With real gas-phase references the two shifts differ by 10.8 eV and the
> potential-determining step is *not* uniform — see *With real references* below.
> The amplification result survives (each configuration's error still passes
> through its own limiting step roughly one-for-one); the PDS uniformity does
> not. Left in place rather than deleted, because it shows exactly how an
> under-determined reference frame produces a self-consistent wrong answer.

**Range compression is the real problem.** Slope of predicted-on-true U_L is
**0.62**:

```
true U_L spread   0.836 V   (−1.102 … −0.266)
predicted spread  0.440 V   (−0.913 … −0.473)
```

Consequences: top-5 overlap 3/5, the genuinely best configuration ranks **3rd**,
and propagating into MAC gives median 6.2%, P90 17.6%, max 34.8% error — from
the surrogate alone, before any reference uncertainty.

## The ceiling is the descriptors, not the model

Two explanations for the compression were tested. **Both were rejected.**

**1. Wasted capacity on arbitrary site ordering.** Replacing the 36 environment
columns with permutation-invariant summaries (element counts, per-descriptor
mean/std/min/max) made things clearly worse:

| sheet | 40 raw features | 24 symmetrised |
|---|---|---|
| \*CO | 0.112 | 0.120 |
| \*CHO | 0.146 | **0.264** |
| \*COOH | 0.106 | **0.196** |

Sites 2–10 are **not** interchangeable; the column order encodes real geometry.

**2. The linear model is the limit.** Four model classes on the same hold-out:

| model | CO CV | COOH CV | U_L RMSE | slope | rank ρ | top-5 |
|---|---|---|---|---|---|---|
| Bayesian linear | 0.108 | 0.102 | 0.102 | 0.62 | 0.80 | 3/5 |
| Random forest | 0.100 | 0.109 | 0.110 | 0.61 | 0.72 | 4/5 |
| Gradient boosting | 0.102 | 0.103 | 0.101 | 0.65 | 0.76 | 3/5 |
| GP with ARD | 0.102 | 0.095 | 0.103 | 0.69 | 0.78 | 3/5 |

All within 0.007 eV on RMSE and 0.08 on slope. **When a linear model and a GP
with ARD agree to that precision, the regressor is not the constraint.** The
descriptors explain ~65% of the U_L variance on this hold-out, consistent with
the degeneracy measured above.

Raising the ceiling needs richer *features* — d-band centre, generalised
coordination number, local strain — not a different regressor. The Bayesian
linear model is retained deliberately: it ties on accuracy, is closed-form, and
supplies the calibrated σ the chain propagates into the cell voltage.

## The reference-frame problem

`proxy.limiting_potential` expects CHE formation free energies referenced to
CO₂(g) + n_H·½H₂. The workbook holds **adsorption** energies. Feeding them
directly gives U_L > 0 for every configuration — unphysical for CO₂ reduction —
which then clips to zero overpotential and yields a **constant** cell voltage,
i.e. silently no catalyst dependence. `hea.check_energy_reference` detects this
and refuses.

Three modes are offered (`chain.ReferenceFrame`):

| mode | needs | gives |
|---|---|---|
| `relative` (default) | nothing | ranking only; V_cell stays user-set |
| `anchored` | one known U_L | absolute U_L, but under-determined — see below |
| `absolute` | your own E(CO₂), E(H₂), E(H₂O) | fully rigorous; **use this** |

Published *total* energies are not portable between DFT setups (code,
functional, pseudopotentials, cutoff), so "use literature values" is not
available for `absolute` — borrowing another group's totals shifts every U_L by
an unknown constant.

### Anchored mode is under-determined

Each species has its **own** reference shift (`hea.che_reference_shift`):
shift(\*CO) ≠ shift(\*COOH), because the balanced half-reactions differ. A single
anchor determines only one of them. It suffices **only while the first PCET step
limits**, where U_L = −(E_ads(X) + shift(X)) and no other shift enters. If the
second step limits, the answer depends on the *difference* of two independent
shifts and is wrong without looking wrong. `run_chain` checks the resulting PDS
and warns.

### With real references: the PDS varies, and the bottleneck is not electrochemical

Gas-phase total energies from the group that ran the slabs (VASP PAW-PBE,
`chain.VASP_PBE_GAS_REFERENCE`) close the reference frame properly. The two
shifts differ by **10.8 eV** — the quantity a single anchor was implicitly
setting to zero:

```
shift(*CO)   = +15.50 eV
shift(*COOH) = +26.34 eV
```

**Verification of the references.** The reference set is checked against a
reaction whose energy is known experimentally:

| CO2 + H2 -> CO + H2O | value |
|---|---|
| VASP PAW-PBE (used by the code) | +0.700 eV |
| experiment (RWGS, 298 K) | +0.427 eV |

The +0.27 eV excess is the documented GGA CO2 overbinding ("OCO backbone")
error, a systematic property of the functional rather than a setup fault — a
mismatched pseudopotential or an unconverged cutoff would give a scattered
deviation, not a characteristic one. **No correction has been applied**, so it
propagates into every ΔG involving CO2. Whether to correct it depends on what
the original slab work did: if no correction was used there, adding one now
would break consistency with the adsorption energies.

With both shifts correct, on the 21 joined configurations:

```
PDS            COOH->CO x16      CO2->COOH x5
dG(CO2->COOH)  -1.084 ... -0.248 eV
dG(COOH->CO)   -0.957 ... -0.226 eV
U_L            +0.226 ... +0.724 V
```

Every proton-coupled step is downhill at zero potential. **The electrochemistry
is not the bottleneck**, and the varying PDS is exactly the scaling-relation
circumvention the source paper describes.

### The real limit: CO desorption

U_L > 0 means no potential is required — which for CO2→CO signals the opposite
problem. `proxy.desorption_free_energy` and `proxy.equilibrium_coverage` evaluate
the chemical step \*CO → CO(g), using G_f(CO(g)) = the reverse water-gas-shift
energy, +0.700 eV:

```
limitation        desorption x21  (all of them)
dG_desorption     +1.767 ... +2.187 eV
coverage (298 K)  1.0000 everywhere
```

Ordering by site-1 element: **Ni least bound (1.77 eV) < Fe < Co (2.19 eV)**.

Desorption transfers no electrons, so its free energy does not shift with
applied potential: it cannot appear in the CHE ladder and cannot be fixed by
polarising the electrode. `chain.run_chain` therefore derives **no cell voltage**
in this regime — fabricating one would misrepresent both the mechanism and the
remedy. The two limits are reported side by side, never combined: different
units, different physics, different fixes. They are also independent, not
complementary — a surface can require a potential *and* hold its product, and
both are reported.

**A structural consequence**, found while writing the tests and now pinned by
one: for CO2→CO you cannot have U_L > 0 *and* weak binding. U_L > 0 requires
G(\*CO) ≤ G(\*COOH) ≤ 0, hence ΔG_des ≥ G_f(CO(g)) > 0, which already saturates
the coverage at 298 K. **An apparently perfect CHE ladder means a poisoned
surface, not a good catalyst.**

### The published band cannot serve as an anchor

The source paper reports |U_L| = **0.29–0.51 V**. Encoded as
`chain.HEA_CO2RR_BAND` and testable with `check_against_published_band`:

```
computed |U_L| width   0.836 V
published band width   0.220 V
width ratio            3.8×
fraction inside band   9.5%
shift can reconcile    False
```

A reference shift *translates* the distribution without narrowing it, so the
disagreement is **independent of any anchor choice**. Two candidate explanations
were proposed; the real references settled one of them:

1. the band describes the paper's *designed* surfaces, not the full sampled set
   (across all 280 \*COOH rows the spread is 1.65 eV);
2. ~~the potential-determining step varies between configurations~~ —
   **confirmed**. With both shifts correct the PDS splits 16/5, and the width
   ratio drops from **3.8× to 2.26×**, with 33% of configurations now inside the
   band. Right direction, still not reconcilable, so (1) is very likely also in
   play.

The residual gap plus the universal CO poisoning points to the same conclusion:
**the workbook's sampled configurations are not the paper's designed catalyst.**
The paper's 0.29–0.51 V surface must sit in a weaker-binding regime than any of
these 21.

This also settles the mode question: `absolute` is not merely cleaner but
**necessary**, because only two independent shifts let the PDS vary at all.

## Applicability domain

The \*CO sheet contains no Cu-terminated adsorption sites, so any Cu-bearing
composition extrapolates for that intermediate. On equimolar FeCoNiCuMo:

```
full ensemble   ΔE *CO  −1.677 ± 0.653 eV
in-domain only  ΔE *CO  −1.993 ± 0.118 eV   (81% of configurations)
```

0.32 eV of bias and 5.5× the spread. **The surrogate's own σ does not detect
it** (±0.168 eV): Cu's descriptor values appear in the environment columns, so
every feature is marginally in range and the novelty is only in the joint
position, which a linear model cannot see. `chain.EnsemblePrediction.warning()`
detects it from site-1 coverage and from the spread ratio, and reports
in-domain-only statistics as a fallback.

## What is still assumed

**Faradaic efficiency is never predicted.** No descriptor→selectivity model
exists; FE passes through untouched and is labelled assumed in
`ChainProvenance`. It enters the cost as 1/FE, so it dominates the TEA. Closing
this needs materials with *both* descriptors and measured FE — a data-collection
problem. `link.descriptor_request_list()` ranks which surfaces would unlock the
most records.

**A composition is a distribution, not a surface.** Sites are drawn
independently (ideal random solid solution); short-range order, segregation and
facet effects are not modelled, so the reported configurational spread is a
lower bound. On equimolar FeCoNiCuMo the configurational spread (±0.22–0.32 eV)
exceeds the model uncertainty (±0.11 eV) by 2–3×.

## Reproduce

```python
from co2dash.hea import load_workbook, join_intermediates, che_reference_shift
from co2dash.chain import (validate_pathway, ReferenceFrame,
                           check_against_published_band,
                           VASP_PBE_GAS_REFERENCE as GAS, VASP_PBE_CO_GAS)
from co2dash.proxy import limiting_analysis

sheets = load_workbook("data/cs2c03675_si_002.xlsx")

# surrogate accuracy through the whole chain (illustrative anchor is fine here:
# the hold-out compares predicted against DFT in the SAME frame)
frame = ReferenceFrame(mode="anchored", anchor_energies={"COOH": -0.90},
                       anchor_U_L=-0.45, anchor_source="illustrative")
v = validate_pathway(sheets, frame, product="CO")
print(v.summary(), v.compression_note(), sep="\n")

# the physics, with the real references
RWGS = (VASP_PBE_CO_GAS + GAS["H2O"]) - (GAS["CO2"] + GAS["H2"])   # +0.700 eV
rows, _ = join_intermediates(sheets, ["CO", "COOH"])
for r in rows[:3]:
    che = {"CO": r["energies"]["CO"] + VASP_PBE_CO_GAS + che_reference_shift("CO", GAS),
           "COOH": r["energies"]["COOH"]}
    la = limiting_analysis(che, "CO", gas_formation_energy=RWGS)
    print(f"{r['site1']:3} U_L {la['U_L']:+.3f} V  dG_des {la['dG_desorption']:+.3f} eV"
          f"  coverage {la['coverage']:.4f}  -> {la['limitation']}")
```

## Assumption still unverified

$E_{ads}$(\*CO) is treated as referenced to CO(g), and $E_{ads}$(\*COOH) as
already referenced to CO₂ + ½H₂ (no stable gas-phase COOH exists). Every result
above is sensitive to this. The source paper's computational-methods section
would settle it, and given how much now rests on it, that check is overdue.
