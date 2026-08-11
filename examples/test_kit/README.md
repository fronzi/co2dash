# Test kit — data to exercise the app end to end

The real DFT workbook is publisher material and is gitignored, so a public clone
cannot use the three DFT tabs at all. This kit fills that gap.

> **The workbook here is SYNTHETIC. It is not DFT.** Adsorption energies come
> from a documented toy function of the site descriptors, chosen so the app has
> a learnable signal with a realistic spread. The file carries a
> `_SYNTHETIC_README` sheet you cannot miss. Use it to test the interface, never
> to draw a chemical conclusion.

Regenerate everything with:

```bash
python examples/test_kit/make_test_data.py
```

## What to upload where

| file | sidebar slot | exercises |
|---|---|---|
| `SYNTHETIC_HEA_workbook.xlsx` | 3 · DFT descriptors | Predict from composition, Next DFT to run, Model reliability |
| `measurements_edge_cases.csv` | 2 · Your measurements | unit fixes, range checks, current-density → CAPEX |
| `measurements_no_product_column.csv` | 2 · Your measurements | the "assume a product" prompt |
| `../example_SCENARIO_CO2-to-CO.yaml` | 1 · Scenario | reads "Marginal" |
| `../example_SCENARIO_CO2-to-CO_favourable.yaml` | 1 · Scenario | reads "Feasible" |

## The workbook

Same structure as the real one: `Labels`, 10 sites × 4 descriptors, `Eads (eV)`.

```
*CO    160 rows   sd 0.104 eV   site-1 elements: Co, Fe, Mo, Ni   (no Cu)
*CHO   200 rows   sd 0.221 eV   all five
*COOH  260 rows   sd 0.185 eV   all five
```

Two quirks are reproduced deliberately, because they are what makes the app's
guards worth having:

**\*CO has no Cu at the adsorption site.** Any Cu-bearing composition therefore
extrapolates for that intermediate, and the applicability guard should flag it —
about 21% of configurations for an equimolar alloy.

**30 configurations are shared across all three sheets**, so the CO pathway is
computable and `validate_pathway` has a hold-out set. A first version of this
generator drew every sheet independently, the sheets never intersected, and the
whole CHE/desorption path was silently untestable.

## What you should see

**Predict from composition** — three ΔE cards; the \*CO one carries an
extrapolation warning for any composition containing Cu, with in-domain-only
statistics offered underneath. Configurational spread exceeds model spread, as
on real data.

**Next DFT to run / Model reliability** — a ranking and a reliability diagram
for whichever intermediate you select in the sidebar.

**Hold-out validation**, on this synthetic data:

```
n=30 held out | E_ads RMSE *CO 0.059, *COOH 0.065 eV
U_L RMSE 0.065 V | slope 0.69 | PDS agreement 100%
```

The slope near 0.69 mirrors the real workbook's 0.62 — regression toward the
mean is a property of noisy data, not of the real system in particular.

**Your measurements** — six rows, each with one deliberate problem:

| row | what it tests |
|---|---|
| `clean-baseline` | nothing wrong |
| `fe-as-percent` | 92 → 0.92 |
| `voltage-in-mV` | 3200 mV → 3.2 V |
| `very-slow-cell` | 20 mA/cm² → CAPEX 2.2×10⁸ $ |
| `very-fast-cell` | 900 mA/cm² → CAPEX 4.8×10⁶ $ |
| `impossible-FE` | 140% → row rejected with an error |

The two current-density rows are the clearest demonstration in the kit: identical
FE and voltage, **45× difference in capital**, because current density sets the
electrode area you must buy.

## Honest limits of testing on synthetic data

It proves the software works. It proves nothing about catalysts. In particular
the desorption verdicts here follow from the toy binding energies, not from
chemistry — on this kit an equimolar alloy comes out electrochemically limited
*and* poisoned, which is an artefact of the chosen coefficients.

For real conclusions, use your own workbook and your own gas-phase references.
