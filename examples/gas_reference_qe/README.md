# Gas-phase reference energies for the CHE reference frame (Quantum ESPRESSO)

Three single-molecule calculations producing `E(CO2)`, `E(H2)`, `E(H2O)` — the
inputs `co2dash.chain.ReferenceFrame(mode="absolute")` needs to convert the
workbook's adsorption energies into CHE formation free energies.

## READ THIS FIRST — consistency is the whole point

A DFT total energy has **no absolute meaning**. It depends on the
pseudopotentials, functional, cutoff, smearing and code. Only differences taken
*within one consistent setup* are physical.

So these three numbers are only valid if they are computed with the **same code,
same functional, same pseudopotentials and same cutoffs as the slab
calculations** that produced `Eads (eV)` in the workbook.

- If the slabs were run in **Quantum ESPRESSO** → edit the marked parameters
  below to match those runs exactly, then use these files.
- If the slabs were run in **VASP** (common for HEA CO2RR work) → **these files
  will not help.** Mixing a QE molecule energy with a VASP slab energy
  reintroduces exactly the offset mismatch the reference frame exists to
  remove. Run the three molecules in VASP instead, with the slab INCAR settings
  and the same POTCARs.

Check the source paper's computational-methods section before running anything.

## What you must edit

Every file has a `!! MATCH THE SLABS` marker on the lines that must agree with
the original calculations:

| Parameter | Why it matters |
|---|---|
| `pseudo_dir`, `UPF` filenames | different pseudopotentials shift totals by tens of eV |
| `ecutwfc`, `ecutrho` | incomplete basis sets shift totals; must be identical |
| `input_dft` | leave unset to use the pseudopotential's functional (PBE); set only if the slabs overrode it |
| `nspin`, `starting_magnetization` | if the slabs were spin-polarised, match it |

The molecules are all closed-shell singlets, so `occupations = 'fixed'` and no
smearing is correct here even if the metallic slabs used smearing — smearing is
a property of the electronic structure being described, not a global setting to
replicate.

`assume_isolated = 'martyna-tuckerman'` removes the spurious interaction between
periodic images. Keep it; it makes the molecular energies correct and does not
affect comparability with the slabs (which are genuinely periodic).

## Run

```bash
cd examples/gas_reference_qe
pw.x -in co2.in > co2.out
pw.x -in h2.in  > h2.out
pw.x -in h2o.in > h2o.out
```

Each takes a few minutes on a laptop. On an Apple-silicon Mac, a Homebrew or
conda-forge `quantum-espresso` build is fine; add `mpirun -np 4` if you have MPI.

## Extract the energies

```bash
python extract_energies.py
```

Prints the final total energies converted from Ry to **eV** (Quantum ESPRESSO
reports Rydberg; co2dash expects eV — a factor of 13.6056931230 that is very
easy to forget and would silently corrupt every U_L).

Then in the app: **Composition tab → reference frame `absolute`** → enter the
three values. Or in code:

```python
from co2dash.chain import ReferenceFrame
frame = ReferenceFrame(mode="absolute",
                       gas_energies={"CO2": ..., "H2": ..., "H2O": ...})
```

## Sanity check before trusting the result

Atomisation-style differences are setup-insensitive enough to catch a blunder.
With PBE the reaction

    CO2 + H2  ->  CO + H2O

should be endothermic by roughly +0.3 to +0.5 eV. If your numbers give something
wildly different, suspect a mismatched pseudopotential or an unconverged cutoff
before proceeding. (This needs E(CO) as well — a fourth trivial calculation,
included as `co.in` for exactly this check.)
