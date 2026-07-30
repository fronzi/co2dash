"""
Physics activity proxy (Computational Hydrogen Electrode / Nørskov framework).

Turns the fetched intermediate energies into a THEORETICAL ACTIVITY target so the
whole loop can run on real Catalysis-Hub data without waiting for experimental FE.

What it computes
----------------
For a chosen CO2RR pathway, each proton-coupled electron-transfer (PCET) step has a
free energy at U = 0 (vs RHE). In the CHE model a reduction step's free energy is
ΔG_i(U) = ΔG_i(0) - eU, so the **limiting potential** (all steps exergonic) is

        U_L = - max_i ΔG_i(0) / e          (volts; e = 1 in eV/V units)

The most endergonic step is the **potential-determining step (PDS)**. The
**overpotential** is η = U_eq - U_L (≥ 0 for a real catalyst), with U_eq the
equilibrium potential of the overall reaction. A less negative U_L / smaller η =
more active catalyst.

Mapping to the engine
---------------------
Activity ≠ selectivity. The proxy therefore predicts a **cell voltage** (which the
TEA already consumes), NOT a faradaic efficiency:

        V_cell ≈ V_baseline + η

i.e. a more active catalyst needs less overpotential → lower V_cell → lower energy
cost → lower MAC. `V_baseline` (anode + ohmic + the rest of the cell) is a
transparent, adjustable parameter, not a first-principles cell model.

HONESTY NOTES
-------------
* Thermochemical corrections Δ(ZPE - TΔS) are DISABLED by default (the proxy then
  uses electronic energies only). `TYPICAL_ZPE_TS_CORRECTIONS` lists commonly-cited
  literature-scale values you can OPT INTO and must verify for your system/method.
* `EQUILIBRIUM_POTENTIALS` are standard textbook values; verify against your
  reference (pH, RHE/SHE convention) before quoting.
* The proxy assumes the intermediate energies you pass are CHE FORMATION free
  energies of *X from CO2(g) + n_H·(½H2) + *. Confirm the Catalysis-Hub reactions
  you fetched use this convention (their `equation` field helps).
* This is an ACTIVITY proxy, not a measurement. Replace with experimental FE when
  available (the pipeline does not change).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

# Equilibrium potentials U_eq (V vs RHE) -- standard values, VERIFY before quoting.
EQUILIBRIUM_POTENTIALS = {"CO": -0.10, "HCOOH": -0.12, "CH3OH": 0.02}

# Commonly-cited Δ(ZPE - TΔS) free-energy corrections (eV) for adsorbates at 298 K.
# DEFAULT IS OFF (zeros). Opt in by passing corrections=TYPICAL_ZPE_TS_CORRECTIONS,
# after verifying the values for your functional/reference.
TYPICAL_ZPE_TS_CORRECTIONS = {"COOH": 0.25, "CO": 0.10, "OCHO": 0.25,
                              "CHO": 0.20, "CH2O": 0.10, "OCH3": 0.25,
                              "H": 0.20, "OH": 0.30}

# Pathways as ordered PCET steps; each step is (reactant_state, product_state).
# 'CO2' is the (zero-energy) initial reference state on the clean surface.
PATHWAYS = {
    "CO":     [("CO2", "COOH"), ("COOH", "CO")],
    "HCOOH":  [("CO2", "OCHO"), ("OCHO", "HCOOH")],
    "CH3OH":  [("CO2", "COOH"), ("COOH", "CO"), ("CO", "CHO"),
               ("CHO", "CH2O"), ("CH2O", "OCH3"), ("OCH3", "CH3OH")],
}


def _formation_free_energies(intermediate_energies: Dict[str, float],
                             corrections: Optional[Dict[str, float]]) -> Dict[str, float]:
    """G_f(*X) = ΔE_f(*X) + Δ(ZPE-TΔS)_X. CO2 reference state = 0."""
    corr = corrections or {}
    g = {"CO2": 0.0}
    for sp, e in intermediate_energies.items():
        if e is None:
            continue
        g[sp] = float(e) + float(corr.get(sp, 0.0))
    return g


def limiting_potential(intermediate_energies: Dict[str, float],
                       product: str,
                       corrections: Optional[Dict[str, float]] = None,
                       equilibrium_potentials: Optional[Dict[str, float]] = None
                       ) -> Optional[Dict]:
    """
    Compute U_L, overpotential and the PDS for `product` from the per-surface
    intermediate formation energies. Returns None if a required step's species is
    missing (so the surface is skipped, not imputed).
    """
    if product not in PATHWAYS:
        raise ValueError(f"Unknown product '{product}'. Options: {list(PATHWAYS)}")
    g = _formation_free_energies(intermediate_energies, corrections)
    Ueq = (equilibrium_potentials or EQUILIBRIUM_POTENTIALS).get(product, 0.0)

    step_dG: List[Tuple[str, float]] = []
    for reactant, prod in PATHWAYS[product]:
        if reactant not in g or prod not in g:
            return None                      # incomplete pathway -> skip surface
        step_dG.append((f"{reactant}->{prod}", g[prod] - g[reactant]))

    pds_name, dG_max = max(step_dG, key=lambda kv: kv[1])
    U_L = -dG_max                            # e = 1 (eV/V)
    overpotential = Ueq - U_L                # >= 0 for a real catalyst
    return {"U_L": U_L, "overpotential": overpotential, "pds": pds_name,
            "steps": step_dG, "U_eq": Ueq}


def proxy_cell_voltage(overpotential: float, v_baseline: float = 2.0) -> float:
    """Transparent activity->voltage map: V_cell = V_baseline + overpotential.
    Floored at a small positive value."""
    return max(0.2, v_baseline + max(0.0, overpotential))


# species label in the descriptor table -> CHE species symbol used above
_DESCRIPTOR_TO_SPECIES = {"dE_CO": "CO", "dE_COOH": "COOH", "dE_OCHO": "OCHO",
                          "dE_CHO": "CHO", "dE_CH2O": "CH2O", "dE_OCH3": "OCH3",
                          "dE_H": "H", "dE_OH": "OH"}


def build_activity_targets(descriptor_table: List[Dict],
                           product: str = "CO",
                           descriptor_keys: Optional[List[str]] = None,
                           corrections: Optional[Dict[str, float]] = None,
                           v_baseline: float = 2.0) -> List[Dict]:
    """
    For each surface with the intermediates required by `product`, compute the CHE
    limiting potential and the proxy cell voltage (the surrogate TARGET y).

    Returns rows: {material_id, descriptors{...}, U_L, overpotential, v_cell,
                   pds} for surfaces where the pathway is computable.
    `descriptor_keys` are the features X the surrogate will train on (default: the
    two descriptors needed by the CO pathway).
    """
    keys = descriptor_keys or ["dE_CO", "dE_COOH"]
    rows: List[Dict] = []
    for i, row in enumerate(descriptor_table):
        inter = {sp: row.get(dk) for dk, sp in _DESCRIPTOR_TO_SPECIES.items()
                 if row.get(dk) is not None}
        lp = limiting_potential(inter, product, corrections)
        if lp is None:
            continue
        if any(row.get(k) is None for k in keys):
            continue                          # need a complete feature vector
        rows.append({
            "material_id": f"{row.get('surface','?')}({row.get('facet','')})#{i}",
            "descriptors": {k: float(row[k]) for k in keys},
            "U_L": lp["U_L"], "overpotential": lp["overpotential"],
            "pds": lp["pds"], "v_cell": proxy_cell_voltage(lp["overpotential"], v_baseline),
        })
    return rows
