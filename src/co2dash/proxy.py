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


# ---------------------------------------------------------------------------
# the non-electrochemical limit: product desorption
# ---------------------------------------------------------------------------
K_B_EV = 8.617333262e-5          # eV/K

# Which adsorbed state must leave the surface for each product, and the free
# energy of the desorbed molecule in the SAME CHE reference frame (CO2(g) +
# n_H*(1/2 H2) = 0). For CO this is the reverse water-gas-shift energy, which the
# gas-phase references already determine, so it is supplied by the caller rather
# than hardcoded.
DESORBING_STATE = {"CO": "CO", "HCOOH": "HCOOH", "CH3OH": "CH3OH"}


def desorption_free_energy(intermediate_energies: Dict[str, float],
                           product: str,
                           gas_formation_energy: float,
                           corrections: Optional[Dict[str, float]] = None
                           ) -> Optional[float]:
    """dG for the CHEMICAL step *X -> X(g), in the CHE reference frame.

        dG_des = G_f(X(g)) - G_f(*X)

    `gas_formation_energy` is G_f of the desorbed molecule referenced to
    CO2(g) + n_H*(1/2 H2); for CO2 -> CO it equals the reverse water-gas-shift
    energy E(CO) + E(H2O) - E(CO2) - E(H2).

    WHY THIS IS SEPARATE FROM U_L. Desorption transfers no electrons, so its free
    energy does NOT shift with applied potential: it cannot appear in the CHE
    ladder and cannot be fixed by polarising the electrode. A surface can have
    every proton-coupled step downhill (U_L > 0, apparently perfect) and still be
    inactive because the product never leaves. Folding it into an overpotential
    would misrepresent both the mechanism and the remedy.
    """
    state = DESORBING_STATE.get(product)
    if state is None or state not in intermediate_energies:
        return None
    corr = (corrections or {}).get(state, 0.0)
    g_ads = float(intermediate_energies[state]) + float(corr)
    return float(gas_formation_energy) - g_ads


def equilibrium_coverage(dg_desorption: float, temperature: float = 298.15) -> float:
    """Equilibrium fractional coverage of the desorbing species, from

        theta / (1 - theta) = exp(dG_des / kT)    =>    theta = 1/(1 + e^{-dG/kT})

    An equilibrium (Langmuir) statement, not a rate: it says how strongly the
    surface holds the product, not how fast it leaves. Preferred to an arbitrary
    eV cut-off because it is dimensionless, temperature-explicit and directly
    interpretable -- theta -> 1 is a poisoned surface.
    """
    x = float(dg_desorption) / (K_B_EV * float(temperature))
    if x > 700:                      # avoid overflow; already saturated
        return 1.0
    if x < -700:
        return 0.0
    import math
    return 1.0 / (1.0 + math.exp(-x))


def limiting_analysis(intermediate_energies: Dict[str, float],
                      product: str,
                      gas_formation_energy: Optional[float] = None,
                      corrections: Optional[Dict[str, float]] = None,
                      equilibrium_potentials: Optional[Dict[str, float]] = None,
                      temperature: float = 298.15,
                      poisoning_coverage: float = 0.99) -> Optional[Dict]:
    """Electrochemical AND desorption limits, reported side by side.

    Returns the `limiting_potential` result plus `dG_desorption`, `coverage` and
    a `limitation` label:

        'electrochemical'  U_L < 0: a potential is required, and the CHE ladder
                           describes the bottleneck.
        'desorption'       every PCET step is downhill but the product stays
                           bound (coverage above `poisoning_coverage`). Applying
                           potential does not help.
        'none'             neither limit binds -- check the inputs before
                           believing it.

    The two limits are deliberately NOT combined into one number: they have
    different units, different physics and different remedies.
    """
    lp = limiting_potential(intermediate_energies, product, corrections,
                            equilibrium_potentials)
    if lp is None:
        return None
    out = dict(lp)
    out["dG_desorption"] = None
    out["coverage"] = None
    if gas_formation_energy is not None:
        dg = desorption_free_energy(intermediate_energies, product,
                                    gas_formation_energy, corrections)
        if dg is not None:
            out["dG_desorption"] = dg
            out["coverage"] = equilibrium_coverage(dg, temperature)

    poisoned = (out["coverage"] is not None and out["coverage"] >= poisoning_coverage)
    if lp["U_L"] < 0:
        out["limitation"] = "electrochemical"
    elif poisoned:
        out["limitation"] = "desorption"
    else:
        out["limitation"] = "none"
    out["poisoned"] = bool(poisoned)
    return out


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
