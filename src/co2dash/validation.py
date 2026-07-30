"""
TEA validation anchor — Jouny, Luc & Jiao (2018),
"A General Techno-Economic Analysis of CO2 Electrolysis Systems"
(Ind. Eng. Chem. Res. 2018; manuscript OSTI 1712664).

Purpose: show the co2dash engine reproduces an established, transparent TEA on
the quantities that are *model-independent physics* (specific electricity
consumption, electricity & CO2 operating cost, electrolyzer electrode area and
stack capital), and to state clearly where co2dash and Jouny legitimately
differ (financing: co2dash uses CRF levelisation; Jouny uses a full MACRS
cash-flow NPV with tax — a scope difference, not an error).

All reference numbers below are taken from the paper's Table 2 (market prices)
and Table 3 (process assumptions). The reference physics implementation here is
INDEPENDENT of co2dash (hand-coded from first principles) so the comparison is a
genuine cross-check, not a re-run of the same code.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

F = 96485.0            # C/mol
J_PER_KWH = 3.6e6
M_CO2 = 0.04401        # kg/mol

# --- Jouny 2018 Table 3: process assumptions ---------------------------------
JOUNY_BASE = dict(elec_price=0.05, current_density_mA=200, cell_voltage=2.3,
                  faradaic=0.90, conversion=0.50, co2_price_per_t=100.0,
                  interest=0.10, electrolyzer_cost_per_m2=2830.0,
                  lifetime_yr=20, days_per_year=350, prod_ton_day=100)
JOUNY_OPT = dict(elec_price=0.03, current_density_mA=300, cell_voltage=2.0,
                 faradaic=0.90, conversion=0.50, co2_price_per_t=50.0,
                 interest=0.075, electrolyzer_cost_per_m2=1415.0,
                 lifetime_yr=20, days_per_year=350, prod_ton_day=100)

# --- Jouny 2018 Table 1/2: products (n electrons, molar mass, market price) ---
@dataclass
class Product:
    key: str
    n: int
    molar_mass: float      # kg/mol
    market_price: float    # $/kg (Table 2)
    co2_per_prod_mol: float = 1.0

PRODUCTS: Dict[str, Product] = {
    "co":       Product("co", 2, 0.02801, 0.60),
    "formate":  Product("formate", 2, 0.04603, 0.70),   # formic acid
    "methanol": Product("methanol", 6, 0.03204, 0.60),
    "ethylene": Product("ethylene", 12, 0.02805, 1.30, co2_per_prod_mol=2.0),
    "ethanol":  Product("ethanol", 12, 0.04607, 1.00, co2_per_prod_mol=2.0),
}


# --- INDEPENDENT first-principles references (do NOT call co2dash) ------------
def ref_specific_energy_kwh_per_kg(n: int, molar_mass: float, cell_voltage: float,
                                   faradaic: float) -> float:
    """E_el = n F V / (M * FE)  [J/kg] -> kWh/kg. Rectifier assumed ideal to
    match Jouny's 'electricity from the Faradaic mass balance'."""
    return (n * F * cell_voltage) / (molar_mass * faradaic) / J_PER_KWH


def ref_electricity_cost(n, molar_mass, cell_voltage, faradaic, elec_price) -> float:
    return ref_specific_energy_kwh_per_kg(n, molar_mass, cell_voltage, faradaic) * elec_price


def ref_co2_cost(molar_mass, co2_per_prod_mol, co2_price_per_t) -> float:
    """Stoichiometric CO2 feedstock cost per kg product ($/kg). Unconverted CO2
    is recycled, so net consumption is stoichiometric."""
    kg_co2_per_kg = co2_per_prod_mol * M_CO2 / molar_mass
    return kg_co2_per_kg * (co2_price_per_t / 1000.0)


def ref_electrolyzer_area_and_capex(prod_ton_day, n, molar_mass, faradaic,
                                    current_density_mA, cost_per_m2):
    """Electrode area and electrolyzer stack capital for a target production
    rate, from Jouny's area-cost method. area = I / j ; I = zF(mdot/M)/FE."""
    mdot = prod_ton_day * 1000.0 / 86400.0          # kg/s
    mol_s = mdot / molar_mass
    current = n * F * mol_s / faradaic               # A
    j_A_per_m2 = current_density_mA * 10.0           # mA/cm2 -> A/m2
    area = current / j_A_per_m2                       # m2
    return area, area * cost_per_m2


# --- validation runners ------------------------------------------------------
def validate_energy(case: Dict = JOUNY_BASE, products: List[str] = None) -> List[Dict]:
    """Compare co2dash specific_electricity_kwh_per_kg against the independent
    reference for each product. Returns rows with relative error."""
    from co2dash.techno_economic import specific_electricity_kwh_per_kg
    products = products or list(PRODUCTS)
    rows = []
    for k in products:
        p = PRODUCTS[k]
        ref = ref_specific_energy_kwh_per_kg(p.n, p.molar_mass, case["cell_voltage"], case["faradaic"])
        got = specific_electricity_kwh_per_kg(p.n, case["cell_voltage"], p.molar_mass,
                                              case["faradaic"], rectifier_eff=1.0)
        rows.append({"product": k, "E_ref_kWh_kg": ref, "E_co2dash_kWh_kg": got,
                     "rel_err": abs(got - ref) / ref})
    return rows


def energetic_efficiency(n, molar_mass, cell_voltage, faradaic, e_equilibrium_cell):
    """Jouny eq. 2 (single product): FE * E_eq / V_cell."""
    return faradaic * e_equilibrium_cell / cell_voltage


# =============================================================================
# SECOND ANCHOR — Osorio-Tejada, Escriba-Gelonch, Vertongen, Bogaerts & Hessel
# (2024), Energy Environ. Sci. 17, 5833; DOI 10.1039/d4ee00164h. Open access.
# Reports a LEVELISED unitary cost of production (UCOP) for CO2->CO electrolysis,
# and — crucially — uses an annual capital charge ratio (ACCR, 10%/20 yr) that is
# identical to co2dash's CRF. So co2dash's LCOP is a like-for-like target here
# (unlike Jouny, whose NPV uses MACRS + tax).
# =============================================================================
OSORIO2024 = dict(
    product="co", cell_voltage=3.0, current_density_mA=250, faradaic=0.85,
    spc=0.25, elec_price=0.03, co2_price_per_t=40.0, water_price_per_m3=14.0,
    water_use_m3_per_kg=0.458e-3, electrolyzer_cost_per_m2=10_000.0,
    plug_to_power=0.80, prod_ton_day=100, tonne_per_year=34_000,
    interest=0.10, lifetime_yr=20,
    # reported results (main text):
    reported_cell_kwh_kg=6.82, reported_total_conv_kwh_kg=8.53,
    reported_cell_area_m2=3791.0, reported_ucop_per_t=962.0,
    reported_ucop_range=(570.0, 1392.0),
    reported_feedstock_per_t=71.9, reported_elec_frac_of_ucop=0.27,
    source="Osorio-Tejada et al., Energy Environ. Sci. 2024, DOI 10.1039/d4ee00164h",
)


def validate_anchor_osorio() -> Dict:
    """Cross-check co2dash against Osorio-Tejada 2024 (CO2->CO), independent of
    the Jouny anchor and at a different operating point (V=3.0, FE=0.85), also
    exercising the plug-to-power / rectifier term."""
    from co2dash.techno_economic import specific_electricity_kwh_per_kg
    o = OSORIO2024
    p = PRODUCTS[o["product"]]

    # specific energy: cell (ideal rectifier) and total (80% plug-to-power)
    cell = specific_electricity_kwh_per_kg(p.n, o["cell_voltage"], p.molar_mass,
                                           o["faradaic"], rectifier_eff=1.0)
    total = specific_electricity_kwh_per_kg(p.n, o["cell_voltage"], p.molar_mass,
                                            o["faradaic"], rectifier_eff=o["plug_to_power"])
    # electrode area for the plant
    area, _ = ref_electrolyzer_area_and_capex(
        o["prod_ton_day"], p.n, p.molar_mass, o["faradaic"],
        o["current_density_mA"], o["electrolyzer_cost_per_m2"])
    # variable OPEX per tonne
    elec_per_t = total * o["elec_price"] * 1000.0
    co2_per_t = ref_co2_cost(p.molar_mass, p.co2_per_prod_mol, o["co2_price_per_t"]) * 1000.0
    water_per_t = o["water_use_m3_per_kg"] * o["water_price_per_m3"] * 1000.0

    return {
        "cell_kwh_kg": (cell, o["reported_cell_kwh_kg"]),
        "total_kwh_kg": (total, o["reported_total_conv_kwh_kg"]),
        "cell_area_m2": (area, o["reported_cell_area_m2"]),
        "elec_per_t": (elec_per_t, o["reported_elec_frac_of_ucop"] * o["reported_ucop_per_t"]),
        "feedstock_per_t": (co2_per_t + water_per_t, o["reported_feedstock_per_t"]),
    }


def co2dash_lcop_in_reported_range(scenario) -> Dict:
    """Check a co2dash CO scenario's LCOP lands inside Osorio-Tejada's reported
    electrolysis UCOP band ($570-1392 /t). Non-circular external corroboration."""
    lcop_per_t = scenario.evaluate()["lcop_usd_per_kg"] * 1000.0
    lo, hi = OSORIO2024["reported_ucop_range"]
    return {"lcop_per_t": lcop_per_t, "range": (lo, hi), "in_range": lo <= lcop_per_t <= hi}


# =====================================================================
# Second anchor: like-for-like LEVELIZED cost (LCOP) vs the published
# literature band for CO2 -> CO. These are LCOP (levelized) values, so
# they ARE comparable to co2dash's LCOP (unlike Jouny's NPV).
# =====================================================================

# Published levelized production costs for CO2->CO (independent TEAs), $/kg CO.
LITERATURE_LCOP_CO = {
    "Nature Sustainability 2021":        0.44,
    "Energy & Fuels 2023 (Dongare)":     0.449,
    "Plasma/electrolysis TEA 2024 (mid)": 0.962,
    "Single-atom NiN3 TEA 2024":         1.08,
    "ACS Energy Lett 2024 (base)":       1.22,
}
LITERATURE_LCOP_FORMATE = {
    "Nature Sustainability 2021":    0.59,
    "Energy & Fuels 2023 (Dongare)": 0.468,
}


def literature_band(product: str = "co"):
    d = LITERATURE_LCOP_CO if product == "co" else LITERATURE_LCOP_FORMATE
    vals = list(d.values())
    return min(vals), max(vals), d


def reconstruct_capex_co(prod_ton_day=100, faradaic=0.90, current_density_mA=200,
                         electrolyzer_cost_per_m2=2830.0, separation_bop_multiple=3.0):
    """Total plant capital for a CO2->CO plant, reconstructed from Jouny's area
    method for the electrolyzer stack times a separation+BoP multiple. The
    multiple is the dominant, uncertain lever (ACS Energy Lett 2024: separations
    dominate capital); sweeping it reproduces the literature LCOP spread."""
    p = PRODUCTS["co"]
    _, stack = ref_electrolyzer_area_and_capex(prod_ton_day, p.n, p.molar_mass,
                                               faradaic, current_density_mA,
                                               electrolyzer_cost_per_m2)
    return stack * separation_bop_multiple


def co2dash_lcop_co(cell_voltage=2.5, faradaic=0.90, elec_price=0.04,
                    co2_price_per_t=50.0, interest=0.075, lifetime_yr=20,
                    prod_ton_day=100, days_per_year=350, current_density_mA=200,
                    electrolyzer_cost_per_m2=2830.0,
                    separation_bop_multiple=3.0, rectifier_eff=0.90) -> dict:
    """Build a co2dash Scenario at a representative CO2->CO configuration and
    return its LCOP and breakdown (dogfoods the engine)."""
    from co2dash.techno_economic import Scenario
    p = PRODUCTS["co"]
    annual = prod_ton_day * 1000.0 * days_per_year
    capex = reconstruct_capex_co(prod_ton_day, faradaic, current_density_mA,
                                 electrolyzer_cost_per_m2, separation_bop_multiple)
    scen = Scenario(
        n_electrons=p.n, molar_mass_prod=p.molar_mass,
        m_co2=M_CO2 / p.molar_mass, m_h2=0.0,
        faradaic_efficiency=faradaic, cell_voltage=cell_voltage,
        capex_total=capex, annual_production_kg=annual,
        opex_fix_per_yr=0.025 * capex,           # Jouny maintenance 2.5%/yr
        disc_rate=interest, lifetime_yr=lifetime_yr,
        c_co2=co2_price_per_t / 1000.0, c_elec=elec_price, c_h2=0.0,
        lcop_conventional=p.market_price, grid_intensity=0.05,
        e_capture=0.0, e_process=0.0, release_fraction=0.0,
        rectifier_eff=rectifier_eff)
    ev = scen.evaluate()
    return {"lcop": ev["lcop_usd_per_kg"], "capex": capex,
            "e_elec": ev["e_elec_kwh_per_kg"], "sep_mult": separation_bop_multiple}


# Representative endpoints matching the assumptions the published TEAs used.
FAVOURABLE_CO = dict(cell_voltage=2.2, faradaic=0.95, elec_price=0.03,
                     co2_price_per_t=40.0, interest=0.075, current_density_mA=300,
                     separation_bop_multiple=3.0)
CONSERVATIVE_CO = dict(cell_voltage=3.2, faradaic=0.90, elec_price=0.06,
                       co2_price_per_t=100.0, interest=0.10, current_density_mA=100,
                       separation_bop_multiple=6.0)


def validate_lcop_band() -> dict:
    """Check co2dash CO LCOP brackets the published literature band as operating
    and capital assumptions move across the range the published TEAs used."""
    lo, hi, sources = literature_band("co")
    fav = co2dash_lcop_co(**FAVOURABLE_CO)["lcop"]
    cons = co2dash_lcop_co(**CONSERVATIVE_CO)["lcop"]
    co2dash_lo, co2dash_hi = min(fav, cons), max(fav, cons)
    brackets = bool((co2dash_lo <= lo) and (co2dash_hi >= hi))   # co2dash spans the band
    return {"lit_lo": lo, "lit_hi": hi, "sources": sources,
            "co2dash_favourable": fav, "co2dash_conservative": cons,
            "co2dash_lo": co2dash_lo, "co2dash_hi": co2dash_hi,
            "brackets_literature": brackets}
