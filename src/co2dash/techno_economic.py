"""
Translation layer: maps technical KPIs (FE, V_cell, current density, stability)
to techno-economic-environmental indicators (LCOP, net abatement, MAC).

Everything is a *pure function* of plain floats so it can be (a) unit-tested,
(b) vectorised for Monte-Carlo uncertainty propagation, and (c) fed to a global
sensitivity analysis. No I/O, no hidden state.

All quantities are on a PER-KG-OF-PRODUCT basis unless stated otherwise.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

F = 96485.0          # C/mol, Faraday constant
J_PER_KWH = 3.6e6


# ---------------------------------------------------------------------------
# Techno-economics
# ---------------------------------------------------------------------------
def capital_recovery_factor(rate: float, lifetime_yr: float) -> float:
    """CRF = i(1+i)^n / ((1+i)^n - 1). Annualises CAPEX."""
    i, n = rate, lifetime_yr
    if i == 0:
        return 1.0 / n
    g = (1.0 + i) ** n
    return i * g / (g - 1.0)


def specific_electricity_kwh_per_kg(n_electrons: float,
                                    cell_voltage: float,
                                    molar_mass_prod: float,
                                    faradaic_efficiency: float,
                                    rectifier_eff: float = 0.95) -> float:
    """
    E_el = n F V_cell / (M * FE * eta_rect)   [J/kg]  ->  kWh/kg

    This is the term with brutal sensitivity to FE and V_cell; exposing it is
    most of the point of the platform.
    """
    fe = np.clip(faradaic_efficiency, 1e-3, 1.0)
    e_j_per_kg = (n_electrons * F * cell_voltage) / (molar_mass_prod * fe * rectifier_eff)
    return e_j_per_kg / J_PER_KWH


def lcop(capex_total: float,
         annual_production_kg: float,
         opex_fix_per_yr: float,
         disc_rate: float,
         lifetime_yr: float,
         c_co2: float, m_co2: float,        # $/kg, kg/kg
         c_elec: float, e_elec: float,      # $/kWh, kWh/kg
         c_h2: float, m_h2: float) -> float:
    """Levelized cost of product [$/kg]."""
    crf = capital_recovery_factor(disc_rate, lifetime_yr)
    fixed_per_kg = (capex_total * crf + opex_fix_per_yr) / annual_production_kg
    variable_per_kg = c_co2 * m_co2 + c_elec * e_elec + c_h2 * m_h2
    return fixed_per_kg + variable_per_kg


# ---------------------------------------------------------------------------
# Life-cycle / climate accounting
# ---------------------------------------------------------------------------
def net_abatement_kg_per_kg(m_co2: float,
                            grid_intensity: float,   # kg CO2 / kWh
                            e_elec: float,           # kWh / kg prod
                            e_capture: float,        # kg CO2 / kg prod (capture penalty)
                            e_process: float,        # kg CO2 / kg prod (other process)
                            release_fraction: float  # phi in [0,1]: 0 durable, 1 fuel
                            ) -> float:
    """
    Net CO2 abatement per kg product:
        (1-phi)*m_co2  -  grid_intensity*e_elec  -  e_capture  -  e_process

    The (1-phi) term is the crux: utilisation != sequestration. A re-oxidised
    fuel (phi->1) keeps almost none of the utilisation credit.
    """
    stored = (1.0 - release_fraction) * m_co2
    emitted = grid_intensity * e_elec + e_capture + e_process
    return stored - emitted


def breakeven_grid_intensity(m_co2: float, e_elec: float,
                             e_capture: float, e_process: float,
                             release_fraction: float) -> float:
    """
    Grid carbon-intensity I* [kg CO2/kWh] at which net abatement = 0.
    Above I*, the conversion *increases* emissions. Closed form -> a first-class
    constraint in the dashboard, not a footnote.
    """
    if e_elec <= 0:
        return np.inf
    return ((1.0 - release_fraction) * m_co2 - e_capture - e_process) / e_elec


# ---------------------------------------------------------------------------
# Marginal abatement cost -- the single number investors & regulators read
# ---------------------------------------------------------------------------
def marginal_abatement_cost(lcop_ccu: float,
                            lcop_conventional: float,
                            net_abatement: float) -> float:
    """
    MAC = (LCOP_CCU - LCOP_conv) / net_abatement   [$/kg CO2 avoided]

    Returns +inf if net_abatement <= 0 (no climate benefit -> abatement cost
    undefined / infinite). Multiply by 1000 for $/tonne.
    """
    if net_abatement <= 0:
        return np.inf
    return (lcop_ccu - lcop_conventional) / net_abatement


# ---------------------------------------------------------------------------
# Scenario container -- one struct that the UI / MC / Sobol all consume
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    # --- reaction / performance (the levers research can move) ---
    n_electrons: float
    molar_mass_prod: float
    m_co2: float                 # kg CO2 / kg prod (from stoichiometry)
    m_h2: float                  # kg H2 / kg prod
    faradaic_efficiency: float
    cell_voltage: float
    # --- plant economics ---
    capex_total: float
    annual_production_kg: float
    opex_fix_per_yr: float
    disc_rate: float
    lifetime_yr: float
    c_co2: float
    c_elec: float
    c_h2: float
    lcop_conventional: float
    # --- environment ---
    grid_intensity: float
    e_capture: float
    e_process: float
    release_fraction: float
    rectifier_eff: float = 0.95

    def evaluate(self) -> dict:
        e_elec = specific_electricity_kwh_per_kg(
            self.n_electrons, self.cell_voltage, self.molar_mass_prod,
            self.faradaic_efficiency, self.rectifier_eff)
        cost = lcop(self.capex_total, self.annual_production_kg, self.opex_fix_per_yr,
                    self.disc_rate, self.lifetime_yr,
                    self.c_co2, self.m_co2, self.c_elec, e_elec, self.c_h2, self.m_h2)
        net = net_abatement_kg_per_kg(self.m_co2, self.grid_intensity, e_elec,
                                      self.e_capture, self.e_process, self.release_fraction)
        mac = marginal_abatement_cost(cost, self.lcop_conventional, net)
        istar = breakeven_grid_intensity(self.m_co2, e_elec, self.e_capture,
                                         self.e_process, self.release_fraction)
        return {"e_elec_kwh_per_kg": e_elec,
                "lcop_usd_per_kg": cost,
                "net_abatement_kg_per_kg": net,
                "mac_usd_per_kg_co2": mac,
                "mac_usd_per_tonne_co2": mac * 1000.0,
                "breakeven_grid_intensity": istar}


# ---------------------------------------------------------------------------
# VECTORISED PATH (piece 2)
# ---------------------------------------------------------------------------
# The scalar functions above are kept unchanged (the existing tests and the
# scalar Scenario.evaluate rely on them). For Monte-Carlo / Sobol we want a
# single vectorised pass instead of a Python loop. These functions accept NumPy
# arrays and broadcast.
def marginal_abatement_cost_array(lcop_ccu, lcop_conventional, net_abatement):
    """Vectorised MAC. Returns +inf wherever net_abatement <= 0 (no climate
    benefit), matching the scalar function's semantics, but element-wise."""
    lcop_ccu = np.asarray(lcop_ccu, float)
    lcop_conventional = np.asarray(lcop_conventional, float)
    net = np.asarray(net_abatement, float)
    safe = np.where(net > 0, net, np.nan)
    mac = (lcop_ccu - lcop_conventional) / safe
    return np.where(net > 0, mac, np.inf)


def evaluate_array(base: "Scenario", overrides: dict | None = None) -> dict:
    """
    Vectorised evaluation. `overrides` maps Scenario field names to NumPy arrays
    (or scalars); any field not overridden falls back to base's scalar value.
    Returns a dict of arrays. Equivalent to calling base.evaluate() per sample,
    but in one broadcast pass.

    NOTE: disc_rate and lifetime_yr must remain scalar (they feed the CRF, whose
    closed form is scalar). All performance/economic/environment levers may be
    arrays.
    """
    overrides = overrides or {}
    def g(name):
        return overrides[name] if name in overrides else getattr(base, name)

    fe = np.clip(g("faradaic_efficiency"), 1e-3, 1.0)
    vcell = np.maximum(g("cell_voltage"), 1e-3)

    e_elec = specific_electricity_kwh_per_kg(
        g("n_electrons"), vcell, g("molar_mass_prod"), fe, g("rectifier_eff"))

    cost = lcop(g("capex_total"), g("annual_production_kg"), g("opex_fix_per_yr"),
                base.disc_rate, base.lifetime_yr,
                g("c_co2"), g("m_co2"), g("c_elec"), e_elec, g("c_h2"), g("m_h2"))

    net = net_abatement_kg_per_kg(g("m_co2"), g("grid_intensity"), e_elec,
                                  g("e_capture"), g("e_process"), g("release_fraction"))

    mac = marginal_abatement_cost_array(cost, g("lcop_conventional"), net)

    return {"e_elec_kwh_per_kg": e_elec,
            "lcop_usd_per_kg": cost,
            "net_abatement_kg_per_kg": net,
            "mac_usd_per_kg_co2": mac,
            "mac_usd_per_tonne_co2": mac * 1000.0}
