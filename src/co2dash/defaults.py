"""
Sourced defaults with provenance.

When a user supplies only a few measured fields (e.g. FE and cell voltage),
the remaining economic/environmental inputs are filled from these documented
defaults. Every default carries a tier and a source so the UI can show the
user exactly what was assumed and let them override it. Values are real but
route/scale dependent — treat ESTIMATED items (esp. CAPEX) as wide.
"""
from __future__ import annotations
from typing import Dict
from .schema import Quantity, DataTier


def _q(v, std, tier, src, unit=""):
    return Quantity(value=v, std=std, tier=tier, unit=unit, source=src)


# Conventional (fossil) market price of the product, $/kg — route dependent.
CONVENTIONAL_PRICE: Dict[str, Quantity] = {
    "co":       _q(0.60, 0.10, DataTier.LIT_EXTRACTED, "Merchant CO ~0.6 $/kg (Jouny, Luc & Jiao TEA 2018)", "$/kg"),
    "methanol": _q(0.45, 0.10, DataTier.LIT_EXTRACTED, "Methanol ~0.35-0.60 $/kg (market/Jouny 2018)", "$/kg"),
    "formate":  _q(0.74, 0.12, DataTier.LIT_EXTRACTED, "Formic acid ~0.74 $/kg (Jouny, Luc & Jiao TEA 2018)", "$/kg"),
}

# Everything else that a user typically will not have measured.
GENERIC_DEFAULTS: Dict[str, Quantity] = {
    "faradaic_efficiency": _q(0.50, 0.20, DataTier.ESTIMATED, "PLACEHOLDER — enter your measured FE", "-"),
    "cell_voltage":     _q(3.00, 0.50, DataTier.ESTIMATED,   "PLACEHOLDER — enter your measured cell voltage", "V"),
    "c_co2":            _q(0.05, 0.02, DataTier.LIT_EXTRACTED, "Point-source capture ~39-51 $/t (PNNL); CCS 15-130 $/t (S&P/IPCC)", "$/kg"),
    "c_elec":           _q(0.06, 0.02, DataTier.ESTIMATED,    "Generic industrial electricity; set to your tariff/PPA", "$/kWh"),
    "c_h2":             _q(4.00, 1.50, DataTier.LIT_EXTRACTED, "Green H2 ~3-6 $/kg (only used for hydrogenation routes)", "$/kg"),
    "grid_intensity":   _q(0.05, 0.02, DataTier.ESTIMATED,    "Low-carbon supply; set via the Grid region selector", "kgCO2e/kWh"),
    "capex_total":      _q(5.0e7, 2.0e7, DataTier.ESTIMATED,  "Order-of-magnitude plant CAPEX (~20 kt/yr); dominant uncertainty", "$"),
    "annual_production_kg": _q(2.0e7, 4.0e6, DataTier.ESTIMATED, "Assumed plant scale ~20 kt/yr", "kg/yr"),
    "opex_fix_per_yr":  _q(3.0e6, 1.0e6, DataTier.ESTIMATED,  "Fixed O&M ~6% of CAPEX/yr", "$/yr"),
    "disc_rate":        _q(0.08, 0.0, DataTier.ESTIMATED,     "Standard project discount rate", "1/yr"),
    "lifetime_yr":      _q(20.0, 0.0, DataTier.ESTIMATED,     "Standard plant life", "yr"),
    "e_capture":        _q(0.10, 0.04, DataTier.ESTIMATED,    "Capture parasitic emissions per kg product", "kgCO2/kg"),
    "e_process":        _q(0.05, 0.02, DataTier.ESTIMATED,    "Other process emissions per kg product", "kgCO2/kg"),
    "release_fraction": _q(0.50, 0.20, DataTier.ESTIMATED,    "Fraction of product carbon re-released; revise per downstream use", "-"),
    "rectifier_eff":    _q(0.95, 0.0, DataTier.ESTIMATED,     "AC/DC rectifier efficiency", "-"),
}


def defaults_for(reaction_key: str) -> Dict[str, Quantity]:
    """Return the full set of default Quantities for a route, including the
    route-specific conventional price. `reaction_key` in {co, methanol, formate}."""
    d = dict(GENERIC_DEFAULTS)
    d["lcop_conventional"] = CONVENTIONAL_PRICE.get(
        reaction_key, CONVENTIONAL_PRICE["co"])
    return d
