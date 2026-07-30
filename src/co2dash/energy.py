"""
Energy & grid module: choose a country/region to set grid carbon intensity (and,
where available, electricity price) for a scenario.

Two layers:
  * STATIC profiles  — sourced annual-average grid carbon intensities (2024) that
    work offline and in the GUI. Values are real and cited; verify before quoting.
  * LIVE connectors  — templates that fetch current values on an OPEN network:
        AU  -> OpenNEM / AEMO (price + emissions intensity, per NEM region)
        many countries -> ElectricityMaps (carbon intensity; needs API token)
    They run on the user's machine; a failed call raises (never fabricates).

Grid intensities are generation/location-based unless noted, in kg CO2(e)/kWh.
Electricity prices are deliberately LEFT UNSET in the static profiles (they are
volatile and not reliably free-queryable): provide them via the slider, a YAML
input, or a live fetch.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional, Dict
from .schema import Quantity, DataTier
from .techno_economic import Scenario

try:
    import requests
except Exception:
    requests = None


@dataclass
class CountryProfile:
    code: str
    name: str
    grid_intensity: Quantity                 # kg CO2e / kWh
    electricity_price: Optional[Quantity] = None   # $/kWh (usually None -> user/live)
    live_provider: str = ""                  # 'opennem' | 'electricitymaps' | ''
    live_zone: str = ""                       # provider-specific zone code


def _gi(value, source):
    return Quantity(value=value, std=0.10 * value, tier=DataTier.LIT_EXTRACTED,
                    unit="kgCO2e/kWh", source=source)


# Sourced 2024 annual-average grid carbon intensities (generation-based unless noted).
COUNTRY_PROFILES: Dict[str, CountryProfile] = {
    "AU": CountryProfile("AU", "Australia (national)",
        _gi(0.62, "DCCEEW NGA Factors 2025, national location-based avg"),
        live_provider="opennem", live_zone="NEM"),
    "AU-NSW": CountryProfile("AU-NSW", "Australia — NSW",
        _gi(0.66, "DCCEEW NGA Factors 2025, NSW"),
        live_provider="opennem", live_zone="NSW1"),
    "JP": CountryProfile("JP", "Japan",
        _gi(0.48, "Ember 2024 (482 gCO2/kWh); ELCS sales-based ~0.42 FY24-25"),
        live_provider="electricitymaps", live_zone="JP"),
    "US": CountryProfile("US", "United States",
        _gi(0.384, "Ember US Electricity 2025 (384 gCO2/kWh, 2024)"),
        live_provider="electricitymaps", live_zone="US"),
    "EU": CountryProfile("EU", "European Union",
        _gi(0.213, "Ember Global Electricity Review 2025 (EU 213 gCO2/kWh, 2024)"),
        live_provider="electricitymaps", live_zone="EU"),
    "DE": CountryProfile("DE", "Germany",
        _gi(0.363, "2024 consumed (~363 gCO2/kWh)"),
        live_provider="electricitymaps", live_zone="DE"),
    "CN": CountryProfile("CN", "China",
        _gi(0.560, "Ember 2024 (560 gCO2/kWh)"),
        live_provider="electricitymaps", live_zone="CN"),
    "IN": CountryProfile("IN", "India",
        _gi(0.713, "Ember 2024 (713 gCO2/kWh)"),
        live_provider="electricitymaps", live_zone="IN"),
    "BR": CountryProfile("BR", "Brazil",
        _gi(0.103, "Ember 2024 (103 gCO2/kWh)"),
        live_provider="electricitymaps", live_zone="BR"),
    "GLOBAL": CountryProfile("GLOBAL", "Global average",
        _gi(0.473, "Ember 2024 (473 gCO2/kWh); IEA 445 gCO2/kWh")),
    "RENEWABLE": CountryProfile("RENEWABLE", "Dedicated renewable (solar/wind PPA)",
        Quantity(0.045, 0.015, DataTier.LIT_EXTRACTED, "kgCO2e/kWh",
                 "Solar PV lifecycle ~40-50 gCO2e/kWh (IPCC)")),
}


def list_regions() -> Dict[str, str]:
    """code -> display name, for a GUI selector."""
    return {c.code: c.name for c in COUNTRY_PROFILES.values()}


# --------------------------------------------------------------------- live
def fetch_opennem_au(region: str = "NSW1", timeout: int = 30) -> Dict:
    """
    Live AU NEM price + emissions intensity via OpenNEM (open network).
    VERIFY the current OpenNEM endpoint/schema before relying on it; some
    deployments now require an API key. Returns {grid_intensity, electricity_price}.
    """
    if requests is None:
        raise RuntimeError("`requests` required: pip install requests")
    url = f"https://api.opennem.org.au/stats/au/NEM/{region}/power"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    # OpenNEM returns time-series; parsing depends on the live schema — adjust here.
    return {"_raw": data, "note": "parse latest emissions_intensity & price from series"}


def fetch_electricitymaps(zone: str, token: str, timeout: int = 30) -> Quantity:
    """
    Live grid carbon intensity for a zone via ElectricityMaps (needs API token).
    Returns a Quantity in kg CO2e/kWh.
    """
    if requests is None:
        raise RuntimeError("`requests` required: pip install requests")
    url = "https://api.electricitymap.org/v3/carbon-intensity/latest"
    r = requests.get(url, params={"zone": zone},
                     headers={"auth-token": token}, timeout=timeout)
    r.raise_for_status()
    ci = r.json().get("carbonIntensity")           # gCO2eq/kWh
    if ci is None:
        raise RuntimeError("ElectricityMaps returned no carbonIntensity")
    return Quantity(value=float(ci) / 1000.0, std=0.0, tier=DataTier.LIT_EXTRACTED,
                    unit="kgCO2e/kWh", source=f"ElectricityMaps live, zone={zone}")


# --------------------------------------------------------------------- apply
def get_energy(country_code: str, live: bool = False,
               em_token: Optional[str] = None) -> Dict:
    """
    Return {grid_intensity: Quantity, electricity_price: Quantity|None, live: bool}
    for a region. Static (sourced) by default; live fetch if requested and
    possible. Never fabricates: unknown code -> KeyError; failed live -> raises.
    """
    prof = COUNTRY_PROFILES[country_code]
    gi, price = prof.grid_intensity, prof.electricity_price
    used_live = False
    if live:
        if prof.live_provider == "electricitymaps" and em_token:
            gi = fetch_electricitymaps(prof.live_zone, em_token); used_live = True
        # OpenNEM live parsing left to the user (schema-dependent); static used otherwise.
    return {"grid_intensity": gi, "electricity_price": price, "live": used_live,
            "name": prof.name}


def apply_to_scenario(base: Scenario, country_code: str, **kwargs) -> Scenario:
    """Return a copy of `base` with grid_intensity (and price if available) set
    from the chosen region."""
    e = get_energy(country_code, **kwargs)
    updates = {"grid_intensity": e["grid_intensity"].value}
    if e["electricity_price"] is not None:
        updates["c_elec"] = e["electricity_price"].value
    return replace(base, **updates)
