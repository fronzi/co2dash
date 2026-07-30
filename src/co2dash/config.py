"""
Config loader with provenance registry (piece 1).

Operationalises the principle "every number has a source". A scenario is read
from a YAML file where each field is either a plain scalar or a tagged record:

    faradaic_efficiency: {value: 0.85, std: 0.04, tier: LAB_VALIDATED, source: "doi:..."}
    capex_total:         {value: 4.2e7, std: 1.5e7, tier: ESTIMATED, source: "NREL TEA 2024"}
    reaction: methanol            # selects stoichiometry (n, M, m_co2) from REACTIONS

The loader returns (Scenario, ProvenanceRegistry). The registry can:
  * render a provenance table (what fed each output, and how trustworthy);
  * AUTO-DERIVE the Monte-Carlo distributions from the tiers -- the tier is the
    noise model, so provenance and uncertainty are not specified twice.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import yaml

from .schema import Quantity, DataTier, REACTIONS, RXN_METHANOL, RXN_FORMATE, RXN_CO
from .techno_economic import Scenario


# Human-friendly aliases -> canonical reaction objects.
_REACTION_ALIASES = {
    "methanol": RXN_METHANOL, "ch3oh": RXN_METHANOL, "meoh": RXN_METHANOL,
    "formate": RXN_FORMATE, "formic": RXN_FORMATE, "hcooh": RXN_FORMATE,
    "co": RXN_CO, "carbon_monoxide": RXN_CO,
}


# Scenario fields whose stoichiometry comes from the chosen reaction, not the YAML.
_REACTION_DERIVED = {"n_electrons", "molar_mass_prod", "m_co2"}
# Fields that must stay scalar (feed the CRF) and are never treated as uncertain.
_SCALAR_ONLY = {"disc_rate", "lifetime_yr", "rectifier_eff"}


@dataclass
class ProvenanceRegistry:
    """field name -> Quantity(value, std, tier, source). The single source of
    truth for 'where did this number come from and how much do we trust it'."""
    entries: Dict[str, Quantity] = field(default_factory=dict)

    def add(self, name: str, q: Quantity) -> None:
        self.entries[name] = q

    def table(self) -> List[dict]:
        return [{"field": k, "value": q.value, "eff_std": q.effective_std(),
                 "tier": q.tier.name, "source": q.source}
                for k, q in self.entries.items()]

    def mc_distributions(self, fields: List[str]) -> Dict[str, tuple]:
        """Build the `uncertain` dict consumed by propagate_mc directly from the
        registry. Cost-like positive quantities (ESTIMATED tier) get a lognormal
        (no negative draws, right-skew); everything else a normal at the
        tier-floored std. This is where provenance becomes uncertainty."""
        dists: Dict[str, tuple] = {}
        for name in fields:
            if name not in self.entries or name in _SCALAR_ONLY:
                continue
            q = self.entries[name]
            sd = q.effective_std()
            if q.tier == DataTier.ESTIMATED and q.value > 0:
                # convert absolute std -> geometric std for a lognormal centred on value
                gstd = float(max(1.05, 1.0 + sd / abs(q.value)))
                dists[name] = ("lognormal", q.value, gstd)
            else:
                dists[name] = ("normal", q.value, sd)
        return dists


def _as_quantity(name: str, raw, unit: str = "") -> Tuple[float, Quantity]:
    """Accept a scalar or a {value,std,tier,source} dict; return (value, Quantity)."""
    if isinstance(raw, dict):
        tier = raw.get("tier", "ESTIMATED")
        tier = DataTier[tier] if isinstance(tier, str) else DataTier(tier)
        q = Quantity(value=float(raw["value"]), std=float(raw.get("std", 0.0)),
                     tier=tier, unit=raw.get("unit", unit),
                     source=raw.get("source", ""), note=raw.get("note", ""))
        return q.value, q
    return float(raw), Quantity(value=float(raw), std=0.0,
                                tier=DataTier.ESTIMATED, unit=unit,
                                source="literal-in-config")


def load_scenario(path: str) -> Tuple[Scenario, ProvenanceRegistry]:
    """Load a Scenario + ProvenanceRegistry from a YAML file."""
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)

    reg = ProvenanceRegistry()
    kwargs: Dict[str, float] = {}

    # 1) reaction stoichiometry (non-empirical) -------------------------------
    rxn_key = cfg.pop("reaction", None)
    if rxn_key is not None:
        key = str(rxn_key).strip().lower()
        rxn = _REACTION_ALIASES.get(key)
        if rxn is None:  # fall back to substring match against canonical names
            rxn = next((r for k, r in REACTIONS.items() if key in k.lower()), None)
        if rxn is None:
            raise ValueError(f"Unknown reaction '{rxn_key}'. "
                             f"Aliases: {sorted(_REACTION_ALIASES)}")
        kwargs.update(n_electrons=rxn.n_electrons,
                      molar_mass_prod=rxn.molar_mass_prod,
                      m_co2=rxn.kg_co2_per_kg_prod)

    # 2) remaining fields ------------------------------------------------------
    for name, raw in cfg.items():
        if name in _REACTION_DERIVED and name in kwargs:
            continue  # already set by the reaction
        value, q = _as_quantity(name, raw)
        kwargs[name] = value
        reg.add(name, q)

    return Scenario(**kwargs), reg
