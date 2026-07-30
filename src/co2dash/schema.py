"""
Canonical data schema for the CO2-utilization techno-economic platform.

Design principle: every quantity carries (i) a *value*, (ii) an *uncertainty*,
and (iii) a *provenance tier*. The tier is not metadata for display only -- it is
part of the noise model. A literature-extracted faradaic efficiency enters the
surrogate / Monte-Carlo propagation with a larger variance than a lab-validated
one, and that variance flows all the way to the MAC distribution.

No empirical values are hard-coded here. Defaults are explicitly flagged as
ILLUSTRATIVE placeholders to be replaced with sourced data.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import math


class DataTier(IntEnum):
    """Provenance tiers, ordered by trust. The integer is used to scale noise."""
    COMPUTED       = 0   # MP / OQMD / OC20 / Catalysis-Hub DFT  -> known functional error
    LAB_VALIDATED  = 1   # your own QE/VASP + collaborators' measurements -> calibration ground truth
    LIT_EXTRACTED  = 2   # FE / STY scraped from papers + SI      -> flagged, human-checked
    ESTIMATED      = 3   # CAPEX / offtake / scenario assumptions -> range, never a point


# Multiplicative inflation applied to a quantity's relative std as a function of
# tier. These are MODELLING CHOICES (priors), not measurements: tune against a
# held-out validation set. Documented so a reviewer can challenge them.
TIER_REL_STD_FLOOR = {
    DataTier.COMPUTED:      0.05,   # ~5% relative spread floor for a good DFT descriptor
    DataTier.LAB_VALIDATED: 0.03,
    DataTier.LIT_EXTRACTED: 0.20,   # large: extraction + reproducibility + half-cell bias
    DataTier.ESTIMATED:     0.40,   # CAPEX-class uncertainty
}


@dataclass
class Quantity:
    """A scalar with uncertainty and provenance. std is absolute (same units)."""
    value: float
    std: float = 0.0
    tier: DataTier = DataTier.ESTIMATED
    unit: str = ""
    source: str = ""          # DOI / API id / 'assumption'
    note: str = ""

    def effective_std(self) -> float:
        """Std, never below the tier-dependent relative floor. This is where the
        tier physically enters the uncertainty budget."""
        floor = TIER_REL_STD_FLOOR[self.tier] * abs(self.value)
        return max(self.std, floor)

    def __repr__(self) -> str:
        return (f"{self.value:.4g}±{self.effective_std():.2g} {self.unit} "
                f"[{self.tier.name}]")


@dataclass
class Reaction:
    """Electrochemical CO2-reduction reaction stoichiometry (per product molecule)."""
    name: str
    n_electrons: int                 # electrons transferred per product molecule
    molar_mass_prod: float           # kg/mol of the target product
    co2_per_prod_mol: float          # mol CO2 incorporated per mol product
    h2_per_prod_mol: float = 0.0     # mol H2 consumed per mol product (thermocat. routes)

    @property
    def kg_co2_per_kg_prod(self) -> float:
        M_CO2 = 0.04401  # kg/mol
        return self.co2_per_prod_mol * M_CO2 / self.molar_mass_prod

    @property
    def kg_h2_per_kg_prod(self) -> float:
        M_H2 = 0.002016
        return self.h2_per_prod_mol * M_H2 / self.molar_mass_prod


# --- Well-defined, non-empirical reference reactions (pure stoichiometry) -------
# These are textbook stoichiometries, not performance claims.
RXN_METHANOL = Reaction(
    name="CO2 -> CH3OH (6e-)", n_electrons=6,
    molar_mass_prod=0.03204, co2_per_prod_mol=1.0)
RXN_FORMATE = Reaction(
    name="CO2 -> HCOOH (2e-)", n_electrons=2,
    molar_mass_prod=0.04603, co2_per_prod_mol=1.0)
RXN_CO = Reaction(
    name="CO2 -> CO (2e-)", n_electrons=2,
    molar_mass_prod=0.02801, co2_per_prod_mol=1.0)

REACTIONS = {r.name: r for r in (RXN_METHANOL, RXN_FORMATE, RXN_CO)}


@dataclass
class Candidate:
    """A catalyst candidate: descriptor vector (from public DFT DBs) + predicted
    performance (filled by the surrogate). No DFT is computed here."""
    material_id: str
    descriptors: dict = field(default_factory=dict)   # e.g. d-band centre, dE(*CO), dE(*COOH)
    faradaic_efficiency: Optional[Quantity] = None    # predicted by surrogate
    cell_voltage: Optional[Quantity] = None
    current_density: Optional[Quantity] = None        # mA/cm^2 -> STY / CAPEX coupling
    source_db: str = ""                               # 'catalysis-hub' | 'oc20' | 'materials-project'
