"""
Composition -> descriptor vectors for the CoCuFeMoNi high-entropy alloy.

WHY
---
The surrogate is trained on 10 surface sites x 4 elemental descriptors. Asking a
user to type 40 numbers is unusable and silently wrong the first time somebody
mis-orders a site. But those 40 numbers are not measurements: they are element
properties looked up from a table. So the honest interface takes a COMPOSITION
and derives the descriptors.

WHAT THIS MODELS
----------------
A composition does not determine a surface -- it determines a DISTRIBUTION over
site occupations. This module samples that distribution and returns the whole
ensemble, so downstream code predicts a DISTRIBUTION of adsorption energies
rather than a single number. Collapsing that ensemble to one scalar before the
user sees it is the failure mode this module exists to prevent.

Sampling model: independent multinomial draws per site, i.e. an ideal random
solid solution. Short-range order (Warren-Cowley alpha != 0), surface
segregation, and facet effects are NOT modelled. On a real HEA all three are
present, so the spread reported here is a LOWER BOUND on configurational spread.
`sro_note()` returns that caveat as text for the UI.

Site 1 is the adsorption site; sites 2..10 are its environment, following the
column order of the published workbook.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .hea import ELEMENT_BY_DESCRIPTOR

# symbol -> (Group, Period, EN, Nied), inverted from the workbook decoding table
DESCRIPTOR_BY_ELEMENT: Dict[str, Tuple[int, int, float, int]] = {
    sym: desc for desc, sym in ELEMENT_BY_DESCRIPTOR.items()
}
ELEMENTS: List[str] = sorted(DESCRIPTOR_BY_ELEMENT)

N_SITES_DEFAULT = 10
DESCRIPTORS_PER_SITE = ("Group", "Period", "EN", "Nied")


def feature_names(n_sites: int = N_SITES_DEFAULT) -> List[str]:
    """Column names in the workbook's order: Site 1 Group ... Site n Nied."""
    return [f"Site {k} {d}"
            for k in range(1, n_sites + 1)
            for d in DESCRIPTORS_PER_SITE]


def sro_note() -> str:
    return ("Sites are drawn independently (ideal random solid solution). "
            "Short-range order, surface segregation and facet effects are not "
            "modelled, so the configurational spread shown is a lower bound.")


@dataclass(frozen=True)
class Composition:
    """Elemental fractions of an HEA surface. Fractions must sum to 1."""
    fractions: Dict[str, float]

    def __post_init__(self):
        unknown = sorted(set(self.fractions) - set(ELEMENTS))
        if unknown:
            raise ValueError(
                f"element(s) {unknown} have no descriptor entry. This model "
                f"covers only {ELEMENTS} -- a material outside that set needs "
                f"its own DFT, not an extrapolation.")
        if any(v < 0 for v in self.fractions.values()):
            raise ValueError("negative fraction")
        total = sum(self.fractions.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"fractions must sum to 1, got {total:.6f}")

    @classmethod
    def equimolar(cls, elements: Optional[Sequence[str]] = None) -> "Composition":
        els = list(elements) if elements else ELEMENTS
        return cls({e: 1.0 / len(els) for e in els})

    @classmethod
    def from_string(cls, text: str) -> "Composition":
        """Parse 'Fe0.2Co0.2Ni0.2Cu0.2Mo0.2' or 'FeCoNiCuMo' (equimolar).

        Fractions are normalised if they sum to something other than 1, so
        'Fe2Co2Ni2Cu2Mo2' and atomic percentages both work; the normalisation is
        arithmetic only and never invents a missing element.
        """
        import re
        tokens = re.findall(r"([A-Z][a-z]?)\s*([0-9]*\.?[0-9]*)", str(text).strip())
        tokens = [(sym, amt) for sym, amt in tokens if sym]
        if not tokens:
            raise ValueError(f"could not parse any element from '{text}'")
        amounts: Dict[str, float] = {}
        for sym, amt in tokens:
            amounts[sym] = amounts.get(sym, 0.0) + (float(amt) if amt else 1.0)
        total = sum(amounts.values())
        if total <= 0:
            raise ValueError(f"all amounts are zero in '{text}'")
        return cls({k: v / total for k, v in amounts.items()})

    def as_vector(self) -> np.ndarray:
        """Fractions over the canonical ELEMENTS order (absent elements -> 0)."""
        return np.array([self.fractions.get(e, 0.0) for e in ELEMENTS], float)

    def label(self) -> str:
        parts = [f"{e}{self.fractions[e]:.2f}" for e in ELEMENTS
                 if self.fractions.get(e, 0.0) > 0]
        return "".join(parts)


def sample_configurations(comp: Composition,
                          n_samples: int = 500,
                          n_sites: int = N_SITES_DEFAULT,
                          fixed_site1: Optional[str] = None,
                          seed: int = 0) -> np.ndarray:
    """Draw site occupations. Returns an (n_samples, n_sites) array of element
    symbols.

    `fixed_site1` pins the adsorption-site element, which is what you want when
    asking 'what does CO binding look like on the Cu sites of this alloy?'. The
    remaining sites are still drawn from the full composition.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    if fixed_site1 is not None and fixed_site1 not in ELEMENTS:
        raise ValueError(f"unknown element '{fixed_site1}'; known: {ELEMENTS}")

    rng = np.random.default_rng(seed)
    p = comp.as_vector()
    p = p / p.sum()
    idx = rng.choice(len(ELEMENTS), size=(n_samples, n_sites), p=p)
    configs = np.array(ELEMENTS, dtype=object)[idx]
    if fixed_site1 is not None:
        configs[:, 0] = fixed_site1
    return configs


def configurations_to_descriptors(configs: np.ndarray) -> np.ndarray:
    """(n_samples, n_sites) element symbols -> (n_samples, 4*n_sites) features,
    in the workbook's column order."""
    configs = np.atleast_2d(configs)
    n, n_sites = configs.shape
    X = np.empty((n, 4 * n_sites), float)
    for k in range(n_sites):
        for i in range(n):
            sym = configs[i, k]
            try:
                X[i, 4 * k:4 * k + 4] = DESCRIPTOR_BY_ELEMENT[sym]
            except KeyError:
                raise KeyError(f"no descriptor for element '{sym}'") from None
    return X


def descriptors_for_composition(comp: Composition,
                                n_samples: int = 500,
                                n_sites: int = N_SITES_DEFAULT,
                                fixed_site1: Optional[str] = None,
                                seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience: composition -> (descriptor matrix, element configurations)."""
    configs = sample_configurations(comp, n_samples, n_sites, fixed_site1, seed)
    return configurations_to_descriptors(configs), configs


def align_to_training_columns(X: np.ndarray,
                              training_feature_names: Sequence[str],
                              n_sites: int = N_SITES_DEFAULT) -> np.ndarray:
    """Reorder/select generated columns to match the trained model's columns.

    Guards the classic silent failure: a model trained on the workbook's column
    order being fed a matrix built in a different order. Raises rather than
    predicting on misaligned features.
    """
    generated = feature_names(n_sites)
    pos = {name: i for i, name in enumerate(generated)}
    missing = [c for c in training_feature_names if c not in pos]
    if missing:
        raise KeyError(
            f"the trained model expects column(s) {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''} which this generator does not "
            f"produce. Refusing to predict on misaligned features.")
    return X[:, [pos[c] for c in training_feature_names]]
