"""
Recommendation synthesis.

Turns the platform's analyses (MC verdict, Sobol sensitivity, breakeven grid,
1-D target search, optional candidate ranking) into a single plain-language
"what to do next" answer for a user who did not build the model.

The engine already knows *which lever matters* (Sobol) and *whether it's
climate-positive* (breakeven); this module composes those into actionable
guidance and, for the dominant controllable lever, computes the target value
that would cross into the viable region.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from .techno_economic import Scenario, evaluate_array
from .uncertainty import propagate_mc, sobol_indices

# Controllable levers, their improvement direction, and how to build a sweep.
# direction: +1 => higher is better, -1 => lower is better.
_LEVERS = {
    "faradaic_efficiency": (+1, lambda s: np.linspace(s.faradaic_efficiency, 0.999, 60)),
    "cell_voltage":        (-1, lambda s: np.linspace(s.cell_voltage, 1.3, 60)),
    "grid_intensity":      (-1, lambda s: np.linspace(s.grid_intensity, 0.0, 60)),
    "c_elec":              (-1, lambda s: np.linspace(s.c_elec, 0.0, 60)),
    "capex_total":         (-1, lambda s: np.linspace(s.capex_total, s.capex_total * 0.2, 60)),
}
_LABEL = {"faradaic_efficiency": "faradaic efficiency", "cell_voltage": "cell voltage (V)",
          "grid_intensity": "grid intensity (kgCO₂/kWh)", "c_elec": "electricity price ($/kWh)",
          "capex_total": "CAPEX ($)"}


@dataclass
class Recommendation:
    verdict: str                    # 'Feasible' | 'Marginal' | 'Not climate-positive'
    p_net_positive: float
    p_feasible: float
    mac_median: float               # $/t CO2
    mac_p05: float
    mac_p95: float
    net_abatement: float
    breakeven_grid: float
    grid_ok: bool
    dominant_lever: Optional[str]
    dominant_ST: float
    target_field: Optional[str]
    target_value: Optional[float]
    target_reachable: bool
    next_candidate: Optional[str] = None
    steps: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(f"• {s}" for s in self.steps)


def _recenter(dists: Dict[str, tuple], base: Scenario) -> Dict[str, tuple]:
    """Re-centre registry-derived distributions on the current base value (so a
    field overridden after YAML load — e.g. grid via the region selector — is
    sampled around its new value, keeping the tier-derived spread shape)."""
    out = {}
    for f, d in dists.items():
        if not hasattr(base, f):
            out[f] = d; continue
        v, kind = getattr(base, f), d[0]
        if kind in ("normal", "lognormal"):
            out[f] = (kind, v, d[2])
        elif kind == "uniform":
            half = (d[2] - d[1]) / 2.0
            out[f] = ("uniform", max(0.0, v - half), v + half)
        else:
            out[f] = d
    return out


def _feasible_mask(base: Scenario, field_name: str, values: np.ndarray,
                   carbon_price: float) -> np.ndarray:
    out = evaluate_array(base, {field_name: values})
    mac = out["mac_usd_per_kg_co2"]
    net = out["net_abatement_kg_per_kg"]
    return np.isfinite(mac) & (mac < carbon_price) & (net > 0)


def _target_value(base: Scenario, field_name: str, carbon_price: float):
    """First value along the improvement sweep that becomes feasible.
    Returns (value|None, reachable)."""
    direction, sweep = _LEVERS[field_name]
    xs = sweep(base)
    feas = _feasible_mask(base, field_name, xs, carbon_price)
    if feas.any():
        return float(xs[np.argmax(feas)]), True     # argmax -> first True along sweep
    return None, False


def recommend(base: Scenario, carbon_price_usd_per_kg: float,
              registry=None, next_candidate: Optional[str] = None,
              n_mc: int = 40_000, seed: int = 0) -> Recommendation:
    """Produce a Recommendation for a scenario.
    `registry` (a ProvenanceRegistry) is used, if given, to derive MC
    distributions from data tiers; otherwise a generic ±spread is used."""
    ev = base.evaluate()
    net = ev["net_abatement_kg_per_kg"]
    bg = ev.get("breakeven_grid_intensity", float("nan"))
    grid_ok = bool(np.isfinite(bg) and base.grid_intensity < bg)

    # uncertainty distributions
    fields = ["faradaic_efficiency", "cell_voltage", "c_elec", "capex_total",
              "grid_intensity", "lcop_conventional"]
    if registry is not None:
        dists = _recenter(registry.mc_distributions(fields), base)
    else:
        dists = {
            "faradaic_efficiency": ("normal", base.faradaic_efficiency, 0.05),
            "cell_voltage": ("normal", base.cell_voltage, 0.3),
            "c_elec": ("normal", base.c_elec, max(1e-3, 0.2 * base.c_elec)),
            "capex_total": ("lognormal", base.capex_total, 1.4),
            "grid_intensity": ("uniform", max(0.0, base.grid_intensity - 0.03),
                               base.grid_intensity + 0.03),
        }
    mc = propagate_mc(base, dists, carbon_price_usd_per_kg, n=n_mc, seed=seed)

    p_net, p_feas = mc["p_net_positive"], mc["p_mac_below_carbon_price"]
    verdict = ("Not climate-positive" if p_net < 0.5
               else "Feasible" if p_feas >= 0.5 else "Marginal")

    # dominant controllable lever via Sobol (only if climate-positive enough to matter)
    dominant, dom_ST = None, 0.0
    tgt_field, tgt_val, reachable = None, None, False
    if p_net >= 0.2:
        bounds = {}
        for f in _LEVERS:
            v = getattr(base, f)
            if f == "faradaic_efficiency":
                bounds[f] = (max(0.05, v * 0.7), min(1.0, v * 1.3))
            elif f == "grid_intensity":
                bounds[f] = (0.0, max(0.1, v * 1.5))
            else:
                bounds[f] = (v * 0.7, v * 1.3)
        # NOTE: a missing SALib must NOT be silently reported as 'no dominant
        # lever' -- that is an unavailable analysis, not a result. Import errors
        # propagate with an actionable message; only genuine numerical failures
        # of the analysis itself degrade to None.
        try:
            S = sobol_indices(base, bounds, n=512)
        except ImportError as exc:                       # dependency, not a finding
            raise ImportError(
                "Sobol sensitivity requires SALib (declared in pyproject "
                "dependencies). Install it with `pip install SALib` -- without "
                "it the dominant-lever recommendation cannot be computed and "
                "must not be reported as absent.") from exc
        try:
            dominant, dom_ST = max(((k, s["ST"]) for k, s in S.items()),
                                   key=lambda kv: kv[1])
        except (ValueError, KeyError, TypeError):        # analysis returned nothing usable
            dominant, dom_ST = None, 0.0
        if dominant is not None:
            tgt_field = dominant
            tgt_val, reachable = _target_value(base, dominant, carbon_price_usd_per_kg)

    rec = Recommendation(
        verdict=verdict, p_net_positive=p_net, p_feasible=p_feas,
        mac_median=mc["mac_median"] * 1000, mac_p05=mc["mac_p05"] * 1000,
        mac_p95=mc["mac_p95"] * 1000, net_abatement=net, breakeven_grid=bg,
        grid_ok=grid_ok, dominant_lever=dominant, dominant_ST=dom_ST,
        target_field=tgt_field, target_value=tgt_val, target_reachable=reachable,
        next_candidate=next_candidate)
    rec.steps = _compose(base, rec, carbon_price_usd_per_kg)
    return rec


def _compose(base: Scenario, r: Recommendation, cp: float) -> List[str]:
    s: List[str] = []
    cp_t = cp * 1000
    # 1. headline verdict
    if r.verdict == "Not climate-positive":
        s.append(f"This route is NOT climate-positive as configured "
                 f"(P(removes CO₂)={r.p_net_positive:.0%}). Net abatement is "
                 f"{r.net_abatement:+.2f} kg CO₂/kg product — fix this before economics.")
    else:
        s.append(f"Verdict: {r.verdict}. It removes CO₂ with probability "
                 f"{r.p_net_positive:.0%}; MAC ≈ {r.mac_median:.0f} $/t "
                 f"(90% CI {r.mac_p05:.0f}–{r.mac_p95:.0f}); "
                 f"P(MAC < {cp_t:.0f} $/t) = {r.p_feasible:.0%}.")

    # 2. the climate gate (electricity carbon)
    if not r.grid_ok and np.isfinite(r.breakeven_grid):
        s.append(f"Electricity carbon is the binding constraint: your grid "
                 f"({base.grid_intensity:.3f} kgCO₂/kWh) exceeds the breakeven "
                 f"({r.breakeven_grid:.3f}). Move to lower-carbon power (renewable "
                 f"PPA) before catalyst work — no catalyst gain fixes a dirty grid.")

    # 3. the dominant lever + target value
    if r.dominant_lever is not None:
        lab = _LABEL.get(r.dominant_lever, r.dominant_lever)
        if r.verdict == "Feasible":
            s.append(f"Biggest lever on the outcome: {lab} (Sobol ST={r.dominant_ST:.2f}). "
                     f"Already in the viable region — protect this value in scale-up.")
        elif r.target_reachable and r.target_value is not None:
            cur = getattr(base, r.dominant_lever)
            s.append(f"Most impactful next target: bring {lab} from {cur:.3g} to "
                     f"≈ {r.target_value:.3g} to cross into the viable region "
                     f"(it is the dominant lever, Sobol ST={r.dominant_ST:.2f}).")
        else:
            s.append(f"The dominant lever is {lab} (Sobol ST={r.dominant_ST:.2f}), but "
                     f"improving it alone does not reach viability at {cp_t:.0f} $/t — "
                     f"you likely need two levers together (use the Feasibility envelope).")

    # 4. which candidate to compute next
    if r.next_candidate:
        s.append(f"Best next calculation/experiment: candidate '{r.next_candidate}' "
                 f"(highest information gain toward the viability decision).")
    else:
        s.append("To choose the next catalyst to compute, load candidate descriptors "
                 "into the Active-learning tab; the top-ranked one is the most "
                 "informative next DFT run.")

    # 5. honesty note
    s.append("Note: MAC uncertainty is dominated by CAPEX (an ESTIMATED input). "
             "Tighten it with a costed design before quoting a single number.")
    return s
