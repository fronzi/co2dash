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
          "capex_total": "CAPEX ($)",
          # uncertain inputs that are NOT controllable levers: they can dominate
          # the variance (and so be worth measuring) without being engineerable
          "lcop_conventional": "conventional product price ($/kg)",
          "c_co2": "CO₂ feedstock cost ($/kg)",
          "c_h2": "H₂ cost ($/kg)"}


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
    top_uncertainty: Optional[str] = None   # biggest driver of MAC variance
    top_uncertainty_ST: float = 0.0
    sobol_reliable: bool = False
    sobol_reason: str = ""
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


_Z90 = 1.6448536269514722          # 5th/95th percentile of the standard normal

# physical clips applied after converting a distribution to a sweep range
_FIELD_CLIPS = {
    "faradaic_efficiency": (1e-3, 1.0),
    "cell_voltage": (1e-2, None),
    "grid_intensity": (0.0, None),
    "c_elec": (0.0, None),
    "capex_total": (0.0, None),
    "c_co2": (0.0, None),
}


def _bounds_from_distributions(dists: Dict[str, tuple], base: Scenario
                               ) -> Dict[str, tuple]:
    """Sobol ranges taken from the SAME distributions the Monte-Carlo uses.

    Previously the ranges were hand-written multipliers: +/-30% for most inputs
    but (0, 1.5x) for grid_intensity. Sobol indices are defined RELATIVE to the
    input ranges you choose, so giving one input a range three times wider than
    the others made it the 'dominant lever' almost by construction -- it came out
    top in 5 of 7 unrelated scenarios, including one where CAPEX had been
    quadrupled.

    Deriving the ranges from the actual uncertainty distributions makes the
    sensitivity analysis answer the decision-relevant question ('given what we
    genuinely do not know, what drives the answer?') and keeps it consistent
    with the MC that produced the verdict.
    """
    out: Dict[str, tuple] = {}
    for f, d in dists.items():
        if not hasattr(base, f):
            continue
        kind = d[0]
        if kind == "normal":
            lo, hi = d[1] - _Z90 * d[2], d[1] + _Z90 * d[2]
        elif kind == "lognormal":                       # d = (median, geo-sd)
            gsd = max(float(d[2]), 1.0 + 1e-9)
            lo, hi = d[1] * gsd ** -_Z90, d[1] * gsd ** _Z90
        elif kind == "uniform":
            lo, hi = d[1], d[2]
        else:
            continue
        clip_lo, clip_hi = _FIELD_CLIPS.get(f, (None, None))
        if clip_lo is not None:
            lo = max(lo, clip_lo)
        if clip_hi is not None:
            hi = min(hi, clip_hi)
        if hi <= lo:                                    # degenerate: nothing to vary
            continue
        out[f] = (float(lo), float(hi))
    return out


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
    top_uncertainty, top_ST = None, 0.0
    sobol_reliable, sobol_reason = False, ""
    tgt_field, tgt_val, reachable = None, None, False
    if p_net >= 0.2:
        bounds = _bounds_from_distributions(dists, base)
        # NOTE: a missing SALib must NOT be silently reported as 'no dominant
        # lever' -- that is an unavailable analysis, not a result. Import errors
        # propagate with an actionable message; only genuine numerical failures
        # of the analysis itself degrade to None.
        try:
            S, sdiag = sobol_indices(base, bounds, n=512,
                                     carbon_price_usd_per_kg=carbon_price_usd_per_kg,
                                     return_diagnostics=True)
        except ImportError as exc:                       # dependency, not a finding
            raise ImportError(
                "Sobol sensitivity requires SALib (declared in pyproject "
                "dependencies). Install it with `pip install SALib` -- without "
                "it the dominant-lever recommendation cannot be computed and "
                "must not be reported as absent.") from exc
        except ValueError:            # e.g. every draw non-finite: no variance to attribute
            S, sdiag = {}, {"reliable": False,
                            "reason": "no MAC variance could be attributed"}
        sobol_reliable = bool(sdiag.get("reliable", False))
        sobol_reason = str(sdiag.get("reason", ""))
        # Two different questions, deliberately answered separately:
        #  * top_uncertainty -- which input's OWN uncertainty drives the spread in
        #    MAC. This is what to go and measure. It may be an input you cannot
        #    control (e.g. electricity price).
        #  * dominant -- which CONTROLLABLE lever matters most. This is what to
        #    go and engineer.
        # Conflating them is how a recommendation ends up telling you to tighten
        # something the sensitivity analysis never identified.
        try:
            top_uncertainty, top_ST = max(((k, s["ST"]) for k, s in S.items()),
                                          key=lambda kv: kv[1])
        except (ValueError, KeyError, TypeError):        # analysis returned nothing usable
            top_uncertainty, top_ST = None, 0.0
        controllable = [(k, s["ST"]) for k, s in S.items() if k in _LEVERS]
        if controllable:
            dominant, dom_ST = max(controllable, key=lambda kv: kv[1])
        if dominant is not None:
            tgt_field = dominant
            tgt_val, reachable = _target_value(base, dominant, carbon_price_usd_per_kg)

    rec = Recommendation(
        verdict=verdict, p_net_positive=p_net, p_feasible=p_feas,
        mac_median=mc["mac_median"] * 1000, mac_p05=mc["mac_p05"] * 1000,
        mac_p95=mc["mac_p95"] * 1000, net_abatement=net, breakeven_grid=bg,
        grid_ok=grid_ok, dominant_lever=dominant, dominant_ST=dom_ST,
        target_field=tgt_field, target_value=tgt_val, target_reachable=reachable,
        next_candidate=next_candidate,
        top_uncertainty=top_uncertainty, top_uncertainty_ST=top_ST,
        sobol_reliable=sobol_reliable, sobol_reason=sobol_reason)
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

    # 5. what to go and MEASURE (as opposed to engineer) -- derived from the
    #    Sobol result, never asserted. This bullet previously hard-coded
    #    "uncertainty is dominated by CAPEX", which contradicted the analysis
    #    printed two bullets above whenever the top contributor was anything else.
    if r.top_uncertainty is not None:
        lab = _LABEL.get(r.top_uncertainty, r.top_uncertainty)
        lab_cap = lab[0].upper() + lab[1:]          # not .capitalize(): keeps kgCO₂/kWh
        # A total-order index above 1 is not a stronger finding, it is a broken
        # estimate: ST is a variance FRACTION. It happens here when the MAC
        # sample contains non-finite draws (net abatement <= 0), which
        # uncertainty.sobol_indices replaces with a large penalty value that
        # inflates the variance. Say so rather than quoting the number.
        if not r.sobol_reliable:
            s.append(f"Sensitivity is not reliable for this scenario ({r.sobol_reason}), "
                     f"so the apparent top contributor ({lab}) is reported as "
                     f"indicative only, not as a finding.")
        elif r.top_uncertainty == r.dominant_lever:
            s.append(f"{lab_cap} is both the biggest lever and the biggest source of "
                     f"spread in MAC (ST={r.top_uncertainty_ST:.2f}) — improving it "
                     f"and pinning it down are the same task.")
        else:
            s.append(f"Most of the spread in MAC comes from {lab} "
                     f"(ST={r.top_uncertainty_ST:.2f}), which is not the lever you "
                     f"would engineer. Narrowing its uncertainty will sharpen the "
                     f"answer more than improving any single lever.")
    else:
        s.append("No global sensitivity was computed for this scenario, so no "
                 "claim is made about what drives the uncertainty.")
    return s
