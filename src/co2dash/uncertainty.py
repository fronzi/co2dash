"""
Uncertainty layer (VECTORISED -- piece 2).

(1) propagate_mc: push input distributions (spread set by the data tier) through
    the nonlinear TEA/LCA chain to a DISTRIBUTION of MAC. Single vectorised pass
    via techno_economic.evaluate_array -- no Python loop. Reports
    P(MAC < carbon_price), the decision-relevant quantity.

(2) sobol_indices: global (variance-based) sensitivity of MAC to the inputs.
    Also vectorised through evaluate_array.

Monte Carlo (not linear error propagation) because the map is strongly nonlinear
(1/FE, division by net_abatement) and tier-driven uncertainties are large.
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np

from .techno_economic import evaluate_array, Scenario


def _sample(rng, dist, size):
    kind, a, b = dist
    if kind == "normal":
        return rng.normal(a, b, size)
    if kind == "lognormal":            # a=median, b=geometric std (>1)
        return a * np.exp(rng.normal(0.0, np.log(b), size))
    if kind == "uniform":
        return rng.uniform(a, b, size)
    raise ValueError(kind)


def propagate_mc(base: Scenario,
                 uncertain: Dict[str, tuple],
                 carbon_price_usd_per_kg: float,
                 n: int = 50_000,
                 seed: int = 0) -> Dict[str, np.ndarray]:
    """
    base       : nominal Scenario
    uncertain  : {field_name: (kind, p1, p2)} distributions overriding nominal
    Vectorised: builds arrays of overrides and evaluates them in one pass.
    """
    rng = np.random.default_rng(seed)
    overrides = {k: _sample(rng, d, n) for k, d in uncertain.items()}
    # physical clips on sampled levers
    if "faradaic_efficiency" in overrides:
        overrides["faradaic_efficiency"] = np.clip(overrides["faradaic_efficiency"], 1e-3, 1.0)
    if "cell_voltage" in overrides:
        overrides["cell_voltage"] = np.maximum(overrides["cell_voltage"], 1e-3)

    out = evaluate_array(base, overrides)
    mac, cost, net = out["mac_usd_per_kg_co2"], out["lcop_usd_per_kg"], out["net_abatement_kg_per_kg"]
    finite = np.isfinite(mac)
    return {
        "mac": mac, "lcop": cost, "net_abatement": net, "e_elec": out["e_elec_kwh_per_kg"],
        "p_mac_below_carbon_price": float(np.mean(mac < carbon_price_usd_per_kg)),
        "p_net_positive": float(np.mean(net > 0)),
        "mac_median": float(np.median(mac[finite])) if finite.any() else np.inf,
        "mac_p05": float(np.percentile(mac[finite], 5)) if finite.any() else np.inf,
        "mac_p95": float(np.percentile(mac[finite], 95)) if finite.any() else np.inf,
    }


SOBOL_TARGETS = ("mac", "feasible")


def sobol_indices(base: Scenario,
                  bounds: Dict[str, Tuple[float, float]],
                  n: int = 1024,
                  target: str = "mac",
                  carbon_price_usd_per_kg: float = 0.0,
                  winsorise_q: float = 99.0,
                  return_diagnostics: bool = False):
    """
    First-order (S1) and total (ST) Sobol indices w.r.t. the inputs in `bounds`.
    Interpretation: high S1 -> directly worth pinning down experimentally;
    high ST with low S1 -> matters mainly through interactions.

    HANDLING NON-FINITE MAC
    -----------------------
    MAC is +inf wherever net abatement <= 0, so any scenario near the
    climate-positive boundary produces a sample with infinities. The previous
    implementation substituted `10 * max(finite Y)`, which injects an enormous
    artificial value: Sobol indices are variance FRACTIONS, and that penalty
    inflates the total variance so much that the estimates stop being meaningful
    -- it can and does produce ST > 1, which is impossible for a real index.

    Two honest options are offered instead:

    `target='mac'`      -- non-finite draws are winsorised to a high finite
                           percentile of the finite draws (default P99) rather
                           than an arbitrary multiple of the maximum. The
                           replacement is still a choice, so the fraction
                           replaced is reported and should be checked.
    `target='feasible'` -- analyse the indicator 1[MAC < carbon_price AND
                           net > 0]. This is bounded in [0, 1] by construction,
                           needs no penalty at all, and is the decision-relevant
                           quantity. Prefer it when the non-finite fraction is
                           material.

    With `return_diagnostics=True` returns (indices, diagnostics) where
    diagnostics carries the non-finite fraction, the output variance, and a
    `reliable` flag that is False when any ST exceeds 1 or too much of the
    sample had to be replaced.
    """
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze

    if target not in SOBOL_TARGETS:
        raise ValueError(f"target must be one of {SOBOL_TARGETS}, got '{target}'")

    names = list(bounds.keys())
    problem = {"num_vars": len(names), "names": names,
               "bounds": [list(bounds[k]) for k in names]}
    X = sobol_sample.sample(problem, n, calc_second_order=False)

    overrides = {name: X[:, j] for j, name in enumerate(names)}
    out = evaluate_array(base, overrides)
    mac = out["mac_usd_per_kg_co2"]
    net = out["net_abatement_kg_per_kg"]

    finite = np.isfinite(mac)
    nonfinite_fraction = float(1.0 - finite.mean())

    if target == "feasible":
        Y = ((mac < carbon_price_usd_per_kg) & finite & (net > 0)).astype(float)
        replaced = 0.0
    else:
        Y = np.array(mac, dtype=float)
        if finite.any() and not finite.all():
            cap = float(np.percentile(mac[finite], winsorise_q))
            Y = np.where(finite, np.minimum(Y, cap), cap)
            replaced = nonfinite_fraction
        elif not finite.any():
            raise ValueError(
                "every MAC draw is non-finite (net abatement <= 0 everywhere); "
                "no sensitivity can be computed. Fix the climate balance first, "
                "or use target='feasible'.")
        else:
            replaced = 0.0

    var_y = float(np.var(Y))
    if var_y <= 0:
        indices = {nm: {"S1": 0.0, "ST": 0.0} for nm in names}
        diag = {"nonfinite_fraction": nonfinite_fraction, "replaced_fraction": replaced,
                "output_variance": var_y, "target": target, "reliable": False,
                "reason": "the output is constant over the sampled ranges, so no "
                          "variance can be attributed"}
        return (indices, diag) if return_diagnostics else indices

    Si = sobol_analyze.analyze(problem, Y, calc_second_order=False,
                               print_to_console=False)
    indices = {names[i]: {"S1": float(Si["S1"][i]), "ST": float(Si["ST"][i])}
               for i in range(len(names))}

    max_st = max((v["ST"] for v in indices.values()), default=0.0)
    reasons = []
    if max_st > 1.0 + 1e-9:
        reasons.append(f"a total-order index came out at {max_st:.2f}; a variance "
                       f"fraction cannot exceed 1, so the estimate has not converged")
    if replaced > 0.05:
        reasons.append(f"{replaced:.0%} of draws had non-finite MAC and were "
                       f"winsorised; prefer target='feasible' here")
    diag = {"nonfinite_fraction": nonfinite_fraction, "replaced_fraction": replaced,
            "output_variance": var_y, "target": target, "max_ST": max_st,
            "reliable": not reasons, "reason": "; ".join(reasons)}
    return (indices, diag) if return_diagnostics else indices


def feasibility_envelope(base: Scenario,
                         x_field: str, x_values: np.ndarray,
                         y_field: str, y_values: np.ndarray,
                         carbon_price_usd_per_kg: float) -> Dict[str, np.ndarray]:
    """
    The keystone deliverable: the region of parameter space where a viable
    industry exists. Sweeps two levers (e.g. faradaic_efficiency x grid_intensity)
    and returns, on the grid, the MAC and a boolean feasibility mask
    (MAC < carbon_price AND climate-positive). Single vectorised pass.

    Returns dict with meshgrids X, Y, the MAC field, and the feasible mask.
    """
    X, Y = np.meshgrid(np.asarray(x_values, float), np.asarray(y_values, float))
    out = evaluate_array(base, {x_field: X.ravel(), y_field: Y.ravel()})
    mac = out["mac_usd_per_kg_co2"].reshape(X.shape)
    feasible = np.isfinite(mac) & (mac < carbon_price_usd_per_kg)
    return {"X": X, "Y": Y, "mac": mac, "feasible": feasible,
            "x_field": x_field, "y_field": y_field}
