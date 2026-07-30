"""
Active-learning layer -- the original, defensible contribution.

The terminal decision is binary and bounded: "does this catalyst achieve
MAC < carbon_price?". The value of computing/measuring a candidate is its
Expected Value Of Information (EVOI) toward THAT decision.

Why not Expected Improvement on the raw MAC? Because MAC = (LCOP-LCOP0)/net is an
unbounded, heavy-tailed transform (1/FE; division by net->0). Gaussian EI on MAC
over-rewards candidates whose only virtue is enormous predictive variance (low-FE
materials whose MAC distribution has a fat tail). The decision-relevant, bounded
quantity is the feasibility probability
        p_feas = P(MAC < carbon_price),
estimated from the candidate's propagated MAC sample. We define the acquisition
on p_feas.

Acquisition (documented VOI proxy):
    a(x) = H(p_feas) * (0.5 + p_feas),     H(p) = -p ln p - (1-p) ln(1-p)
  * H(p_feas) -> 0 for already-decided candidates (p~0 hopeless, p~1 certain).
  * (0.5 + p_feas) tilts attention toward *promising* uncertain candidates.
Bounded, free of the heavy-tail pathology. With INDEPENDENT beliefs it ranks
"which feasibility am I most unsure about, among the promising ones". With a
CORRELATED surrogate (GP/BNN) the principled generalisation is the Knowledge
Gradient on the feasibility decision (also credits learning about neighbours).
Classic EI is kept for well-behaved sub-objectives (e.g. LCOP).

Pipeline per candidate:
    descriptors --surrogate--> FE (mean,std) --MC through TEA/LCA--> MAC sample
              --> p_feas, mac_median, acquisition -> rank.
Only the top-ranked candidate(s) go to DFT/experiment.
"""
from __future__ import annotations
from typing import List, Dict
import numpy as np
from scipy.stats import norm

from .schema import Candidate
from .surrogate import BayesianLinearSurrogate
from .techno_economic import Scenario, evaluate_array


def candidate_mac_sample(target_mean: float, target_std: float, base: Scenario,
                         target_field: str = "faradaic_efficiency",
                         clip: tuple = (1e-3, 1.0),
                         n: int = 4000, seed: int = 0) -> np.ndarray:
    """Propagate one candidate's surrogate prediction (for `target_field`) through
    TEA/LCA to a MAC sample ($/kg CO2). Vectorised. +inf draws (net<=0) are kept so
    they correctly count as 'not feasible' in p_feas.

    `target_field` is the Scenario field the surrogate predicts (e.g.
    'faradaic_efficiency' for a selectivity surrogate, or 'cell_voltage' for the
    physics activity proxy). `clip` bounds the sampled values physically."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(target_mean, max(target_std, 1e-6), n)
    lo, hi = clip
    samples = np.clip(samples, lo, hi)
    return evaluate_array(base, {target_field: samples})["mac_usd_per_kg_co2"]


def _bernoulli_entropy(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def expected_improvement(obj_mean: float, obj_std: float, best_so_far: float) -> float:
    """Classic EI for a MINIMISATION objective. Use only on well-behaved,
    light-tailed objectives (e.g. LCOP), NOT on raw MAC -- see module docstring."""
    if obj_std <= 1e-12:
        return max(best_so_far - obj_mean, 0.0)
    z = (best_so_far - obj_mean) / obj_std
    return (best_so_far - obj_mean) * norm.cdf(z) + obj_std * norm.pdf(z)


def rank_candidates(candidates: List[Candidate],
                    surrogate: BayesianLinearSurrogate,
                    descriptor_keys: List[str],
                    base: Scenario,
                    carbon_price_usd_per_kg: float,
                    target_field: str = "faradaic_efficiency",
                    clip: tuple = (1e-3, 1.0),
                    n_mc: int = 4000,
                    seed: int = 0) -> List[Dict]:
    """Rank candidates by EVOI (feasibility-information acquisition) toward the
    MAC<carbon_price decision. Top entry = next thing worth a DFT run.
    `target_field` is the Scenario field the surrogate predicts (e.g.
    'faradaic_efficiency' or 'cell_voltage')."""
    X = np.array([[c.descriptors[k] for k in descriptor_keys] for c in candidates])
    pred_mean, pred_std = surrogate.predict(X)

    rows = []
    for c, pm, ps in zip(candidates, pred_mean, pred_std):
        macs = candidate_mac_sample(float(pm), float(ps), base, target_field=target_field,
                                    clip=clip, n=n_mc, seed=seed)
        finite = np.isfinite(macs)
        p_feas = float(np.mean(macs < carbon_price_usd_per_kg))   # inf -> infeasible
        mac_median = float(np.median(macs[finite])) if finite.any() else np.inf
        acq = _bernoulli_entropy(p_feas) * (0.5 + p_feas)
        rows.append({"material_id": c.material_id,
                     "pred_mean": float(pm), "pred_std": float(ps),
                     "target_field": target_field,
                     "mac_median": mac_median, "p_feas": p_feas,
                     "acquisition": float(acq)})
    return sorted(rows, key=lambda r: r["acquisition"], reverse=True)
