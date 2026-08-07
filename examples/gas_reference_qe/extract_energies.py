#!/usr/bin/env python3
"""
Pull the final total energies out of Quantum ESPRESSO outputs and convert to eV.

QE reports total energies in Rydberg; co2dash expects eV. The conversion is a
factor of 13.6057, and forgetting it does not produce an error -- it produces a
plausible-looking U_L that is wrong by a factor of two. Hence this script rather
than a grep.

Usage:
    python extract_energies.py [directory]
"""
from __future__ import annotations

import os
import re
import sys

RY_TO_EV = 13.6056931230
RY_BOHR_TO_EV_ANG = RY_TO_EV / 0.529177210903     # 1 Ry/au = 25.711 eV/Angstrom

# QE marks converged total energies with a leading '!'
_TOTAL = re.compile(r"^!\s+total energy\s*=\s*(-?\d+\.\d+)\s*Ry", re.M)
_JOB_DONE = re.compile(r"JOB DONE", re.M)
_UNFINISHED = re.compile(r"convergence NOT achieved", re.M)
_BFGS_OK = re.compile(r"bfgs converged in\s+(\d+) scf cycles and\s+(\d+) bfgs steps", re.M)
_FORCE = re.compile(r"Total force\s*=\s*(\d+\.\d+)\s+Total SCF correction\s*=\s*(\d+\.\d+)", re.M)

SPECIES = {"co2": "CO2", "h2": "H2", "h2o": "H2O", "co": "CO"}


def read_energy(path: str):
    """Final converged total energy in eV, or (None, reason)."""
    with open(path, "r", errors="replace") as fh:
        text = fh.read()
    if _UNFINISHED.search(text):
        return None, "electronic convergence NOT achieved"
    hits = _TOTAL.findall(text)
    if not hits:
        return None, "no '! total energy' line found (did the run start?)"
    if not _JOB_DONE.search(text):
        return None, f"found {len(hits)} energies but no 'JOB DONE' -- run incomplete"
    return float(hits[-1]) * RY_TO_EV, ""


def read_convergence(path: str) -> dict:
    """Geometry- and force-convergence diagnostics for one run.

    Two independent things must hold, and only one of them is obvious:

      * the residual force must be below the threshold -- otherwise the energy
        belongs to an unrelaxed structure;
      * the 'Total SCF correction' must be MUCH smaller than the total force.
        It is the estimated error in the force from incomplete SCF convergence.
        If the two are comparable, the forces are numerical noise and BFGS has
        been optimising against noise -- a run can 'converge' this way and still
        be meaningless. Tighten conv_thr, not forc_conv_thr.
    """
    with open(path, "r", errors="replace") as fh:
        text = fh.read()
    forces = _FORCE.findall(text)
    bfgs = _BFGS_OK.search(text)
    out = {
        "bfgs_converged": bool(bfgs),
        "scf_cycles": int(bfgs.group(1)) if bfgs else None,
        "bfgs_steps": int(bfgs.group(2)) if bfgs else None,
        "n_force_evals": len(forces),
        "total_force_ry_au": float(forces[-1][0]) if forces else None,
        "scf_correction_ry_au": float(forces[-1][1]) if forces else None,
    }
    if out["total_force_ry_au"] is not None:
        out["total_force_ev_ang"] = out["total_force_ry_au"] * RY_BOHR_TO_EV_ANG
        corr, f = out["scf_correction_ry_au"], out["total_force_ry_au"]
        out["force_is_resolved"] = (f <= 0.0) or (corr < 0.1 * max(f, 1e-12))
    return out


def main(directory: str = ".") -> int:
    found, problems = {}, []
    for stem, name in SPECIES.items():
        path = os.path.join(directory, f"{stem}.out")
        if not os.path.exists(path):
            problems.append(f"{name}: {stem}.out not found")
            continue
        energy, why = read_energy(path)
        if energy is None:
            problems.append(f"{name}: {why}")
        else:
            found[name] = energy

    print(f"{'species':8} {'energy (eV)':>15} {'force (eV/A)':>13} {'bfgs':>6}  notes")
    for name in ("CO2", "H2", "H2O", "CO"):
        if name not in found:
            continue
        stem = [k for k, v in SPECIES.items() if v == name][0]
        c = read_convergence(os.path.join(directory, f"{stem}.out"))
        f_ev = c.get("total_force_ev_ang")
        notes = []
        if not c["bfgs_converged"]:
            notes.append("BFGS did NOT converge")
        if c.get("force_is_resolved") is False:
            notes.append(f"SCF correction ({c['scf_correction_ry_au']:.2e}) is not "
                         f"small vs the force ({c['total_force_ry_au']:.2e}) -- "
                         f"forces are numerical noise; tighten conv_thr")
        f_txt = f"{f_ev:13.5f}" if f_ev is not None else f"{'n/a':>13}"
        bfgs_txt = "yes" if c["bfgs_converged"] else "NO"
        print(f"{name:8} {found[name]:15.6f} {f_txt} {bfgs_txt:>6}  {'; '.join(notes)}")
    for p in problems:
        print(f"  !! {p}")

    required = ("CO2", "H2", "H2O")
    if all(k in found for k in required):
        print("\nPaste into the app (reference frame = 'absolute'), or in code:\n")
        print("    from co2dash.chain import ReferenceFrame")
        print("    frame = ReferenceFrame(mode='absolute', gas_energies={")
        for k in required:
            print(f"        {k!r}: {found[k]:.6f},")
        print("    })")

    # sanity check: CO2 + H2 -> CO + H2O should be endothermic by ~0.3-0.5 eV (PBE)
    if all(k in found for k in ("CO2", "H2", "CO", "H2O")):
        dE = (found["CO"] + found["H2O"]) - (found["CO2"] + found["H2"])
        verdict = "plausible" if 0.1 <= dE <= 0.8 else "SUSPECT"
        print(f"\nSanity check  CO2 + H2 -> CO + H2O : {dE:+.3f} eV  [{verdict}]")
        if verdict == "SUSPECT":
            print("  Expected roughly +0.3 to +0.5 eV with PBE. A large deviation "
                  "usually means mismatched pseudopotentials or an unconverged "
                  "cutoff. Fix that before using these references.")
    else:
        print("\n(run co.in as well to enable the CO2 + H2 -> CO + H2O sanity check)")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
