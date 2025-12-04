#!/usr/bin/env python3
"""
Standalone parameter update step for MaxEnt IC loop.

This version only computes observables and loss (creates state.json) without
updating parameters or submitting the next iteration. Used for standalone simulations.
"""

import argparse, os, json
from pathlib import Path
import numpy as np
from chunkchromatin.maxent_IC_loop.bin.utils import load_config, write_json, format_iter

def load_phi(iter_dir: Path):
    obs = iter_dir / "obs"
    phi_mean = np.load(obs / "phi_mean.npy")  # required
    cov_diag_path = obs / "phi_cov_diag.npy"
    if cov_diag_path.exists():
        phi_cov_diag = np.load(cov_diag_path)
    else:
        # Fallback: identity (uninformative)
        phi_cov_diag = np.ones_like(phi_mean) * np.nan
    return phi_mean, phi_cov_diag

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--iter", required=True, type=int)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    cfg = load_config(Path(args.config))
    it = args.iter
    iterd = run_root / format_iter(it)
    dmax = cfg["ideal_chromosome"]["d_end"] - cfg["ideal_chromosome"]["d_init"]

    # Load the lambda_IC_next.npy produced by the Newton update
    obs_dir = iterd / "obs"
    lambda_next_path = obs_dir / "lambda_IC_next.npy"
    
    if not lambda_next_path.exists():
        raise FileNotFoundError(f"lambda_IC_next.npy not found at {lambda_next_path}. "
                              "The Newton update step must complete before this update step.")
    
    lambda_next = np.load(lambda_next_path)
    
    # Load current lambda_IC for comparison/state tracking
    lambda_current = np.load(iterd / "params" / "lambda_IC.npy")
    if lambda_current.shape[0] != dmax:
        raise ValueError(f"lambda_current shape {lambda_current.shape} != ({dmax},)")

    # Load targets and simulated means for diagnostics/state tracking
    targets_path = run_root / "exp_targets" / "phi_exp_IC.npy"
    if not targets_path.exists():
        raise FileNotFoundError(f"Experimental targets not found: {targets_path}")
    
    phi_exp = np.load(targets_path)
    if phi_exp.shape[0] != dmax:
        raise ValueError(f"phi_exp shape {phi_exp.shape} != ({dmax},)")
    
    phi, cov_diag = load_phi(iterd)
    if phi.shape[0] != dmax:
        raise ValueError(f"phi shape {phi.shape} != ({dmax},)")

    # Gradient: g = phi_exp - <phi>_sim (for diagnostics)
    g = phi_exp - phi
    
    # Compute delta for state tracking
    delta = lambda_next - lambda_current

    # Compute relative error: sum(|gradients|) / sum(|experimental|) * 100%
    sum_abs_grad = np.sum(np.abs(g))
    sum_abs_exp = np.sum(np.abs(phi_exp))
    relative_error_pct = (sum_abs_grad / sum_abs_exp * 100.0) if sum_abs_exp > 0 else float('inf')

    # Save update artifacts
    upd = iterd / "update"
    np.save(upd / "grad.npy", g)
    np.save(upd / "delta_vec.npy", delta)
    np.save(upd / "lambda_vec.npy", lambda_next)
    
    state = {
        "iteration": it,
        "eta": "N/A (using Newton update)",
        "lambda_vec": lambda_next.tolist(),
        "grad_vec": g.tolist(),
        "precond_diag": "N/A (using Newton update)",
        "max_abs_residual": float(np.max(np.abs(g))),
        "l2_residual": float(np.linalg.norm(g)),
        "relative_error_pct": float(relative_error_pct),
        "max_param_step": float(np.max(np.abs(delta))),
        "update_method": "Newton (from process_IC_update.py)",
        "standalone": True,
    }
    (upd / "state.json").write_text(json.dumps(state, indent=2))

    print(f"[update:standalone] Created state.json for iteration {it}")
    print(f"  max_abs_residual: {state['max_abs_residual']:.6f}")
    print(f"  l2_residual: {state['l2_residual']:.6f}")
    print(f"  relative_error_pct: {state['relative_error_pct']:.2f}%")
    print(f"  max_param_step: {state['max_param_step']:.6f}")
    print(f"[update:standalone] Standalone mode - not updating parameters or submitting next iteration")

if __name__ == "__main__":
    main()

