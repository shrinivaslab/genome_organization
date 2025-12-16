#!/usr/bin/env python3
"""
Standalone parameter update step for combined MaxEnt TKL + IC loop.

This script mirrors `update_step_tkl_IC.py` but:
  * runs a single update for a given iteration,
  * computes diagnostics and writes `state.json`,
  * DOES NOT create the next iteration directory,
  * DOES NOT submit another iteration driver.

Use this from `iteration_driver_tkl_IC_standalone.py` when you only want to
compute observables and a single-step state summary for inspection.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from chunkchromatin.maxent_tkl_IC_loop.bin.utils import (
    ensure_dir,
    load_config,
    vectorize_upper_tri,
    write_json,
    format_iter,
)


def load_phi(iter_dir: Path, observable_type: str):
    """
    Load phi_mean and phi_cov_diag for either TKL or IC observable.

    Args:
        iter_dir: Iteration directory
        observable_type: Either "tkl" or "IC"
    """
    obs = iter_dir / "obs"
    if observable_type == "tkl":
        phi_mean_path = obs / "phi_mean.npy"
        cov_diag_path = obs / "phi_cov_diag.npy"
    elif observable_type == "IC":
        phi_mean_path = obs / "phi_mean_IC.npy"
        cov_diag_path = obs / "phi_cov_diag_IC.npy"
    else:
        raise ValueError(f"Unknown observable_type: {observable_type}")

    if not phi_mean_path.exists():
        raise FileNotFoundError(f"phi_mean file not found: {phi_mean_path}")

    phi_mean = np.load(phi_mean_path)

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
    ensure_dir(iterd / "update")

    K = cfg["simulation"]["n_types"]
    d_init = cfg["ideal_chromosome"]["d_init"]
    d_end = cfg["ideal_chromosome"]["d_end"]
    dmax = d_end - d_init
    method = cfg.get("update", {}).get("method", "newton")
    method_str = method.lower()
    adam_cfg = cfg.get("update", {}).get("adam", {}) if method_str == "adam" else {}

    obs_dir = iterd / "obs"

    # ============================================================
    # Process EPSILON (TKL) update
    # ============================================================
    epsilon_next_path = obs_dir / "epsilon_next.npy"
    if not epsilon_next_path.exists():
        raise FileNotFoundError(
            f"epsilon_next.npy not found at {epsilon_next_path}. "
            "The Newton/Adam update step (process_tkl_IC_update.py --mode reduce) "
            "must complete before this standalone update step."
        )

    eps_next = np.load(epsilon_next_path)

    # Load current epsilon for comparison
    eps_mat = np.load(iterd / "params" / "epsilon.npy")
    lam_tkl = vectorize_upper_tri(eps_mat)
    lam_next_tkl = vectorize_upper_tri(eps_next)

    # Load TKL targets and simulated means for diagnostics
    targets_path_tkl = run_root / "exp_targets" / "T_type_kl.npy"
    if not targets_path_tkl.exists():
        raise FileNotFoundError(f"Experimental TKL targets not found: {targets_path_tkl}")

    T_full = np.load(targets_path_tkl)
    # Convert T to vector form if it's a matrix
    if T_full.ndim == 2:
        # Use the same convention as in the full update_step_tkl_IC
        from chunkchromatin.maxent_tkl_IC_loop.bin.utils import vectorize_upper_tri as _vec

        T = _vec(T_full)
    else:
        T = T_full

    phi_tkl, cov_diag_tkl = load_phi(iterd, "tkl")
    if phi_tkl.shape[0] != T.shape[0]:
        raise ValueError(f"phi_tkl shape {phi_tkl.shape} != T shape {T.shape}")

    # Gradient: g = T - <phi>_sim
    g_tkl = T - phi_tkl

    # Compute delta for state tracking
    delta_tkl = lam_next_tkl - lam_tkl

    # Compute relative error
    sum_abs_grad_tkl = np.sum(np.abs(g_tkl))
    sum_abs_exp_tkl = np.sum(np.abs(T))
    relative_error_pct_tkl = (
        (sum_abs_grad_tkl / sum_abs_exp_tkl * 100.0) if sum_abs_exp_tkl > 0 else float("inf")
    )

    # ============================================================
    # Process LAMBDA_IC update
    # ============================================================
    lambda_next_path = obs_dir / "lambda_IC_next.npy"
    if not lambda_next_path.exists():
        raise FileNotFoundError(
            f"lambda_IC_next.npy not found at {lambda_next_path}. "
            "The Newton/Adam update step (process_tkl_IC_update.py --mode reduce) "
            "must complete before this standalone update step."
        )

    lambda_next = np.load(lambda_next_path)

    # Load current lambda_IC for comparison
    lambda_current = np.load(iterd / "params" / "lambda_IC.npy")
    if lambda_current.shape[0] != dmax:
        raise ValueError(f"lambda_current shape {lambda_current.shape} != ({dmax},)")
    if lambda_next.shape[0] != dmax:
        raise ValueError(f"lambda_next shape {lambda_next.shape} != ({dmax},)")

    # Load IC targets and simulated means for diagnostics
    targets_path_ic = run_root / "exp_targets" / "phi_exp_IC.npy"
    if not targets_path_ic.exists():
        raise FileNotFoundError(f"Experimental IC targets not found: {targets_path_ic}")

    phi_exp = np.load(targets_path_ic)
    if phi_exp.shape[0] != dmax:
        raise ValueError(f"phi_exp shape {phi_exp.shape} != ({dmax},)")

    phi_ic, cov_diag_ic = load_phi(iterd, "IC")
    if phi_ic.shape[0] != dmax:
        raise ValueError(f"phi_ic shape {phi_ic.shape} != ({dmax},)")

    # Gradient: g = phi_exp - <phi>_sim
    g_ic = phi_exp - phi_ic

    # Compute delta for state tracking
    delta_ic = lambda_next - lambda_current

    # Compute relative error
    sum_abs_grad_ic = np.sum(np.abs(g_ic))
    sum_abs_exp_ic = np.sum(np.abs(phi_exp))
    relative_error_pct_ic = (
        (sum_abs_grad_ic / sum_abs_exp_ic * 100.0) if sum_abs_exp_ic > 0 else float("inf")
    )

    # ============================================================
    # Save update artifacts and diagnostics
    # ============================================================
    upd = iterd / "update"

    # Save TKL artifacts
    np.save(upd / "grad_tkl.npy", g_tkl)
    np.save(upd / "delta_vec_tkl.npy", delta_tkl)
    np.save(upd / "lambda_vec_tkl.npy", lam_next_tkl)

    # Save IC artifacts
    np.save(upd / "grad_ic.npy", g_ic)
    np.save(upd / "delta_vec_ic.npy", delta_ic)
    np.save(upd / "lambda_vec_ic.npy", lambda_next)

    # Combined state tracking
    update_method = "Adam" if method_str == "adam" else "Newton"
    eta_tkl = (
        adam_cfg.get("learning_rate")
        if method_str == "adam"
        else "N/A (using Newton update)"
    )
    eta_ic = (
        adam_cfg.get("learning_rate_ic", adam_cfg.get("learning_rate"))
        if method_str == "adam"
        else "N/A (using Newton update)"
    )

    state = {
        "iteration": it,
        "epsilon": {
            "eta": eta_tkl,
            "lambda_vec": lam_next_tkl.tolist(),
            "grad_vec": g_tkl.tolist(),
            "precond_diag": "N/A (Adam first/second moments)"
            if method_str == "adam"
            else "N/A (using Newton update)",
            "max_abs_residual": float(np.max(np.abs(g_tkl))),
            "l2_residual": float(np.linalg.norm(g_tkl)),
            "relative_error_pct": float(relative_error_pct_tkl),
            "max_param_step": float(np.max(np.abs(delta_tkl))),
            "update_method": f"{update_method} (from process_tkl_IC_update.py)",
        },
        "lambda_IC": {
            "eta": eta_ic,
            "lambda_vec": lambda_next.tolist(),
            "grad_vec": g_ic.tolist(),
            "precond_diag": "N/A (Adam first/second moments)"
            if method_str == "adam"
            else "N/A (using Newton update)",
            "max_abs_residual": float(np.max(np.abs(g_ic))),
            "l2_residual": float(np.linalg.norm(g_ic)),
            "relative_error_pct": float(relative_error_pct_ic),
            "max_param_step": float(np.max(np.abs(delta_ic))),
            "update_method": f"{update_method} (from process_tkl_IC_update.py)",
        },
    }
    (upd / "state.json").write_text(json.dumps(state, indent=2))

    # Optionally, also write a minimal summary for convenience
    summary = {
        "iteration": it,
        "epsilon": {
            "max_abs_residual": state["epsilon"]["max_abs_residual"],
            "l2_residual": state["epsilon"]["l2_residual"],
            "max_param_step": state["epsilon"]["max_param_step"],
            "relative_error_pct": state["epsilon"]["relative_error_pct"],
        },
        "lambda_IC": {
            "max_abs_residual": state["lambda_IC"]["max_abs_residual"],
            "l2_residual": state["lambda_IC"]["l2_residual"],
            "max_param_step": state["lambda_IC"]["max_param_step"],
            "relative_error_pct": state["lambda_IC"]["relative_error_pct"],
        },
    }
    write_json(run_root / "standalone_update_summary.json", summary)


if __name__ == "__main__":
    main()


