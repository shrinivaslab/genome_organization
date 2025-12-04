#!/usr/bin/env python3
"""
Parameter update step for combined MaxEnt TKL and IC loop.

The Newton update from process_tkl_IC_update.py produces both epsilon_next.npy 
and lambda_IC_next.npy. This script handles copying both to the next iteration 
and provides diagnostics/convergence tracking for both observables.
"""

import argparse, os, json, shutil, math, subprocess
from pathlib import Path
import numpy as np
from chunkchromatin.maxent_tkl_IC_loop.bin.utils import ensure_dir, load_config, vectorize_upper_tri, devectorize_upper_tri, write_json, format_iter, delete_dir_if_exists

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
        raise FileNotFoundError(f"epsilon_next.npy not found at {epsilon_next_path}. "
                              "The Newton update step must complete before this update step.")
    
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
        T = vectorize_upper_tri(T_full)
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
    relative_error_pct_tkl = (sum_abs_grad_tkl / sum_abs_exp_tkl * 100.0) if sum_abs_exp_tkl > 0 else float('inf')

    # ============================================================
    # Process LAMBDA_IC update
    # ============================================================
    lambda_next_path = obs_dir / "lambda_IC_next.npy"
    if not lambda_next_path.exists():
        raise FileNotFoundError(f"lambda_IC_next.npy not found at {lambda_next_path}. "
                              "The Newton update step must complete before this update step.")
    
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
    relative_error_pct_ic = (sum_abs_grad_ic / sum_abs_exp_ic * 100.0) if sum_abs_exp_ic > 0 else float('inf')

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
    # Record which optimizer produced these updates; previous code always wrote "Newton"
    update_method = "Adam" if method_str == "adam" else "Newton"
    eta_tkl = adam_cfg.get("learning_rate") if method_str == "adam" else "N/A (using Newton update)"
    eta_ic = adam_cfg.get("learning_rate_ic", adam_cfg.get("learning_rate")) if method_str == "adam" else "N/A (using Newton update)"

    state = {
        "iteration": it,
        "epsilon": {
            "eta": eta_tkl,
            "lambda_vec": lam_next_tkl.tolist(),
            "grad_vec": g_tkl.tolist(),
            "precond_diag": "N/A (Adam first/second moments)" if method_str == "adam" else "N/A (using Newton update)",
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
            "precond_diag": "N/A (Adam first/second moments)" if method_str == "adam" else "N/A (using Newton update)",
            "max_abs_residual": float(np.max(np.abs(g_ic))),
            "l2_residual": float(np.linalg.norm(g_ic)),
            "relative_error_pct": float(relative_error_pct_ic),
            "max_param_step": float(np.max(np.abs(delta_ic))),
            "update_method": f"{update_method} (from process_tkl_IC_update.py)",
        },
    }
    (upd / "state.json").write_text(json.dumps(state, indent=2))

    # ============================================================
    # Convergence check
    # ============================================================
    conv_cfg = cfg["convergence"]
    
    # Check convergence for epsilon
    residual_ok_tkl = (state["epsilon"]["max_abs_residual"] <= conv_cfg["max_abs_residual"]) and \
                      (state["epsilon"]["l2_residual"] <= conv_cfg["l2_residual"])
    step_ok_tkl = (state["epsilon"]["max_param_step"] <= conv_cfg["max_param_step"])
    converged_tkl = bool(residual_ok_tkl or step_ok_tkl)
    
    # Check convergence for lambda_IC
    residual_ok_ic = (state["lambda_IC"]["max_abs_residual"] <= conv_cfg["max_abs_residual"]) and \
                     (state["lambda_IC"]["l2_residual"] <= conv_cfg["l2_residual"])
    step_ok_ic = (state["lambda_IC"]["max_param_step"] <= conv_cfg["max_param_step"])
    converged_ic = bool(residual_ok_ic or step_ok_ic)
    
    # Combined convergence: both must converge
    converged = converged_tkl and converged_ic

    # Track consecutive passes
    track_file = run_root / "convergence_track.json"
    if track_file.exists():
        track = json.loads(track_file.read_text())
    else:
        track = {"streak": 0, "last_iter": -1, "streak_tkl": 0, "streak_ic": 0}
    
    # Update streaks for each observable separately
    if converged_tkl:
        streak_tkl = (track.get("streak_tkl", 0) + 1) if (track.get("last_iter", -1) == it-1) else 1
    else:
        streak_tkl = 0
    
    if converged_ic:
        streak_ic = (track.get("streak_ic", 0) + 1) if (track.get("last_iter", -1) == it-1) else 1
    else:
        streak_ic = 0
    
    # Combined streak: increment only if both converged
    if converged:
        streak = (track.get("streak", 0) + 1) if (track.get("last_iter", -1) == it-1) else 1
    else:
        streak = 0
    
    track = {
        "streak": streak,
        "streak_tkl": streak_tkl,
        "streak_ic": streak_ic,
        "last_iter": it,
        "converged_tkl": converged_tkl,
        "converged_ic": converged_ic,
    }
    track_file.write_text(json.dumps(track, indent=2))

    # ============================================================
    # Prepare next iteration folder
    # ============================================================
    it_next = it + 1
    iterd_next = run_root / format_iter(it_next)
    (iterd_next / "params").mkdir(parents=True, exist_ok=True)
    (iterd_next / "sims").mkdir(parents=True, exist_ok=True)
    (iterd_next / "obs").mkdir(parents=True, exist_ok=True)
    (iterd_next / "update").mkdir(parents=True, exist_ok=True)
    
    # Copy both parameters to next iteration
    np.save(iterd_next / "params" / "epsilon.npy", eps_next)
    np.save(iterd_next / "params" / "lambda_IC.npy", lambda_next)
    
    # Also ensure both are available for next iteration's Newton update
    next_epsilon_path = iterd_next / "update" / f"epsilon_tk_{it_next}.npy"
    np.save(next_epsilon_path, eps_next)
    
    next_lambda_path = iterd_next / "update" / f"lambda_IC_tk_{it_next}.npy"
    np.save(next_lambda_path, lambda_next)

    # Storage policy: delete frames from i-1 (now two iterations behind next run)
    i_minus_1 = it - 1
    if i_minus_1 >= 0:
        old_iter = run_root / format_iter(i_minus_1)
        old_sims = old_iter / "sims"
        if old_sims.exists():
            try:
                delete_dir_if_exists(old_sims)
                (old_iter / "sims").mkdir(exist_ok=True)  # leave empty dir for provenance
            except Exception as e:
                print(f"[warn] failed to delete old frames at {old_sims}: {e}")

    # ============================================================
    # Decide to continue
    # ============================================================
    done = (streak >= conv_cfg["consecutive"])
    summary = {
        "iteration": it,
        "converged_this_iter": converged,
        "converged_tkl": converged_tkl,
        "converged_ic": converged_ic,
        "consecutive_streak": streak,
        "consecutive_streak_tkl": streak_tkl,
        "consecutive_streak_ic": streak_ic,
        "done": done,
    }
    (run_root / "last_update_summary.json").write_text(json.dumps(summary, indent=2))

    if done:
        print(f"[update] Converged with combined streak {streak} (TKL: {streak_tkl}, IC: {streak_ic}). Stopping.")
        return

    # ============================================================
    # Submit next iteration driver
    # ============================================================
    driver = Path(__file__).resolve().parent / "iteration_driver_tkl_IC.py"
    cfg_path = Path(args.config).resolve()
    
    # Try to get name from run_manifest.json, fallback to generic name
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        try:
            name = json.loads(manifest_path.read_text())["name"]
        except (json.JSONDecodeError, KeyError):
            name = f"resumed_iter{it}"
            print(f"Warning: Could not read name from run_manifest.json, using: {name}")
    else:
        name = f"resumed_iter{it}"
        print(f"Warning: run_manifest.json not found, using generic name: {name}")

    proj_root = Path(__file__).resolve().parent.parent
    cmd = ["sbatch",
           "--job-name", f"{name}_iter{it_next:03d}_driver",
           "--account", cfg["slurm"]["account"],
           "--partition", cfg["slurm"]["partition"],
           "--time", "00:10:00",
           "--cpus-per-task", "1",
           "--mem", "1G",
           "--output", str((run_root / "logs" / f"driver_%j.out")),
           "--error",  str((run_root / "logs" / f"driver_%j.err")),
           str(driver),
           "--run-root", str(run_root),
           "--iter", str(it_next),
           "--config", str(cfg_path),
           "--name", name,
           "--proj-root", str(proj_root),
           ]
    # Optional constraints
    if cfg["slurm"].get("constraint"):
        cmd[1:1] = ["--constraint", cfg["slurm"]["constraint"]]
    if cfg["slurm"].get("qos"):
        cmd[1:1] = ["--qos", cfg["slurm"]["qos"]]

    print("[update] Submitting next iteration driver:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=run_root)

if __name__ == "__main__":
    main()
