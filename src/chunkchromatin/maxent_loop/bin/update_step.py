#!/usr/bin/env python3
"""
Parameter update step for MaxEnt loop.

The Newton update from process_tkl_update.py produces both epsilon_tk_*.npy 
files and epsilon_next.npy. This script handles copying epsilon_next to the 
next iteration and provides diagnostics/convergence tracking, rather than 
computing its own update.
"""

import argparse, os, json, shutil, math, subprocess
from pathlib import Path
import numpy as np
from chunkchromatin.maxent_loop.bin.utils import ensure_dir, load_config, vectorize_upper_tri, devectorize_upper_tri, write_json, format_iter, delete_dir_if_exists

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

def bb_eta(prev_eta, s, y, eta_min, eta_max, clip_frac):
    # Barzilai–Borwein step size
    denom = float(np.dot(y, y))
    if denom <= 0 or not np.isfinite(denom):
        return prev_eta
    eta = float(np.dot(s, y) / denom)
    # Clip aggressively to avoid pathologies
    lo = max(eta_min, (1.0 - clip_frac) * prev_eta)
    hi = min(eta_max, (1.0 + clip_frac) * prev_eta)
    return float(np.clip(eta, lo, hi))

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

    # Load the epsilon_next.npy produced by the Newton update
    obs_dir = iterd / "obs"
    epsilon_next_path = obs_dir / "epsilon_next.npy"
    
    if not epsilon_next_path.exists():
        raise FileNotFoundError(f"epsilon_next.npy not found at {epsilon_next_path}. "
                              "The Newton update step must complete before this update step.")
    
    eps_next = np.load(epsilon_next_path)
    
    # Load current epsilon for comparison/state tracking
    eps_mat = np.load(iterd / "params" / "epsilon.npy")
    lam = vectorize_upper_tri(eps_mat)
    lam_next = vectorize_upper_tri(eps_next)

    # Load targets and simulated means for diagnostics/state tracking
    targets_path = run_root / "exp_targets" / "T_type_kl.npy"
    if not targets_path.exists():
        raise FileNotFoundError(f"Experimental targets not found: {targets_path}")
    
    T_full = np.load(targets_path)
    # Convert T to vector form if it's a matrix
    if T_full.ndim == 2:
        T = vectorize_upper_tri(T_full)
    else:
        T = T_full
    
    phi, cov_diag = load_phi(iterd)

    # Gradient: g = T - <phi>_sim (for diagnostics)
    g = T - phi
    
    # Compute delta for state tracking
    delta = lam_next - lam

    # Save update artifacts (now mainly for diagnostics and state tracking)
    upd = iterd / "update"
    np.save(upd / "grad.npy", g)
    np.save(upd / "delta_vec.npy", delta)
    np.save(upd / "lambda_vec.npy", lam_next)
    # Note: epsilon_next.npy is already created by the Newton update
    state = {
        "iteration": it,
        "eta": "N/A (using Newton update)",
        "lambda_vec": lam_next.tolist(),
        "grad_vec": g.tolist(),
        "precond_diag": "N/A (using Newton update)",
        "max_abs_residual": float(np.max(np.abs(g))),
        "l2_residual": float(np.linalg.norm(g)),
        "max_param_step": float(np.max(np.abs(delta))),
        "update_method": "Newton (from process_tkl_update.py)",
    }
    (upd / "state.json").write_text(json.dumps(state, indent=2))

    # Convergence check
    conv_cfg = cfg["convergence"]
    residual_ok = (state["max_abs_residual"] <= conv_cfg["max_abs_residual"]) and (state["l2_residual"] <= conv_cfg["l2_residual"])
    step_ok = (state["max_param_step"] <= conv_cfg["max_param_step"])
    converged = bool(residual_ok or step_ok)

    # Track consecutive passes
    track_file = run_root / "convergence_track.json"
    if track_file.exists():
        track = json.loads(track_file.read_text())
    else:
        track = {"streak": 0, "last_iter": -1}
    if converged:
        streak = (track["streak"] + 1) if (track["last_iter"] == it-1) else 1
    else:
        streak = 0
    track = {"streak": streak, "last_iter": it}
    track_file.write_text(json.dumps(track, indent=2))

    # Prepare next iteration folder and copy epsilon_next (which came from Newton update)
    it_next = it + 1
    iterd_next = run_root / format_iter(it_next)
    (iterd_next / "params").mkdir(parents=True, exist_ok=True)
    (iterd_next / "sims").mkdir(parents=True, exist_ok=True)
    (iterd_next / "obs").mkdir(parents=True, exist_ok=True)
    (iterd_next / "update").mkdir(parents=True, exist_ok=True)
    np.save(iterd_next / "params" / "epsilon.npy", eps_next)
    
    # Also ensure epsilon is available for next iteration's Newton update
    next_epsilon_path = iterd_next / "update" / f"epsilon_tk_{it_next}.npy"
    np.save(next_epsilon_path, eps_next)

    # Storage policy: delete frames from i-1 (now two iterations behind next run), keep frames for current i
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

    # Decide to continue
    done = (streak >= conv_cfg["consecutive"])
    summary = {
        "iteration": it,
        "converged_this_iter": converged,
        "consecutive_streak": streak,
        "done": done,
    }
    (run_root / "last_update_summary.json").write_text(json.dumps(summary, indent=2))

    if done:
        print(f"[update] Converged with streak {streak}. Stopping.")
        return

    # Submit next iteration driver
    driver = Path(__file__).resolve().parent / "iteration_driver.py"
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
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
