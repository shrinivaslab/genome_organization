#!/usr/bin/env python3
import os
import re
import sys
import time
import json
import glob
import math
import socket
import struct
import argparse
import resource
import numpy as np
from pathlib import Path
from functools import partial
import multiprocessing as mp
from scipy.spatial import cKDTree
from scipy.linalg import cho_factor, cho_solve

# ==========================
# Constants / defaults
# ==========================
MU_DEFAULT   = 4.22
RC_DEFAULT   = 1.82
RCUT_DEFAULT = 3.0  
BETA_DEFAULT = 1.0
GAMMA        = 0.33    # damping factor for Newton step
LAMBDA_REG_SCALE = 1e-10

# ==========================
# Core kernels & helpers
# ==========================
def f_switch(r, mu=MU_DEFAULT, rc=RC_DEFAULT):
    # MiChroM "switch" (contact probability proxy)
    return 0.5 * (1.0 + np.tanh(mu * (rc - r)))

def _load_exp_Tkl(exp_Tkl_path, expected_K=None):
    if not os.path.exists(exp_Tkl_path):
        raise FileNotFoundError(exp_Tkl_path)
    if exp_Tkl_path.endswith(".npy"):
        T = np.load(exp_Tkl_path)
    elif exp_Tkl_path.endswith(".npz"):
        z = np.load(exp_Tkl_path)
        for k in ("Tkl_exp", "Tkl", "experimental_Tkl"):
            if k in z:
                T = z[k]; break
        else:
            T = z[sorted(z.files)[0]]
    else:
        raise ValueError("Experimental Tkl path must be .npy or .npz")

    T = np.asarray(T, float)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("Experimental Tkl must be square.")
    iu = np.triu_indices(T.shape[0], 1)
    T[(iu[1], iu[0])] = T[iu]  # reflect upper -> lower

    if expected_K is not None and T.shape[0] != expected_K:
        K = expected_K
        if T.shape[0] > K:
            T = T[:K, :K]
        else:
            pad = K - T.shape[0]
            T = np.pad(T, ((0, pad), (0, pad)), constant_values=0.0)
    return T

def _flatten_upper(M):
    iu = np.triu_indices(M.shape[0])
    return M[iu], iu

class UpperTriOnlineCov:
    """Online mean/covariance over the vectorized upper-tri entries (incl diag)."""
    def __init__(self, K):
        self.K = K
        self.iu = np.triu_indices(K)
        self.M = len(self.iu[0])
        self.n = 0
        self.mean = np.zeros(self.M, float)
        self.M2   = np.zeros((self.M, self.M), float)

    def add_frame_from_upper_mat(self, T_upper_only):
        v = T_upper_only[self.iu]
        self.n += 1
        if self.n == 1:
            self.mean[:] = v
        else:
            d = v - self.mean
            self.mean += d / self.n
            d2 = v - self.mean
            self.M2 += np.outer(d, d2)

    def finalize(self, beta=1.0):
        cov = np.zeros((self.M, self.M), float) if self.n < 2 else self.M2 / (self.n - 1)
        mean_upper = self.mean.copy()
        mean_T = np.zeros((self.K, self.K), float)
        mean_T[self.iu] = mean_upper
        mean_T = mean_T + mean_T.T - np.diag(np.diag(mean_T))
        hess = (beta**2) * cov
        return mean_T, cov, hess, self.iu

def _covariance_pass_upper(positions, monomer_types, mu=MU_DEFAULT, rc=RC_DEFAULT, rcut=None):
    F, N, _ = positions.shape
    type_labels, inv = np.unique(monomer_types, return_inverse=True)
    K = len(type_labels)
    acc = UpperTriOnlineCov(K)
    if rcut is None:
        rcut = rc + 4.0 / mu

    iuK = np.triu_indices(K)
    for f in range(F):
        X = positions[f]
        tree = cKDTree(X, leafsize=40)
        pairs = tree.query_pairs(rcut, output_type='ndarray')

        T_up = np.zeros((K, K), float)
        if pairs.size != 0:
            i = pairs[:, 0]; j = pairs[:, 1]
            rij = np.linalg.norm(X[i] - X[j], axis=1)
            fij = f_switch(rij, mu=mu, rc=rc)
            ti = inv[i]; tj = inv[j]
            k = np.minimum(ti, tj)
            l = np.maximum(ti, tj)
            flat = k * K + l
            sums = np.bincount(flat, weights=fij, minlength=K*K).reshape(K, K)
            T_up[iuK] = sums[iuK]
        acc.add_frame_from_upper_mat(T_up)

    return acc.finalize(beta=1.0) + (type_labels,)

def process_one_replicate(positions, monomer_types, exp_Tkl_path, mu=MU_DEFAULT, rc=RC_DEFAULT, rcut=RCUT_DEFAULT, beta=BETA_DEFAULT):
    Tkl_sim, Cov_upper, Hess_upper, iu, type_labels = _covariance_pass_upper(
        positions, monomer_types, mu=mu, rc=rc, rcut=rcut
    )
    K = len(type_labels)
    Tkl_exp = _load_exp_Tkl(exp_Tkl_path, expected_K=K)
    Delta = Tkl_exp - Tkl_sim
    delta_vec, _ = _flatten_upper(Delta)
    grad_vec = beta * delta_vec
    Hess_upper = (beta**2) * Cov_upper
    return {
        "type_labels": type_labels,
        "Tkl_sim": Tkl_sim,
        "Tkl_exp": Tkl_exp,
        "Delta": Delta,
        "upper_indices": iu,
        "grad_vec": grad_vec,
        "Hess_upper": Hess_upper,
        "Cov_upper": Cov_upper,
        "mu": mu, "rc": rc, "rcut": rcut if rcut is not None else rc + 4.0/mu
    }

def load_all_positions(filename):
    """
    Load all particle positions from a binary .traj file.
    Returns (n_frames, n_particles, 3) float32 -> float64
    """
    HEADER_FORMAT = "<4sBHII16s"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    with open(filename, 'rb') as f:
        header = f.read(HEADER_SIZE)
        magic, version, n_particles, frame_size, n_frames, _ = struct.unpack(HEADER_FORMAT, header)
        assert magic == b'CHRM'
        metadata_len = struct.unpack("<I", f.read(4))[0]
        f.seek(HEADER_SIZE + 4 + metadata_len)
        data = np.frombuffer(f.read(), dtype=np.float32)
        return data.reshape((n_frames, n_particles, 3)).astype(np.float64, copy=False)

# ==========================
# Per-replicate worker plumbing (multiprocessing)
# ==========================
_IO_SEMA = None
_K_IO = None

def _maybe_write_manifest_header(path):
    if not os.path.exists(path):
        with open(path, "w") as mf:
            mf.write(
                "timestamp\tk_io\thostname\tpid\trep\ttraj_size_mb\t"
                "read_s\tcompute_s\twrite_s\ttotal_s\tmax_rss_mb\tstatus\tmessage\n"
            )

def _rep_dir_and_path(replicate_root, rep_idx):
    rep_str = f"rep{rep_idx:02d}"
    rep_dir = os.path.join(replicate_root, rep_str)
    traj_path = os.path.join(rep_dir, "trajectory.traj")
    return rep_str, traj_path

def _process_replicate_entry(rep_idx, replicate_root, output_dir, monomer_types, exp_Tkl_path, mu, rc, rcut, beta, manifest_path):
    rep_str, traj_path = _rep_dir_and_path(replicate_root, rep_idx)
    out_npz   = os.path.join(output_dir, f"{rep_str}_upper_grad_hess.npz")
    out_touch = os.path.join(output_dir, f"{rep_str}.READY")

    if os.path.exists(out_npz):
        print(f"[SKIP] {rep_str} already processed.")
        return

    hostname = socket.gethostname()
    pid = os.getpid()

    if not os.path.exists(traj_path):
        with open(manifest_path, "a") as mf:
            mf.write(f"{time.time()}\t{_K_IO}\t{hostname}\t{pid}\t{rep_str}\t0\t0\t0\t0\t0\t0\tMISSING\t{traj_path}\n")
        print(f"[SKIP] Missing {traj_path}")
        return

    traj_size_mb = os.path.getsize(traj_path) / (1024*1024.0)
    t0 = time.time()
    read_s = compute_s = write_s = 0.0
    status = "OK"
    message = ""

    try:
        t_read0 = time.time()
        if _IO_SEMA is not None: _IO_SEMA.acquire()
        try:
            positions = load_all_positions(traj_path)
        finally:
            if _IO_SEMA is not None: _IO_SEMA.release()
        read_s = time.time() - t_read0

        t_comp0 = time.time()
        out = process_one_replicate(
            positions=positions,
            monomer_types=monomer_types,
            exp_Tkl_path=exp_Tkl_path,
            mu=mu, rc=rc, rcut=rcut, beta=beta
        )
        compute_s = time.time() - t_comp0

        t_wr0 = time.time()
        if _IO_SEMA is not None: _IO_SEMA.acquire()
        try:
            np.savez_compressed(
                out_npz,
                grad_vec=out["grad_vec"],
                Hess_upper=out["Hess_upper"],
                upper_indices_row=out["upper_indices"][0],
                upper_indices_col=out["upper_indices"][1],
                type_labels=out["type_labels"],
                mu=out["mu"], rc=out["rc"], rcut=out["rcut"],
                K=len(out["type_labels"]),
                rep=rep_idx,
            )
            Path(out_touch).write_text("ready\n")
        finally:
            if _IO_SEMA is not None: _IO_SEMA.release()
        write_s = time.time() - t_wr0

        usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        max_rss_mb = usage_kb / 1024.0
        total_s = time.time() - t0
        with open(manifest_path, "a") as mf:
            mf.write(
                f"{time.time()}\t{_K_IO}\t{hostname}\t{pid}\t{rep_str}\t{traj_size_mb:.2f}\t"
                f"{read_s:.3f}\t{compute_s:.3f}\t{write_s:.3f}\t{total_s:.3f}\t"
                f"{max_rss_mb:.2f}\t{status}\t{message}\n"
            )
        print(f"[DONE] {rep_str} | total {total_s:.1f}s | read {read_s:.1f}s | compute {compute_s:.1f}s | write {write_s:.1f}s")

    except Exception as e:
        status = "FAIL"
        message = str(e)
        try:
            usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            max_rss_mb = usage_kb / 1024.0
        except Exception:
            max_rss_mb = 0.0
        with open(manifest_path, "a") as mf:
            mf.write(
                f"{time.time()}\t{_K_IO}\t{hostname}\t{pid}\t{rep_str}\t{traj_size_mb:.2f}\t"
                f"{read_s:.3f}\t{compute_s:.3f}\t{write_s:.3f}\t{time.time()-t0:.3f}\t"
                f"{max_rss_mb:.2f}\tFAIL\t{message}\n"
            )
        print(f"[FAIL] {rep_str}: {e}")

# ==========================
# Reduce step: aggregate 50 reps and update epsilon
# ==========================
def _find_latest_epsilon(epsilon_dir: Path, stem: str = "epsilon_tk_", ext: str = ".npy") -> Path:
    patt = re.compile(rf"^{re.escape(stem)}(\d+){re.escape(ext)}$")
    latest = None
    max_n = -1
    for p in epsilon_dir.glob(f"{stem}*{ext}"):
        m = patt.match(p.name)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
                latest = p
    if latest is None:
        raise FileNotFoundError(f"No prior epsilon_tk_*.npy found in {epsilon_dir}")
    return latest

def _next_version_path(dirpath: Path, stem: str = "epsilon_tk_", ext: str = ".npy", iteration_idx: int = None) -> Path:
    """
    Generate the next epsilon filename. For iteration i, should produce epsilon_tk_{i+1}.npy.
    If iteration_idx is provided, uses that as the target. Otherwise, uses the legacy +1 logic.
    
    When using iteration_idx, will overwrite existing files to prevent version escalation
    during resume operations.
    """
    if iteration_idx is not None:
        # For iteration i, create epsilon_tk_{i+1}.npy
        target_n = iteration_idx + 1
        target_path = dirpath / f"{stem}{target_n}{ext}"
        
        # Clean up any higher-numbered epsilon files that shouldn't exist
        # (These can occur from repeated resume operations)
        patt = re.compile(rf"^{re.escape(stem)}(\d+){re.escape(ext)}$")
        for p in dirpath.glob(f"{stem}*{ext}"):
            m = patt.match(p.name)
            if m:
                n = int(m.group(1))
                if n > target_n:
                    print(f"[CLEANUP] Removing unexpected epsilon file: {p.name}")
                    p.unlink()
        
        return target_path
    else:
        # Legacy behavior: find max and add 1
        patt = re.compile(rf"^{re.escape(stem)}(\d+){re.escape(ext)}$")
        max_n = -1
        for p in dirpath.glob(f"{stem}*{ext}"):
            m = patt.match(p.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
        return dirpath / f"{stem}{max_n + 1}{ext}"

def reduce_and_update(output_dir, epsilon_dir, beta=BETA_DEFAULT, iteration_idx=None):
    """
    Read all repXX_upper_grad_hess.npz in output_dir, average grad & Hess,
    apply one damped Newton step, and save epsilon_tk_{n+1}.npy in epsilon_dir.
    
    Also saves the result as epsilon_next.npy for use by the subsequent 
    update_step.py in the pipeline.
    """
    files = sorted(glob.glob(os.path.join(output_dir, "rep??_upper_grad_hess.npz")))
    if len(files) == 0:
        raise RuntimeError(f"No per-replicate .npz found in {output_dir}")

    grads = []
    Hlist = []
    iu_row = iu_col = None
    K = None
    for fpath in files:
        z = np.load(fpath)
        grad_vec  = z["grad_vec"]
        Hess_up   = z["Hess_upper"]
        this_K    = int(z["K"])
        if K is None:
            K = this_K
        elif this_K != K:
            raise ValueError(f"Inconsistent K in {fpath}: {this_K} vs {K}")
        if iu_row is None:
            iu_row = z["upper_indices_row"]
            iu_col = z["upper_indices_col"]
        grads.append(grad_vec)
        Hlist.append(Hess_up)

    grads = np.stack(grads, axis=0)       # (R, M)
    Hlist = np.stack(Hlist, axis=0)       # (R, M, M)
    g_mean = grads.mean(axis=0)           # (M,)
    B_mean = Hlist.mean(axis=0)           # (M,M)

    # Spectral conditioning to target condition number
    M = B_mean.shape[0]
    w = np.linalg.eigvalsh(B_mean)
    lam_min, lam_max = w[0], w[-1]
    kappa_raw = lam_max / max(abs(lam_min), 1e-12)

    # Target condition number and floor for PSD
    kappa_target = 1e4
    eps_floor = 1e-5 * w.mean()

    # Compute required damping
    lam_psd = max(0.0, -lam_min + eps_floor)  # Ensure positive definite
    if kappa_raw > kappa_target:
        lam_kappa = max(0.0, (lam_max / kappa_target) - lam_min)
    else:
        lam_kappa = 0.0  # No conditioning needed

    lambda_reg = max(lam_psd, lam_kappa)
    B_reg = B_mean + lambda_reg * np.eye(M)

    # Solve Δλ = -γ * B^{-1} g using Cholesky for numerical stability
    try:
        cho_fac = cho_factor(B_reg)
        delta_vec = -cho_solve(cho_fac, g_mean)
    except np.linalg.LinAlgError:
        print("[WARNING] Cholesky failed, falling back to standard solve")
        delta_vec = -np.linalg.solve(B_reg, g_mean)
    
    # Parameter-dependent scaling: limit max change per parameter
    max_change_per_param = 0.5
    max_proposed_change = np.max(np.abs(delta_vec))
    if max_proposed_change > 0:
        adaptive_gamma = min(max_change_per_param / max_proposed_change, GAMMA)
    else:
        adaptive_gamma = GAMMA
    
    delta_vec *= adaptive_gamma
    
    # Calculate final condition number after regularization
    kappa_after = (lam_max + lambda_reg) / (abs(lam_min) + lambda_reg)
    
    print(f"[NEWTON] max_proposed_change: {max_proposed_change:.3f}")
    print(f"[NEWTON] adaptive_gamma: {adaptive_gamma:.3f} (base_gamma: {GAMMA:.3f})")
    print(f"[SPECTRAL] κ_raw: {kappa_raw:.2e}, λ_reg: {lambda_reg:.2e}, κ_after: {kappa_after:.2e}")
    print(f"[SPECTRAL] eigenvalue range: [{lam_min:.2e}, {lam_max:.2e}]")

    # Map to symmetric KxK
    iu = (iu_row, iu_col)
    delta_mat = np.zeros((K, K), float)
    delta_mat[iu] = delta_vec
    delta_mat = delta_mat + delta_mat.T - np.diag(np.diag(delta_mat))

    epsilon_dir = Path(epsilon_dir)
    epsilon_dir.mkdir(parents=True, exist_ok=True)
    epsilon_old_path = _find_latest_epsilon(epsilon_dir)
    epsilon_old = np.load(epsilon_old_path)
    if epsilon_old.shape != (K, K):
        raise ValueError(f"epsilon_old shape {epsilon_old.shape} != ({K},{K})")
    epsilon_new = epsilon_old + delta_mat

    save_path = _next_version_path(epsilon_dir, iteration_idx=iteration_idx)
    np.save(save_path, epsilon_new)

    # Also save as epsilon_next.npy for the pipeline
    epsilon_next_path = os.path.join(output_dir, "epsilon_next.npy")
    np.save(epsilon_next_path, epsilon_new)
    
    # IMPORTANT: Also create phi_mean.npy from the simulated observables
    # This is needed by update_step.py for convergence tracking
    # Since grad = beta * (phi_sim - T_exp), we have phi_sim = (grad / beta) + T_exp
    
    # Try to find experimental targets in standard locations
    exp_targets_path = None
    for potential_path in [
        os.path.join(output_dir, "..", "..", "exp_targets", "T_type_kl.npy"),  # run_root/exp_targets/
        os.path.join(output_dir, "T_type_kl.npy")  # obs/ directory
    ]:
        if os.path.exists(potential_path):
            exp_targets_path = potential_path
            break
    
    if exp_targets_path:
        try:
            T_exp_full = np.load(exp_targets_path)  # Experimental targets (could be matrix or vector)
            
            # Convert T_exp to vector format if it's a matrix
            if T_exp_full.ndim == 2:
                # Convert matrix to upper triangular vector
                T_exp_vec, _ = _flatten_upper(T_exp_full)
                print(f"[REDUCE] Converted T_exp matrix {T_exp_full.shape} to vector {T_exp_vec.shape}")
            else:
                # Already in vector format
                T_exp_vec = T_exp_full
                print(f"[REDUCE] Using T_exp vector shape {T_exp_vec.shape}")
            
            phi_sims = []
            
            # Validate shapes before processing
            sample_file = files[0]
            z_sample = np.load(sample_file)
            grad_sample = z_sample["grad_vec"]
            print(f"[REDUCE] grad_vec shape: {grad_sample.shape}, T_exp_vec shape: {T_exp_vec.shape}")
            
            if grad_sample.shape != T_exp_vec.shape:
                raise ValueError(f"Shape mismatch: grad_vec {grad_sample.shape} != T_exp_vec {T_exp_vec.shape}")
            
            for fpath in files:
                z = np.load(fpath)
                grad_vec = z["grad_vec"] 
                # Reconstruct simulated observables: phi_sim = (grad / beta) + T_exp
                phi_sim = (grad_vec / beta) + T_exp_vec
                phi_sims.append(phi_sim)
            
            # Compute mean and save
            phi_mean = np.mean(phi_sims, axis=0)
            phi_mean_path = os.path.join(output_dir, "phi_mean.npy")
            np.save(phi_mean_path, phi_mean)
            print(f"[REDUCE] Created phi_mean.npy from {len(files)} replicates")
            
            # Also compute covariance diagonal if we have enough data
            if len(phi_sims) > 1:
                phi_sims_array = np.array(phi_sims)
                phi_cov_diag = np.var(phi_sims_array, axis=0, ddof=1)
                phi_cov_path = os.path.join(output_dir, "phi_cov_diag.npy")
                np.save(phi_cov_path, phi_cov_diag)
                print(f"[REDUCE] Created phi_cov_diag.npy")
                
        except Exception as e:
            print(f"[WARNING] Failed to create phi_mean.npy: {e}")
            print(f"[WARNING] update_step.py may fail without phi_mean.npy")
    else:
        print(f"[WARNING] Could not find experimental targets to create phi_mean.npy")
        print(f"[WARNING] update_step.py may fail without phi_mean.npy")

    meta = {
        "gamma_base": GAMMA,
        "gamma_adaptive": float(adaptive_gamma),
        "max_proposed_change": float(max_proposed_change),
        "max_change_per_param": max_change_per_param,
        "lambda_reg": float(lambda_reg),
        "n_replicates": int(len(files)),
        "K": int(K),
        "epsilon_old_path": str(epsilon_old_path),
        "epsilon_new_path": str(save_path),
        "epsilon_next_path": str(epsilon_next_path),
        "B_trace": float(np.trace(B_mean)),
        "M": int(M),
        "spectral_conditioning": {
            "kappa_raw": float(kappa_raw),
            "kappa_target": float(kappa_target),
            "kappa_after": float(kappa_after),
            "lambda_min": float(lam_min),
            "lambda_max": float(lam_max),
            "lambda_reg_spectral": float(lambda_reg),
            "eps_floor": float(eps_floor)
        }
    }
    with open(os.path.join(output_dir, "reduce_summary.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[REDUCE] gamma_base={GAMMA:.2f}, gamma_adaptive={adaptive_gamma:.3f}, lambda_reg={lambda_reg:.3e}, reps={len(files)}")
    print(f"[REDUCE] epsilon_old: {epsilon_old_path.name}")
    print(f"[REDUCE] epsilon_new: {save_path.name}")
    print(f"[REDUCE] epsilon_next: epsilon_next.npy")
    return save_path

# ==========================
# CLI
# ==========================
def parse_args():
    p = argparse.ArgumentParser(description="Process MiChroM-type observables across replicates and update interaction parameters.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # worker mode (used by SLURM array tasks)
    pw = sub.add_parser("worker", help="Process a slice of replicates (per SLURM array task).")
    pw.add_argument("--replicate-root", type=str, required=True,
                    help="Root directory with repXX/trajectory.traj (e.g., /home/.../first_replicate_run/data/first_replicate_run)")
    pw.add_argument("--output-dir", type=str, required=True,
                    help="Directory to write per-rep artifacts and manifest.")
    pw.add_argument("--monomer-types", type=str, required=True,
                    help="Path to ME_bed_types.npy")
    pw.add_argument("--exp-tkl", type=str, required=True,
                    help="Path to Tkl_exp.npy or .npz")
    pw.add_argument("--start-rep", type=int, default=None, help="Start rep index (1-based, inclusive). If omitted, computed from array index.")
    pw.add_argument("--end-rep", type=int, default=None, help="End rep index (inclusive). If omitted, computed from array index.")
    pw.add_argument("--array-index", type=int, default=None, help="SLURM_ARRAY_TASK_ID (0-based or 1-based). If provided, will compute range.")
    pw.add_argument("--array-count", type=int, default=None, help="Total tasks in array.")
    pw.add_argument("--n-total-reps", type=int, default=50)
    pw.add_argument("--workers", type=int, default=7, help="Multiprocessing workers inside this task.")
    pw.add_argument("--io-k", type=int, default=2, help="Max concurrent I/O (reads/writes) per node.")
    pw.add_argument("--mu", type=float, default=MU_DEFAULT)
    pw.add_argument("--rc", type=float, default=RC_DEFAULT)
    pw.add_argument("--rcut", type=float, default=RCUT_DEFAULT)
    pw.add_argument("--beta", type=float, default=BETA_DEFAULT)

    # reduce mode
    pr = sub.add_parser("reduce", help="Aggregate all per-rep artifacts and write epsilon_tk_{n+1}.npy")
    pr.add_argument("--output-dir", type=str, required=True)
    pr.add_argument("--epsilon-dir", type=str, required=True)
    pr.add_argument("--iteration", type=int, help="Current iteration index (for proper epsilon naming)")

    return p.parse_args()

def compute_chunk_for_array(n_total, array_idx, array_count):
    """
    Split n_total items across array_count buckets as evenly as possible.
    First (n_total % array_count) buckets get +1 item.
    Returns (start_idx, end_idx) 1-based inclusive.
    """
    if array_idx is None or array_count is None:
        raise ValueError("array_idx and array_count must be provided to auto-chunk.")
    # Handle 1-based SLURM array indices (convert to 0-based)
    # SLURM arrays typically start at 1, so we need to convert to 0-based indexing
    if array_idx >= array_count:
        # Likely 1-based indexing, convert to 0-based
        array_idx = array_idx - 1
    elif array_idx < 0:
        # Invalid negative index
        raise ValueError(f"array_idx={array_idx} cannot be negative")
    if not (0 <= array_idx < array_count):
        raise ValueError(f"array_idx={array_idx} out of range for array_count={array_count}")

    base = n_total // array_count
    extra = n_total % array_count
    # first 'extra' buckets have size base+1
    if array_idx < extra:
        size = base + 1
        start0 = array_idx * size
    else:
        size = base
        start0 = extra * (base + 1) + (array_idx - extra) * base

    start = start0 + 1            # 1-based inclusive
    end   = start0 + size         # 1-based inclusive
    return start, end

def main():
    args = parse_args()

    if args.cmd == "worker":
        # Resolve replicate range
        if args.start_rep is None or args.end_rep is None:
            if args.array_index is None or args.array_count is None:
                # Try to read SLURM env if not explicitly provided
                try:
                    array_index_env = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
                    array_count_env = int(os.environ.get("SLURM_ARRAY_TASK_COUNT"))
                except Exception:
                    raise SystemExit("Provide --start-rep/--end-rep OR --array-index/--array-count, or run under SLURM array.")
                s, e = compute_chunk_for_array(args.n_total_reps, array_index_env, array_count_env)
            else:
                s, e = compute_chunk_for_array(args.n_total_reps, args.array_index, args.array_count)
        else:
            s, e = args.start_rep, args.end_rep

        # Announce assignment
        print(f"[WORKER] Assigned replicates: {s}..{e} (inclusive) out of {args.n_total_reps}")

        # Prep paths
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = os.path.join(output_dir, "replicate_manifest.txt")
        _maybe_write_manifest_header(manifest_path)

        # Load monomer types once
        monomer_types = np.load(args.monomer_types)

        # Single-thread math libs inside each worker process
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        # Fork-safe semaphore for I/O throttling
        ctx = mp.get_context("fork")
        global _IO_SEMA, _K_IO
        _IO_SEMA = ctx.Semaphore(args.io_k)
        _K_IO = args.io_k

        targets = list(range(s, e + 1))
        n_workers = min(args.workers, max(1, len(targets)))
        with ctx.Pool(processes=n_workers) as pool:
            pool.map(
                partial(
                    _process_replicate_entry,
                    replicate_root=args.replicate_root,
                    output_dir=str(output_dir),
                    monomer_types=monomer_types,
                    exp_Tkl_path=args.exp_tkl,
                    mu=args.mu, rc=args.rc, rcut=args.rcut, beta=args.beta,
                    manifest_path=manifest_path,
                ),
                targets,
                chunksize=1,
            )

    elif args.cmd == "reduce":
        save_path = reduce_and_update(args.output_dir, args.epsilon_dir, iteration_idx=args.iteration)
        print(f"[DONE] Wrote updated parameters to: {save_path}")

if __name__ == "__main__":
    main()
