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

def _load_exp_phi_IC(exp_phi_IC_path, expected_dmax=None):
    """Load experimental phi_exp_IC (1D vector)."""
    if not os.path.exists(exp_phi_IC_path):
        raise FileNotFoundError(exp_phi_IC_path)
    if exp_phi_IC_path.endswith(".npy"):
        phi = np.load(exp_phi_IC_path)
    elif exp_phi_IC_path.endswith(".npz"):
        z = np.load(exp_phi_IC_path)
        for k in ("phi_exp_IC", "phi_exp", "phi"):
            if k in z:
                phi = z[k]; break
        else:
            phi = z[sorted(z.files)[0]]
    else:
        raise ValueError("Experimental phi_IC path must be .npy or .npz")

    phi = np.asarray(phi, float)
    if phi.ndim != 1:
        raise ValueError("Experimental phi_IC must be 1D vector.")
    
    if expected_dmax is not None and phi.shape[0] != expected_dmax:
        raise ValueError(f"phi_IC length ({phi.shape[0]}) != expected dmax ({expected_dmax})")
    
    return phi

class PhiICOnlineCov:
    """Online mean/covariance for phi_IC[d] observables (1D vector)."""
    def __init__(self, dmax):
        self.dmax = dmax
        self.n = 0
        self.mean = np.zeros(dmax, float)
        self.M2   = np.zeros((dmax, dmax), float)

    def add_frame(self, phi_frame):
        """Add a frame's phi[d] values."""
        if phi_frame.shape[0] != self.dmax:
            raise ValueError(f"phi_frame length {phi_frame.shape[0]} != dmax {self.dmax}")
        self.n += 1
        if self.n == 1:
            self.mean[:] = phi_frame
        else:
            d = phi_frame - self.mean
            self.mean += d / self.n
            d2 = phi_frame - self.mean
            self.M2 += np.outer(d, d2)

    def finalize(self, beta=1.0):
        """Return mean, covariance, and Hessian."""
        cov = np.zeros((self.dmax, self.dmax), float) if self.n < 2 else self.M2 / (self.n - 1)
        hess = (beta**2) * cov
        return self.mean.copy(), cov, hess

def _compute_phi_IC_from_positions(positions, d_init, d_end, mu=MU_DEFAULT, rc=RC_DEFAULT, rcut=None, cutoff=0.0):
    """
    Compute phi_IC[d] from positions for all frames (OPTIMIZED VERSION).
    
    For each frame:
    1. Use spatial indexing (cKDTree) to find pairs within rcut
    2. Compute contact probabilities only for nearby pairs
    3. Vectorize diagonal extraction using advanced indexing
    
    Parameters
    ----------
    positions : (F, N, 3) ndarray
        Positions for F frames, N particles.
    d_init : int
        Minimum genomic distance.
    d_end : int
        Maximum genomic distance.
    mu : float
        Tanh kernel parameter.
    rc : float
        Tanh kernel parameter.
    rcut : float, optional
        Cutoff distance. If None, computed as rc + 4.0/mu.
    cutoff : float
        Minimum contact probability to keep (default 0.0).
    
    Returns
    -------
    phi_frames : (F, dmax) ndarray
        phi[d] for each frame.
    """
    F, N, _ = positions.shape
    dmax = d_end - d_init
    
    if rcut is None:
        rcut = rc + 4.0 / mu
    
    phi_frames = np.zeros((F, dmax), dtype=float)
    
    # Pre-allocate arrays for vectorized diagonal extraction
    # For each genomic distance d, we need to track which pairs (i,j) have |j-i| = d
    d_values = np.arange(d_init, d_end)
    
    for f in range(F):
        X = positions[f]
        
        # Use spatial indexing instead of computing full distance matrix
        tree = cKDTree(X, leafsize=40)
        pairs = tree.query_pairs(rcut, output_type='ndarray')
        
        if pairs.size == 0:
            # No pairs found, all phi[d] remain zero
            continue
        
        i = pairs[:, 0]
        j = pairs[:, 1]
        
        # Compute distances and contact probabilities only for nearby pairs
        rij = np.linalg.norm(X[i] - X[j], axis=1)
        fij = f_switch(rij, mu=mu, rc=rc)
        
        # Apply cutoff
        if cutoff > 0.0:
            mask = fij >= cutoff
            i = i[mask]
            j = j[mask]
            fij = fij[mask]
        
        if i.size == 0:
            continue
        
        # Compute genomic distances |j - i| for all pairs
        # query_pairs returns i < j, so genomic distance = j - i (always positive)
        genomic_dists = j - i
        
        # Vectorized accumulation: for each d in [d_init, d_end), compute mean of fij
        # where genomic_dists == d. Use bincount for efficient aggregation.
        # Filter to only genomic distances in our range
        valid_mask = (genomic_dists >= d_init) & (genomic_dists < d_end)
        if np.any(valid_mask):
            genomic_dists_valid = genomic_dists[valid_mask]
            fij_valid = fij[valid_mask]
            
            # Use bincount to sum fij for each genomic distance, then divide by counts
            # Shift indices so d_init maps to 0
            dist_indices = genomic_dists_valid - d_init
            sums = np.bincount(dist_indices, weights=fij_valid, minlength=dmax)
            counts = np.bincount(dist_indices, minlength=dmax)
            
            # Compute means (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                phi_frames[f, :] = np.where(counts > 0, sums / counts, 0.0)
    
    return phi_frames

def process_one_replicate_IC(positions, exp_phi_IC_path, d_init, d_end, mu=MU_DEFAULT, rc=RC_DEFAULT, rcut=RCUT_DEFAULT, beta=BETA_DEFAULT, cutoff=0.0):
    """
    Process one replicate to compute phi_sim, grad, and Hess for IC optimization.
    
    Returns
    -------
    dict with keys:
        phi_sim: (dmax,) mean phi[d] across frames
        grad_vec: (dmax,) gradient vector = beta * (phi_exp - phi_sim)
        Hess: (dmax, dmax) Hessian matrix
        Cov: (dmax, dmax) covariance matrix
    """
    dmax = d_end - d_init
    
    # Compute phi[d] for each frame
    phi_frames = _compute_phi_IC_from_positions(
        positions, d_init, d_end, mu=mu, rc=rc, rcut=rcut, cutoff=cutoff
    )
    
    # Accumulate mean and covariance
    acc = PhiICOnlineCov(dmax)
    for f in range(phi_frames.shape[0]):
        acc.add_frame(phi_frames[f])
    
    phi_sim, Cov, Hess = acc.finalize(beta=beta)
    
    # Load experimental target
    phi_exp = _load_exp_phi_IC(exp_phi_IC_path, expected_dmax=dmax)
    
    # Compute gradient: grad = beta * (phi_exp - phi_sim)
    grad_vec = beta * (phi_exp - phi_sim)
    
    return {
        "phi_sim": phi_sim,
        "phi_exp": phi_exp,
        "grad_vec": grad_vec,
        "Hess": Hess,
        "Cov": Cov,
        "d_init": d_init,
        "d_end": d_end,
        "dmax": dmax,
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

def _process_replicate_entry_IC(rep_idx, replicate_root, output_dir, exp_phi_IC_path, d_init, d_end, mu, rc, rcut, beta, cutoff, manifest_path):
    rep_str, traj_path = _rep_dir_and_path(replicate_root, rep_idx)
    out_npz   = os.path.join(output_dir, f"{rep_str}_IC_grad_hess.npz")
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
        out = process_one_replicate_IC(
            positions=positions,
            exp_phi_IC_path=exp_phi_IC_path,
            d_init=d_init,
            d_end=d_end,
            mu=mu, rc=rc, rcut=rcut, beta=beta, cutoff=cutoff
        )
        compute_s = time.time() - t_comp0

        t_wr0 = time.time()
        if _IO_SEMA is not None: _IO_SEMA.acquire()
        try:
            np.savez_compressed(
                out_npz,
                grad_vec=out["grad_vec"],
                Hess=out["Hess"],
                phi_sim=out["phi_sim"],
                dmax=out["dmax"],
                d_init=out["d_init"],
                d_end=out["d_end"],
                mu=out["mu"], rc=out["rc"], rcut=out["rcut"],
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
# Reduce step: aggregate reps and update lambda_IC
# ==========================
def _find_latest_lambda_IC(lambda_dir: Path, stem: str = "lambda_IC_tk_", ext: str = ".npy") -> Path:
    patt = re.compile(rf"^{re.escape(stem)}(\d+){re.escape(ext)}$")
    latest = None
    max_n = -1
    for p in lambda_dir.glob(f"{stem}*{ext}"):
        m = patt.match(p.name)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
                latest = p
    if latest is None:
        raise FileNotFoundError(f"No prior lambda_IC_tk_*.npy found in {lambda_dir}")
    return latest

def _next_version_path(dirpath: Path, stem: str = "lambda_IC_tk_", ext: str = ".npy", iteration_idx: int = None) -> Path:
    """Generate the next lambda_IC filename."""
    if iteration_idx is not None:
        target_n = iteration_idx + 1
        target_path = dirpath / f"{stem}{target_n}{ext}"
        
        # Clean up any higher-numbered files
        patt = re.compile(rf"^{re.escape(stem)}(\d+){re.escape(ext)}$")
        for p in dirpath.glob(f"{stem}*{ext}"):
            m = patt.match(p.name)
            if m:
                n = int(m.group(1))
                if n > target_n:
                    print(f"[CLEANUP] Removing unexpected lambda_IC file: {p.name}")
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

def reduce_and_update_IC(output_dir, lambda_dir, beta=BETA_DEFAULT, iteration_idx=None):
    """
    Read all repXX_IC_grad_hess.npz in output_dir, average grad & Hess,
    apply one damped Newton step, and save lambda_IC_tk_{n+1}.npy in lambda_dir.
    """
    files = sorted(glob.glob(os.path.join(output_dir, "rep??_IC_grad_hess.npz")))
    if len(files) == 0:
        raise RuntimeError(f"No per-replicate .npz found in {output_dir}")

    grads = []
    Hlist = []
    dmax = None
    for fpath in files:
        z = np.load(fpath)
        grad_vec  = z["grad_vec"]
        Hess      = z["Hess"]
        this_dmax = int(z["dmax"])
        if dmax is None:
            dmax = this_dmax
        elif this_dmax != dmax:
            raise ValueError(f"Inconsistent dmax in {fpath}: {this_dmax} vs {dmax}")
        grads.append(grad_vec)
        Hlist.append(Hess)

    grads = np.stack(grads, axis=0)       # (R, dmax)
    Hlist = np.stack(Hlist, axis=0)       # (R, dmax, dmax)
    g_mean = grads.mean(axis=0)           # (dmax,)
    B_mean = Hlist.mean(axis=0)           # (dmax, dmax)

    # Spectral conditioning (same as process_tkl_update.py)
    M = B_mean.shape[0]
    w = np.linalg.eigvalsh(B_mean)
    lam_min, lam_max = w[0], w[-1]
    kappa_raw = lam_max / max(abs(lam_min), 1e-12)

    if kappa_raw > 1e7:
        kappa_target = 5e2
    elif kappa_raw > 1e5:
        kappa_target = 1e3
    else:
        kappa_target = 1e4
    eps_floor = 1e-5 * w.mean()

    lam_psd = max(0.0, -lam_min + eps_floor)
    if kappa_raw > kappa_target:
        lam_kappa = max(0.0, (lam_max / kappa_target) - lam_min)
    else:
        lam_kappa = 0.0

    lambda_reg = max(lam_psd, lam_kappa)
    B_reg = B_mean + lambda_reg * np.eye(M)

    # Solve Δλ = -γ * B^{-1} g
    try:
        cho_fac = cho_factor(B_reg)
        delta_vec = -cho_solve(cho_fac, g_mean)
    except np.linalg.LinAlgError:
        print("[WARNING] Cholesky failed, falling back to standard solve")
        delta_vec = -np.linalg.solve(B_reg, g_mean)
    
    # Parameter-dependent scaling
    max_change_per_param = 0.5
    max_proposed_change = np.max(np.abs(delta_vec))
    if max_proposed_change > 0:
        adaptive_gamma = min(max_change_per_param / max_proposed_change, GAMMA)
    else:
        adaptive_gamma = GAMMA
    
    delta_vec *= adaptive_gamma
    
    kappa_after = (lam_max + lambda_reg) / (abs(lam_min) + lambda_reg)
    
    print(f"[NEWTON] max_proposed_change: {max_proposed_change:.3f}")
    print(f"[NEWTON] adaptive_gamma: {adaptive_gamma:.3f} (base_gamma: {GAMMA:.3f})")
    print(f"[SPECTRAL] κ_raw: {kappa_raw:.2e}, λ_reg: {lambda_reg:.2e}, κ_after: {kappa_after:.2e}")

    lambda_dir = Path(lambda_dir)
    lambda_dir.mkdir(parents=True, exist_ok=True)
    lambda_old_path = _find_latest_lambda_IC(lambda_dir)
    lambda_old = np.load(lambda_old_path)
    if lambda_old.shape[0] != dmax:
        raise ValueError(f"lambda_old shape {lambda_old.shape} != ({dmax},)")
    lambda_new = lambda_old + delta_vec

    save_path = _next_version_path(lambda_dir, iteration_idx=iteration_idx)
    np.save(save_path, lambda_new)

    # Also save as lambda_IC_next.npy for the pipeline
    lambda_next_path = os.path.join(output_dir, "lambda_IC_next.npy")
    np.save(lambda_next_path, lambda_new)
    
    # Create phi_mean.npy from simulated observables
    exp_targets_path = None
    for potential_path in [
        os.path.join(output_dir, "..", "..", "exp_targets", "phi_exp_IC.npy"),
        os.path.join(output_dir, "phi_exp_IC.npy")
    ]:
        if os.path.exists(potential_path):
            exp_targets_path = potential_path
            break
    
    if exp_targets_path:
        try:
            phi_exp = np.load(exp_targets_path)
            
            phi_sims = []
            for fpath in files:
                z = np.load(fpath)
                grad_vec = z["grad_vec"]
                # phi_sim = phi_exp - (grad_vec / beta)
                phi_sim = phi_exp - (grad_vec / beta)
                phi_sims.append(phi_sim)
            
            phi_mean = np.mean(phi_sims, axis=0)
            phi_mean_path = os.path.join(output_dir, "phi_mean.npy")
            np.save(phi_mean_path, phi_mean)
            print(f"[REDUCE] Created phi_mean.npy from {len(files)} replicates")
            
            if len(phi_sims) > 1:
                phi_sims_array = np.array(phi_sims)
                phi_cov_diag = np.var(phi_sims_array, axis=0, ddof=1)
                phi_cov_path = os.path.join(output_dir, "phi_cov_diag.npy")
                np.save(phi_cov_path, phi_cov_diag)
                print(f"[REDUCE] Created phi_cov_diag.npy")
                
        except Exception as e:
            print(f"[WARNING] Failed to create phi_mean.npy: {e}")

    meta = {
        "gamma_base": GAMMA,
        "gamma_adaptive": float(adaptive_gamma),
        "max_proposed_change": float(max_proposed_change),
        "max_change_per_param": max_change_per_param,
        "lambda_reg": float(lambda_reg),
        "n_replicates": int(len(files)),
        "dmax": int(dmax),
        "lambda_old_path": str(lambda_old_path),
        "lambda_new_path": str(save_path),
        "lambda_next_path": str(lambda_next_path),
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
    print(f"[REDUCE] lambda_old: {lambda_old_path.name}")
    print(f"[REDUCE] lambda_new: {save_path.name}")
    print(f"[REDUCE] lambda_next: lambda_IC_next.npy")
    return save_path

# ==========================
# CLI
# ==========================
def parse_args():
    p = argparse.ArgumentParser(description="Process ideal chromosome observables and update lambda_IC parameters.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # worker mode
    pw = sub.add_parser("worker", help="Process a slice of replicates (per SLURM array task).")
    pw.add_argument("--replicate-root", type=str, required=True)
    pw.add_argument("--output-dir", type=str, required=True)
    pw.add_argument("--exp-phi-IC", type=str, required=True, help="Path to phi_exp_IC.npy")
    pw.add_argument("--d-init", type=int, required=True)
    pw.add_argument("--d-end", type=int, required=True)
    pw.add_argument("--start-rep", type=int, default=None)
    pw.add_argument("--end-rep", type=int, default=None)
    pw.add_argument("--array-index", type=int, default=None)
    pw.add_argument("--array-count", type=int, default=None)
    pw.add_argument("--n-total-reps", type=int, default=50)
    pw.add_argument("--workers", type=int, default=7)
    pw.add_argument("--io-k", type=int, default=2)
    pw.add_argument("--mu", type=float, default=MU_DEFAULT)
    pw.add_argument("--rc", type=float, default=RC_DEFAULT)
    pw.add_argument("--rcut", type=float, default=RCUT_DEFAULT)
    pw.add_argument("--beta", type=float, default=BETA_DEFAULT)
    pw.add_argument("--cutoff", type=float, default=0.0)

    # reduce mode
    pr = sub.add_parser("reduce", help="Aggregate all per-rep artifacts and write lambda_IC_tk_{n+1}.npy")
    pr.add_argument("--output-dir", type=str, required=True)
    pr.add_argument("--lambda-dir", type=str, required=True)
    pr.add_argument("--iteration", type=int, help="Current iteration index")

    return p.parse_args()

def compute_chunk_for_array(n_total, array_idx, array_count):
    """Split n_total items across array_count buckets."""
    if array_idx is None or array_count is None:
        raise ValueError("array_idx and array_count must be provided to auto-chunk.")
    if array_idx >= array_count:
        array_idx = array_idx - 1
    elif array_idx < 0:
        raise ValueError(f"array_idx={array_idx} cannot be negative")
    if not (0 <= array_idx < array_count):
        raise ValueError(f"array_idx={array_idx} out of range for array_count={array_count}")

    base = n_total // array_count
    extra = n_total % array_count
    if array_idx < extra:
        size = base + 1
        start0 = array_idx * size
    else:
        size = base
        start0 = extra * (base + 1) + (array_idx - extra) * base

    start = start0 + 1
    end   = start0 + size
    return start, end

def main():
    args = parse_args()

    if args.cmd == "worker":
        # Resolve replicate range
        if args.start_rep is None or args.end_rep is None:
            if args.array_index is None or args.array_count is None:
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

        print(f"[WORKER] Assigned replicates: {s}..{e} (inclusive) out of {args.n_total_reps}")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = os.path.join(output_dir, "replicate_manifest.txt")
        _maybe_write_manifest_header(manifest_path)

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        ctx = mp.get_context("fork")
        global _IO_SEMA, _K_IO
        _IO_SEMA = ctx.Semaphore(args.io_k)
        _K_IO = args.io_k

        targets = list(range(s, e + 1))
        n_workers = min(args.workers, max(1, len(targets)))
        with ctx.Pool(processes=n_workers) as pool:
            pool.map(
                partial(
                    _process_replicate_entry_IC,
                    replicate_root=args.replicate_root,
                    output_dir=str(output_dir),
                    exp_phi_IC_path=args.exp_phi_IC,
                    d_init=args.d_init,
                    d_end=args.d_end,
                    mu=args.mu, rc=args.rc, rcut=args.rcut, beta=args.beta, cutoff=args.cutoff,
                    manifest_path=manifest_path,
                ),
                targets,
                chunksize=1,
            )

    elif args.cmd == "reduce":
        save_path = reduce_and_update_IC(args.output_dir, args.lambda_dir, iteration_idx=args.iteration)
        print(f"[DONE] Wrote updated parameters to: {save_path}")

if __name__ == "__main__":
    main()

