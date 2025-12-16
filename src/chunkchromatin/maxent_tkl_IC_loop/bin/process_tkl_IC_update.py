#!/usr/bin/env python3
"""
Combined processing script for both TKL (type-type) and IC (ideal chromosome) observables.
This script processes both observables from the same trajectory files and performs
combined Newton updates for both epsilon and lambda_IC parameters.
"""
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

# Adam optimizer constants
ADAM_LR_DEFAULT = 1e-3
ADAM_BETA1_DEFAULT = 0.9
ADAM_BETA2_DEFAULT = 0.999
ADAM_EPS_DEFAULT = 1e-8
RELSTEP_TARGET_DEFAULT = 0.02   # target rms step as fraction of rms(param)
RELSTEP_MAX_FRAC_DEFAULT = 0.05 # per-parameter cap as fraction of |param|

# ==========================
# Core kernels & helpers (shared)
# ==========================
def f_switch(r, mu=MU_DEFAULT, rc=RC_DEFAULT):
    # MiChroM "switch" (contact probability proxy)
    return 0.5 * (1.0 + np.tanh(mu * (rc - r)))

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
# TKL-specific functions
# ==========================
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

def _compute_monomer_contact_matrix(positions, mu=MU_DEFAULT, rc=RC_DEFAULT, rcut=None):
    """
    Compute average N×N monomer-level contact matrix across all frames.
    Returns: (F, N, N) array where F is number of frames, N is number of monomers.
    """
    F, N, _ = positions.shape
    if rcut is None:
        rcut = rc + 4.0 / mu
    
    # Store per-frame contact matrices for covariance calculation
    contact_matrices = np.zeros((F, N, N), dtype=np.float64)
    
    for f in range(F):
        X = positions[f]
        tree = cKDTree(X, leafsize=40)
        pairs = tree.query_pairs(rcut, output_type='ndarray')
        
        if pairs.size != 0:
            i = pairs[:, 0]
            j = pairs[:, 1]
            rij = np.linalg.norm(X[i] - X[j], axis=1)
            fij = f_switch(rij, mu=mu, rc=rc)
            
            # Fill symmetric matrix
            contact_matrices[f, i, j] = fij
            contact_matrices[f, j, i] = fij
    
    return contact_matrices

def _block_sum_and_normalize(matrix, block_size=5):
    """
    Block-sum matrix with blocks of size block_size × block_size,
    then normalize each row by its maximum value.
    
    Args:
        matrix: (N, N) array at 100kb resolution
        block_size: number of bins to sum (default 5 for 100kb -> 500kb)
    
    Returns:
        (N//block_size, N//block_size) array normalized by row maxima
    """
    N = matrix.shape[0]
    N_new = N // block_size
    
    # Truncate to evenly divisible size
    N_trunc = N_new * block_size
    matrix_trunc = matrix[:N_trunc, :N_trunc]
    
    # Reshape and sum over blocks: (N_new, block_size, N_new, block_size) -> (N_new, N_new)
    matrix_blocked = matrix_trunc.reshape(N_new, block_size, N_new, block_size).sum(axis=(1, 3))
    
    # Normalize each row by its maximum
    row_maxes = matrix_blocked.max(axis=1, keepdims=True)
    row_maxes = np.where(row_maxes > 0, row_maxes, 1.0)  # Avoid division by zero
    matrix_normalized = matrix_blocked / row_maxes
    
    return matrix_normalized

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

def _covariance_pass_upper_500kb(
    positions,
    monomer_types,
    mu=MU_DEFAULT,
    rc=RC_DEFAULT,
    rcut=None,
    block_size=5,
):
    """
    Compute type-type observables at 500kb resolution:
    1. Use monomer-level positions (100kb beads) and the same distance kernel f_switch
       to compute contact probabilities.
    2. Block-sum the monomer contacts in 5x5 blocks to obtain a 500kb contact matrix.
    3. Normalize each 500kb-row by its maximum (Hi-C style).
    4. Aggregate the normalized 500kb contact matrix by chromatin type of the 500kb bins
       to obtain a KxK Tkl for each frame.
    5. Use UpperTriOnlineCov over the KxK upper triangle, exactly as in the 100kb case.

    Returns
    -------
    Tkl_mean : (K, K) float
        Mean type-type observable at 500kb.
    Cov_upper : (M, M) float
        Covariance over the flattened upper triangle (M = K*(K+1)//2).
    Hess_upper : (M, M) float
        Same as Cov_upper initially; caller rescales by beta^2.
    iu : tuple of arrays
        Indices of the upper triangle (k,l) with k <= l.
    type_labels : (K,) array
        Sorted unique chromatin type labels at 500kb (downsampled).
    N_500kb : int
        Number of 500kb bins (N_monomers // block_size).
    """
    F, N, _ = positions.shape
    if rcut is None:
        rcut = rc + 4.0 / mu

    # 500kb binning along the chain
    N_500 = N // block_size
    if N_500 * block_size != N:
        # Truncate monomer_types and positions to a multiple of block_size
        N_trunc = N_500 * block_size
        positions = positions[:, :N_trunc, :]
        monomer_types = monomer_types[:N_trunc]

    # Downsample monomer types: take every 5th monomer as the "500kb bead" type
    monomer_types_500 = monomer_types[::block_size]
    type_labels, inv_500 = np.unique(monomer_types_500, return_inverse=True)
    K = len(type_labels)

    acc = UpperTriOnlineCov(K)
    iuK = np.triu_indices(K)

    for f in range(F):
        X = positions[f]
        tree = cKDTree(X, leafsize=40)
        pairs = tree.query_pairs(rcut, output_type='ndarray')

        # Build 500kb contact matrix for this frame by block-summing monomer contacts
        B = np.zeros((N_500, N_500), dtype=np.float64)

        if pairs.size != 0:
            i = pairs[:, 0]
            j = pairs[:, 1]
            rij = np.linalg.norm(X[i] - X[j], axis=1)
            fij = f_switch(rij, mu=mu, rc=rc)

            # Map monomer indices -> 500kb bin indices
            bi = i // block_size
            bj = j // block_size

            # Accumulate into a dense 500kb matrix (upper triangle)
            flat_ij = bi * N_500 + bj
            sums = np.bincount(flat_ij, weights=fij, minlength=N_500 * N_500)
            sums = sums.reshape(N_500, N_500)

            # Symmetrize: we only had i<j pairs originally
            B = sums + sums.T
            # Do not double count diagonal contributions
            diag_idx = np.diag_indices(N_500)
            B[diag_idx] = sums[diag_idx]

        # Row-normalize so each row's max is 1 (if any contacts exist)
        row_max = B.max(axis=1, keepdims=True)
        row_max = np.where(row_max > 0.0, row_max, 1.0)
        B_norm = B / row_max

        # Aggregate 500kb contact probabilities into KxK type-type observables
        T_frame = np.zeros((K, K), dtype=np.float64)

        # We want to sum B_norm[I,J] for all I,J belonging to type pairs (k,l).
        # Do it in one shot with bincount over the upper triangle of 500kb bins.
        I_bins, J_bins = np.triu_indices(N_500)
        if I_bins.size != 0:
            weights = B_norm[I_bins, J_bins].ravel()
            tI = inv_500[I_bins]
            tJ = inv_500[J_bins]
            k = np.minimum(tI, tJ)
            l = np.maximum(tI, tJ)
            flat_types = k * K + l
            sums_types = np.bincount(
                flat_types,
                weights=weights,
                minlength=K * K
            ).reshape(K, K)
            T_frame[iuK] = sums_types[iuK]

        acc.add_frame_from_upper_mat(T_frame)

    Tkl_mean, Cov_upper, Hess_upper, iu = acc.finalize(beta=1.0)
    return Tkl_mean, Cov_upper, Hess_upper, iu, type_labels, N_500

def process_one_replicate(
    positions,
    monomer_types,
    exp_Tkl_path,
    mu=MU_DEFAULT,
    rc=RC_DEFAULT,
    rcut=RCUT_DEFAULT,
    beta=BETA_DEFAULT,
    resolution=None,
):
    """
    Process one replicate to compute gradients and Hessian for TKL optimization.

    Parameters
    ----------
    positions : (F, N, 3) array
        Trajectory in reduced units.
    monomer_types : (N,) array
        Integer chromatin type labels at 100kb resolution.
    exp_Tkl_path : str
        Path to experimental Tkl target (K x K) at the chosen resolution
        (100kb or 500kb), aggregated by chromatin type.
    mu, rc, rcut, beta : float
        Parameters for the distance kernel and MaxEnt.
    resolution : None or "500kb"
        If None:
            Use the standard 100kb observable definition:
            - distances → f_switch → sum by chromatin type (100kb).
        If "500kb":
            Use 500kb Hi-C bin observables:
            - distances → f_switch at 100kb
            - block-sum 5x5 to 500kb
            - row-normalize per 500kb row
            - sum by chromatin type of 500kb bins.

    Returns
    -------
    dict with keys:
        type_labels, Tkl_sim, Tkl_exp, Delta,
        upper_indices, grad_vec, Hess_upper, Cov_upper,
        mu, rc, rcut, resolution, (and N_500kb for 500kb mode).
    """
    if resolution == "500kb":
        # 500kb: coarse-grain monomer contacts to 500kb bins, normalize, then
        # compute type-type observable exactly as in 100kb case.
        Tkl_sim, Cov_upper, Hess_upper, iu, type_labels, N_500 = _covariance_pass_upper_500kb(
            positions,
            monomer_types,
            mu=mu,
            rc=rc,
            rcut=rcut,
            block_size=5,
        )
        K = len(type_labels)

        # Experimental target should already be the type-type Tkl at 500kb
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
            "mu": mu,
            "rc": rc,
            "rcut": rcut if rcut is not None else rc + 4.0 / mu,
            "resolution": "500kb",
            "N_500kb": int(N_500),
        }

    else:
        # Default 100kb pathway: type-averaged observables from monomer contacts.
        Tkl_sim, Cov_upper, Hess_upper, iu, type_labels = _covariance_pass_upper(
            positions,
            monomer_types,
            mu=mu,
            rc=rc,
            rcut=rcut,
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
            "mu": mu,
            "rc": rc,
            "rcut": rcut if rcut is not None else rc + 4.0 / mu,
            "resolution": None,
        }

# ==========================
# IC-specific functions
# ==========================
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

def _compute_phi_IC_from_positions(positions, d_init, d_end, mu=MU_DEFAULT, rc=RC_DEFAULT, rcut=None):
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

def process_one_replicate_IC(positions, exp_phi_IC_path, d_init, d_end, mu=MU_DEFAULT, rc=RC_DEFAULT, rcut=RCUT_DEFAULT, beta=BETA_DEFAULT):
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
        positions, d_init, d_end, mu=mu, rc=rc, rcut=rcut
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

# ==========================
# Shared helper functions
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

def compute_chunk_for_array(n_total, array_idx, array_count):
    """
    Split n_total items across array_count buckets as evenly as possible.
    First (n_total % array_count) buckets get +1 item.
    Returns (start_idx, end_idx) 1-based inclusive.
    """
    if array_idx is None or array_count is None:
        raise ValueError("array_idx and array_count must be provided to auto-chunk.")
    # Handle 1-based SLURM array indices (convert to 0-based)
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

    start = start0 + 1            # 1-based inclusive
    end   = start0 + size         # 1-based inclusive
    return start, end

# ==========================
# File versioning helpers (generalized)
# ==========================
def _find_latest_param(param_dir: Path, stem: str, ext: str = ".npy") -> Path:
    """Generic function to find latest versioned parameter file."""
    patt = re.compile(rf"^{re.escape(stem)}(\d+){re.escape(ext)}$")
    latest = None
    max_n = -1
    for p in param_dir.glob(f"{stem}*{ext}"):
        m = patt.match(p.name)
        if m:
            n = int(m.group(1))
            if n > max_n:
                max_n = n
                latest = p
    if latest is None:
        raise FileNotFoundError(f"No prior {stem}*.{ext} found in {param_dir}")
    return latest

def _next_version_path(dirpath: Path, stem: str, ext: str = ".npy", iteration_idx: int = None) -> Path:
    """
    Generate the next versioned filename. For iteration i, produces {stem}{i+1}{ext}.
    If iteration_idx is provided, uses that as the target. Otherwise, uses legacy +1 logic.
    """
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
                    print(f"[CLEANUP] Removing unexpected file: {p.name}")
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

# Convenience wrappers
def _find_latest_epsilon(epsilon_dir: Path) -> Path:
    return _find_latest_param(epsilon_dir, "epsilon_tk_", ".npy")

def _find_latest_lambda_IC(lambda_dir: Path) -> Path:
    return _find_latest_param(lambda_dir, "lambda_IC_tk_", ".npy")

# ==========================
# Relative error-based step size scaling
# ==========================
def _compute_relative_error_step_sizes(
    base_max_step_size,
    iteration_idx,
    output_dir,
    adaptive_config=None
):
    """
    Compute separate step sizes for TKL and IC based on their relative errors.
    
    The idea: larger relative errors need larger step sizes (more correction needed),
    smaller relative errors need smaller step sizes (fine-tuning, prevent overshooting).
    
    Parameters
    ----------
    base_max_step_size : float
        Base maximum step size from config
    iteration_idx : int
        Current iteration index
    output_dir : str
        Output directory (to find previous iteration's state.json)
    adaptive_config : dict, optional
        Configuration for relative error scaling:
        - relative_error_scaling: bool - enable/disable this feature
        - reference_threshold: float (default 100.0) - reference relative error (%)
        - min_scale_factor: float (default 0.1) - minimum step size multiplier
        - max_scale_factor: float (default 2.0) - maximum step size multiplier
        - scaling_function: str (default "sqrt") - "linear", "sqrt", or "ratio"
    
    Returns
    -------
    max_step_tkl : float
        Adjusted maximum step size for TKL
    max_step_ic : float
        Adjusted maximum step size for IC
    scaling_info : dict
        Information about the scaling applied
    """
    if base_max_step_size is None:
        return None, None, {"reason": "no_base_step_size"}

    if adaptive_config is None or iteration_idx is None or iteration_idx == 0:
        return base_max_step_size, base_max_step_size, {"reason": "first_iteration"}
    
    # Check if relative error scaling is enabled
    relative_error_scaling = adaptive_config.get("relative_error_scaling", False)
    if not relative_error_scaling:
        return base_max_step_size, base_max_step_size, {"reason": "disabled"}
    
    # Get scaling parameters
    reference_threshold = adaptive_config.get("reference_threshold", 100.0)
    min_scale = adaptive_config.get("min_scale_factor", 0.1)
    max_scale = adaptive_config.get("max_scale_factor", 2.0)
    scaling_function = adaptive_config.get("scaling_function", "sqrt")
    
    # Try to load previous iteration's state.json to get relative errors
    prev_iter = iteration_idx - 1
    prev_state_path = os.path.join(output_dir, "..", f"iter_{prev_iter:03d}", "update", "state.json")
    
    if not os.path.exists(prev_state_path):
        # Try alternative location
        prev_state_path = os.path.join(output_dir, "..", f"iter_{prev_iter:03d}", "obs", "reduce_summary.json")
    
    if not os.path.exists(prev_state_path):
        return base_max_step_size, base_max_step_size, {"reason": "no_previous_state"}
    
    try:
        with open(prev_state_path, 'r') as f:
            prev_state = json.load(f)
        
        # Extract relative errors
        epsilon_data = prev_state.get("epsilon", {})
        lambda_IC_data = prev_state.get("lambda_IC", {})
        
        rel_err_tkl = epsilon_data.get("relative_error_pct")
        rel_err_ic = lambda_IC_data.get("relative_error_pct")
        
        if rel_err_tkl is None or rel_err_ic is None:
            return base_max_step_size, base_max_step_size, {"reason": "no_relative_errors"}
        
        # Compute scaling factors based on relative errors
        if scaling_function == "linear":
            # Linear scaling: scale proportionally to relative error
            scale_tkl = rel_err_tkl / reference_threshold
            scale_ic = rel_err_ic / reference_threshold
        elif scaling_function == "sqrt":
            # Square root scaling: more conservative, prevents extreme values
            scale_tkl = np.sqrt(rel_err_tkl / reference_threshold)
            scale_ic = np.sqrt(rel_err_ic / reference_threshold)
        elif scaling_function == "ratio":
            # Ratio-based: scale based on ratio of relative errors
            # If IC has 5x larger error, give it 5x larger step (capped)
            if rel_err_tkl > 1e-10:
                rel_err_ratio = rel_err_ic / rel_err_tkl
                scale_ic = min(rel_err_ratio, max_scale)
                scale_tkl = min(rel_err_tkl / rel_err_ic, max_scale) if rel_err_ic > 1e-10 else 1.0
            else:
                scale_tkl = 1.0
                scale_ic = min(rel_err_ic / reference_threshold, max_scale) if reference_threshold > 1e-10 else 1.0
        else:
            # Default: sqrt
            scale_tkl = np.sqrt(rel_err_tkl / reference_threshold)
            scale_ic = np.sqrt(rel_err_ic / reference_threshold)
        
        # Apply bounds
        scale_tkl = np.clip(scale_tkl, min_scale, max_scale)
        scale_ic = np.clip(scale_ic, min_scale, max_scale)
        
        # Compute final step sizes
        max_step_tkl = base_max_step_size * scale_tkl
        max_step_ic = base_max_step_size * scale_ic
        
        scaling_info = {
            "reason": "relative_error_scaling",
            "rel_err_tkl": float(rel_err_tkl),
            "rel_err_ic": float(rel_err_ic),
            "scale_tkl": float(scale_tkl),
            "scale_ic": float(scale_ic),
            "max_step_tkl": float(max_step_tkl),
            "max_step_ic": float(max_step_ic),
            "scaling_function": scaling_function
        }
        
        return max_step_tkl, max_step_ic, scaling_info
        
    except Exception as e:
        print(f"[WARNING] Failed to compute relative-error-based step sizes: {e}")
        return base_max_step_size, base_max_step_size, {"reason": f"error: {str(e)}"}

# ==========================
# Adaptive step size reduction
# ==========================
def _compute_adaptive_step_size(
    current_max_step_size,
    iteration_idx,
    output_dir,
    adaptive_config=None
):
    """
    Compute adaptive step size based on convergence metrics.
    
    Parameters
    ----------
    current_max_step_size : float
        Current maximum step size
    iteration_idx : int
        Current iteration index
    output_dir : str
        Output directory (to find previous iteration's state.json)
    adaptive_config : dict, optional
        Configuration for adaptive reduction:
        - reduce_when_relative_error_below: float (e.g., 100.0) - reduce when relative_error < this
        - reduction_factor: float (e.g., 0.8) - multiply step size by this when reducing
        - min_step_size: float (e.g., 0.01) - minimum allowed step size
        - reduce_when_loss_plateau: bool - reduce when losses plateau
        - plateau_patience: int - number of iterations without improvement before reducing
    
    Returns
    -------
    new_max_step_size : float
        Adjusted maximum step size
    reduction_reason : str
        Reason for reduction (or None if no reduction)
    """
    if adaptive_config is None or iteration_idx is None or iteration_idx == 0:
        return current_max_step_size, None
    
    # Default config
    reduce_when_relative_error_below = adaptive_config.get("reduce_when_relative_error_below", None)
    reduction_factor = adaptive_config.get("reduction_factor", 0.8)
    min_step_size = adaptive_config.get("min_step_size", 0.01)
    reduce_when_loss_plateau = adaptive_config.get("reduce_when_loss_plateau", False)
    plateau_patience = adaptive_config.get("plateau_patience", 3)
    
    new_step_size = current_max_step_size
    reduction_reason = None
    prev_state = {}  # Initialize prev_state to avoid UnboundLocalError
    
    # Try to read previous iteration's state.json
    prev_iter = iteration_idx - 1
    prev_state_path = os.path.join(output_dir, "..", f"iter_{prev_iter:03d}", "update", "state.json")
    
    if not os.path.exists(prev_state_path):
        # Try alternative location
        prev_state_path = os.path.join(output_dir, "..", f"iter_{prev_iter:03d}", "obs", "reduce_summary.json")
    
    if os.path.exists(prev_state_path):
        try:
            with open(prev_state_path, 'r') as f:
                prev_state = json.load(f)
            
            # Get current residuals (from current iteration's reduce_summary if available)
            current_reduce_summary = os.path.join(output_dir, "reduce_summary.json")
            if os.path.exists(current_reduce_summary):
                with open(current_reduce_summary, 'r') as f:
                    current_summary = json.load(f)
                
                # Extract relative error from current iteration
                epsilon_data = current_summary.get("epsilon", {})
                lambda_IC_data = current_summary.get("lambda_IC", {})
                
                # Try to get relative error (may not be in reduce_summary, need to compute from gradients)
                # For now, use max_proposed_change as a proxy
                current_max_change = max(
                    epsilon_data.get("max_proposed_change", 0),
                    lambda_IC_data.get("max_proposed_change", 0)
                )
            else:
                current_max_change = None
            
            # Strategy 1: Reduce when relative error is below threshold
            if reduce_when_relative_error_below is not None:
                # Try to get relative error from previous state
                prev_epsilon = prev_state.get("epsilon", {})
                prev_lambda_IC = prev_state.get("lambda_IC", {})
                prev_relative_error = prev_epsilon.get("relative_error_pct") or prev_lambda_IC.get("relative_error_pct")
                
                if prev_relative_error is not None and prev_relative_error < reduce_when_relative_error_below:
                    new_step_size = max(current_max_step_size * reduction_factor, min_step_size)
                    reduction_reason = f"relative_error ({prev_relative_error:.1f}%) < threshold ({reduce_when_relative_error_below}%)"
            
            # Strategy 2: Reduce when losses plateau (no improvement for N iterations)
            if reduce_when_loss_plateau and prev_iter >= plateau_patience:
                # Check if losses have been increasing or flat for plateau_patience iterations
                prev_epsilon = prev_state.get("epsilon", {})
                prev_lambda_IC = prev_state.get("lambda_IC", {})
                prev_max_abs = max(
                    prev_epsilon.get("max_abs_residual", 0),
                    prev_lambda_IC.get("max_abs_residual", 0)
                )
                
                # Check previous plateau_patience iterations
                losses_increasing = True
                for check_iter in range(max(0, prev_iter - plateau_patience + 1), prev_iter):
                    check_state_path = os.path.join(output_dir, "..", f"iter_{check_iter:03d}", "update", "state.json")
                    if os.path.exists(check_state_path):
                        try:
                            with open(check_state_path, 'r') as f:
                                check_state = json.load(f)
                            check_epsilon = check_state.get("epsilon", {})
                            check_lambda_IC = check_state.get("lambda_IC", {})
                            check_max_abs = max(
                                check_epsilon.get("max_abs_residual", 0),
                                check_lambda_IC.get("max_abs_residual", 0)
                            )
                            if check_max_abs < prev_max_abs * 0.99:  # At least 1% improvement
                                losses_increasing = False
                                break
                        except Exception:
                            pass
                
                if losses_increasing:
                    new_step_size = max(current_max_step_size * reduction_factor, min_step_size)
                    reduction_reason = f"losses plateaued (no improvement for {plateau_patience} iterations)"
        
        except Exception as e:
            print(f"[WARNING] Could not read previous state for adaptive step size: {e}")
    
    # Strategy 3: Exponential decay (only if BOTH relative errors are below threshold)
    if adaptive_config.get("exponential_decay", False) and prev_state:
        decay_rate = adaptive_config.get("decay_rate", 0.995)
        decay_relative_error_threshold = adaptive_config.get("decay_relative_error_threshold", 100.0)
        
        # Check if both relative errors are below threshold
        prev_epsilon = prev_state.get("epsilon", {})
        prev_lambda_IC = prev_state.get("lambda_IC", {})
        epsilon_rel_err = prev_epsilon.get("relative_error_pct")
        lambda_IC_rel_err = prev_lambda_IC.get("relative_error_pct")
        
        both_below_threshold = (
            epsilon_rel_err is not None and 
            lambda_IC_rel_err is not None and
            epsilon_rel_err < decay_relative_error_threshold and
            lambda_IC_rel_err < decay_relative_error_threshold
        )
        
        if both_below_threshold:
            new_step_size = max(current_max_step_size * (decay_rate ** iteration_idx), min_step_size)
            if new_step_size < current_max_step_size:
                reduction_reason = f"exponential decay (iter {iteration_idx}, rate={decay_rate}, both errors < {decay_relative_error_threshold}%)"
        else:
            # Don't apply exponential decay if either error is above threshold
            if epsilon_rel_err is not None and lambda_IC_rel_err is not None:
                reduction_reason = f"exponential decay skipped (epsilon={epsilon_rel_err:.1f}%, IC={lambda_IC_rel_err:.1f}%, threshold={decay_relative_error_threshold}%)"
    
    return new_step_size, reduction_reason

# ==========================
# Shared Newton update logic (extracted from both reduce functions)
# ==========================
def _apply_newton_update(g_mean, B_mean, max_step_size=None):
    """
    Apply spectral conditioning and Newton update to compute parameter change.
    
    Parameters
    ----------
    g_mean : (M,) array
        Mean gradient vector
    B_mean : (M, M) array
        Mean Hessian matrix
    max_step_size : float, optional
        Maximum allowed step size per parameter. If None, uses default adaptive scaling.
    
    Returns
    -------
    delta_vec : (M,) array
        Parameter update vector
    meta : dict
        Metadata about the update (gamma, regularization, etc.)
    """
    M = B_mean.shape[0]
    w = np.linalg.eigvalsh(B_mean)
    lam_min, lam_max = w[0], w[-1]
    kappa_raw = lam_max / max(abs(lam_min), 1e-12)

    # Adaptive target condition number based on severity of ill-conditioning
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
    max_change_per_param = max_step_size if max_step_size is not None else 0.5
    max_proposed_change = np.max(np.abs(delta_vec))
    if max_proposed_change > 0:
        adaptive_gamma = min(max_change_per_param / max_proposed_change, GAMMA)
    else:
        adaptive_gamma = GAMMA
    
    delta_vec *= adaptive_gamma
    
    kappa_after = (lam_max + lambda_reg) / (abs(lam_min) + lambda_reg)
    
    meta = {
        "gamma_base": GAMMA,
        "gamma_adaptive": float(adaptive_gamma),
        "max_proposed_change": float(max_proposed_change),
        "max_change_per_param": float(max_change_per_param),
        "lambda_reg": float(lambda_reg),
        "kappa_raw": float(kappa_raw),
        "kappa_target": float(kappa_target),
        "kappa_after": float(kappa_after),
        "lambda_min": float(lam_min),
        "lambda_max": float(lam_max),
        "eps_floor": float(eps_floor),
    }
    
    return delta_vec, meta

# ==========================
# Adam optimizer update logic
# ==========================
def _apply_adam_update(
    g_mean,
    iteration,
    m_prev=None,
    v_prev=None,
    learning_rate=ADAM_LR_DEFAULT,
    beta1=ADAM_BETA1_DEFAULT,
    beta2=ADAM_BETA2_DEFAULT,
    epsilon=ADAM_EPS_DEFAULT,
    max_step_size=None
):
    """
    Apply Adam optimizer update.
    
    Parameters
    ----------
    g_mean : (M,) array
        Mean gradient vector
    iteration : int
        Current iteration (0-indexed, used for bias correction)
    m_prev : (M,) array, optional
        Previous first moment estimate (momentum). If None, initializes to zeros.
    v_prev : (M,) array, optional
        Previous second moment estimate (RMSprop). If None, initializes to zeros.
    learning_rate : float
        Learning rate (alpha)
    beta1 : float
        Exponential decay rate for first moment
    beta2 : float
        Exponential decay rate for second moment
    epsilon : float
        Small constant for numerical stability
    max_step_size : float, optional
        Maximum allowed step size per parameter (clips update if needed)
    
    Returns
    -------
    delta_vec : (M,) array
        Parameter update vector
    m_new : (M,) array
        Updated first moment estimate
    v_new : (M,) array
        Updated second moment estimate
    meta : dict
        Metadata about the update
    """
    M = g_mean.shape[0]
    
    # Initialize momentum terms if not provided
    if m_prev is None:
        m_prev = np.zeros(M, dtype=float)
    if v_prev is None:
        v_prev = np.zeros(M, dtype=float)
    
    # Adam update equations
    # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    m_new = beta1 * m_prev + (1 - beta1) * g_mean
    
    # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    v_new = beta2 * v_prev + (1 - beta2) * (g_mean ** 2)
    
    # Bias correction (iteration is 0-indexed, so use iteration + 1)
    t = iteration + 1
    m_hat = m_new / (1 - beta1 ** t)
    v_hat = v_new / (1 - beta2 ** t)
    
    # Compute update: delta = -alpha * m_hat / (sqrt(v_hat) + epsilon)
    v_hat_sqrt = np.sqrt(v_hat) + epsilon
    delta_vec = -learning_rate * m_hat / v_hat_sqrt
    
    # Apply step size constraint if provided
    max_proposed_change = np.max(np.abs(delta_vec))
    if max_step_size is not None and max_proposed_change > max_step_size:
        scale_factor = max_step_size / max_proposed_change
        delta_vec *= scale_factor
        max_proposed_change = max_step_size
    
    meta = {
        "learning_rate": float(learning_rate),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "epsilon": float(epsilon),
        "max_proposed_change": float(max_proposed_change),
        "max_change_per_param": float(max_step_size) if max_step_size is not None else None,
        "m_norm": float(np.linalg.norm(m_new)),
        "v_norm": float(np.linalg.norm(v_new)),
    }
    
    return delta_vec, m_new, v_new, meta

# ==========================
# Relative step capping
# ==========================
def _apply_relative_step_cap(delta_vec, param_vec, target_rms_frac=None, max_frac=None, label=""):
    """
    Rescale/clamp update relative to parameter magnitudes.
    - target_rms_frac: scale update so rms(delta) <= target_rms_frac * rms(param)
    - max_frac: clip each element to max_frac * max(|param_i|, rms(param)) to allow motion near zero params
    """
    if delta_vec is None or param_vec is None or delta_vec.size == 0:
        return delta_vec

    delta_vec = delta_vec.copy()
    param_rms = float(np.sqrt(np.mean(param_vec ** 2))) if param_vec.size > 0 else 0.0
    safe_scale = max(param_rms, 1e-12)

    if target_rms_frac is not None:
        target_rms = target_rms_frac * safe_scale
        delta_rms = float(np.sqrt(np.mean(delta_vec ** 2)))
        if delta_rms > target_rms and delta_rms > 0:
            scale = target_rms / delta_rms
            delta_vec *= scale
            print(f"[RELSTEP {label}] Scaled to target RMS: {delta_rms:.3e} -> {target_rms:.3e} (scale={scale:.3f})")

    if max_frac is not None:
        cap_vec = max_frac * np.maximum(np.abs(param_vec), safe_scale)
        before = float(np.max(np.abs(delta_vec)))
        delta_vec = np.clip(delta_vec, -cap_vec, cap_vec)
        after = float(np.max(np.abs(delta_vec)))
        if after < before - 1e-12:
            print(f"[RELSTEP {label}] Clipped per-param to {max_frac*100:.1f}% of |param| (or rms); max |Δ| {before:.3e} -> {after:.3e}")

    return delta_vec

# ==========================
# Combined worker function
# ==========================
def _process_replicate_entry_tkl_IC(
    rep_idx, 
    replicate_root, 
    output_dir, 
    monomer_types,
    exp_Tkl_path,
    exp_phi_IC_path,
    d_init,
    d_end,
    mu, 
    rc, 
    rcut, 
    beta, 
    manifest_path, 
    resolution=None
):
    """
    Process one replicate to compute both TKL and IC observables from the same trajectory.
    """
    rep_str, traj_path = _rep_dir_and_path(replicate_root, rep_idx)
    out_npz_tkl = os.path.join(output_dir, f"{rep_str}_upper_grad_hess.npz")
    out_npz_ic  = os.path.join(output_dir, f"{rep_str}_IC_grad_hess.npz")
    out_touch   = os.path.join(output_dir, f"{rep_str}.READY")

    if os.path.exists(out_npz_tkl) and os.path.exists(out_npz_ic):
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
        # Read trajectory once (shared by both observables)
        t_read0 = time.time()
        if _IO_SEMA is not None: _IO_SEMA.acquire()
        try:
            positions = load_all_positions(traj_path)
        finally:
            if _IO_SEMA is not None: _IO_SEMA.release()
        read_s = time.time() - t_read0

        # Process both observables
        t_comp0 = time.time()
        
        # Process TKL observable
        out_tkl = None
        if not os.path.exists(out_npz_tkl):
            out_tkl = process_one_replicate(
                positions=positions,
                monomer_types=monomer_types,
                exp_Tkl_path=exp_Tkl_path,
                mu=mu, rc=rc, rcut=rcut, beta=beta,
                resolution=resolution
            )
        
        # Process IC observable
        out_ic = None
        if not os.path.exists(out_npz_ic):
            out_ic = process_one_replicate_IC(
                positions=positions,
                exp_phi_IC_path=exp_phi_IC_path,
                d_init=d_init,
                d_end=d_end,
                mu=mu, rc=rc, rcut=rcut, beta=beta
            )
        
        compute_s = time.time() - t_comp0

        # Write outputs
        t_wr0 = time.time()
        if _IO_SEMA is not None: _IO_SEMA.acquire()
        try:
            # Save TKL results
            if out_tkl is not None:
                save_dict_tkl = {
                    "grad_vec": out_tkl["grad_vec"],
                    "Hess_upper": out_tkl["Hess_upper"],
                    "upper_indices_row": out_tkl["upper_indices"][0],
                    "upper_indices_col": out_tkl["upper_indices"][1],
                    "mu": out_tkl["mu"],
                    "rc": out_tkl["rc"],
                    "rcut": out_tkl["rcut"],
                    "rep": rep_idx,
                    "resolution": out_tkl.get("resolution"),
                    "type_labels": out_tkl["type_labels"],
                    "K": len(out_tkl["type_labels"]),
                }
                if out_tkl.get("resolution") == "500kb" and "N_500kb" in out_tkl:
                    save_dict_tkl["N_500kb"] = out_tkl["N_500kb"]
                np.savez_compressed(out_npz_tkl, **save_dict_tkl)
            
            # Save IC results
            if out_ic is not None:
                np.savez_compressed(
                    out_npz_ic,
                    grad_vec=out_ic["grad_vec"],
                    Hess=out_ic["Hess"],
                    phi_sim=out_ic["phi_sim"],
                    dmax=out_ic["dmax"],
                    d_init=out_ic["d_init"],
                    d_end=out_ic["d_end"],
                    mu=out_ic["mu"], rc=out_ic["rc"], rcut=out_ic["rcut"],
                    rep=rep_idx,
                )
            
            # Mark as ready
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
# Combined reduce function
# ==========================
def reduce_and_update_both(
    output_dir, 
    epsilon_dir, 
    lambda_dir, 
    beta=BETA_DEFAULT, 
    iteration_idx=None,
    max_lambda_step_size=None,
    gradient_normalization=None,
    method="newton",  # "newton" or "adam"
    adam_lr=None,
    adam_beta1=None,
    adam_beta2=None,
    adam_epsilon=None,
    adaptive_step_size_config=None,  # Dict with reduction strategy config
    use_separate_updates=False,  # If True, update TKL and IC separately (no cross-covariance)
    adam_lr_ic=None,  # Separate learning rate for IC (if None, uses adam_lr)
    relstep_target_frac=None,
    relstep_max_frac=None
):
    """
    Read all repXX_*_grad_hess.npz files, aggregate gradients and Hessians,
    apply a single joint Newton update to both epsilon and lambda_IC, and save updated parameters.
    
    This function performs a joint update that includes cross-covariance information between
    TKL and IC gradients, making the Hessian more expressive and accounting for correlations
    between epsilon and lambda_IC parameters.
    
    Parameters
    ----------
    output_dir : str
        Directory containing rep??_upper_grad_hess.npz and rep??_IC_grad_hess.npz files
    epsilon_dir : str
        Directory for epsilon versioning (epsilon_tk_*.npy)
    lambda_dir : str
        Directory for lambda_IC versioning (lambda_IC_tk_*.npy)
    beta : float
        MaxEnt beta parameter
    iteration_idx : int, optional
        Current iteration index for proper versioning
    max_lambda_step_size : float, optional
        Maximum allowed step size per parameter (applies to both epsilon and lambda_IC)
    """
    # Process TKL/epsilon updates
    files_tkl = sorted(glob.glob(os.path.join(output_dir, "rep??_upper_grad_hess.npz")))
    files_ic = sorted(glob.glob(os.path.join(output_dir, "rep??_IC_grad_hess.npz")))
    
    if len(files_tkl) == 0:
        raise RuntimeError(f"No TKL per-replicate .npz found in {output_dir}")
    if len(files_ic) == 0:
        raise RuntimeError(f"No IC per-replicate .npz found in {output_dir}")

    # Relative step parameters (enable by default)
    relstep_target_frac = RELSTEP_TARGET_DEFAULT if relstep_target_frac is None else relstep_target_frac
    relstep_max_frac = RELSTEP_MAX_FRAC_DEFAULT if relstep_max_frac is None else relstep_max_frac
    
    print(f"[REDUCE] Processing {len(files_tkl)} TKL replicates and {len(files_ic)} IC replicates")
    
    # ========== Apply adaptive step size reduction ==========
    # Try to load adaptive config from JSON file if not provided
    if adaptive_step_size_config is None:
        adaptive_config_path = os.path.join(output_dir, "adaptive_step_size_config.json")
        if os.path.exists(adaptive_config_path):
            try:
                with open(adaptive_config_path, 'r') as f:
                    adaptive_step_size_config = json.load(f)
            except Exception as e:
                print(f"[WARNING] Could not load adaptive_step_size_config.json: {e}")

    original_max_step_size = max_lambda_step_size

    # ========== Compute relative-error-based step sizes (separate for TKL and IC) ==========
    max_step_tkl, max_step_ic, rel_err_scaling_info = _compute_relative_error_step_sizes(
        max_lambda_step_size,
        iteration_idx,
        output_dir,
        adaptive_step_size_config
    )
    
    if rel_err_scaling_info.get("reason") == "relative_error_scaling":
        print(f"[RELATIVE ERROR STEP SIZE] TKL: {rel_err_scaling_info['rel_err_tkl']:.1f}% -> step_size={max_step_tkl:.6f} (scale={rel_err_scaling_info['scale_tkl']:.3f})")
        print(f"[RELATIVE ERROR STEP SIZE] IC: {rel_err_scaling_info['rel_err_ic']:.1f}% -> step_size={max_step_ic:.6f} (scale={rel_err_scaling_info['scale_ic']:.3f})")
    else:
        # No relative error scaling, use same step size for both
        max_step_tkl = max_lambda_step_size
        max_step_ic = max_lambda_step_size
    
    # ========== Apply global adaptive step size reduction (if enabled) ==========
    # This reduces both TKL and IC step sizes together (for fine-tuning near convergence)
    if adaptive_step_size_config is not None and max_step_tkl is not None and max_step_ic is not None:
        # Apply reduction to both (they'll be scaled proportionally)
        max_step_tkl_reduced, reduction_reason_tkl = _compute_adaptive_step_size(
            max_step_tkl,
            iteration_idx,
            output_dir,
            adaptive_step_size_config
        )
        max_step_ic_reduced, reduction_reason_ic = _compute_adaptive_step_size(
            max_step_ic,
            iteration_idx,
            output_dir,
            adaptive_step_size_config
        )
        
        if reduction_reason_tkl or reduction_reason_ic:
            if reduction_reason_tkl:
                print(f"[ADAPTIVE STEP SIZE] Reduced TKL max_step_size: {max_step_tkl:.6f} -> {max_step_tkl_reduced:.6f}")
                print(f"[ADAPTIVE STEP SIZE] Reason: {reduction_reason_tkl}")
            if reduction_reason_ic:
                print(f"[ADAPTIVE STEP SIZE] Reduced IC max_step_size: {max_step_ic:.6f} -> {max_step_ic_reduced:.6f}")
                print(f"[ADAPTIVE STEP SIZE] Reason: {reduction_reason_ic}")
            max_step_tkl = max_step_tkl_reduced
            max_step_ic = max_step_ic_reduced

    # Use the computed step sizes (either from relative error scaling or base value)
    max_lambda_step_size_tkl = max_step_tkl
    max_lambda_step_size_ic = max_step_ic
    
    # ========== Update epsilon (TKL) ==========
    print("\n[REDUCE] Updating epsilon (TKL)...")
    grads_tkl = []
    Hlist_tkl = []
    iu_row = iu_col = None
    K = None
    for fpath in files_tkl:
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
        grads_tkl.append(grad_vec)
        Hlist_tkl.append(Hess_up)

    grads_tkl = np.stack(grads_tkl, axis=0)       # (R, M)
    Hlist_tkl = np.stack(Hlist_tkl, axis=0)       # (R, M, M)
    g_mean_tkl = grads_tkl.mean(axis=0)           # (M,)
    B_mean_tkl = Hlist_tkl.mean(axis=0)           # (M, M)
    
    # Create phi_mean.npy for TKL
    exp_targets_path = None
    for potential_path in [
        os.path.join(output_dir, "..", "..", "exp_targets", "T_type_kl.npy"),
        os.path.join(output_dir, "T_type_kl.npy")
    ]:
        if os.path.exists(potential_path):
            exp_targets_path = potential_path
            break
    
    if exp_targets_path:
        try:
            T_exp_full = np.load(exp_targets_path)
            if T_exp_full.ndim == 2:
                T_exp_vec, _ = _flatten_upper(T_exp_full)
            else:
                T_exp_vec = T_exp_full
            
            phi_sims_tkl = []
            for fpath in files_tkl:
                z = np.load(fpath)
                grad_vec = z["grad_vec"]
                phi_sim = T_exp_vec - (grad_vec / beta)
                phi_sims_tkl.append(phi_sim)
            
            phi_mean_tkl = np.mean(phi_sims_tkl, axis=0)
            phi_mean_path = os.path.join(output_dir, "phi_mean.npy")
            np.save(phi_mean_path, phi_mean_tkl)
            print(f"[REDUCE TKL] Created phi_mean.npy from {len(files_tkl)} replicates")
            
            if len(phi_sims_tkl) > 1:
                phi_sims_array = np.array(phi_sims_tkl)
                phi_cov_diag = np.var(phi_sims_array, axis=0, ddof=1)
                phi_cov_path = os.path.join(output_dir, "phi_cov_diag.npy")
                np.save(phi_cov_path, phi_cov_diag)
        except Exception as e:
            print(f"[WARNING] Failed to create phi_mean.npy for TKL: {e}")

    # ========== Update lambda_IC ==========
    print("\n[REDUCE] Updating lambda_IC...")
    grads_ic = []
    Hlist_ic = []
    dmax = None
    for fpath in files_ic:
        z = np.load(fpath)
        grad_vec  = z["grad_vec"]
        Hess      = z["Hess"]
        this_dmax = int(z["dmax"])
        if dmax is None:
            dmax = this_dmax
        elif this_dmax != dmax:
            raise ValueError(f"Inconsistent dmax in {fpath}: {this_dmax} vs {dmax}")
        grads_ic.append(grad_vec)
        Hlist_ic.append(Hess)

    grads_ic = np.stack(grads_ic, axis=0)       # (R, dmax)
    Hlist_ic = np.stack(Hlist_ic, axis=0)       # (R, dmax, dmax)
    g_mean_ic = grads_ic.mean(axis=0)           # (dmax,)
    B_mean_ic = Hlist_ic.mean(axis=0)           # (dmax, dmax)

    # Load current parameters for relative step capping
    epsilon_old_path = _find_latest_epsilon(Path(epsilon_dir))
    epsilon_old = np.load(epsilon_old_path)
    if epsilon_old.shape != (K, K):
        raise ValueError(f"epsilon_old shape {epsilon_old.shape} != ({K},{K})")
    epsilon_vec, _ = _flatten_upper(epsilon_old)

    lambda_old_path = _find_latest_lambda_IC(Path(lambda_dir))
    lambda_old = np.load(lambda_old_path)
    if lambda_old.shape[0] != dmax:
        raise ValueError(f"lambda_old shape {lambda_old.shape} != ({dmax},)")
    
    # ========== Choose Update Strategy ==========
    if use_separate_updates:
        print("\n[REDUCE] Using SEPARATE updates (no cross-covariance, independent control)")
        # For separate updates, we don't need cross-covariance
        B_cross = None
        M_tkl = B_mean_tkl.shape[0]
        M_ic = B_mean_ic.shape[0]
        B_joint = None  # Not used for separate updates
    else:
        # ========== Joint Update: Compute Cross-Covariance and Build Joint Hessian ==========
        print("\n[REDUCE] Computing joint update with cross-covariance...")
        
        # Ensure files are aligned (they should be since both are sorted)
        if len(files_tkl) != len(files_ic):
            raise RuntimeError(f"Mismatch: {len(files_tkl)} TKL files vs {len(files_ic)} IC files")
        
        # Compute cross-covariance: B_cross = Cov(grad_tkl, grad_ic)
        grads_tkl_centered = grads_tkl - g_mean_tkl  # (R, M_tkl)
        grads_ic_centered = grads_ic - g_mean_ic      # (R, M_ic)
        B_cross = (grads_tkl_centered.T @ grads_ic_centered) / len(files_tkl)  # (M_tkl, M_ic)
        
        # Build joint Hessian matrix
        M_tkl = B_mean_tkl.shape[0]  # Number of TKL parameters
        M_ic = B_mean_ic.shape[0]     # Number of IC parameters
        
        # Build joint Hessian: [[B_tkl, B_cross], [B_cross.T, B_ic]]
        B_joint = np.zeros((M_tkl + M_ic, M_tkl + M_ic), dtype=float)
        B_joint[:M_tkl, :M_tkl] = B_mean_tkl
        B_joint[:M_tkl, M_tkl:] = B_cross
        B_joint[M_tkl:, :M_tkl] = B_cross.T
        B_joint[M_tkl:, M_tkl:] = B_mean_ic
    
    # Apply gradient normalization if requested
    if gradient_normalization and gradient_normalization.upper() == "L2":
        # L2 normalization: normalize each gradient vector by its L2 norm
        norm_tkl = np.linalg.norm(g_mean_tkl)
        norm_ic = np.linalg.norm(g_mean_ic)
        
        eps_norm = 1e-10
        
        # Normalize gradients
        if norm_tkl > eps_norm:
            g_mean_tkl_normalized = g_mean_tkl / norm_tkl
            norm_tkl_for_hessian = norm_tkl
        else:
            g_mean_tkl_normalized = g_mean_tkl.copy()
            norm_tkl_for_hessian = 1.0
        
        if norm_ic > eps_norm:
            g_mean_ic_normalized = g_mean_ic / norm_ic
            norm_ic_for_hessian = norm_ic
        else:
            g_mean_ic_normalized = g_mean_ic.copy()
            norm_ic_for_hessian = 1.0
        
        # Normalize Hessian blocks to match (if g -> g/α, then B -> B/α²)
        B_mean_tkl_normalized = B_mean_tkl / (norm_tkl_for_hessian ** 2)
        B_mean_ic_normalized = B_mean_ic / (norm_ic_for_hessian ** 2)
        B_cross_normalized = B_cross / (norm_tkl_for_hessian * norm_ic_for_hessian)
        
        # Rebuild normalized joint Hessian
        B_joint_normalized = np.zeros((M_tkl + M_ic, M_tkl + M_ic), dtype=float)
        B_joint_normalized[:M_tkl, :M_tkl] = B_mean_tkl_normalized
        B_joint_normalized[:M_tkl, M_tkl:] = B_cross_normalized
        B_joint_normalized[M_tkl:, :M_tkl] = B_cross_normalized.T
        B_joint_normalized[M_tkl:, M_tkl:] = B_mean_ic_normalized
        
        # Build normalized joint gradient
        g_joint = np.concatenate([g_mean_tkl_normalized, g_mean_ic_normalized])
        
        # Use normalized versions for Newton update
        B_joint = B_joint_normalized
        
        print(f"[NORMALIZATION] L2 normalization applied")
        print(f"[NORMALIZATION] TKL norm: {norm_tkl:.2e} -> {np.linalg.norm(g_mean_tkl_normalized):.2e}")
        print(f"[NORMALIZATION] IC norm: {norm_ic:.2e} -> {np.linalg.norm(g_mean_ic_normalized):.2e}")
        print(f"[NORMALIZATION] Original TKL/IC norm ratio: {norm_tkl/norm_ic:.2e}")
        
        # Store normalization factors for denormalizing updates later
        norm_factors_tkl = norm_tkl_for_hessian
        norm_factors_ic = norm_ic_for_hessian
    else:
        # No normalization: use original gradients and Hessian
        norm_factors_tkl = None
        norm_factors_ic = None
        # Ensure normalized variables exist (set to original if not normalizing)
        g_mean_tkl_normalized = g_mean_tkl
        g_mean_ic_normalized = g_mean_ic
    
    # ========== Apply update based on method ==========
    if method and method.lower() == "adam":
        # Adam optimization
        print(f"[ADAM] Using Adam optimizer")
        
        # Load previous Adam state if available
        adam_state_tkl_path = os.path.join(epsilon_dir, "adam_state_tkl.npz")
        adam_state_ic_path = os.path.join(lambda_dir, "adam_state_ic.npz")
        
        m_tkl_prev = None
        v_tkl_prev = None
        m_ic_prev = None
        v_ic_prev = None
        
        if os.path.exists(adam_state_tkl_path):
            state_tkl = np.load(adam_state_tkl_path)
            m_tkl_prev = state_tkl.get("m", None)
            v_tkl_prev = state_tkl.get("v", None)
        
        if os.path.exists(adam_state_ic_path):
            state_ic = np.load(adam_state_ic_path)
            m_ic_prev = state_ic.get("m", None)
            v_ic_prev = state_ic.get("v", None)
        
        # Use default Adam hyperparameters if not provided
        lr = adam_lr if adam_lr is not None else ADAM_LR_DEFAULT
        beta1 = adam_beta1 if adam_beta1 is not None else ADAM_BETA1_DEFAULT
        beta2 = adam_beta2 if adam_beta2 is not None else ADAM_BETA2_DEFAULT
        eps = adam_epsilon if adam_epsilon is not None else ADAM_EPS_DEFAULT
        
        iter_num = iteration_idx if iteration_idx is not None else 0
        
        # Apply Adam update separately for TKL and IC (Adam naturally handles scale differences)
        # Note: With L2 normalization, we still normalize gradients but Adam adapts per parameter
        if gradient_normalization and gradient_normalization.upper() == "L2" and norm_factors_tkl is not None:
            # Use already normalized gradients if normalization was applied
            g_mean_tkl_adam = g_mean_tkl_normalized
            g_mean_ic_adam = g_mean_ic_normalized
            print(f"[ADAM] Using L2-normalized gradients for Adam update")
        else:
            g_mean_tkl_adam = g_mean_tkl
            g_mean_ic_adam = g_mean_ic
        
        # Determine IC learning rate: use explicit value, or adapt based on gradient magnitudes
        if adam_lr_ic is not None:
            # Use explicitly provided IC learning rate
            lr_ic = adam_lr_ic
            print(f"[ADAM] Using separate learning rates: TKL={lr:.4f}, IC={lr_ic:.4f} (IC is {lr_ic/lr:.1f}x TKL)")
        else:
            # Adaptive learning rate: scale IC LR based on gradient magnitude ratio
            # Goal: make update magnitudes more similar despite different gradient scales
            norm_tkl_raw = np.linalg.norm(g_mean_tkl) if not (gradient_normalization and gradient_normalization.upper() == "L2") else norm_factors_tkl
            norm_ic_raw = np.linalg.norm(g_mean_ic) if not (gradient_normalization and gradient_normalization.upper() == "L2") else norm_factors_ic
            
            if norm_tkl_raw > 1e-10 and norm_ic_raw > 1e-10:
                # Scale IC LR to compensate for gradient scale difference
                # This makes update magnitudes more similar
                gradient_scale_ratio = norm_tkl_raw / norm_ic_raw
                # Cap the scaling to avoid extreme values (max 20x)
                adaptive_lr_multiplier = min(gradient_scale_ratio, 20.0)
                lr_ic = lr * adaptive_lr_multiplier
                print(f"[ADAM] Adaptive IC learning rate: TKL={lr:.4f}, IC={lr_ic:.4f} (gradient norm ratio={gradient_scale_ratio:.1f}x, multiplier={adaptive_lr_multiplier:.1f}x)")
            else:
                # Fallback: use same LR if gradients are too small
                lr_ic = lr
                print(f"[ADAM] Using same learning rate for both (gradients too small for adaptive scaling)")
        
        # Apply Adam update to TKL
        # If normalization is used, don't apply step size constraint yet - we'll do it after denormalization
        max_step_size_for_adam_tkl = None if (gradient_normalization and gradient_normalization.upper() == "L2" and norm_factors_tkl is not None) else max_lambda_step_size_tkl
        delta_vec_tkl, m_tkl_new, v_tkl_new, meta_tkl = _apply_adam_update(
            g_mean_tkl_adam,
            iter_num,
            m_prev=m_tkl_prev,
            v_prev=v_tkl_prev,
            learning_rate=lr,
            beta1=beta1,
            beta2=beta2,
            epsilon=eps,
            max_step_size=max_step_size_for_adam_tkl
        )
        max_step_size_for_adam_ic = None if (gradient_normalization and gradient_normalization.upper() == "L2" and norm_factors_ic is not None) else max_lambda_step_size_ic
        delta_vec_ic, m_ic_new, v_ic_new, meta_ic = _apply_adam_update(
            g_mean_ic_adam,
            iter_num,
            m_prev=m_ic_prev,
            v_prev=v_ic_prev,
            learning_rate=lr_ic,  # Use separate LR for IC
            beta1=beta1,
            beta2=beta2,
            epsilon=eps,
            max_step_size=max_step_size_for_adam_ic
        )
        
        # Denormalize if normalization was applied
        if gradient_normalization and gradient_normalization.upper() == "L2" and norm_factors_tkl is not None:
            if norm_factors_tkl > 1e-10:
                delta_vec_tkl = delta_vec_tkl * norm_factors_tkl
            if norm_factors_ic > 1e-10:
                delta_vec_ic = delta_vec_ic * norm_factors_ic
            print(f"[ADAM] Denormalized updates: TKL scale={norm_factors_tkl:.2e}, IC scale={norm_factors_ic:.2e}")
            
            # CRITICAL: Re-apply step size constraint in parameter space after denormalization
            # The constraint was applied in normalized space, but after denormalization it may be violated
            # Use separate step sizes for TKL and IC (from relative error scaling)
            max_change_per_param_tkl = max_lambda_step_size_tkl
            max_change_per_param_ic = max_lambda_step_size_ic
            max_proposed_change_tkl = np.max(np.abs(delta_vec_tkl))
            max_proposed_change_ic = np.max(np.abs(delta_vec_ic))
            
            # For Adam with L2 normalization, IC updates can be too small because:
            # 1. L2 normalization equalizes gradients to unit vectors
            # 2. Adam computes conservative updates in normalized space
            # 3. After denormalizing by small norm_factors_ic, updates remain tiny
            # Solution: Scale IC updates to make them more similar to TKL updates
            if norm_factors_tkl is not None and norm_factors_ic is not None:
                if norm_factors_tkl > 1e-10 and norm_factors_ic > 1e-10:
                    # Strategy 1: Scale by original gradient norm ratio (compensate for normalization)
                    original_scale_ratio = norm_factors_tkl / norm_factors_ic
                    
                    # Strategy 2: Scale to match TKL update magnitude (make updates similar)
                    # This ensures both parameters update at similar rates
                    tkl_update_magnitude = np.max(np.abs(delta_vec_tkl))
                    ic_update_magnitude = np.max(np.abs(delta_vec_ic))
                    
                    if ic_update_magnitude > 1e-10:
                        # Scale IC to match TKL magnitude (but don't exceed max_step_size)
                        magnitude_ratio = tkl_update_magnitude / ic_update_magnitude
                        # Use the minimum of both strategies, capped appropriately
                        ic_compensation_scale = min(original_scale_ratio, magnitude_ratio, 50.0)
                        delta_vec_ic *= ic_compensation_scale
                        max_proposed_change_ic = np.max(np.abs(delta_vec_ic))
                        print(f"[ADAM] Balanced IC updates: scaled by {ic_compensation_scale:.2f} (norm_ratio={original_scale_ratio:.2f}, magnitude_ratio={magnitude_ratio:.2f})")
                        print(f"[ADAM] Update magnitudes after balancing: TKL={tkl_update_magnitude:.6f}, IC={max_proposed_change_ic:.6f}, ratio={tkl_update_magnitude/max_proposed_change_ic:.2f}")

            # Relative step cap (RMS + per-parameter fraction of current params)
            delta_vec_tkl = _apply_relative_step_cap(delta_vec_tkl, epsilon_vec, relstep_target_frac, relstep_max_frac, label="TKL")
            delta_vec_ic = _apply_relative_step_cap(delta_vec_ic, lambda_old, relstep_target_frac, relstep_max_frac, label="IC")
            max_proposed_change_tkl = np.max(np.abs(delta_vec_tkl))
            max_proposed_change_ic = np.max(np.abs(delta_vec_ic))
            
            if max_change_per_param_tkl is not None and max_proposed_change_tkl > max_change_per_param_tkl:
                scale_factor_tkl = max_change_per_param_tkl / max_proposed_change_tkl
                delta_vec_tkl *= scale_factor_tkl
                print(f"[ADAM] Re-applied step size constraint to TKL: {max_proposed_change_tkl:.6f} -> {np.max(np.abs(delta_vec_tkl)):.6f} (scale={scale_factor_tkl:.3f}, max_step={max_change_per_param_tkl:.6f})")
            
            if max_change_per_param_ic is not None and max_proposed_change_ic > max_change_per_param_ic:
                scale_factor_ic = max_change_per_param_ic / max_proposed_change_ic
                delta_vec_ic *= scale_factor_ic
                print(f"[ADAM] Re-applied step size constraint to IC: {max_proposed_change_ic:.6f} -> {np.max(np.abs(delta_vec_ic)):.6f} (scale={scale_factor_ic:.3f}, max_step={max_change_per_param_ic:.6f})")
            
            # Update metadata with final values
            meta_tkl['max_proposed_change'] = float(np.max(np.abs(delta_vec_tkl)))
            meta_ic['max_proposed_change'] = float(np.max(np.abs(delta_vec_ic)))
        
        else:
            # No normalization path: apply relative cap and optional absolute cap
            delta_vec_tkl = _apply_relative_step_cap(delta_vec_tkl, epsilon_vec, relstep_target_frac, relstep_max_frac, label="TKL")
            delta_vec_ic = _apply_relative_step_cap(delta_vec_ic, lambda_old, relstep_target_frac, relstep_max_frac, label="IC")
            max_proposed_change_tkl = np.max(np.abs(delta_vec_tkl))
            max_proposed_change_ic = np.max(np.abs(delta_vec_ic))

            if max_lambda_step_size_tkl is not None and max_proposed_change_tkl > max_lambda_step_size_tkl:
                scale_factor_tkl = max_lambda_step_size_tkl / max_proposed_change_tkl
                delta_vec_tkl *= scale_factor_tkl
                print(f"[ADAM] Re-applied step size constraint to TKL: {max_proposed_change_tkl:.6f} -> {np.max(np.abs(delta_vec_tkl)):.6f} (scale={scale_factor_tkl:.3f}, max_step={max_lambda_step_size_tkl:.6f})")

            if max_lambda_step_size_ic is not None and max_proposed_change_ic > max_lambda_step_size_ic:
                scale_factor_ic = max_lambda_step_size_ic / max_proposed_change_ic
                delta_vec_ic *= scale_factor_ic
                print(f"[ADAM] Re-applied step size constraint to IC: {max_proposed_change_ic:.6f} -> {np.max(np.abs(delta_vec_ic)):.6f} (scale={scale_factor_ic:.3f}, max_step={max_lambda_step_size_ic:.6f})")

            meta_tkl['max_proposed_change'] = float(np.max(np.abs(delta_vec_tkl)))
            meta_ic['max_proposed_change'] = float(np.max(np.abs(delta_vec_ic)))

        # Save Adam state for next iteration
        np.savez(adam_state_tkl_path, m=m_tkl_new, v=v_tkl_new)
        np.savez(adam_state_ic_path, m=m_ic_new, v=v_ic_new)
        
        print(f"[ADAM TKL] max_proposed_change={meta_tkl['max_proposed_change']:.6f}, lr={lr:.4f}")
        print(f"[ADAM IC] max_proposed_change={meta_ic['max_proposed_change']:.6f}, lr={lr:.4f}")
        
    else:
        # Newton optimization (existing code)
        # Build joint gradient
        if gradient_normalization and gradient_normalization.upper() == "L2":
            # Use already normalized gradients and Hessian
            g_joint = np.concatenate([g_mean_tkl_normalized, g_mean_ic_normalized])
            # Don't apply step size constraint in normalized space - we'll do it in parameter space after denormalization
            # Pass a very large value so _apply_newton_update applies only base GAMMA scaling (no additional constraint)
            # Then we'll undo that, denormalize, and apply the real constraint in parameter space
            max_step_size_normalized = 1e10  # Very large value so adaptive_gamma = GAMMA (no extra constraint)
        else:
            g_joint = np.concatenate([g_mean_tkl, g_mean_ic])
            max_step_size_normalized = max_lambda_step_size
        
        # Apply single Newton update to joint system
        delta_joint, meta_joint = _apply_newton_update(g_joint, B_joint, max_step_size_normalized)
        
        # Split the joint update back into TKL and IC components
        delta_vec_tkl_raw = delta_joint[:M_tkl]
        delta_vec_ic_raw = delta_joint[M_tkl:]
        
        # Denormalize updates if normalization was applied
        if norm_factors_tkl is not None and norm_factors_ic is not None:
            # The delta from _apply_newton_update has adaptive_gamma applied in normalized space
            # Since we passed a very large max_step_size, adaptive_gamma = GAMMA (no extra constraint)
            # Undo that GAMMA scaling, denormalize, then apply constraint in parameter space
            
            # Undo the GAMMA that was applied in normalized space
            gamma_applied = meta_joint.get('gamma_adaptive', GAMMA)
            if gamma_applied > 1e-10:
                delta_vec_tkl_raw = delta_vec_tkl_raw / gamma_applied
                delta_vec_ic_raw = delta_vec_ic_raw / gamma_applied
            
            # Denormalize: if we normalized g -> g/α and B -> B/α², then:
            # delta_normalized = -B_normalized^{-1} * g_normalized = α * delta_original
            # Therefore: delta_original = delta_normalized / α
            delta_vec_tkl = delta_vec_tkl_raw / norm_factors_tkl if norm_factors_tkl > 1e-10 else delta_vec_tkl_raw
            delta_vec_ic = delta_vec_ic_raw / norm_factors_ic if norm_factors_ic > 1e-10 else delta_vec_ic_raw
            print(f"[NORMALIZATION] Denormalized updates: TKL scale=1/{norm_factors_tkl:.2e}, IC scale=1/{norm_factors_ic:.2e}")
            
            # Relative step cap before applying adaptive gamma
            delta_vec_tkl = _apply_relative_step_cap(delta_vec_tkl, epsilon_vec, relstep_target_frac, relstep_max_frac, label="TKL")
            delta_vec_ic = _apply_relative_step_cap(delta_vec_ic, lambda_old, relstep_target_frac, relstep_max_frac, label="IC")
            
            # Now compute adaptive_gamma in parameter space using denormalized updates
            # Use separate step sizes for TKL and IC (from relative error scaling)
            max_change_per_param_tkl = max_lambda_step_size_tkl
            max_change_per_param_ic = max_lambda_step_size_ic
            max_proposed_change_tkl = np.max(np.abs(delta_vec_tkl))
            max_proposed_change_ic = np.max(np.abs(delta_vec_ic))
            
            if max_proposed_change_tkl > 0:
                adaptive_gamma_tkl = min(max_change_per_param_tkl / max_proposed_change_tkl, GAMMA) if max_change_per_param_tkl is not None else GAMMA
            else:
                adaptive_gamma_tkl = GAMMA
            
            if max_proposed_change_ic > 0:
                adaptive_gamma_ic = min(max_change_per_param_ic / max_proposed_change_ic, GAMMA) if max_change_per_param_ic is not None else GAMMA
            else:
                adaptive_gamma_ic = GAMMA
            
            # Apply adaptive_gamma to the denormalized updates in parameter space
            delta_vec_tkl *= adaptive_gamma_tkl
            delta_vec_ic *= adaptive_gamma_ic
            
            print(f"[NORMALIZATION] Applied adaptive_gamma in parameter space: TKL={adaptive_gamma_tkl:.3f}, IC={adaptive_gamma_ic:.3f}")
            print(f"[NORMALIZATION] Final max_proposed_change: TKL={np.max(np.abs(delta_vec_tkl)):.6f}, IC={np.max(np.abs(delta_vec_ic)):.6f}")
        else:
            # No normalization: adaptive_gamma already applied in _apply_newton_update
            # But we still need to check if it respects separate step sizes for TKL and IC
            delta_vec_tkl = delta_vec_tkl_raw
            delta_vec_ic = delta_vec_ic_raw
            # Relative step cap before any extra constraint
            delta_vec_tkl = _apply_relative_step_cap(delta_vec_tkl, epsilon_vec, relstep_target_frac, relstep_max_frac, label="TKL")
            delta_vec_ic = _apply_relative_step_cap(delta_vec_ic, lambda_old, relstep_target_frac, relstep_max_frac, label="IC")
            max_proposed_change_tkl = np.max(np.abs(delta_vec_tkl))
            max_proposed_change_ic = np.max(np.abs(delta_vec_ic))
            
            # Re-apply constraints with separate step sizes
            max_change_per_param_tkl = max_lambda_step_size_tkl
            max_change_per_param_ic = max_lambda_step_size_ic
            
            adaptive_gamma_base = meta_joint.get('gamma_adaptive', GAMMA)
            
            if max_change_per_param_tkl is not None and max_proposed_change_tkl > max_change_per_param_tkl:
                adaptive_gamma_tkl = min(max_change_per_param_tkl / max_proposed_change_tkl, GAMMA)
                delta_vec_tkl *= adaptive_gamma_tkl / adaptive_gamma_base
                print(f"[NEWTON] Re-applied step size constraint to TKL: {max_proposed_change_tkl:.6f} -> {np.max(np.abs(delta_vec_tkl)):.6f} (max_step={max_change_per_param_tkl:.6f})")
            else:
                adaptive_gamma_tkl = adaptive_gamma_base
            
            if max_change_per_param_ic is not None and max_proposed_change_ic > max_change_per_param_ic:
                adaptive_gamma_ic = min(max_change_per_param_ic / max_proposed_change_ic, GAMMA)
                delta_vec_ic *= adaptive_gamma_ic / adaptive_gamma_base
                print(f"[NEWTON] Re-applied step size constraint to IC: {max_proposed_change_ic:.6f} -> {np.max(np.abs(delta_vec_ic)):.6f} (max_step={max_change_per_param_ic:.6f})")
            else:
                adaptive_gamma_ic = adaptive_gamma_base
        
        # Extract metadata for separate logging
        meta_tkl = meta_joint.copy()
        meta_ic = meta_joint.copy()
        
        # Update with correct values after denormalization/recomputation
        if norm_factors_tkl is not None:
            meta_tkl['max_proposed_change'] = float(np.max(np.abs(delta_vec_tkl)))
            meta_tkl['gamma_adaptive'] = float(adaptive_gamma_tkl)
            meta_ic['max_proposed_change'] = float(np.max(np.abs(delta_vec_ic)))
            meta_ic['gamma_adaptive'] = float(adaptive_gamma_ic)
        else:
            meta_tkl['max_proposed_change'] = float(max_proposed_change_tkl)
            meta_tkl['gamma_adaptive'] = float(adaptive_gamma_tkl)
            meta_ic['max_proposed_change'] = float(max_proposed_change_ic)
            meta_ic['gamma_adaptive'] = float(adaptive_gamma_ic)
        
        print(f"[NEWTON JOINT] Joint update applied (M_tkl={M_tkl}, M_ic={M_ic}, M_total={M_tkl+M_ic})")
        print(f"[NEWTON TKL] max_proposed_change: {meta_tkl['max_proposed_change']:.3f}")
        print(f"[NEWTON TKL] adaptive_gamma: {meta_tkl['gamma_adaptive']:.3f} (base_gamma: {GAMMA:.3f})")
        print(f"[SPECTRAL TKL] κ_raw: {meta_tkl['kappa_raw']:.2e}, λ_reg: {meta_tkl['lambda_reg']:.2e}, κ_after: {meta_tkl['kappa_after']:.2e}")
        print(f"[NEWTON IC] max_proposed_change: {meta_ic['max_proposed_change']:.3f}")
        print(f"[NEWTON IC] adaptive_gamma: {meta_ic['gamma_adaptive']:.3f} (base_gamma: {GAMMA:.3f})")
        print(f"[SPECTRAL IC] κ_raw: {meta_ic['kappa_raw']:.2e}, λ_reg: {meta_ic['lambda_reg']:.2e}, κ_after: {meta_ic['kappa_after']:.2e}")
    
    # ========== Apply updates to epsilon (TKL) ==========
    # Map to symmetric KxK
    iu = (iu_row, iu_col)
    delta_mat = np.zeros((K, K), float)
    delta_mat[iu] = delta_vec_tkl
    delta_mat = delta_mat + delta_mat.T - np.diag(np.diag(delta_mat))

    epsilon_dir = Path(epsilon_dir)
    epsilon_dir.mkdir(parents=True, exist_ok=True)
    epsilon_old_path = _find_latest_epsilon(epsilon_dir)
    epsilon_old = np.load(epsilon_old_path)
    if epsilon_old.shape != (K, K):
        raise ValueError(f"epsilon_old shape {epsilon_old.shape} != ({K},{K})")
    epsilon_new = epsilon_old + delta_mat

    epsilon_save_path = _next_version_path(epsilon_dir, "epsilon_tk_", ".npy", iteration_idx)
    np.save(epsilon_save_path, epsilon_new)

    epsilon_next_path = os.path.join(output_dir, "epsilon_next.npy")
    np.save(epsilon_next_path, epsilon_new)

    # ========== Apply updates to lambda_IC ==========
    lambda_dir = Path(lambda_dir)
    lambda_dir.mkdir(parents=True, exist_ok=True)
    lambda_old_path = _find_latest_lambda_IC(lambda_dir)
    lambda_old = np.load(lambda_old_path)
    if lambda_old.shape[0] != dmax:
        raise ValueError(f"lambda_old shape {lambda_old.shape} != ({dmax},)")
    lambda_new = lambda_old + delta_vec_ic

    lambda_save_path = _next_version_path(lambda_dir, "lambda_IC_tk_", ".npy", iteration_idx)
    np.save(lambda_save_path, lambda_new)

    lambda_next_path = os.path.join(output_dir, "lambda_IC_next.npy")
    np.save(lambda_next_path, lambda_new)
    
    # Create phi_mean.npy for IC
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
            phi_sims_ic = []
            for fpath in files_ic:
                z = np.load(fpath)
                grad_vec = z["grad_vec"]
                phi_sim = phi_exp - (grad_vec / beta)
                phi_sims_ic.append(phi_sim)
            
            phi_mean_ic = np.mean(phi_sims_ic, axis=0)
            phi_mean_path_ic = os.path.join(output_dir, "phi_mean_IC.npy")
            np.save(phi_mean_path_ic, phi_mean_ic)
            print(f"[REDUCE IC] Created phi_mean_IC.npy from {len(files_ic)} replicates")
            
            if len(phi_sims_ic) > 1:
                phi_sims_array = np.array(phi_sims_ic)
                phi_cov_diag = np.var(phi_sims_array, axis=0, ddof=1)
                phi_cov_path = os.path.join(output_dir, "phi_cov_diag_IC.npy")
                np.save(phi_cov_path, phi_cov_diag)
        except Exception as e:
            print(f"[WARNING] Failed to create phi_mean_IC.npy: {e}")

    # Save metadata
    meta = {
        "epsilon": {
            **meta_tkl,
            "n_replicates": int(len(files_tkl)),
            "K": int(K),
            "epsilon_old_path": str(epsilon_old_path),
            "epsilon_new_path": str(epsilon_save_path),
            "epsilon_next_path": str(epsilon_next_path),
            "B_trace": float(np.trace(B_mean_tkl)),
            "M": int(B_mean_tkl.shape[0]),
        },
        "lambda_IC": {
            **meta_ic,
            "n_replicates": int(len(files_ic)),
            "dmax": int(dmax),
            "lambda_old_path": str(lambda_old_path),
            "lambda_new_path": str(lambda_save_path),
            "lambda_next_path": str(lambda_next_path),
            "B_trace": float(np.trace(B_mean_ic)),
            "M": int(dmax),
        },
        "max_lambda_step_size": float(max_lambda_step_size) if max_lambda_step_size is not None else None,
    }
    with open(os.path.join(output_dir, "reduce_summary.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n[REDUCE] epsilon_old: {epsilon_old_path.name}")
    print(f"[REDUCE] epsilon_new: {epsilon_save_path.name}")
    print(f"[REDUCE] lambda_IC_old: {lambda_old_path.name}")
    print(f"[REDUCE] lambda_IC_new: {lambda_save_path.name}")
    
    return epsilon_save_path, lambda_save_path

# ==========================
# CLI
# ==========================
def parse_args():
    p = argparse.ArgumentParser(description="Process both TKL and IC observables and update interaction parameters.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # worker mode
    pw = sub.add_parser("worker", help="Process a slice of replicates (per SLURM array task).")
    pw.add_argument("--replicate-root", type=str, required=True)
    pw.add_argument("--output-dir", type=str, required=True)
    pw.add_argument("--monomer-types", type=str, required=True, help="Path to ME_bed_types.npy")
    pw.add_argument("--exp-tkl", type=str, required=True, help="Path to Tkl_exp.npy or .npz")
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
    pw.add_argument("--resolution", type=str, default=None, 
                    help="Observable resolution: None (default) for K×K type-averaged observables, '500kb' for (N/5)×(N/5) monomer-resolution observables")

    # reduce mode
    pr = sub.add_parser("reduce", help="Aggregate all per-rep artifacts and update both epsilon and lambda_IC")
    pr.add_argument("--output-dir", type=str, required=True)
    pr.add_argument("--epsilon-dir", type=str, required=True)
    pr.add_argument("--lambda-dir", type=str, required=True)
    pr.add_argument("--iteration", type=int, help="Current iteration index")
    pr.add_argument("--max-lambda-step-size", type=float, default=None,
                    help="Maximum allowed step size per parameter (applies to both epsilon and lambda_IC)")
    pr.add_argument("--gradient-normalization", type=str, default=None,
                    help="Gradient normalization method: 'L2' for L2 normalization, or leave unset for no normalization")
    pr.add_argument("--method", type=str, default="newton",
                    help="Optimization method: 'newton' or 'adam'")
    pr.add_argument("--adam-lr", type=float, default=None,
                    help="Adam learning rate for TKL (default: 0.001)")
    pr.add_argument("--adam-lr-ic", type=float, default=None,
                    help="Adam learning rate for IC (lambda_IC). If not provided, uses --adam-lr")
    pr.add_argument("--adam-beta1", type=float, default=None,
                    help="Adam beta1 parameter (default: 0.9)")
    pr.add_argument("--adam-beta2", type=float, default=None,
                    help="Adam beta2 parameter (default: 0.999)")
    pr.add_argument("--adam-epsilon", type=float, default=None,
                    help="Adam epsilon parameter (default: 1e-8)")
    pr.add_argument("--relstep-target-frac", type=float, default=None,
                    help="Target RMS step as fraction of RMS(param) for relative capping")
    pr.add_argument("--relstep-max-frac", type=float, default=None,
                    help="Max per-parameter step as fraction of |param| (or RMS if param≈0)")

    return p.parse_args()

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

        # Load monomer types once
        monomer_types = np.load(args.monomer_types)

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
                    _process_replicate_entry_tkl_IC,
                    replicate_root=args.replicate_root,
                    output_dir=str(output_dir),
                    monomer_types=monomer_types,
                    exp_Tkl_path=args.exp_tkl,
                    exp_phi_IC_path=args.exp_phi_IC,
                    d_init=args.d_init,
                    d_end=args.d_end,
                    mu=args.mu, rc=args.rc, rcut=args.rcut, beta=args.beta,
                    manifest_path=manifest_path,
                    resolution=args.resolution,
                ),
                targets,
                chunksize=1,
            )

    elif args.cmd == "reduce":
        epsilon_path, lambda_path = reduce_and_update_both(
            args.output_dir, 
            args.epsilon_dir, 
            args.lambda_dir, 
            iteration_idx=args.iteration,
            max_lambda_step_size=args.max_lambda_step_size,
            gradient_normalization=args.gradient_normalization,
            method=args.method,
            adam_lr=args.adam_lr,
            adam_lr_ic=getattr(args, 'adam_lr_ic', None),
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_epsilon=args.adam_epsilon,
            relstep_target_frac=getattr(args, "relstep_target_frac", None),
            relstep_max_frac=getattr(args, "relstep_max_frac", None)
        )
        print(f"[DONE] Wrote updated epsilon to: {epsilon_path}")
        print(f"[DONE] Wrote updated lambda_IC to: {lambda_path}")

if __name__ == "__main__":
    main()
