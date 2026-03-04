#!/usr/bin/env python3
"""
JAX-accelerated contact map generation from trajectory files.
Uses cell-list neighbor search with overflow fallback to dense computation.
"""
from __future__ import annotations
import argparse
import glob
import struct
from typing import Optional

import os
import sys
from jax import config as jax_config
_use_fp32_env = os.environ.get("CHUNKCHROMATIN_USE_FP32", "1").lower() not in ("0", "false", "no")
jax_config.update("jax_enable_x64", not _use_fp32_env)

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from tqdm import tqdm

# --- Defaults ---
MU_DEFAULT = 4.22
RC_DEFAULT = 1.82

_CELL_KEY_PAD = int(jnp.iinfo(jnp.int32).max)  # keeps padded cell hashes sorted


def f_switch(r: jnp.ndarray, mu: float, rc: float) -> jnp.ndarray:
    """
    Compute a smooth contact-switching weight f(r) = 0.5 * (1 + tanh(mu * (rc - r))).
    Shapes: r: (*,), returns: (*,).
    """
    return 0.5 * (1.0 + jnp.tanh(mu * (rc - r)))


def _cell_hash(cx: jnp.ndarray, cy: jnp.ndarray, cz: jnp.ndarray,
               nx: int, ny: int, nz: int) -> jnp.ndarray:
    """
    Encode 3D cell coordinates into a single increasing integer key (lexicographic by x→y→z).
    Shapes: cx,cy,cz: (P,), returns: (P,).
    """
    return (cx * ny + cy) * nz + cz


def _build_neighbor_list(
    positions: jnp.ndarray,
    rcut: float,
    max_cell_particles: int = 96,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build a padded neighbor candidate index list using cubic cells of edge rcut and flag overflow.
    Shapes:
        positions: (N, 3)
        returns: (nbr_indices: (N, 27*max_cell_particles), overflow: ())
    For each particle, gather indices from its own cell and the 26 adjacent cells,
    padded to a fixed size, and report if any cell exceeded the per-cell cap.
    """
    N = positions.shape[0]
    cell_size = rcut

    # Shift to non-negative grid and compute integer cell coords
    pos_min = jnp.min(positions, axis=0)                   # (3,)
    rel = (positions - pos_min) / cell_size                # (N,3)
    cx = jnp.floor(rel[:, 0]).astype(jnp.int32)            # (N,)
    cy = jnp.floor(rel[:, 1]).astype(jnp.int32)            # (N,)
    cz = jnp.floor(rel[:, 2]).astype(jnp.int32)            # (N,)

    # Normalize cell indices to start at zero for compact hashing
    cx0 = cx - jnp.min(cx)
    cy0 = cy - jnp.min(cy)
    cz0 = cz - jnp.min(cz)

    nx = jnp.max(cx0) + 1
    ny = jnp.max(cy0) + 1
    nz = jnp.max(cz0) + 1

    keys = _cell_hash(cx0, cy0, cz0, nx, ny, nz)           # (N,)
    order = jnp.argsort(keys, stable=False)                # (N,)
    keys_sorted = keys[order]                              # (N,)

    # CSR-like ranges for occupied cells
    # Use size=N (worst case: one cell per particle) with sentinel fill value
    # Sentinel is chosen > any possible hash to keep padding sorted to the end.
    cell_keys, starts, counts = jnp.unique(
        keys_sorted,
        return_index=True,
        return_counts=True,
        size=N,
        fill_value=_CELL_KEY_PAD,
    )                                                      # (N,), (N,), (N,)
    # Find actual number of unique cells (count non-fill entries)
    n_unique = jnp.sum(cell_keys != _CELL_KEY_PAD)         # ()
    ends = starts + counts                                 # (N,)
    # 27 neighbor offsets (self + face + edge + corner)
    offs = jnp.array(
        [[dx, dy, dz] for dx in (-1, 0, 1)
                     for dy in (-1, 0, 1)
                     for dz in (-1, 0, 1)],
        dtype=jnp.int32
    )                                                      # (27,3)

    # Each particle's cell triplet
    cell_triplets = jnp.stack([cx0, cy0, cz0], axis=1)     # (N,3)

    def particle_neighbors(p_triplet: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Gather up to max_cell_particles indices from each of 27 neighbor cells for one particle.
        Shapes: p_triplet: (3,), returns (idxs: (27*max_cell_particles,), overflow_local: ()).
        """
        # Neighbor cells (27,3); invalidate those outside of the grid bounds
        neigh = p_triplet[None, :] + offs                  # (27,3)
        valid_neigh = (
            (neigh[:, 0] >= 0) & (neigh[:, 0] < nx) &
            (neigh[:, 1] >= 0) & (neigh[:, 1] < ny) &
            (neigh[:, 2] >= 0) & (neigh[:, 2] < nz)
        )
        neigh = jnp.where(valid_neigh[:, None], neigh, 0)  # (27,3)

        # Map to keys and locate their CSR ranges
        nkeys = jnp.where(
            valid_neigh,
            _cell_hash(neigh[:, 0], neigh[:, 1], neigh[:, 2], nx, ny, nz),
            _CELL_KEY_PAD,
        )                                                  # (27,)
        # Search in full cell_keys array (sorted, padded with -1)
        # searchsorted will work correctly on sorted array even with padding
        pos_left = jnp.searchsorted(cell_keys, nkeys, side="left")              # (27,)
        # Clip for safe indexing, then check if key was found and index is valid
        pos_safe = jnp.clip(pos_left, 0, N - 1)                                  # (27,)
        found = (
            valid_neigh
            & (pos_left < n_unique)
            & (cell_keys[pos_safe] == nkeys)
        )                                                   # (27,)
        c_st = jnp.where(found, starts[pos_safe], 0)                            # (27,)
        c_en = jnp.where(found, ends[pos_safe], 0)                              # (27,)
        c_len = c_en - c_st                                                    # (27,)

        # Gather indices per neighbor cell with fixed cap (pad with -1)
        a = jnp.arange(max_cell_particles, dtype=jnp.int32)                    # (M,)
        rel_idx = a[None, :]                                                   # (27,M)
        in_cell = rel_idx < c_len[:, None]                                     # (27,M)
        pick = jnp.clip(c_st[:, None] + rel_idx, 0, order.size - 1)            # (27,M)
        idxs = jnp.where(in_cell, order[pick], -1)                             # (27,M)

        overflow_local = jnp.any(c_len > max_cell_particles)                   # ()
        return idxs.reshape((-1,)), overflow_local

    idxs, overflows = jax.vmap(particle_neighbors, in_axes=(0,))(cell_triplets)  # (N,27*M),(N,)
    overflow = jnp.any(overflows)                                                 # ()
    return idxs.astype(jnp.int32), overflow


def _frame_contact_map_dense(
    positions: jnp.ndarray,
    mu: float,
    rc: float,
    rcut: float,
) -> jnp.ndarray:
    """
    Compute the per-frame contact map using a dense O(N^2) distance pass.
    Shapes:
        positions: (N,3), returns: (N, N)
    Use full pairwise distances, process upper triangle only, then make symmetric.
    """
    N = positions.shape[0]
    dif = positions[:, None, :] - positions[None, :, :]               # (N,N,3)
    D = jnp.linalg.norm(dif, axis=-1)                                  # (N,N)
    # Upper triangle mask (i < j)
    iu = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)                   # (N,N)
    mask = iu & (D <= rcut)                                            # (N,N)
    fij = f_switch(D, mu, rc) * mask.astype(jnp.float64)              # (N,N)
    # Make symmetric (upper triangle + lower triangle)
    return fij + fij.T


def _frame_contact_map_neighbor(
    positions: jnp.ndarray,
    nbr_idx: jnp.ndarray,
    mu: float,
    rc: float,
    rcut: float,
) -> jnp.ndarray:
    """
    Compute the per-frame contact map using neighbor list.
    Shapes:
        positions: (N,3), nbr_idx: (N, C), returns: (N, N)
    """
    N = positions.shape[0]
    # Unique pairs i<j inside rcut using neighbor candidates (upper triangle only, like query_pairs)
    i = jnp.arange(N, dtype=jnp.int32)[:, None]                           # (N,1)
    j_idx = nbr_idx                                                       # (N,C)
    valid = (j_idx >= 0) & (j_idx < N) & (j_idx != i) & (i < j_idx)       # (N,C)

    ri = positions[:, None, :]                                            # (N,1,3)
    rj = positions[jnp.clip(j_idx, 0, N - 1), :]                          # (N,C,3)
    dij = jnp.linalg.norm(ri - rj, axis=-1)                               # (N,C)
    valid = valid & (dij <= rcut)                                         # (N,C)

    fij = f_switch(dij, mu=mu, rc=rc) * valid.astype(jnp.float64)         # (N,C)

    # Accumulate into full (N, N) contact map using scatter-add
    contact_map = jnp.zeros((N, N), dtype=jnp.float64)
    
    # Create indices for scatter-add: (i, j_idx) pairs
    i_expanded = jnp.broadcast_to(i, j_idx.shape)                         # (N,C)
    
    # Flatten
    i_flat = i_expanded.ravel()                                           # (N*C,)
    j_flat = j_idx.ravel()                                                # (N*C,)
    fij_flat = fij.ravel()                                                # (N*C,)
    valid_flat = valid.ravel()                                            # (N*C,)
    
    # Filter to only valid pairs (avoid scatter with -1 indices)
    # Use boolean mask to select valid entries
    i_valid = jnp.where(valid_flat, i_flat, 0)                            # (N*C,)
    j_valid = jnp.where(valid_flat, j_flat, 0)                            # (N*C,)
    fij_valid = jnp.where(valid_flat, fij_flat, 0.0)                      # (N*C,)
    
    # Use scatter-add to accumulate (symmetric: add to both (i,j) and (j,i))
    # Only scatter where valid_flat is True
    contact_map = contact_map.at[i_valid, j_valid].add(fij_valid)
    contact_map = contact_map.at[j_valid, i_valid].add(fij_valid)
    
    return contact_map


def frame_contact_map(
    positions: jnp.ndarray,
    mu: float,
    rc: float,
    rcut: float,
    max_cell_particles: int,
) -> jnp.ndarray:
    """
    Compute the per-frame contact map using a cell-list neighbor search with overflow fallback.
    JIT-compiled for efficiency. This is the single-frame worker function.
    
    Shapes:
        positions: (N,3), returns: (N, N)
    Accumulate f_switch over unique pairs within rcut via a fast cell list, falling back to dense if any cell overflows.
    """
    # Build neighbor candidates and detect overflow
    nbr_idx, overflow = _build_neighbor_list(positions, rcut, max_cell_particles)  # (N,C), ()

    def _neighbor_path(_: None) -> jnp.ndarray:
        return _frame_contact_map_neighbor(positions, nbr_idx, mu, rc, rcut)

    def _dense_path(_: None) -> jnp.ndarray:
        return _frame_contact_map_dense(positions, mu, rc, rcut)

    # Overflow gate: if any cell exceeded the cap, switch to dense for correctness
    contact_map = lax.cond(overflow, _dense_path, _neighbor_path, operand=None)

    # Match OpenMiChroM: include i==j with f_switch(0) on the diagonal.
    f0 = 0.5 * (1.0 + jnp.tanh(mu * rc))
    idx = jnp.arange(positions.shape[0], dtype=jnp.int32)
    contact_map = contact_map.at[idx, idx].set(f0)
    return contact_map


def load_all_positions(filename: str) -> np.ndarray:
    """Read .traj binary file into array (F, N, 3)."""
    HEADER_FORMAT = "<4sBHII16s"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    with open(filename, 'rb') as f:
        header = f.read(HEADER_SIZE)
        magic, version, n_particles, frame_size, n_frames, _ = struct.unpack(HEADER_FORMAT, header)
        assert magic == b'CHRM', f"Bad magic in {filename}"

        metadata_len = struct.unpack("<I", f.read(4))[0]
        f.seek(HEADER_SIZE + 4 + metadata_len)

        data = np.frombuffer(f.read(), dtype=np.float32)
        return data.reshape((n_frames, n_particles, 3))


# Global cache for pre-compiled functions
_COMPILED_FRAME_VMAP_CACHE = {}
_COMPILED_FRAME_VMAP_DENSE_CACHE = {}


def _get_compiled_frame_vmap(mu: float, rc: float, rcut: float, max_cell_particles: int):
    """
    Get or create a JIT-compiled vmap function for processing frames.
    
    This function caches compiled vmap functions to avoid recompilation overhead.
    The vmap applies the single-frame worker (frame_contact_map) to all frames in a chunk.
    
    Optimization: JIT compiles the entire vmap operation once, which automatically
    includes optimization of the inner frame_contact_map function.
    """
    cache_key = (mu, rc, rcut, max_cell_particles)
    if cache_key not in _COMPILED_FRAME_VMAP_CACHE:
        # Create vmap over frame_contact_map
        # vmap will vectorize over the first axis (frames)
        vmap_fn = jax.vmap(
            frame_contact_map,
            in_axes=(0, None, None, None, None),
        )
        
        # JIT compile the vmap - this automatically optimizes frame_contact_map too
        # Static arguments don't change between calls, allowing better optimization
        _COMPILED_FRAME_VMAP_CACHE[cache_key] = jax.jit(
            vmap_fn,
            static_argnames=('mu', 'rc', 'rcut', 'max_cell_particles')
        )
    return _COMPILED_FRAME_VMAP_CACHE[cache_key]


def _get_compiled_frame_vmap_dense(mu: float, rc: float, rcut: float):
    """
    Get or create a JIT-compiled vmap function for dense per-frame contact maps.
    """
    cache_key = (mu, rc, rcut)
    if cache_key not in _COMPILED_FRAME_VMAP_DENSE_CACHE:
        vmap_fn = jax.vmap(
            _frame_contact_map_dense,
            in_axes=(0, None, None, None),
        )
        _COMPILED_FRAME_VMAP_DENSE_CACHE[cache_key] = jax.jit(
            vmap_fn,
            static_argnames=('mu', 'rc', 'rcut')
        )
    return _COMPILED_FRAME_VMAP_DENSE_CACHE[cache_key]


def _calculate_optimal_chunk_size(
    n_particles: int,
    available_memory_gb: float = None,
    min_chunk: int = 10,
    max_chunk: int = 150,
) -> int:
    """
    Calculate optimal chunk size based on memory constraints.
    
    Memory requirements for a chunk of size F:
    - Input positions: F * N * 3 * 4 bytes (float32) = F * N * 3 * 4
    - Intermediate contact maps: F * N * N * 8 bytes (float64) = F * N^2 * 8 (largest component)
    - Neighbor lists: F * N * 27 * 96 * 4 bytes (int32) = F * N * 27 * 96 * 4
    - Output contact map: N * N * 8 bytes (float64) = N^2 * 8
    - JAX/GPU overhead: ~30-40% buffer for compilation and fragmentation
    
    Total ≈ F * (N * 3 * 4 + N^2 * 8 + N * 27 * 96 * 4) + N^2 * 8
    
    The dominant term is F * N^2 * 8 (intermediate contact maps), so we solve:
    F * N^2 * 8 ≈ available_memory * 0.4 (use 40% for intermediate arrays, conservative)
    """
    if available_memory_gb is None:
        # Default: assume ~12GB available for GPU computation (A100 has 40GB)
        # Reserve significant overhead for JAX/GPU system usage and fragmentation
        available_memory_gb = 12.0
    
    # Convert to bytes
    available_memory_bytes = available_memory_gb * (1024**3)
    
    # Use only 40% of available memory for intermediate arrays (conservative)
    # JAX needs significant overhead for compilation, fragmentation, and other operations
    # Since we sum immediately after computing each chunk, we don't need to store all chunks
    usable_memory_bytes = available_memory_bytes * 0.40
    
    # Size of output contact map (constant overhead)
    output_memory = n_particles * n_particles * 8  # float64
    
    # Memory per frame (excluding the intermediate (F, N, N) array which dominates)
    # These are smaller components
    per_frame_input = n_particles * 3 * 4  # float32 positions
    per_frame_neighbors = n_particles * 27 * 96 * 4  # int32 neighbor lists
    per_frame_other = per_frame_input + per_frame_neighbors
    
    # The dominant memory usage is the intermediate array: chunk_size * N^2 * 8
    # Solve: chunk_size * N^2 * 8 + chunk_size * per_frame_other + output_memory <= usable_memory
    # chunk_size * (N^2 * 8 + per_frame_other) <= usable_memory - output_memory
    memory_per_frame_dominant = n_particles * n_particles * 8  # float64 (F, N, N) array
    memory_per_frame_total = memory_per_frame_dominant + per_frame_other
    
    # Calculate max chunk size
    max_chunk_by_memory = int((usable_memory_bytes - output_memory) / memory_per_frame_total)
    max_chunk_by_memory = max(1, max_chunk_by_memory)  # Ensure at least 1
    
    # Also limit by kernel launch limits (avoid too many blocks)
    # Conservative limit: don't exceed 150 frames in a batch to avoid kernel launch errors
    max_chunk_by_kernel = min(150, max_chunk)
    
    # Take the minimum of all constraints
    optimal_chunk = min(max_chunk_by_memory, max_chunk_by_kernel, max_chunk)
    optimal_chunk = max(optimal_chunk, min_chunk)  # Ensure minimum
    
    return optimal_chunk


def contact_map_from_traj(
    positions: np.ndarray,
    mu: float = MU_DEFAULT,
    rc: float = RC_DEFAULT,
    rcut: Optional[float] = None,
    max_cell_particles: int = 96,
    available_memory_gb: float = None,
    chunk_size: Optional[int] = None,
    use_dense: bool = False,
) -> np.ndarray:
    """
    Calculate contact map from positions array (F, N, 3) using JAX with optimized chunking.
    
    This function:
    1. JIT-compiles a single-frame worker function once
    2. Uses vmap to apply it to chunks of frames
    3. Dynamically calculates optimal chunk size based on available memory
    
    Args:
        positions: Array of shape (F, N, 3) with frame positions
        mu: Switch parameter for contact function
        rc: Switch parameter for contact function
        rcut: Cutoff distance (default: rc + 4/mu)
        max_cell_particles: Maximum particles per cell in neighbor list
        available_memory_gb: Available GPU memory in GB (auto-detected if None)
        chunk_size: Force chunk size (auto-calculated if None)
    
    Returns:
        Contact map array of shape (N, N)
    """
    F, N, _ = positions.shape
    if rcut is None:
        rcut = rc + 4.0 / mu
    
    # Calculate optimal chunk size if not provided
    if chunk_size is None:
        chunk_size = _calculate_optimal_chunk_size(N, available_memory_gb)
        print(f"Calculated optimal chunk size: {chunk_size} frames for {N} particles", flush=True)
    
    # Get pre-compiled vmap function (or compile it)
    # This function is cached, so compilation happens only once
    if use_dense:
        frame_vmap = _get_compiled_frame_vmap_dense(mu, rc, rcut)
    else:
        frame_vmap = _get_compiled_frame_vmap(mu, rc, rcut, max_cell_particles)
    
    # Initialize output contact map
    contact_map = jnp.zeros((N, N), dtype=jnp.float64)
    
    # Process in chunks
    total_chunks = (F + chunk_size - 1) // chunk_size
    for i in range(0, F, chunk_size):
        end_idx = min(i + chunk_size, F)
        chunk_positions = positions[i:end_idx]
        chunk_num = i // chunk_size + 1
        
        # Convert to JAX array (only when needed, not cached)
        chunk_positions_jax = jnp.asarray(chunk_positions)
        
        # Process chunk: vmap applies frame_contact_map to all frames in chunk
        # This returns (chunk_size, N, N)
        if use_dense:
            chunk_maps = frame_vmap(chunk_positions_jax, mu, rc, rcut)
        else:
            chunk_maps = frame_vmap(chunk_positions_jax, mu, rc, rcut, max_cell_particles)
        
        # Sum chunk maps and accumulate into total contact map
        # Sum along frame axis first, then add to total (more efficient than storing all)
        contact_map = contact_map + jnp.sum(chunk_maps, axis=0)
        
        if chunk_num % max(1, total_chunks // 20) == 0 or chunk_num == total_chunks:
            print(f"Processed chunk {chunk_num}/{total_chunks} (frames {i} to {end_idx-1})", flush=True)
    
    return np.asarray(contact_map)

def _detect_available_memory_gb() -> float:
    """Estimate available memory for chunk sizing (prefer SLURM request)."""
    try:
        slurm_mem = os.environ.get('SLURM_MEM_PER_NODE')
        if slurm_mem:
            return float(slurm_mem) / 1024.0
        if os.environ.get('SLURM_JOB_ID'):
            return 16.0
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                total_memory_mb = float(result.stdout.strip().split('\n')[0])
                return total_memory_mb / 1024.0 * 0.5
        except Exception:
            pass
    except Exception:
        pass
    return 16.0

def main():
    parser = argparse.ArgumentParser(
        description="Build a contact map from trajectory .traj files using JAX"
    )
    parser.add_argument("traj_glob", help="Glob path to trajectory files, e.g. 'sims/rep*/trajectory.traj'")
    parser.add_argument("output", help="Output .npy filename for contact map")
    parser.add_argument("--mu", type=float, default=None, help="Switch mu parameter (default: config kernel_flags or MU_DEFAULT)")
    parser.add_argument("--rc", type=float, default=None, help="Switch rc parameter (default: config kernel_flags or RC_DEFAULT)")
    parser.add_argument("--rcut", type=float, default=None, help="Cutoff distance (default: config kernel_flags or rc + 4/mu)")
    parser.add_argument("--config", type=str, default=None, help="Config YAML with processing_inputs.kernel_flags")
    parser.add_argument("--use-fp32", action=argparse.BooleanOptionalAction, default=None, help="Use float32 (default: config or True)")
    parser.add_argument("--max-cell-particles", type=int, default=96, help="Max particles per cell (default: 96)")
    parser.add_argument("--skip-frames", type=int, default=None, help="Number of initial frames to skip (default: config burnin_frames or 400)")
    parser.add_argument("--method", type=str, choices=["hash", "dense"], default="hash",
                        help="Contact map method (default: hash)")
    args = parser.parse_args()

    if args.config:
        try:
            from jax_compare_contact_maps import load_config
            config = load_config(args.config)
            kernel_flags = config.get("processing_inputs", {}).get("kernel_flags", {})
        except Exception as e:
            raise RuntimeError(f"Failed to load config kernel_flags from {args.config}: {e}")
    else:
        kernel_flags = {}

    use_fp32 = args.use_fp32
    if use_fp32 is None and args.config:
        use_fp32 = config.get("processing_inputs", {}).get("use_fp32", True)
    if use_fp32 is None:
        use_fp32 = True

    env_use_fp32 = os.environ.get("CHUNKCHROMATIN_USE_FP32", "1").lower() not in ("0", "false", "no")
    if use_fp32 != env_use_fp32:
        os.environ["CHUNKCHROMATIN_USE_FP32"] = "1" if use_fp32 else "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)

    mu = args.mu if args.mu is not None else kernel_flags.get("mu", MU_DEFAULT)
    rc = args.rc if args.rc is not None else kernel_flags.get("rc", RC_DEFAULT)
    rcut = args.rcut if args.rcut is not None else kernel_flags.get("rcut", None)
    if args.config and args.skip_frames is None:
        args.skip_frames = config.get("simulation", {}).get("burnin_frames", None)
    if args.skip_frames is None:
        args.skip_frames = 400

    # Find files
    all_traj_paths = glob.glob(args.traj_glob)
    if len(all_traj_paths) == 0:
        raise FileNotFoundError(f"No trajectory files found for {args.traj_glob}")
    print(f"Found {len(all_traj_paths)} trajectory files")

    # Load positions
    results = []
    for traj_path in tqdm(all_traj_paths, desc="Loading trajectories"):
        pos = load_all_positions(traj_path)
        if pos.shape[0] <= args.skip_frames:
            print(f"Warning: {traj_path} has only {pos.shape[0]} frames, skipping all frames")
            continue
        # Discard burn-in frames
        pos = pos[args.skip_frames:]
        results.append(pos)

    positions = np.concatenate(results, axis=0)
    print("Loaded positions:", positions.shape)

    # Build contact map
    print("Computing contact map with JAX...")
    available_memory_gb = _detect_available_memory_gb()
    print(f"Using available memory estimate: {available_memory_gb:.1f} GB", flush=True)
    print("Warming up JIT compilation with small batch (5 frames)...", flush=True)
    _ = contact_map_from_traj(
        positions[:min(5, positions.shape[0])],
        mu=mu,
        rc=rc,
        rcut=rcut,
        max_cell_particles=args.max_cell_particles,
        available_memory_gb=available_memory_gb,
        chunk_size=5,
    )
    print("JIT compilation complete", flush=True)
    contact_map = contact_map_from_traj(
        positions,
        mu=mu,
        rc=rc,
        rcut=rcut,
        max_cell_particles=args.max_cell_particles,
        available_memory_gb=available_memory_gb,
        use_dense=(args.method == "dense"),
    )

    # Save as .npy
    np.save(args.output, contact_map)
    print(f"Saved contact map to {args.output}")

if __name__ == "__main__":
    main()
