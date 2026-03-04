from __future__ import annotations

import os
from typing import Optional

import jax
import jax.numpy as jnp


def f_switch(r: jnp.ndarray, mu: float, rc: float) -> jnp.ndarray:
    """
    Smooth contact kernel: f(r) = 0.5 * (1 + tanh(mu * (rc - r))).
    Shapes: r: (*,) -> (*,)
    """
    return 0.5 * (1.0 + jnp.tanh(mu * (rc - r)))


def _detect_available_memory_gb() -> float:
    env_override = os.environ.get("DIFFTRE_AVAILABLE_MEM_GB")
    if env_override:
        try:
            return float(env_override)
        except ValueError:
            pass
    try:
        slurm_mem = os.environ.get("SLURM_MEM_PER_NODE")
        if slurm_mem:
            return float(slurm_mem) / 1024.0
        slurm_mem_gpu = os.environ.get("SLURM_MEM_PER_GPU")
        if slurm_mem_gpu:
            return float(slurm_mem_gpu) / 1024.0
        if os.environ.get("SLURM_JOB_ID"):
            return 16.0
    except Exception:
        pass
    return 16.0


def calculate_chunk_size(
    n_particles: int,
    dtype_bytes: int,
    available_memory_gb: Optional[float] = None,
    headroom_frac: float = 0.5,
    max_chunk: Optional[int] = None,
) -> int:
    """
    Calculate safe chunk size for vmap operations on contact maps.
    
    Dense contact maps scale as O(N^2). When vmap processes multiple frames,
    it needs memory for all contact maps in the chunk simultaneously.
    
    Memory calculation:
    - per_frame_bytes = N^2 * dtype_bytes (one contact map per frame)
    - chunk_size = (available_memory * headroom_frac) / per_frame_bytes
    
    Note: This calculation accounts for contact map memory only. For observable
    calculations that need additional arrays (type_onehot, chain_mask, etc.),
    use a more conservative headroom_frac (e.g., 0.3-0.4 instead of 0.5).
    
    Args:
        n_particles: Number of particles (N)
        dtype_bytes: Bytes per element (4 for float32, 8 for float64)
        available_memory_gb: Available GPU memory in GB (None to auto-detect)
        headroom_frac: Fraction of available memory to use (default 0.5)
        max_chunk: Maximum chunk size to return (None for no limit)
    
    Returns:
        Number of frames that can be processed in one chunk
    """
    if available_memory_gb is None:
        available_memory_gb = _detect_available_memory_gb()
    
    # Convert GB to bytes
    available_bytes = available_memory_gb * (1024**3)
    
    # Use only a fraction to leave headroom for JAX overhead, intermediate arrays, etc.
    usable_bytes = available_bytes * headroom_frac
    
    # Memory needed per frame: one NxN contact map
    per_frame_bytes = n_particles * n_particles * dtype_bytes
    
    if per_frame_bytes <= 0:
        return 1
    
    # Calculate how many frames fit in usable memory
    chunk = max(1, int(usable_bytes // per_frame_bytes))
    
    # Apply maximum chunk size limit if specified
    if max_chunk is not None:
        chunk = min(chunk, max_chunk)
    
    return max(1, chunk)

def calculate_trajectory_chunk_size(
    n_particles: int,
    n_frames_per_traj: int,
    dtype_bytes: int = 8,
    available_memory_gb: Optional[float] = None,
    headroom_frac: float = 0.5,
    max_chunk: Optional[int] = None,
) -> int:
    """
    Calculate how many trajectory files can be loaded into memory at once.
    
    Each trajectory file contains positions with shape (n_frames, n_particles, 3).
    When loading multiple trajectories, we need memory for all of them simultaneously.
    
    Memory calculation:
    - per_trajectory_bytes = n_frames * n_particles * 3 * dtype_bytes
    - chunk_size = (available_memory * headroom_frac) / per_trajectory_bytes
    
    Args:
        n_particles: Number of particles per frame
        n_frames_per_traj: Number of frames in each trajectory file
        dtype_bytes: Bytes per element (8 for float64, 4 for float32)
        available_memory_gb: Available memory in GB (None to auto-detect)
        headroom_frac: Fraction of available memory to use (default 0.5)
        max_chunk: Maximum number of trajectories to load at once (None for no limit)
    
    Returns:
        Number of trajectory files that can be loaded in one chunk
    """
    
    # Convert GB to bytes
    available_bytes = available_memory_gb * (1024**3)
    
    # Use only a fraction to leave headroom for overhead, intermediate arrays, etc.
    usable_bytes = available_bytes * headroom_frac
    
    # Memory needed per trajectory file: (n_frames, n_particles, 3) array
    per_trajectory_bytes = n_frames_per_traj * n_particles * 3 * dtype_bytes
    
    if per_trajectory_bytes <= 0:
        return 1
    
    # Calculate how many trajectory files fit in usable memory
    chunk = max(1, int(usable_bytes // per_trajectory_bytes))
    
    # Apply maximum chunk size limit if specified
    if max_chunk is not None:
        chunk = min(chunk, max_chunk)
    
    return max(1, chunk)


def frame_contact_map_dense(frame: jnp.ndarray, mu: float, rc: float) -> jnp.ndarray:
    """
    Dense contact map with no hard cutoff. Shapes: frame (N,3) -> (N,N)
    """
    dif = frame[:, None, :] - frame[None, :, :]
    dist = jnp.linalg.norm(dif, axis=-1)
    cmap = f_switch(dist, mu=mu, rc=rc)
    f0 = 0.5 * (1.0 + jnp.tanh(mu * rc))
    diag_idx = jnp.arange(frame.shape[0], dtype=jnp.int32)
    return cmap.at[diag_idx, diag_idx].set(f0)


def build_vmap_contact_map_fn(mu: float, rc: float):
    """
    Returns a JIT-compiled vmapped function over frames.
    """
    def _frame_fn(frame: jnp.ndarray) -> jnp.ndarray:
        return frame_contact_map_dense(frame, mu=mu, rc=rc)

    return jax.jit(jax.vmap(_frame_fn, in_axes=(0,)))
