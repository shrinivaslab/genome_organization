from __future__ import annotations

from typing import Iterable, Tuple, Optional, Union

import numpy as np
import jax
import jax.numpy as jnp

from .contact_map import build_vmap_contact_map_fn, calculate_chunk_size


def _normalize_chains(
    chains: Iterable[Tuple[int, int, bool]] | None,
    n_particles: int,
) -> list[Tuple[int, int, bool]]:
    if not chains:
        return [(0, n_particles, False)]
    norm = []
    for entry in chains:
        if len(entry) == 2:
            start, end = entry
            is_ring = False
        else:
            start, end, is_ring = entry
        end = n_particles if end is None else end
        norm.append((int(start), int(end), bool(is_ring)))
    return norm


def _build_chain_mask(
    n_particles: int,
    chains: Iterable[Tuple[int, int, bool]] | None,
) -> np.ndarray:
    if not chains:
        return np.ones((n_particles, n_particles), dtype=bool)
    chain_id = np.full(n_particles, -1, dtype=int)
    for cid, (start, end, _) in enumerate(chains):
        chain_id[start:end] = cid
    return chain_id[:, None] == chain_id[None, :]

def _build_chain_mask_jax(
    n_particles: int,
    chains: Iterable[Tuple[int, int, bool]] | None,
) -> jnp.ndarray:
    """Build chain mask in JAX."""
    if not chains:
        return jnp.ones((n_particles, n_particles), dtype=bool)
    chain_id = jnp.full(n_particles, -1, dtype=jnp.int32)
    for cid, (start, end, _) in enumerate(chains):
        chain_id = chain_id.at[start:end].set(cid)
    return chain_id[:, None] == chain_id[None, :]

def _map_type_ids(monomer_types: Union[np.ndarray, jnp.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    type_labels = np.unique(monomer_types)
    type_map = {int(t): i for i, t in enumerate(type_labels)}
    type_ids = np.array([type_map[int(t)] for t in monomer_types], dtype=int)
    return type_ids, type_labels


def compute_observables_per_frame(
    positions: Union[np.ndarray, jnp.ndarray],  # Accept both
    monomer_types: Union[np.ndarray, jnp.ndarray],
    loop_pairs: Union[np.ndarray, jnp.ndarray],
    chains: Iterable[Tuple[int, int, bool]] | None,
    d_init: int,
    d_end: int,
    mu: float,
    rc: float,
    use_fp32: bool = False,
    available_memory_gb: Optional[float] = None,
    headroom_frac: float = 0.5,
) -> dict:
    """
    Compute per-frame observables from raw (unnormalized) contact maps.
    Returns dict with tkl_frames, phi_frames, loop_frames, type_labels.
    """
    dtype = jnp.float32 if use_fp32 else jnp.float64
    n_frames, n_particles, _ = positions.shape
    dmax = d_end - d_init

    type_ids, type_labels = _map_type_ids(monomer_types)
    type_ids_jax = jnp.asarray(type_ids, dtype=jnp.int32)
    k = int(type_labels.shape[0])
    type_onehot = jnp.eye(k, dtype=dtype)[type_ids_jax].T  # K x N
    counts = jnp.sum(type_onehot, axis=1)
    denom = counts[:, None] * counts[None, :]

    chains_norm = _normalize_chains(chains, n_particles)
    chain_mask = _build_chain_mask_jax(n_particles, chains_norm).astype(dtype)

    # Compute genomic distances in JAX
    idx_jax = jnp.arange(n_particles, dtype=jnp.int32)
    dist_jax = jnp.abs(idx_jax[:, None] - idx_jax[None, :]).astype(jnp.int32)
    dist_flat = dist_jax.reshape(-1)
    
    # Count pairs by genomic distance using JAX bincount
    count_by_d = jnp.bincount(
        dist_flat,
        weights=chain_mask.reshape(-1),
        length=n_particles,
    ).astype(dtype)

    loop_pairs_jax = jnp.asarray(loop_pairs, dtype=jnp.int32)
    has_loops = loop_pairs_jax.size > 0
    if has_loops:
        loop_i = loop_pairs_jax[:, 0]
        loop_j = loop_pairs_jax[:, 1]
    else:
        loop_i = jnp.zeros((1,), dtype=jnp.int32)
        loop_j = jnp.zeros((1,), dtype=jnp.int32)

    frame_vmap = build_vmap_contact_map_fn(mu=mu, rc=rc)

    def _loop_mean(cmap: jnp.ndarray) -> jnp.ndarray:
        if has_loops:
            return jnp.mean(cmap[loop_i, loop_j])
        return jnp.asarray(0.0, dtype=dtype)

    def _frame_observables(cmap: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        sums = type_onehot @ cmap @ type_onehot.T
        tkl = jnp.where(denom > 0, sums / denom, 0.0)

        weighted = cmap * chain_mask
        sum_by_d = jnp.bincount(
            dist_flat,
            weights=weighted.reshape(-1),
            length=n_particles,
        )
        phi_full = jnp.where(count_by_d > 0, sum_by_d / count_by_d, 0.0)
        phi = phi_full[d_init:d_end]
        loop_mean = _loop_mean(cmap)
        return tkl, phi, loop_mean

    frame_obs_vmap = jax.jit(jax.vmap(_frame_observables, in_axes=(0,)))

    dtype_bytes = 4 if use_fp32 else 8
    chunk_size = calculate_chunk_size(
        n_particles,
        dtype_bytes=dtype_bytes,
        available_memory_gb=available_memory_gb,
        headroom_frac=headroom_frac,
        max_chunk=n_frames,
    )

    tkl_frames_list = []
    phi_frames_list = []
    loop_frames_list = []

    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        chunk = positions[start:end]  # Already JAX, no conversion needed
        cmap_chunk = frame_vmap(chunk)
        tkl_chunk, phi_chunk, loop_chunk = frame_obs_vmap(cmap_chunk)

        # Accumulate JAX arrays
        tkl_frames_list.append(tkl_chunk)
        phi_frames_list.append(phi_chunk)
        loop_frames_list.append(loop_chunk)

    # Concatenate using JAX
    tkl_frames = jnp.concatenate(tkl_frames_list, axis=0)
    phi_frames = jnp.concatenate(phi_frames_list, axis=0)
    loop_frames = jnp.concatenate(loop_frames_list, axis=0)

    return {
        "tkl_frames": tkl_frames,  # JAX array
        "phi_frames": phi_frames,  # JAX array
        "loop_frames": loop_frames,  # JAX array
        "type_labels": type_labels,  # numpy array (metadata)
    }
