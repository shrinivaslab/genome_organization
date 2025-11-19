# jax_O_calc.py
from __future__ import annotations
from typing import Sequence, Tuple, Optional, Dict, List

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import lax


__all__ = [
    "f_switch",
    "frame_typepair_observable",
    "compute_observables_chunk",
    "compute_observables_from_chunks",
    "compute_observables_all",
]


# ---------------------------------------------------------------------------
# Switch function
# ---------------------------------------------------------------------------

def f_switch(r: jnp.ndarray, mu: float, rc: float) -> jnp.ndarray:
    """
    Compute a smooth contact-switching weight f(r) = 0.5 * (1 + tanh(mu * (rc - r))).
    Shapes: r: (*,), returns: (*,).
    """
    return 0.5 * (1.0 + jnp.tanh(mu * (rc - r)))


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _upper_tri_indices(K: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Return upper-triangle (including diagonal) indices for a K×K matrix.
    Shapes: returns (row_idx: (M,), col_idx: (M,)) with M = K*(K+1)//2.
    """
    return jnp.triu_indices(K, k=0)


def _prepare_types(monomer_types: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """
    Map raw monomer types to contiguous [0..K-1] ids.
    Shapes: monomer_types: (N,), returns (type_labels: (K,), inv_ids: (N,), K: int).
    """
    type_labels, inv = jnp.unique(monomer_types, return_inverse=True)
    K = int(type_labels.shape[0])
    return type_labels, inv.astype(jnp.int32), K


# ---------------------------------------------------------------------------
# Neighbor list via uniform cubic cell grid (cell size = rcut) with overflow gate
# ---------------------------------------------------------------------------

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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
    order = jnp.argsort(keys, kind="quicksort")            # (N,)
    keys_sorted = keys[order]                              # (N,)

    # CSR-like ranges for occupied cells
    cell_keys, starts, counts = jnp.unique(
        keys_sorted, return_index=True, return_counts=True
    )                                                      # (C,), (C,), (C,)
    ends = starts + counts                                 # (C,)
    # 27 neighbor offsets (self + face + edge + corner)
    offs = jnp.array(
        [[dx, dy, dz] for dx in (-1, 0, 1)
                     for dy in (-1, 0, 1)
                     for dz in (-1, 0, 1)],
        dtype=jnp.int32
    )                                                      # (27,3)

    # Each particle's cell triplet
    cell_triplets = jnp.stack([cx0, cy0, cz0], axis=1)     # (N,3)

    def particle_neighbors(p_triplet: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Gather up to max_cell_particles indices from each of 27 neighbor cells for one particle.
        Shapes: p_triplet: (3,), returns (idxs: (27*max_cell_particles,), overflow_local: ()).
        """
        # Neighbor cells (27,3), clipped to grid
        neigh = p_triplet[None, :] + offs                  # (27,3)
        neigh = jnp.stack([
            jnp.clip(neigh[:, 0], 0, nx - 1),
            jnp.clip(neigh[:, 1], 0, ny - 1),
            jnp.clip(neigh[:, 2], 0, nz - 1),
        ], axis=1)                                         # (27,3)

        # Map to keys and locate their CSR ranges
        nkeys = _cell_hash(neigh[:, 0], neigh[:, 1], neigh[:, 2], nx, ny, nz)  # (27,)
        pos_left = jnp.searchsorted(cell_keys, nkeys, side="left")             # (27,)
        found = (pos_left < cell_keys.size) & (cell_keys[pos_left] == nkeys)   # (27,)
        c_st = jnp.where(found, starts[pos_left], 0)                           # (27,)
        c_en = jnp.where(found, ends[pos_left], 0)                             # (27,)
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


# ---------------------------------------------------------------------------
# Dense fallback (used only when overflow detected)
# ---------------------------------------------------------------------------

def _frame_typepair_observable_dense(
    positions: jnp.ndarray,
    type_ids: jnp.ndarray,
    K: int,
    mu: float,
    rc: float,
    rcut: float,
) -> jnp.ndarray:
    """
    Compute the per-frame type–type observable using a dense O(N^2) distance pass.
    Shapes:
        positions: (N,3), type_ids: (N,), K: (), returns: (K*(K+1)//2,)
    Use full pairwise distances with masking to accumulate upper-triangular type–type sums.
    """
    N = positions.shape[0]
    dif = positions[:, None, :] - positions[None, :, :]               # (N,N,3)
    D = jnp.linalg.norm(dif, axis=-1)                                  # (N,N)
    iu = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)                   # (N,N)
    mask = iu & (D <= rcut)                                            # (N,N)
    fij = f_switch(D, mu, rc) * mask.astype(jnp.float64)               # (N,N)

    # Map (i,j) to type-pair bins (upper-triangular in type space)
    ti = type_ids[:, None]                                             # (N,1)
    tj = type_ids[None, :]                                             # (1,N)
    k = jnp.minimum(ti, tj)                                            # (N,N)
    l = jnp.maximum(ti, tj)                                            # (N,N)
    flat = (k * K + l).astype(jnp.int32)                               # (N,N)

    sums = jnp.bincount(flat[mask].ravel(),
                        weights=fij[mask].ravel(),
                        length=K * K).reshape(K, K)                     # (K,K)

    iuK = _upper_tri_indices(K)
    return sums[iuK]                                                   # (M,)


# ---------------------------------------------------------------------------
# Per-frame observable (neighbor list + overflow gate to dense)
# ---------------------------------------------------------------------------

def frame_typepair_observable(
    positions: jnp.ndarray,
    type_ids: jnp.ndarray,
    K: int,
    mu: float,
    rc: float,
    rcut: Optional[float] = None,
    max_cell_particles: int = 96,
) -> jnp.ndarray:
    """
    Compute the per-frame upper-triangular type–type observable using a cell-list neighbor search with overflow fallback.
    Shapes:
        positions: (N,3), type_ids: (N,), K: (), returns: (K*(K+1)//2,)
    Accumulate f_switch over unique pairs within rcut via a fast cell list, falling back to dense if any cell overflows.
    """
    if rcut is None:
        rcut = rc + 4.0 / mu

    N = positions.shape[0]
    # Build neighbor candidates and detect overflow
    nbr_idx, overflow = _build_neighbor_list(positions, rcut, max_cell_particles)  # (N,C), ()

    def _neighbor_path(_: None) -> jnp.ndarray:
        # Unique pairs i<j inside rcut using neighbor candidates
        i = jnp.arange(N, dtype=jnp.int32)[:, None]                           # (N,1)
        j_idx = nbr_idx                                                       # (N,C)
        valid = (j_idx >= 0) & (j_idx < N) & (j_idx != i) & (i < j_idx)       # (N,C)

        ri = positions[:, None, :]                                            # (N,1,3)
        rj = positions[jnp.clip(j_idx, 0, N - 1), :]                          # (N,C,3)
        dij = jnp.linalg.norm(ri - rj, axis=-1)                               # (N,C)
        valid = valid & (dij <= rcut)                                         # (N,C)

        fij = f_switch(dij, mu=mu, rc=rc) * valid.astype(jnp.float64)         # (N,C)

        ti = type_ids[:, None]                                                # (N,1)
        tj = type_ids[jnp.clip(j_idx, 0, N - 1)]                              # (N,C)
        k = jnp.minimum(ti, tj)
        l = jnp.maximum(ti, tj)
        flat = (k * K + l).astype(jnp.int32)                                  # (N,C)

        sums = jnp.bincount(
            flat.ravel(),
            weights=fij.ravel(),
            length=K * K
        ).reshape(K, K)                                                       # (K,K)

        iuK = _upper_tri_indices(K)
        return sums[iuK]                                                      # (M,)

    def _dense_path(_: None) -> jnp.ndarray:
        return _frame_typepair_observable_dense(positions, type_ids, K, mu, rc, rcut)

    # Overflow gate: if any cell exceeded the cap, switch to dense for correctness
    return lax.cond(overflow, _dense_path, _neighbor_path, operand=None)


# ---------------------------------------------------------------------------
# Chunking (per-replicate) and multi-replicate computation
# ---------------------------------------------------------------------------

def compute_observables_chunk(
    frames_chunk: jnp.ndarray,          # (n_frames, n_particles, 3)
    monomer_types: jnp.ndarray,         # (n_particles,)
    mu: float,
    rc: float,
    rcut: Optional[float] = None,
    max_cell_particles: int = 96,
) -> jnp.ndarray:
    """
    Compute per-frame observables for one replicate using vmap over frames.
    Input shapes:
        frames_chunk    : (n_frames, n_particles, 3)
        monomer_types   : (n_particles,)
    Returns:
        (n_frames, n_tt_combinations) where n_tt_combinations = K*(K+1)//2 and K is the number of unique monomer types.
    """
    # Prepare contiguous type ids (shared across all frames in this replicate)
    _, inv_ids, K = _prepare_types(jnp.asarray(monomer_types))
    rcut_val = rcut if rcut is not None else (rc + 4.0 / mu)

    # Vectorize the per-frame kernel across the frame axis
    frame_vmap = jax.vmap(
        frame_typepair_observable,
        in_axes=(0, None, None, None, None, None, None),
    )
    return jax.jit(frame_vmap)(
        frames_chunk, inv_ids, K, mu, rc, rcut_val, max_cell_particles
    )  # (n_frames, n_tt_combinations)


def compute_observables_all(
    positions_list: Sequence[jnp.ndarray],  # list length = n_replicates; each (n_frames, n_particles, 3)
    monomer_types: jnp.ndarray,             # (n_particles,)
    mu: float,
    rc: float,
    rcut: Optional[float] = None,
    max_cell_particles: int = 96,
    rep_chunk_size: Optional[int] = None,   # number of replicates per batch; None = all at once
) -> jnp.ndarray:
    """
    Compute observables for all replicates using nested vmaps (replicates × frames).
    Input:
        positions_list : list of length n_replicates; each element is (n_frames, n_particles, 3)
        monomer_types  : (n_particles,)
    Returns:
        (n_replicates, n_frames, n_tt_combinations)
    Notes:
        • All replicates in positions_list must share the same n_frames to stack.
        • Only the outer replicate batching is a Python loop (when rep_chunk_size is set).
    """
    # Shared type mapping and constants
    _, inv_ids, K = _prepare_types(jnp.asarray(monomer_types))
    rcut_val = rcut if rcut is not None else (rc + 4.0 / mu)

    # Per-replicate kernel: vmap over frames (builds directly on the per-frame kernel)
    def _per_replicate(frames_chunk: jnp.ndarray) -> jnp.ndarray:
        frame_vmap = jax.vmap(
            frame_typepair_observable,
            in_axes=(0, None, None, None, None, None, None),
        )
        return frame_vmap(frames_chunk, inv_ids, K, mu, rc, rcut_val, max_cell_particles)  # (n_frames, n_tt_combinations)

    # vmap over replicates expects a stacked array (n_replicates_or_batch, n_frames, n_particles, 3)
    reps_vmap = jax.jit(jax.vmap(_per_replicate, in_axes=(0,)))

    # No batching over replicates → single stack and single vmap
    if rep_chunk_size is None:
        frames_stack = jnp.stack(positions_list, axis=0)  # (n_replicates, n_frames, n_particles, 3)
        return reps_vmap(frames_stack)                    # (n_replicates, n_frames, n_tt_combinations)

    # Batching over replicates to control memory/compile size; only this outer loop is used
    outs = []
    n_replicates = len(positions_list)
    for start in range(0, n_replicates, rep_chunk_size):
        batch = positions_list[start : start + rep_chunk_size]
        frames_stack = jnp.stack(batch, axis=0)          # (rep_chunk_size, n_frames, n_particles, 3)
        outs.append(reps_vmap(frames_stack))             # (rep_chunk_size, n_frames, n_tt_combinations)
    return jnp.concatenate(outs, axis=0)                 # (n_replicates, n_frames, n_tt_combinations)




