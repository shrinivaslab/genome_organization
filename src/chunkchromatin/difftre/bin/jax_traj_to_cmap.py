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

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

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
    rcut: Optional[float] = None,
    max_cell_particles: int = 96,
) -> jnp.ndarray:
    """
    Compute the per-frame contact map using a cell-list neighbor search with overflow fallback.
    Shapes:
        positions: (N,3), returns: (N, N)
    Accumulate f_switch over unique pairs within rcut via a fast cell list, falling back to dense if any cell overflows.
    """
    if rcut is None:
        rcut = rc + 4.0 / mu

    # Build neighbor candidates and detect overflow
    nbr_idx, overflow = _build_neighbor_list(positions, rcut, max_cell_particles)  # (N,C), ()

    def _neighbor_path(_: None) -> jnp.ndarray:
        return _frame_contact_map_neighbor(positions, nbr_idx, mu, rc, rcut)

    def _dense_path(_: None) -> jnp.ndarray:
        return _frame_contact_map_dense(positions, mu, rc, rcut)

    # Overflow gate: if any cell exceeded the cap, switch to dense for correctness
    return lax.cond(overflow, _dense_path, _neighbor_path, operand=None)


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


def contact_map_from_traj(
    positions: np.ndarray,
    mu: float = MU_DEFAULT,
    rc: float = RC_DEFAULT,
    rcut: Optional[float] = None,
    max_cell_particles: int = 96,
) -> np.ndarray:
    """
    Calculate contact map from positions array (F, N, 3) using JAX.
    Returns (N, N) contact map.
    """
    F, N, _ = positions.shape
    if rcut is None:
        rcut = rc + 4.0 / mu

    # Convert to JAX array
    positions_jax = jnp.asarray(positions)
    
    # JIT compile the per-frame function
    frame_fn = jax.jit(frame_contact_map, static_argnames=('mu', 'rc', 'rcut', 'max_cell_particles'))
    
    # Process frames in batches to avoid memory issues
    contact_map = jnp.zeros((N, N), dtype=jnp.float64)
    
    # Process all frames using vmap for efficiency
    frame_vmap = jax.jit(jax.vmap(
        frame_contact_map,
        in_axes=(0, None, None, None, None),
    ), static_argnames=('mu', 'rc', 'rcut', 'max_cell_particles'))
    
    # Compute contact maps for all frames at once
    frame_maps = frame_vmap(positions_jax, mu, rc, rcut, max_cell_particles)  # (F, N, N)
    
    # Sum over frames
    contact_map = jnp.sum(frame_maps, axis=0)  # (N, N)
    
    return np.asarray(contact_map)


def main():
    parser = argparse.ArgumentParser(
        description="Build a contact map from trajectory .traj files using JAX"
    )
    parser.add_argument("traj_glob", help="Glob path to trajectory files, e.g. 'sims/rep*/trajectory.traj'")
    parser.add_argument("output", help="Output .npy filename for contact map")
    parser.add_argument("--mu", type=float, default=MU_DEFAULT, help="Switch mu parameter (default: 4.22)")
    parser.add_argument("--rc", type=float, default=RC_DEFAULT, help="Switch rc parameter (default: 1.82)")
    parser.add_argument("--rcut", type=float, default=None, help="Cutoff distance (default: rc + 4/mu)")
    parser.add_argument("--max-cell-particles", type=int, default=96, help="Max particles per cell (default: 96)")
    args = parser.parse_args()

    # Find files
    all_traj_paths = glob.glob(args.traj_glob)
    if len(all_traj_paths) == 0:
        raise FileNotFoundError(f"No trajectory files found for {args.traj_glob}")
    print(f"Found {len(all_traj_paths)} trajectory files")

    # Load positions
    results = []
    skip_frames = 400
    for traj_path in tqdm(all_traj_paths, desc="Loading trajectories"):
        pos = load_all_positions(traj_path)
        if pos.shape[0] <= skip_frames:
            print(f"Warning: {traj_path} has only {pos.shape[0]} frames, skipping all frames")
            continue
        # Discard first 400 frames
        pos = pos[skip_frames:]
        results.append(pos)

    positions = np.concatenate(results, axis=0)
    print("Loaded positions:", positions.shape)

    # Build contact map
    print("Computing contact map with JAX...")
    contact_map = contact_map_from_traj(
        positions,
        mu=args.mu,
        rc=args.rc,
        rcut=args.rcut,
        max_cell_particles=args.max_cell_particles,
    )

    # Save as .npy
    np.save(args.output, contact_map)
    print(f"Saved contact map to {args.output}")

if __name__ == "__main__":
    main()

