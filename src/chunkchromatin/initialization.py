from __future__ import annotations

from typing import List, Tuple
import numpy as np

from math import ceil


# ================================================================
# Helpers
# ================================================================

def _validate_chains(chains: List[Tuple[int, int, bool]]) -> int:
    """Validate that chains cover a contiguous index range [0, N_tot)."""
    if not chains:
        raise ValueError("chains list must not be empty")

    starts = [s for (s, _, _) in chains]
    ends = [e for (_, e, _) in chains]

    if min(starts) != 0:
        raise ValueError("chains must start at index 0")

    N_tot = max(ends)
    covered = np.zeros(N_tot, dtype=bool)
    for s, e, _ in chains:
        if e <= s:
            raise ValueError(f"Non-positive chain length in ({s}, {e})")
        covered[s:e] = True

    if not np.all(covered):
        raise ValueError("Chains must cover a contiguous block of indices with no gaps")

    return N_tot


def _random_unit_vectors(n: int) -> np.ndarray:
    """Sample n random unit vectors uniformly on the sphere."""
    theta = np.random.uniform(0.0, 2.0 * np.pi, n)
    u = np.random.uniform(-1.0, 1.0, n)  # cos(polar angle)
    r_xy = np.sqrt(1.0 - u * u)
    return np.stack((r_xy * np.cos(theta), r_xy * np.sin(theta), u), axis=1)


def _sample_in_sphere(n: int, radius: float, center: np.ndarray) -> np.ndarray:
    """Sample n points uniformly inside a sphere of given radius and center."""
    directions = _random_unit_vectors(n)
    radii = radius * np.random.rand(n) ** (1.0 / 3.0)  # correct r^2 dr distribution
    return center[None, :] + directions * radii[:, None]


def _choose_farthest_point(
    existing: np.ndarray, radius: float, center: np.ndarray, min_sep: float, n_candidates: int = 512
) -> np.ndarray:
    """
    Choose a point inside the sphere that maximizes the minimum distance
    to existing points, subject to a minimum separation if possible.
    """
    candidates = _sample_in_sphere(n_candidates, radius, center)
    if existing.size == 0:
        # No constraints yet
        return candidates[np.random.randint(0, n_candidates)]

    diff = candidates[:, None, :] - existing[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    min_d2 = d2.min(axis=1)

    min_sep2 = min_sep * min_sep
    good = min_d2 >= min_sep2

    if np.any(good):
        # Among points respecting min_sep, choose the farthest
        idx = np.argmax(min_d2[good])
        return candidates[good][idx]

    # If nothing meets min_sep, just pick the farthest candidate overall
    idx = np.argmax(min_d2)
    return candidates[idx]


def _inside_sphere(p: np.ndarray, radius: float, center: np.ndarray) -> bool:
    """Check if p lies inside a sphere of given radius and center."""
    return np.dot(p - center, p - center) <= radius * radius


def _respects_excluded_volume(
    p: np.ndarray, all_positions: np.ndarray, min_sep: float
) -> bool:
    """Check that p is at least min_sep away from all non-NaN positions."""
    mask = ~np.isnan(all_positions[:, 0])
    if not np.any(mask):
        return True
    diff = all_positions[mask] - p
    d2 = np.einsum("ij,ij->i", diff, diff)
    return np.all(d2 >= min_sep * min_sep)


# ================================================================
# Multi-chain constrained random walk in a spherical confinement
# ================================================================

def create_multi_constrained_random_walk(
    chains: List[Tuple[int, int, bool]],
    density: float,
    k_wall: float = 5.0,
    step_size: float = 1.0,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    min_sep: float = 0.8,
    max_attempts_per_bead: int = 10_000,
) -> np.ndarray:
    """
    Initialize multiple chains as constrained random walks inside a spherical
    confinement determined by the same density logic as add_spherical_confinement.

    Parameters
    ----------
    chains : list of (start, end, isRing)
        Chain index ranges. The total number of beads is max(end).
        Currently, isRing is not used (chains are initialized linearly).
    density : float
        Particle number density used to define the confinement radius:
            R = (3 * N_tot / (4*pi*density))^(1/3)
    k_wall : float
        Stiffness parameter from the spherical confinement. Used here to
        stay comfortably inside the wall:
            R_inner ~ R - 2 / k_wall
    step_size : float
        Step size for the random walk.
    center : tuple of float
        Center of the confinement sphere (reduced units).
    min_sep : float
        Minimal allowed distance between any two beads (excluded volume).
    max_attempts_per_bead : int
        Maximum number of trial steps per bead before giving up.

    Returns
    -------
    positions : (N_tot, 3) array
        Initial positions in reduced units. Chain k occupies indices [start:end].
    """
    N_tot = _validate_chains(chains)
    center_arr = np.asarray(center, dtype=float)

    # Same radius as used in add_spherical_confinement for r="density"
    R = (3.0 * N_tot / (4.0 * np.pi * density)) ** (1.0 / 3.0)
    # Stay comfortably inside the wall; avoid degenerate radius
    R_inner = max(R - 2.0 / k_wall, 0.5 * R)

    positions = np.full((N_tot, 3), np.nan, dtype=float)

    for start, end, is_ring in chains:
        if is_ring:
            raise NotImplementedError("Ring chains are not currently supported.")

        L = end - start
        if L <= 0:
            raise ValueError(f"Non-positive chain length for ({start}, {end}).")

        # Choose starting point far from existing beads
        existing_mask = ~np.isnan(positions[:, 0])
        existing_positions = positions[existing_mask]
        p0 = _choose_farthest_point(existing_positions, R_inner, center_arr, min_sep)
        positions[start] = p0

        # Grow the chain bead by bead
        for i in range(start + 1, end):
            attempts = 0
            while True:
                attempts += 1
                if attempts > max_attempts_per_bead:
                    raise RuntimeError(
                        f"Failed to place bead {i} after {max_attempts_per_bead} attempts. "
                        "Try decreasing min_sep or density."
                    )

                step_dir = _random_unit_vectors(1)[0]
                candidate = positions[i - 1] + step_size * step_dir

                if not _inside_sphere(candidate, R_inner, center_arr):
                    continue
                if not _respects_excluded_volume(candidate, positions, min_sep):
                    continue

                positions[i] = candidate
                break

    return positions


# ================================================================
# Multi-chain cubic lattice initializer
# ================================================================

def _grow_cubic_single_chain(
    length: int,
    box_size: int,
    method: str,
    occ: np.ndarray,
    max_seed_attempts: int = 1_000,
) -> np.ndarray:
    """
    Grow a single lattice chain of a given length using a shared occupancy grid.

    Coordinates are maintained in the occupancy grid index space [1..box_size],
    and shifted to [0..box_size-1] at the end, matching the original grow_cubic.
    """
    if length > box_size**3:
        raise ValueError("Chain length exceeds box capacity.")

    # Choose a non-overlapping seed
    for _ in range(max_seed_attempts):
        # Keep neighbors away from the boundary: y,z in [1, box_size-1], x in [1, box_size]
        tx = np.random.randint(1, box_size + 1)
        ty = np.random.randint(1, box_size)
        tz = np.random.randint(1, box_size)

        if method == "standard":
            seed = [
                (tx, ty, tz),
                (tx, ty, tz + 1),
                (tx, ty + 1, tz + 1),
                (tx, ty + 1, tz),
            ]
        elif method == "linear":
            seed = [(tx, ty, z) for z in range(1, box_size + 1)]
            if (len(seed) % 2) != (length % 2):
                seed = seed[1:]
            if len(seed) > length:
                raise ValueError("Chain length too short for 'linear' seed.")
        elif method == "extended":
            seed = [(tx, ty, z) for z in range(1, box_size)] + [
                (tx, ty - 1, z) for z in range(box_size - 1, 0, -1)
            ]
            if len(seed) > length:
                raise ValueError("Chain length too short for 'extended' seed.")
        else:
            raise ValueError("method must be 'standard', 'linear', or 'extended'.")

        # Check overlap
        if all(occ[idx] == 0 for idx in seed):
            break
    else:
        raise RuntimeError("Could not find a non-overlapping seed for this chain.")

    # Mark seed and grow if needed
    chain = list(seed)
    for idx in seed:
        occ[idx] = 1

    # Number of extra pairs to insert (each iteration inserts 2 beads)
    extra_pairs = max(0, ceil((length - len(chain)) / 2.0))

    for _ in range(extra_pairs):
        while True:
            # Choose an edge along the contour
            if method == "linear":
                t = np.random.randint(0, len(chain) - 1)
            else:
                t = np.random.randint(0, len(chain))

            if t != len(chain) - 1:
                c = np.abs(np.array(chain[t]) - np.array(chain[t + 1]))
                t0 = np.array(chain[t])
                t1 = np.array(chain[t + 1])
            else:
                c = np.abs(np.array(chain[t]) - np.array(chain[0]))
                t0 = np.array(chain[t])
                t1 = np.array(chain[0])

            cur_dir = np.argmax(c)  # axis of the current edge
            # Choose a transverse direction
            while True:
                new_dir = np.random.randint(0, 3)
                if new_dir != cur_dir:
                    break

            shift = 1 if np.random.rand() > 0.5 else -1
            delta = np.zeros(3, dtype=int)
            delta[new_dir] = shift

            t3 = t0 + delta
            t4 = t1 + delta

            # Stay within [1, box_size] and avoid overlap
            if (
                np.all(t3 >= 1)
                and np.all(t4 >= 1)
                and np.all(t3 <= box_size)
                and np.all(t4 <= box_size)
                and occ[tuple(t3)] == 0
                and occ[tuple(t4)] == 0
            ):
                chain.insert(t + 1, tuple(t3))
                chain.insert(t + 2, tuple(t4))
                occ[tuple(t3)] = 1
                occ[tuple(t4)] = 1
                break

    # Convert from [1..box_size] to [0..box_size-1]
    return np.array(chain[:length]) - 1


def grow_cubic_multi(
    chains: List[Tuple[int, int, bool]],
    box_size: int,
    method: str = "standard",
) -> np.ndarray:
    """
    Multi-chain version of grow_cubic using a shared occupancy grid.

    Parameters
    ----------
    chains : list of (start, end, isRing)
        Chain index ranges. Total beads = max(end), and ranges must cover
        [0, N_tot) contiguously. isRing is currently ignored.
    box_size : int
        Size of the cubic lattice (number of sites along one axis).
    method : str
        'standard', 'linear', or 'extended', matching the original grow_cubic
        behavior for the seed pattern.

    Returns
    -------
    positions : (N_tot, 3) int array
        Lattice coordinates in [0, box_size-1] for all beads.
    """
    N_tot = _validate_chains(chains)
    if N_tot > box_size**3:
        raise ValueError("Total number of beads exceeds box capacity.")

    occ = np.zeros((box_size + 2, box_size + 2, box_size + 2), dtype=int)
    positions = np.empty((N_tot, 3), dtype=int)

    for start, end, is_ring in chains:
        if is_ring:
            raise NotImplementedError("Ring chains are not currently supported.")
        length = end - start
        chain_coords = _grow_cubic_single_chain(length, box_size, method, occ)
        positions[start:end] = chain_coords

    return positions

# ================================================================
# Voronoi-based multi-chain initializers (territories per chromosome)
# ================================================================


def _compute_confinement_radii(
    N_tot: int, density: float, k_wall: float
) -> tuple[float, float]:
    """Return (R, R_inner) consistent with add_spherical_confinement."""
    R = (3.0 * N_tot / (4.0 * np.pi * density)) ** (1.0 / 3.0)
    R_inner = max(R - 2.0 / k_wall, 0.5 * R)
    return R, R_inner

#usually doesn't work
def _sample_territory_centers_random(
    n_chains: int,
    R_inner: float,
    center: np.ndarray,
    frac: float = 0.7,
) -> np.ndarray:
    """
    Sample one territory center per chain inside a smaller sphere of
    radius frac * R_inner around center.
    """
    R_centers = frac * R_inner
    return _sample_in_sphere(n_chains, R_centers, center)

def _sample_territory_centers(
    n_chains: int,
    R_inner: float,
    center: np.ndarray,
    frac: float = 0.7,
    n_candidates: int = 4096,
) -> np.ndarray:
    """
    Sample one territory center per chain using farthest-point sampling
    inside a smaller sphere of radius frac * R_inner around center.

    This makes Voronoi cells more balanced and avoids pathologically tiny
    regions that can trap random walks.
    """
    R_centers = frac * R_inner

    # Draw a large pool of candidate points inside the inner sphere
    pool = _sample_in_sphere(n_candidates, R_centers, center)

    # First center: random from pool
    centers = [pool[np.random.randint(0, n_candidates)]]

    # Subsequent centers: farthest from existing centers
    for _ in range(1, n_chains):
        diff = pool[:, None, :] - np.array(centers)[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        min_d2 = d2.min(axis=1)
        idx = np.argmax(min_d2)
        centers.append(pool[idx])

    return np.asarray(centers)

def _in_voronoi_region(p: np.ndarray, k: int, centers: np.ndarray) -> bool:
    """Return True if p lies in Voronoi cell of center k (Euclidean metric)."""
    diff = centers - p
    d2 = np.einsum("ij,ij->i", diff, diff)
    return np.argmin(d2) == k


def _choose_farthest_point_in_voronoi(
    existing: np.ndarray,
    R_inner: float,
    center: np.ndarray,
    min_sep: float,
    centers: np.ndarray,
    k: int,
    n_candidates: int = 2048,
    interior_margin: float = 0.5,
) -> np.ndarray:
    """
    Choose a starting point for chain k that is:
      - inside its Voronoi region,
      - inside the nuclear sphere,
      - as far as possible from existing beads,
      - and not too close to Voronoi boundaries.

    The 'interior_margin' parameter controls how far inside the Voronoi
    cell we prefer to be, by preferring points that are noticeably closer
    to center_k than to any other center.
    """
    # Sample candidate points inside sphere
    candidates = _sample_in_sphere(n_candidates, R_inner, center)

    # Compute squared distances to all centers
    diff_cent = candidates[:, None, :] - centers[None, :, :]
    d2_cent = np.einsum("ijk,ijk->ij", diff_cent, diff_cent)
    idx_closest = np.argmin(d2_cent, axis=1)

    # Keep only candidates in Voronoi cell k
    mask_voronoi = idx_closest == k
    candidates = candidates[mask_voronoi]
    d2_cent = d2_cent[mask_voronoi]

    if candidates.shape[0] == 0:
        # Fallback: ignore Voronoi for the starting point if the cell is extremely small
        return _choose_farthest_point(existing, R_inner, center, min_sep)

    # Prefer points that are "deep" in the Voronoi cell: distance to center_k
    # is noticeably less than to any other center.
    d2_k = d2_cent[:, k]
    d2_others = np.delete(d2_cent, k, axis=1) if centers.shape[0] > 1 else None

    if d2_others is not None and d2_others.size > 0:
        # Margin: we want d_k^0.5 + interior_margin <= d_j^0.5 for all j != k
        # Equivalent squared inequality: d_k + 2*m*sqrt(d_k) + m^2 <= d_j (approx),
        # but we'll use a simpler heuristic: require d_j - d_k >= margin^2
        margin2 = interior_margin * interior_margin
        good_interior = np.all(d2_others - d2_k[:, None] >= margin2, axis=1)
        interior_candidates = candidates[good_interior]
        if interior_candidates.shape[0] > 0:
            candidates = interior_candidates

    # Now, among remaining candidates, pick the one farthest from existing beads
    if existing.size == 0:
        # No beads yet; any interior candidate is fine
        return candidates[np.random.randint(0, candidates.shape[0])]

    diff = candidates[:, None, :] - existing[None, :, :]
    d2 = np.einsum("ijk,ijk->ij", diff, diff)
    min_d2 = d2.min(axis=1)

    min_sep2 = min_sep * min_sep
    good = min_d2 >= min_sep2

    if np.any(good):
        idx = np.argmax(min_d2[good])
        return candidates[good][idx]

    # If nothing respects min_sep, just pick farthest candidate overall
    idx = np.argmax(min_d2)
    return candidates[idx]


# ----------------------------------------------------------------
# 1. Voronoi territory – constrained random walk
# ----------------------------------------------------------------

def init_multi_territory_rw(
    chains: list[tuple[int, int, bool]],
    density: float,
    k_wall: float = 5.0,
    step_size: float = 1.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    min_sep: float = 0.8,
    max_attempts_per_bead: int = 10_000,
    territory_center_frac: float = 0.7,
) -> np.ndarray:
    """
    Initialize multiple chains as Voronoi-territory-constrained random walks
    inside a density-based spherical confinement.

    Each chain k is confined to its own Voronoi cell Ω_k defined by centers[k],
    plus the global nuclear sphere of radius R (inner radius R_inner).

    Parameters
    ----------
    chains : list of (start, end, isRing)
        Chain index ranges; must cover [0, N_tot) without gaps.
    density : float
        Particle number density used to define the confinement radius:
            R = (3 * N_tot / (4*pi*density))^(1/3)
    k_wall : float
        Stiffness parameter from the spherical confinement; used to define
        R_inner ~ R - 2 / k_wall.
    step_size : float
        Step size for the random walk.
    center : tuple of float
        Center of the confinement sphere.
    min_sep : float
        Minimal allowed distance between any two beads.
    max_attempts_per_bead : int
        Maximum number of trial steps per bead.
    territory_center_frac : float
        Territory centers are sampled inside a sphere of radius
        territory_center_frac * R_inner.

    Returns
    -------
    positions : (N_tot, 3) array
        Initial positions in reduced units. Chain k occupies indices [start:end].
    """
    N_tot = _validate_chains(chains)
    center_arr = np.asarray(center, dtype=float)

    R, R_inner = _compute_confinement_radii(N_tot, density, k_wall)

    n_chains = len(chains)
    territory_centers = _sample_territory_centers(
        n_chains, R_inner, center_arr, frac=territory_center_frac
    )

    positions = np.full((N_tot, 3), np.nan, dtype=float)

    for k, (start, end, is_ring) in enumerate(chains):
        if is_ring:
            raise NotImplementedError("Ring chains are not currently supported.")
        L = end - start

        # Starting point for this chain
        existing_mask = ~np.isnan(positions[:, 0])
        existing_positions = positions[existing_mask]
        p0 = _choose_farthest_point_in_voronoi(
            existing=existing_positions,
            R_inner=R_inner,
            center=center_arr,
            min_sep=min_sep,
            centers=territory_centers,
            k=k,
        )
        positions[start] = p0

        # Grow the chain bead by bead
        for i in range(start + 1, end):
            attempts = 0
            while True:
                attempts += 1
                if attempts > max_attempts_per_bead:
                    raise RuntimeError(
                        f"Failed to place bead {i} after {max_attempts_per_bead} attempts. "
                        "Try relaxing min_sep, density, or k_wall."
                    )

                step_dir = _random_unit_vectors(1)[0]
                candidate = positions[i - 1] + step_size * step_dir

                if not _inside_sphere(candidate, R_inner, center_arr):
                    continue
                if not _in_voronoi_region(candidate, k, territory_centers):
                    continue
                if not _respects_excluded_volume(candidate, positions, min_sep):
                    continue

                positions[i] = candidate
                break

    return positions


# ----------------------------------------------------------------
# 2. Voronoi territory – crumpled-globule-like configuration
# ----------------------------------------------------------------

def _biased_step_towards(
    prev: np.ndarray,
    target: np.ndarray,
    step_size: float,
    bias_strength: float,
) -> np.ndarray:
    """
    Generate a step of length step_size that interpolates between a random
    direction and the direction towards `target` (crumpling bias).
    """
    rand_dir = _random_unit_vectors(1)[0]
    to_target = target - prev
    norm = np.linalg.norm(to_target)
    if norm < 1e-8:
        dir_vec = rand_dir
    else:
        to_target /= norm
        dir_vec = (1.0 - bias_strength) * rand_dir + bias_strength * to_target
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm < 1e-8:
            dir_vec = rand_dir
        else:
            dir_vec /= dir_norm
    return prev + step_size * dir_vec


def init_multi_territory_crumpled(
    chains: list[tuple[int, int, bool]],
    density: float,
    k_wall: float = 5.0,
    step_size: float = 1.0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    min_sep: float = 0.8,
    block_size: int = 20,
    bias_strength: float = 0.6,
    max_attempts_per_bead: int = 10_000,
    territory_center_frac: float = 0.7,
) -> np.ndarray:
    """
    Initialize multiple chains as crumpled-globule-like configurations,
    each confined to its own Voronoi territory and to a spherical nucleus.

    This construction encourages local compaction by biasing steps toward
    the recent center-of-mass of each chain segment, while respecting:
      - global excluded volume,
      - nuclear confinement (R_inner),
      - Voronoi territory per chain.

    Parameters
    ----------
    chains : list of (start, end, isRing)
        Chain index ranges; must cover [0, N_tot) without gaps.
    density : float
        Particle density for confinement radius.
    k_wall : float
        Stiffness from spherical confinement.
    step_size : float
        Base step size.
    center : tuple of float
        Nuclear center.
    min_sep : float
        Minimal allowed distance between any two beads.
    block_size : int
        Number of recent beads used to define a local "cluster center"
        for the crumpling bias.
    bias_strength : float
        Weight in [0, 1] of the bias towards the local center.
    max_attempts_per_bead : int
        Maximum trial steps per bead.
    territory_center_frac : float
        Radius fraction for sampling territory centers.

    Returns
    -------
    positions : (N_tot, 3) array
        Crumpled-globule-like initial positions.
    """
    N_tot = _validate_chains(chains)
    center_arr = np.asarray(center, dtype=float)

    R, R_inner = _compute_confinement_radii(N_tot, density, k_wall)

    n_chains = len(chains)
    territory_centers = _sample_territory_centers(
        n_chains, R_inner, center_arr, frac=territory_center_frac
    )

    positions = np.full((N_tot, 3), np.nan, dtype=float)

    for k, (start, end, is_ring) in enumerate(chains):
        if is_ring:
            raise NotImplementedError("Ring chains are not currently supported.")
        L = end - start

        # Starting point for this chain (also its initial local cluster)
        existing_mask = ~np.isnan(positions[:, 0])
        existing_positions = positions[existing_mask]
        p0 = _choose_farthest_point_in_voronoi(
            existing=existing_positions,
            R_inner=R_inner,
            center=center_arr,
            min_sep=min_sep,
            centers=territory_centers,
            k=k,
        )
        positions[start] = p0

        for i in range(start + 1, end):
            attempts = 0
            while True:
                attempts += 1
                if attempts > max_attempts_per_bead:
                    raise RuntimeError(
                        f"Failed to place bead {i} after {max_attempts_per_bead} attempts. "
                        "Try relaxing min_sep, density, or k_wall."
                    )

                # Define local "cluster center" for crumpling bias:
                # recent block_size beads or territory center if too early.
                if i - start >= block_size:
                    local_center = positions[i - block_size : i].mean(axis=0)
                else:
                    local_center = territory_centers[k]

                candidate = _biased_step_towards(
                    prev=positions[i - 1],
                    target=local_center,
                    step_size=step_size,
                    bias_strength=bias_strength,
                )

                if not _inside_sphere(candidate, R_inner, center_arr):
                    continue
                if not _in_voronoi_region(candidate, k, territory_centers):
                    continue
                if not _respects_excluded_volume(candidate, positions, min_sep):
                    continue

                positions[i] = candidate
                break

    return positions

