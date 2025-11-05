# jax_impl.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Sequence

import jax
import jax.numpy as jnp

###########
# Containers
###########

@dataclass(frozen=True)
class Params:
    # Bonds/angles
    bond_wiggle: float = 0.05               # -> k_bond = 1/bond_wiggle^2
    bond_length: float = 1.0
    angle_k: float = 1.5                    # kT / rad^2
    angle_theta0: float = jnp.pi

    # Confinement
    conf_k: float = 5.0                     # kT / σ (linear-smoothed)
    conf_density: float = 0.30              # particles per σ^3
    conf_center: Tuple[float,float,float] = (0.0, 0.0, 0.0)

    # Polynomial repulsion
    rep_trunc: float = 3.0                  # kT at r->0
    rep_sigma: float = 1.0                  # σ ; equals radiusMult * σ

    # Tanh type-dependent
    tanh_mu: float = 4.22                   # 1/σ
    tanh_rc: float = 1.82                   # σ

    interaction_matrix: jnp.ndarray = None  # (T, T) float64

@dataclass(frozen=True)
class Static:
    #Changing likely requires a recompile of the jit functions
    # Topology
    bonds: jnp.ndarray              # (M_bonds, 2) int32
    angles: jnp.ndarray             # (M_angles, 3) int32
    types: jnp.ndarray              # (N,) int32 in [0, T-1]
    interaction_matrix: jnp.ndarray # (T, T) float64

    # Cutoffs (hard, no switching)
    rep_cutoff: float = 1.0         # σ
    tanh_cutoff: float = 3.0        # σ

    # Constants for smoothed linear wall
    conf_tt: float = 0.01           # regularizer inside r
    # Note: conf_t = (1/conf_k)/10 computed on the fly per frame via Params.conf_k

#############
# Utilities
#############

def _pairwise_distances(positions: jnp.ndarray) -> jnp.ndarray:
    """
    Dense pairwise distances (no PBC). positions: (N,3)
    Returns: (N,N) matrix with zeros on diagonal.
    """
    # (N,1,3) - (1,N,3) -> (N,N,3)
    d = positions[:, None, :] - positions[None, :, :]
    r2 = jnp.sum(d*d, axis=-1)
    r = jnp.sqrt(jnp.maximum(r2, 0.0))
    return r

def _compute_R_from_density(N: int, density: float) -> float:
    # R = (3N / (4π ρ))^(1/3)
    return (3.0 * N / (4.0 * jnp.pi * density)) ** (1.0 / 3.0)

#########################
# Energy component kernels
#########################

def energy_bonds(positions: jnp.ndarray, bonds: jnp.ndarray, bond_wiggle: float, bond_length: float) -> float:
    # k = 1 / wiggle^2  (kT/σ^2)
    k = 1.0 / (bond_wiggle * bond_wiggle)
    # gather
    pi = positions[bonds[:,0]]
    pj = positions[bonds[:,1]]
    rij = jnp.linalg.norm(pj - pi, axis=-1)
    e = 0.5 * k * (rij - bond_length) ** 2
    return jnp.sum(e)

def energy_angles(positions: jnp.ndarray, angles: jnp.ndarray, k: float, theta0: float) -> float:
    i = positions[angles[:,0]]
    j = positions[angles[:,1]]
    kpos = positions[angles[:,2]]
    v1 = i - j
    v2 = kpos - j
    # angles via cosine
    dot = jnp.sum(v1 * v2, axis=-1)
    n1 = jnp.linalg.norm(v1, axis=-1)
    n2 = jnp.linalg.norm(v2, axis=-1)
    cos_th = jnp.clip(dot / (n1 * n2), -1.0, 1.0)
    theta = jnp.arccos(cos_th)
    e = 0.5 * k * (theta - theta0) ** 2
    return jnp.sum(e)

def energy_confinement(positions: jnp.ndarray, conf_k: float, density: float, center: jnp.ndarray, tt: float) -> float:
    # smoothed linear wall: U = step(r - a) * k * (sqrt((r - a)^2 + t^2) - t)
    # with a = R - 1/k, t = (1/k)/10
    N = positions.shape[0]
    R = _compute_R_from_density(N, density)
    a = R - 1.0 / conf_k
    t = (1.0 / conf_k) / 10.0
    # shift to center
    r = jnp.sqrt(jnp.sum((positions - center) ** 2, axis=1) + tt**2)
    excess = r - a
    contrib = jnp.where(excess > 0.0, conf_k * (jnp.sqrt(excess*excess + t*t) - t), 0.0)
    return jnp.sum(contrib)

def energy_repulsive_dense(positions: jnp.ndarray, rep_trunc: float, rep_sigma: float, cutoff: float) -> float:
    """
    Polynomial repulsion applied to *all pairs* within cutoff (no exclusions).
    U = trunc * (1 + rsc^12 * (rsc^2 - 1) / emin12), rsc = (r/rep_sigma)*sqrt(6/7)

    NOTE: Hard cutoff at r <= cutoff, no shift/switch.
    """
    r = _pairwise_distances(positions)  # (N,N)
    # mask strictly upper triangle to avoid double-counting & self
    N = r.shape[0]
    iu = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
    within = (r <= cutoff) & iu

    rmin12 = jnp.sqrt(6.0 / 7.0)
    emin12 = 46656.0 / 823543.0

    rsc = (r / rep_sigma) * rmin12
    rsc2 = rsc * rsc
    rsc12 = rsc2 ** 6
    U = rep_trunc * (1.0 + rsc12 * (rsc2 - 1.0) / emin12)

    return jnp.sum(jnp.where(within, U, 0.0))

def energy_tanh_dense(positions: jnp.ndarray,
                      types: jnp.ndarray,
                      alpha: jnp.ndarray,
                      mu: float,
                      rc: float,
                      cutoff: float) -> float:
    """
    Tanh type-dependent force on *all pairs* within cutoff (no exclusions).
    U_ij(r) = 0.5 * (1 + tanh(mu * (rc - r))) * alpha[type_i, type_j]
    """
    r = _pairwise_distances(positions)  # (N,N)
    N = r.shape[0]
    iu = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
    within = (r <= cutoff) & iu

    f = 0.5 * (1.0 + jnp.tanh(mu * (rc - r)))  # (N,N)

    # type mixing: alpha[type_i, type_j] for all pairs
    ti = types[:, None]  # (N,1)
    tj = types[None, :]  # (1,N)
    A = alpha[ti, tj]    # (N,N)

    U = f * A
    return jnp.sum(jnp.where(within, U, 0.0))

#############################
# Per-frame energy: main entry
#############################

def energy_components_one_frame(positions: jnp.ndarray, params: Params, static: Static) -> Dict[str, jnp.ndarray]:
    """
    positions: (N,3) float64
    returns dict of components (float64 scalars) and 'total'
    """
    # Cast center and constants
    center = jnp.asarray(params.conf_center, dtype=positions.dtype)

    e_bond = energy_bonds(
        positions, static.bonds, params.bond_wiggle, params.bond_length
    )

    e_angle = energy_angles(
        positions, static.angles, params.angle_k, params.angle_theta0
    )

    e_conf = energy_confinement(
        positions, params.conf_k, params.conf_density, center, static.conf_tt
    )

    # Nonbonded (dense masked)
    e_rep = energy_repulsive_dense(
        positions, params.rep_trunc, params.rep_sigma, static.rep_cutoff
    )

    e_tanh = energy_tanh_dense(
        positions, static.types, static.interaction_matrix,
        params.tanh_mu, params.tanh_rc, static.tanh_cutoff
    )

    total = e_bond + e_angle + e_conf + e_rep + e_tanh
    return {
        "bond": e_bond,
        "angle": e_angle,
        "conf": e_conf,
        "rep": e_rep,
        "tanh": e_tanh,
        "total": total,
    }

# vmapped, jitted version over frames
_batched_components = jax.jit(
    jax.vmap(energy_components_one_frame, in_axes=(0, None, None)),  # frames, shared params/static
    static_argnames=("static",)
)

def compute_energies_chunk(frames_chunk: jnp.ndarray, params: Params, static: Static) -> Dict[str, jnp.ndarray]:
    """
    frames_chunk: (B, N, 3) float64 frames (B = frames * reps in this chunk)
    Returns dict of arrays (B,) for each component and total.
    """
    comps = _batched_components(frames_chunk, params, static)
    # jax.vmap over dict returns a dict of (B,) via PyTree mapping
    return {k: v for k, v in comps.items()}

#############################
# End-to-end batching helpers
#############################

def compute_energies_all(frame_blocks: Sequence[jnp.ndarray],
                         params: Params,
                         static: Static,
                         chunk_reps: int = 10) -> Dict[str, jnp.ndarray]:
    """
    frame_blocks: list of per-replicate arrays, each (3500, N, 3) float64 for frames 500..3999
    We process chunk_reps replicates at a time and return a flat vector over all replicates.
    """
    per_key = []
    for i in range(0, len(frame_blocks), chunk_reps):
        reps = frame_blocks[i:i+chunk_reps]  # list of arrays
        # stack along frames-first axis: (chunk_reps*3500, N, 3)
        chunk = jnp.concatenate(reps, axis=0)
        out = compute_energies_chunk(chunk, params, static)  # dict of (B,)
        per_key.append(out)

    # concatenate across chunks
    keys = per_key[0].keys()
    merged = {k: jnp.concatenate([d[k] for d in per_key], axis=0) for k in keys}
    return merged
