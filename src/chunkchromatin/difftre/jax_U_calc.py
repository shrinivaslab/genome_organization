# jax_U_calc.py
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Dict, Tuple, Optional, Sequence, List

from jax import config as jax_config
jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from scipy import constants

###########
# Containers
###########

@dataclass(frozen=True)
class Params:
    # Bonds/angles (all as jnp arrays for traceability)
    bond_wiggle: jnp.ndarray = jnp.array(0.05)               # -> k_bond = 1/bond_wiggle^2
    bond_length: jnp.ndarray = jnp.array(1.0)
    angle_k: jnp.ndarray = jnp.array(1.5)                    # kT / rad^2
    angle_theta0: jnp.ndarray = jnp.array(jnp.pi)
    angle_global: jnp.ndarray = jnp.array(1.0)               # OpenMM global scale (e.g., force_kT)

    # Confinement
    conf_k: jnp.ndarray = jnp.array(5.0)                     # kT / σ (linear-smoothed)
    conf_density: jnp.ndarray = jnp.array(0.30)              # particles per σ^3
    conf_center: jnp.ndarray = jnp.array([0.0, 0.0, 0.0])    # (3,) array

    # Polynomial repulsion
    rep_trunc: jnp.ndarray = jnp.array(3.0)                  # kT at r->0
    rep_sigma: jnp.ndarray = jnp.array(1.0)                  # σ ; equals radiusMult * σ

    # Tanh type-dependent
    tanh_mu: jnp.ndarray = jnp.array(4.22)                   # 1/σ
    tanh_rc: jnp.ndarray = jnp.array(1.82)                   # σ

    # Not used in kernels (kept in Static); left for compatibility
    interaction_matrix: Optional[jnp.ndarray] = None  # (T, T) float64


@dataclass(frozen=True)
class Static:
    # Changing likely requires a recompile of the jit functions
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

    # Optional: reproduce OpenMM-style exclusions or scaling
    allowed_mask: Optional[jnp.ndarray] = None  # (N,N) bool
    tanh_scale: float = 1.0                     # global scale on tanh nonbonded

    def __hash__(self):
        """Make Static hashable for JAX static_argnames."""
        def _hash_array(arr):
            if arr is None:
                return hash(None)
            return hash(np.asarray(arr).tobytes())

        return hash((
            _hash_array(self.bonds),
            _hash_array(self.angles),
            _hash_array(self.types),
            _hash_array(self.interaction_matrix),
            self.rep_cutoff,
            self.tanh_cutoff,
            self.conf_tt,
            _hash_array(self.allowed_mask),
            self.tanh_scale,
        ))

    def __eq__(self, other):
        """Equality comparison for Static objects."""
        if not isinstance(other, Static):
            return False

        def _arrays_equal(a, b):
            if a is None and b is None:
                return True
            if a is None or b is None:
                return False
            return np.array_equal(np.asarray(a), np.asarray(b))

        return (
            _arrays_equal(self.bonds, other.bonds) and
            _arrays_equal(self.angles, other.angles) and
            _arrays_equal(self.types, other.types) and
            _arrays_equal(self.interaction_matrix, other.interaction_matrix) and
            self.rep_cutoff == other.rep_cutoff and
            self.tanh_cutoff == other.tanh_cutoff and
            self.conf_tt == other.conf_tt and
            _arrays_equal(self.allowed_mask, other.allowed_mask) and
            self.tanh_scale == other.tanh_scale
        )


# Register Params as a JAX pytree so it can be traced through JIT
def _params_flatten(params):
    """Flatten Params into (values, metadata) for pytree registration."""
    values = (
        params.bond_wiggle, params.bond_length, params.angle_k, params.angle_theta0, params.angle_global,
        params.conf_k, params.conf_density, params.conf_center,
        params.rep_trunc, params.rep_sigma,
        params.tanh_mu, params.tanh_rc, params.interaction_matrix
    )
    return values, None

def _params_unflatten(aux, values):
    """Unflatten values back into a Params object."""
    return Params(*values)

tree_util.register_pytree_node(Params, _params_flatten, _params_unflatten)


#############
# Utilities
#############

def _pairwise_distances(positions: jnp.ndarray) -> jnp.ndarray:
    """
    Dense pairwise distances (no PBC). positions: (N,3)
    Returns: (N,N) matrix with zeros on diagonal.
    """
    d = positions[:, None, :] - positions[None, :, :]
    r2 = jnp.sum(d*d, axis=-1)
    r = jnp.sqrt(jnp.maximum(r2, 0.0))
    return r

def _compute_R_from_density(N: int, density: float) -> float:
    # R = (3N / (4π ρ))^(1/3)
    return (3.0 * N / (4.0 * jnp.pi * density)) ** (1.0 / 3.0)

def make_linear_bonds_angles_from_chains(chains: List[Tuple[int,int,bool]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    chains: list of (start, end, is_ring). end is exclusive.
    Returns (bonds:(M,2), angles:(K,3)) int32
    """
    bonds = []
    angles = []
    for start, end, is_ring in chains:
        L = end - start
        # bonds
        for i in range(L - 1):
            bonds.append((start + i, start + i + 1))
        if is_ring and L > 2:
            bonds.append((start + L - 1, start))
        # angles
        for i in range(L - 2):
            angles.append((start + i, start + i + 1, start + i + 2))
        if is_ring and L > 2:
            angles.append((start + L - 2, start + L - 1, start))
            angles.append((start + L - 1, start, start + 1))
    if len(bonds) == 0:
        b = jnp.zeros((0,2), dtype=jnp.int32)
    else:
        b = jnp.asarray(bonds, dtype=jnp.int32)
    if len(angles) == 0:
        a = jnp.zeros((0,3), dtype=jnp.int32)
    else:
        a = jnp.asarray(angles, dtype=jnp.int32)
    return b, a

def make_nonbonded_mask(N: int, bonds: jnp.ndarray, angles: jnp.ndarray) -> jnp.ndarray:
    """Upper-triangle boolean mask for allowed nonbonded pairs, excluding 1–2 and 1–3."""
    iu = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)

    # 1-2 exclusions
    ex12 = jnp.zeros((N, N), dtype=bool)
    if bonds.size:
        ex12 = ex12.at[bonds[:,0], bonds[:,1]].set(True)
        ex12 = ex12 | ex12.T

    # 1-3 exclusions
    ex13 = jnp.zeros((N, N), dtype=bool)
    if angles.size:
        ex13 = ex13.at[angles[:,0], angles[:,2]].set(True)
        ex13 = ex13 | ex13.T

    return iu & ~(ex12 | ex13)

def build_params_static_from_inputs(
    monomer_types: np.ndarray,
    interaction_matrix: np.ndarray,
    chains: List[Tuple[int,int,bool]],
    force_kwargs: dict,
    density: float,
) -> Tuple[Params, Static]:
    """
    Map OpenMM-style inputs to JAX Params/Static, with exclusions mask.
    """
    # Topology
    bonds, angles = make_linear_bonds_angles_from_chains(chains)

    # Invariant: for any (a,b,c), they must be consecutive: a+1==b and b+1==c (non-ring angles)
    # We allow the two ring wraparounds only when is_ring=True
    if angles.size:
        a = np.asarray(angles)
        diffs = a[:,1] - a[:,0], a[:,2] - a[:,1]
        # Count how many are 1-1 vs wraparounds (we don't know chain ring flags here,
        # so just report; you can eyeball quickly).
        n_11 = int(np.sum((diffs[0]==1) & (diffs[1]==1)))
        n_wrap = int(a.shape[0] - n_11)
        print(f"[JAX] consecutive 1-1 angles: {n_11}, wraparounds (ring-only): {n_wrap}")

    #########################################################
    types = jnp.asarray(monomer_types.astype(np.int32))
    alpha = jnp.asarray(interaction_matrix, dtype=jnp.float64)

    # Forces mapping
    bondWiggle = float(force_kwargs["harmonic_bonds"]["bondWiggleDistance"])
    bondLength = float(force_kwargs["harmonic_bonds"]["bondLength"])

    angle_k     = float(force_kwargs["angle_force"]["k"])
    angle_theta = float(force_kwargs["angle_force"]["theta_0"])
    # IMPORTANT: keep angle_global dimensionless (no kT here)
    angle_global = 1.0

    conf_k      = float(force_kwargs["spherical_confinement"]["k"])
    conf_center = tuple(force_kwargs["spherical_confinement"]["center"])
    conf_density = float(density)

    rep_trunc   = float(force_kwargs["polynomial_repulsive"]["trunc"])
    rep_sigma   = float(force_kwargs["polynomial_repulsive"]["radiusMult"])
    rep_cutoff  = rep_sigma  # match JAX kernel behavior

    tanh_mu     = float(force_kwargs["tanh_type_force"]["mu"])
    tanh_rc     = float(force_kwargs["tanh_type_force"]["rc"])
    tanh_cutoff = float(force_kwargs["tanh_type_force"]["rCutoff"])
    tanh_scale  = float(force_kwargs["tanh_type_force"].get("scale", 1.0))   # NEW

    # OpenMM CustomNonbondedForce does NOT exclude bonded pairs by default
    # So we set allowed_mask = None to include all pairs (matching OpenMM behavior)
    allowed_mask = None

    params = Params(
        bond_wiggle=jnp.array(bondWiggle),
        bond_length=jnp.array(bondLength),
        angle_k=jnp.array(angle_k),
        angle_theta0=jnp.array(angle_theta),
        angle_global=jnp.array(angle_global), # must be 1.0 (dimensionless)
        conf_k=jnp.array(conf_k),
        conf_density=jnp.array(conf_density),
        conf_center=jnp.array(conf_center),
        rep_trunc=jnp.array(rep_trunc),
        rep_sigma=jnp.array(rep_sigma),
        tanh_mu=jnp.array(tanh_mu),
        tanh_rc=jnp.array(tanh_rc),
        interaction_matrix=None,
    )

    static = Static(
        bonds=bonds,
        angles=angles,
        types=types,
        interaction_matrix=alpha,
        rep_cutoff=rep_cutoff,
        tanh_cutoff=tanh_cutoff,
        conf_tt=0.01,
        allowed_mask=allowed_mask,
        tanh_scale=tanh_scale,                                # NEW
    )

    return params, static


#########################
# Energy component kernels
#########################

def energy_bonds(positions: jnp.ndarray, bonds: jnp.ndarray, bond_wiggle: float, bond_length: float) -> float:
    """Harmonic bond energy in kT units."""
    if bonds.size == 0:
        return jnp.array(0.0, dtype=positions.dtype)
    k = 1.0 / (bond_wiggle * bond_wiggle)
    pi = positions[bonds[:,0]]
    pj = positions[bonds[:,1]]
    rij = jnp.linalg.norm(pj - pi, axis=-1)
    # OpenMM HarmonicBondForce: U = 0.5 * k * (r - r0)^2
    e = 0.5 * k * (rij - bond_length) ** 2
    return jnp.sum(e)

def energy_angles(
    positions: jnp.ndarray,
    angles: jnp.ndarray,
    k: float,
    theta0: float,
    global_scale: float = 1.0
) -> float:
    """Angle force energy in kT units, using atan2(|v1×v2|, v1·v2) for OpenMM-like geometry."""
    if angles.size == 0:
        return jnp.array(0.0, dtype=positions.dtype)

    # global_scale must be dimensionless (~1.0); do not pass kT here
    if isinstance(global_scale, (float, int)) and global_scale > 1.1:
        raise ValueError("angle global_scale must be dimensionless (~1.0). Do not pass kT here.")

    i = positions[angles[:, 0]]
    j = positions[angles[:, 1]]
    kpos = positions[angles[:, 2]]

    v1 = i - j
    v2 = kpos - j

    # Stable angle: theta = atan2(|v1×v2|, v1·v2)
    cross = jnp.linalg.norm(jnp.cross(v1, v2), axis=-1)
    dot   = jnp.sum(v1 * v2, axis=-1)

    # Tiny epsilons avoid NaNs when bonds are nearly colinear or zero-length
    eps = jnp.finfo(positions.dtype).eps
    theta = jnp.arctan2(jnp.maximum(cross, eps), jnp.where(dot == 0.0, eps, dot))

    e = global_scale * 0.5 * k * (theta - theta0) ** 2  # remains in kT
    return jnp.sum(e)

def energy_confinement(positions: jnp.ndarray, conf_k: float, density: float, center: jnp.ndarray, tt: float) -> float:
    """Spherical confinement energy in kT units."""
    # Smoothed linear wall: U = step(r - a) * k * (sqrt((r - a)^2 + t^2) - t)
    # with a = R - 1/k, t = (1/k)/10
    N = positions.shape[0]
    R = _compute_R_from_density(N, density)
    a = R - 1.0 / conf_k
    t = (1.0 / conf_k) / 10.0
    r = jnp.sqrt(jnp.sum((positions - center) ** 2, axis=1) + tt**2)
    excess = r - a
    contrib = jnp.where(excess > 0.0, conf_k * (jnp.sqrt(excess*excess + t*t) - t), 0.0)
    return jnp.sum(contrib)

def energy_repulsive_dense(positions: jnp.ndarray, rep_trunc: float, rep_sigma: float, cutoff: float,
                           allowed_mask: Optional[jnp.ndarray]=None) -> float:
    """
    Polynomial repulsion for allowed pairs within cutoff in kT units.
    U = trunc * (1 + rsc^12 * (rsc^2 - 1) / emin12), rsc = (r/rep_sigma)*sqrt(6/7)
    
    Note: OpenMM CustomNonbondedForce does NOT exclude bonded pairs by default,
    so we include all pairs (i < j) to match OpenMM behavior.
    """
    r = _pairwise_distances(positions)  # (N,N)
    N = r.shape[0]
    iu = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
    # Always use upper triangle mask - no exclusions to match OpenMM
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
                      cutoff: float,
                      allowed_mask: Optional[jnp.ndarray]=None,
                      scale: float = 1.0) -> float:
    """
    Tanh type-dependent force on allowed pairs within cutoff in kT units.
    U_ij(r) = scale * 0.5 * (1 + tanh(mu * (rc - r))) * alpha[type_i, type_j]
    
    Note: OpenMM CustomNonbondedForce does NOT exclude bonded pairs by default,
    so we include all pairs (i < j) to match OpenMM behavior.
    """
    r = _pairwise_distances(positions)  # (N,N)
    N = r.shape[0]
    iu = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
    # Always use upper triangle mask - no exclusions to match OpenMM
    within = (r <= cutoff) & iu

    f = 0.5 * (1.0 + jnp.tanh(mu * (rc - r)))  # (N,N)

    ti = types[:, None]  # (N,1)
    tj = types[None, :]  # (1,N)
    A = alpha[ti, tj]    # (N,N)

    U = scale * f * A
    return jnp.sum(jnp.where(within, U, 0.0))

#############################
# Per-frame energy: main entry
#############################


def energy_components_one_frame(
    positions: jnp.ndarray,
    params: Params,
    static: Static,
    return_components: bool = False
) -> Dict[str, jnp.ndarray] | jnp.ndarray:
    """
    Calculate energy components for one frame.

    Parameters
    ----------
    positions : jnp.ndarray
        (N,3) float64 particle positions
    params : Params
        Force parameters
    static : Static
        Static simulation parameters
    return_components : bool, optional
        If True, return a dictionary of all components.
        If False (default), return only the total energy (float64 scalar).

    Returns
    -------
    Union[Dict[str, jnp.ndarray], jnp.ndarray]
        - If return_components=True: dict with energies in kJ/mol.
        - If return_components=False: total energy (kJ/mol).
    """
    # Unit conversion: kT -> kJ/mol at T=300K
    T = 300.0  # K
    kB_kJ_per_mol_K = constants.k * constants.N_A / 1000.0  # kJ/(mol·K)
    kT_to_kJmol = kB_kJ_per_mol_K * T  # kJ/mol

    center = jnp.asarray(params.conf_center, dtype=positions.dtype)

    # Calculate energies in kT units
    e_bond_kT = energy_bonds(positions, static.bonds, params.bond_wiggle, params.bond_length)
    e_angle_kT = energy_angles(positions, static.angles, params.angle_k, params.angle_theta0, params.angle_global)
    e_conf_kT = energy_confinement(positions, params.conf_k, params.conf_density, center, static.conf_tt)
    e_rep_kT = energy_repulsive_dense(
        positions, params.rep_trunc, params.rep_sigma, static.rep_cutoff, static.allowed_mask
    )
    e_tanh_kT = energy_tanh_dense(
        positions, static.types, static.interaction_matrix,
        params.tanh_mu, params.tanh_rc, static.tanh_cutoff, static.allowed_mask, static.tanh_scale
    )

    # Convert to kJ/mol
    e_bond = e_bond_kT * kT_to_kJmol
    e_angle = e_angle_kT * kT_to_kJmol
    e_conf = e_conf_kT * kT_to_kJmol
    e_rep = e_rep_kT * kT_to_kJmol
    e_tanh = e_tanh_kT * kT_to_kJmol
    total = e_bond + e_angle + e_conf + e_rep + e_tanh

    if not return_components:
        return total

    return {
        "bond": e_bond,
        "angle": e_angle,
        "conf": e_conf,
        "rep": e_rep,
        "tanh": e_tanh,
        "total": total,
    }


# =========================
# Batched JIT-compiled versions
# =========================

# Batched total energies only (default)
_batched_total = jax.jit(
    jax.vmap(
        lambda x, p, s: energy_components_one_frame(x, p, s, return_components=False),
        in_axes=(0, None, None)
    ),
    static_argnames=("s",)
)

# Batched full component energies (on-demand)
_batched_full = jax.jit(
    jax.vmap(
        lambda x, p, s: energy_components_one_frame(x, p, s, return_components=True),
        in_axes=(0, None, None)
    ),
    static_argnames=("s",)
)


def compute_energies_chunk(
    frames_chunk: jnp.ndarray,
    params: Params,
    static: Static,
    return_components: bool = False
) -> Dict[str, jnp.ndarray] | jnp.ndarray:
    """
    Compute energies for a batch of frames.

    Parameters
    ----------
    frames_chunk : (B, N, 3)
    params : Params
    static : Static
    return_components : bool, optional
        If True, return a dict of arrays for each component.
        If False (default), return only total energies (B,).

    Returns
    -------
    Union[Dict[str, jnp.ndarray], jnp.ndarray]
    """
    if return_components:
        comps = _batched_full(frames_chunk, params, static)
        return {k: v for k, v in comps.items()}
    else:
        return _batched_total(frames_chunk, params, static)


def compute_energies_all(
    frame_blocks: Sequence[jnp.ndarray],
    params: Params,
    static: Static,
    chunk_reps: int = 10,
    return_components: bool = False
) -> Dict[str, jnp.ndarray] | jnp.ndarray:
    """
    Compute energies across all replicates.

    Parameters
    ----------
    frame_blocks : list of arrays (frames, N, 3)
    params : Params
    static : Static
    chunk_reps : int
    return_components : bool, optional
        If True, return dict of concatenated arrays per component.
        If False (default), return 1D array of total energies.

    Returns
    -------
    Union[Dict[str, jnp.ndarray], jnp.ndarray]
    shape of the returned array is (n_frames*n_replicates,)
    """
    per_key = []
    for i in range(0, len(frame_blocks), chunk_reps):
        reps = frame_blocks[i:i+chunk_reps]
        chunk = jnp.concatenate(reps, axis=0)
        out = compute_energies_chunk(chunk, params, static, return_components)
        per_key.append(out)

    if not return_components:
        # Concatenate arrays of total energies
        return jnp.concatenate(per_key, axis=0)

    # Merge dicts of components
    keys = per_key[0].keys()
    merged = {k: jnp.concatenate([d[k] for d in per_key], axis=0) for k in keys}
    return merged

