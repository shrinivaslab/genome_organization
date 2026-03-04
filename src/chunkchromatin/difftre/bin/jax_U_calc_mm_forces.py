# jax_U_calc_mm_forces_test.py
"""
JAX-based energy calculations for OpenMiChroM forces.

This module implements exact JAX replicas of the OpenMiChroM force field
implementations from chromosome_michrom.py, ensuring energy calculations
match OpenMM energies for use in the DiffTre pipeline.
"""
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
class ParamsMM:
    """Parameters for OpenMiChroM forces (all in reduced units, matching OpenMiChroM)."""
    # FENE bonds
    kFb: jnp.ndarray = jnp.array(30.0)              # FENE bond coefficient
    fr0: jnp.ndarray = jnp.array(1.5)               # FENE bond maximum extension
    epsilon: jnp.ndarray = jnp.array(1.0)            # LJ epsilon (reduced units)
    sigma: jnp.ndarray = jnp.array(1.0)             # LJ sigma (reduced units)
    
    # Angles (cosine-based)
    kA: jnp.ndarray = jnp.array(2.0)                 # Angle coefficient (can be per-angle array)
    
    # Repulsive soft-core
    eCut: jnp.ndarray = jnp.array(4.0)               # Energy cost for chain crossing (kT)
    
    # Type-to-type interactions
    mu: jnp.ndarray = jnp.array(3.22)                # Tanh kernel parameter
    rc: jnp.ndarray = jnp.array(1.78)                 # Tanh kernel parameter
    
    # Ideal chromosome
    gamma1: jnp.ndarray = jnp.array(-0.030)          # Ideal chromosome parameter 1
    gamma2: jnp.ndarray = jnp.array(-0.351)          # Ideal chromosome parameter 2
    gamma3: jnp.ndarray = jnp.array(-3.727)          # Ideal chromosome parameter 3
    lambda_IC: Optional[jnp.ndarray] = None          # Full-phi IC parameters (per genomic distance)
    d_init: jnp.ndarray = jnp.array(3)               # Minimum genomic distance
    d_end: jnp.ndarray = jnp.array(500)               # Maximum genomic distance
    
    # Loops
    qsi: jnp.ndarray = jnp.array(-1.612990)          # Loop interaction parameter
    
    # Flat bottom harmonic confinement
    kR: jnp.ndarray = jnp.array(5e-3)                # Spring constant
    nRad: jnp.ndarray = jnp.array(10.0)               # Nucleus radius
    
    # Interaction matrix (stored in Static, but kept here for compatibility)
    interaction_matrix: Optional[jnp.ndarray] = None


@dataclass(frozen=True)
class StaticMM:
    """Static parameters for OpenMiChroM forces (topology, cutoffs, etc.)."""
    # Topology
    bonds: jnp.ndarray              # (M_bonds, 2) int32
    triplets: jnp.ndarray           # (M_angles, 3) int32
    types: jnp.ndarray              # (N,) int32 in [0, T-1]
    interaction_matrix: jnp.ndarray # (T, T) float64
    
    # Loop pairs (from loop files)
    loop_pairs: jnp.ndarray         # (M_loops, 2) int32 (0-indexed)
    
    # Cutoffs
    rep_cutoff: float = 3.0         # Repulsive soft-core cutoff
    type_cutoff: float = 3.0        # Type-to-type cutoff
    ic_cutoff: float = 3.0          # Ideal chromosome cutoff
    lim: float = 1.0                 # Minimum distance for type-to-type (step(r-lim))
    
    # Exclusions (only bonded pairs, d=1, matching OpenMiChroM)
    # OpenMiChroM only excludes bonded pairs, not d=2
    allowed_mask: Optional[jnp.ndarray] = None  # (N,N) bool for nonbonded pairs
    
    def __hash__(self):
        """Make StaticMM hashable for JAX static_argnames."""
        def _hash_array(arr):
            if arr is None:
                return hash(None)
            return hash(np.asarray(arr).tobytes())
        
        return hash((
            _hash_array(self.bonds),
            _hash_array(self.triplets),
            _hash_array(self.types),
            _hash_array(self.interaction_matrix),
            _hash_array(self.loop_pairs),
            self.rep_cutoff,
            self.type_cutoff,
            self.ic_cutoff,
            self.lim,
            _hash_array(self.allowed_mask),
        ))
    
    def __eq__(self, other):
        """Equality comparison for StaticMM objects."""
        if not isinstance(other, StaticMM):
            return False
        
        def _arrays_equal(a, b):
            if a is None and b is None:
                return True
            if a is None or b is None:
                return False
            return np.array_equal(np.asarray(a), np.asarray(b))
        
        return (
            _arrays_equal(self.bonds, other.bonds) and
            _arrays_equal(self.triplets, other.triplets) and
            _arrays_equal(self.types, other.types) and
            _arrays_equal(self.interaction_matrix, other.interaction_matrix) and
            _arrays_equal(self.loop_pairs, other.loop_pairs) and
            self.rep_cutoff == other.rep_cutoff and
            self.type_cutoff == other.type_cutoff and
            self.ic_cutoff == other.ic_cutoff and
            self.lim == other.lim and
            _arrays_equal(self.allowed_mask, other.allowed_mask)
        )


# Register ParamsMM as a JAX pytree
def _params_mm_flatten(params):
    """Flatten ParamsMM into (values, metadata) for pytree registration."""
    values = (
        params.kFb, params.fr0, params.epsilon, params.sigma,
        params.kA,
        params.eCut,
        params.mu, params.rc,
        params.gamma1, params.gamma2, params.gamma3, params.lambda_IC, params.d_init, params.d_end,
        params.qsi,
        params.kR, params.nRad,
        params.interaction_matrix
    )
    return values, None

def _params_mm_unflatten(aux, values):
    """Unflatten values back into a ParamsMM object."""
    return ParamsMM(*values)

tree_util.register_pytree_node(ParamsMM, _params_mm_flatten, _params_mm_unflatten)


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
    r = jnp.sqrt(jnp.maximum(r2, 1e-10))  # Small epsilon to avoid exact zeros
    return r

def _genomic_distance(i: int, j: int) -> int:
    """Compute genomic distance |i - j|."""
    return abs(i - j)

def make_bonds_triplets_from_chains(chains: List[Tuple[int,int,bool]], N: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate bonds and triplets from chains, matching ChromosomeMichroM._generate_bonds/_generate_triplets.
    
    chains: list of (start, end, is_ring). end is exclusive.
    Returns (bonds:(M,2), triplets:(K,3)) int32
    """
    bonds = []
    triplets = []
    for start, end, is_ring in chains:
        end = N if end is None else end
        # Linear bonds: (j, j+1) for j in [start, end-1)
        for j in range(start, end - 1):
            bonds.append((j, j + 1))
        # Ring closure: OpenMiChroM uses (start, end) directly
        if is_ring:
            bonds.append((start, end))
        
        # Linear chain angles: (j-1, j, j+1) for j in [start+1, end-1)
        for j in range(start + 1, end - 1):
            triplets.append((j - 1, j, j + 1))
        # Ring angles: OpenMiChroM uses (end-1, end, start) and (end, start, start+1)
        if is_ring:
            triplets.append((end - 1, end, start))
            triplets.append((end, start, start + 1))
    
    if len(bonds) == 0:
        b = jnp.zeros((0,2), dtype=jnp.int32)
    else:
        b = jnp.asarray(bonds, dtype=jnp.int32)
    if len(triplets) == 0:
        t = jnp.zeros((0,3), dtype=jnp.int32)
    else:
        t = jnp.asarray(triplets, dtype=jnp.int32)
    return b, t

def make_nonbonded_mask(N: int, bonds: jnp.ndarray) -> jnp.ndarray:
    """
    Upper-triangle boolean mask for allowed nonbonded pairs.
    OpenMiChroM only excludes bonded pairs (d=1), not d=2.
    """
    iu = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
    
    # 1-2 exclusions (bonded pairs only)
    ex12 = jnp.zeros((N, N), dtype=bool)
    if bonds.size:
        ex12 = ex12.at[bonds[:,0], bonds[:,1]].set(True)
        ex12 = ex12 | ex12.T
    
    return iu & ~ex12


#########################
# Energy component kernels
#########################

def energy_fene_bonds(
    positions: jnp.ndarray,
    bonds: jnp.ndarray,
    kFb: float,
    fr0: float,
    epsilon: float,
    sigma: float
) -> float:
    """
    FENE bond energy exactly as in OpenMiChroM.
    
    Energy: -0.5 * kFb * fr0^2 * log(1 - (r / fr0)^2) + LJ * step(cutoff - r)
    where LJ = 4 * epsilon * ((sigma / r)^12 - (sigma / r)^6) + epsilon
    cutoff = 2^(1/6) * sigma
    """
    if bonds.size == 0:
        return jnp.array(0.0, dtype=positions.dtype)
    
    cutoff = 2.0 ** (1.0 / 6.0)  # 2^(1/6) * sigma
    
    pi = positions[bonds[:,0]]
    pj = positions[bonds[:,1]]
    rij = jnp.linalg.norm(pj - pi, axis=-1)
    
    # FENE term: -0.5 * kFb * fr0^2 * log(1 - (r / fr0)^2)
    # OpenMM handles this directly - bonds should always satisfy r < fr0
    r_over_fr0 = rij / fr0
    r_over_fr0_sq = r_over_fr0 * r_over_fr0
    # Use a small epsilon to prevent numerical issues at boundary, but don't clamp
    # This matches OpenMM's behavior more closely
    log_arg = 1.0 - r_over_fr0_sq
    # Add tiny epsilon to prevent log(0) but don't change valid values
    log_arg = jnp.maximum(log_arg, jnp.finfo(positions.dtype).eps)
    fene_term = -0.5 * kFb * fr0 * fr0 * jnp.log(log_arg)
    
    # LJ term: 4 * epsilon * ((sigma / r)^12 - (sigma / r)^6) + epsilon
    # Only applied when r <= cutoff
    sigma_over_r = sigma / jnp.maximum(rij, 1e-10)
    sigma_over_r6 = sigma_over_r ** 6
    sigma_over_r12 = sigma_over_r6 ** 2
    lj_term = 4.0 * epsilon * (sigma_over_r12 - sigma_over_r6) + epsilon
    lj_term = lj_term * jnp.where(rij <= cutoff, 1.0, 0.0)
    
    e = fene_term + lj_term
    return jnp.sum(e)

def energy_angles_cosine(
    positions: jnp.ndarray,
    triplets: jnp.ndarray,
    kA: jnp.ndarray  # Can be scalar or per-angle array
) -> float:
    """
    Angle force energy exactly as in OpenMiChroM.
    
    Energy: kA * (1 - cos(theta - pi))
    where theta is the angle between the three particles.
    """
    if triplets.size == 0:
        return jnp.array(0.0, dtype=positions.dtype)
    
    # Ensure kA is array
    if jnp.isscalar(kA) or kA.ndim == 0:
        kA_array = jnp.full(len(triplets), kA, dtype=positions.dtype)
    else:
        kA_array = jnp.asarray(kA, dtype=positions.dtype)
        if len(kA_array) != len(triplets):
            raise ValueError(f"kA length ({len(kA_array)}) must match triplets ({len(triplets)})")
    
    i = positions[triplets[:, 0]]
    j = positions[triplets[:, 1]]
    k = positions[triplets[:, 2]]
    
    v1 = i - j
    v2 = k - j
    
    # Compute angle using dot product: cos(theta) = (v1 · v2) / (|v1| |v2|)
    dot = jnp.sum(v1 * v2, axis=-1)
    norm1 = jnp.linalg.norm(v1, axis=-1)
    norm2 = jnp.linalg.norm(v2, axis=-1)
    
    # Avoid division by zero
    eps = jnp.finfo(positions.dtype).eps
    cos_theta = dot / jnp.maximum(norm1 * norm2, eps)
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)  # Clamp for numerical stability
    
    theta = jnp.arccos(cos_theta)
    
    # Energy: kA * (1 - cos(theta - pi))
    e = kA_array * (1.0 - jnp.cos(theta - jnp.pi))
    return jnp.sum(e)

def energy_repulsive_softcore(
    positions: jnp.ndarray,
    eCut: float,
    epsilon: float,
    sigma: float,
    cutoff: float,
    allowed_mask: Optional[jnp.ndarray] = None
) -> float:
    """
    Repulsive soft-core energy exactly as in OpenMiChroM.
    
    Energy: LJ * step(r - r0) * step(cutoff - r) + step(r0 - r) * 0.5 * eCut * (1.0 + tanh((2.0 * LJ / eCut) - 1.0))
    where LJ = 4.0 * epsilon * ((sigma / r)^12 - (sigma / r)^6) + epsilon
    and r0 is calculated from eCut and epsilon.
    """
    r = _pairwise_distances(positions)  # (N,N)
    N = r.shape[0]
    
    # Calculate r0 exactly as in OpenMiChroM
    eCut_scaled = eCut * epsilon
    r0 = sigma * (((0.5 * eCut_scaled) / (4.0 * epsilon) - 0.25 + (0.5) ** 2.0) ** 0.5 + 0.5) ** (-1.0 / 6.0)
    
    # In OpenMM: cutoff parameter in energy expression is nbCutoffDist, not cutoffDistance
    # cutoffDistance (3.0) is used for OpenMM's setCutoffDistance (limits evaluation)
    # cutoff (nbCutoffDist) is used in the energy expression step function
    nbCutoffDist = sigma * 2.0 ** (1.0 / 6.0)
    
    # Use upper triangle mask
    iu = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
    if allowed_mask is not None:
        iu = iu & allowed_mask
    
    # OpenMM only evaluates pairs within cutoffDistance (3.0)
    within_cutoff = (r <= cutoff) & iu
    
    # Compute LJ term
    sigma_over_r = sigma / jnp.maximum(r, 1e-10)
    sigma_over_r6 = sigma_over_r ** 6
    sigma_over_r12 = sigma_over_r6 ** 2
    lj = 4.0 * epsilon * (sigma_over_r12 - sigma_over_r6) + epsilon
    
    # Energy expression exactly as in OpenMM:
    # LJ * step(r - r0) * step(cutoff - r) + step(r0 - r) * 0.5 * eCut * (1.0 + tanh((2.0 * LJ / eCut) - 1.0))
    # where cutoff = nbCutoffDist in the expression
    # Note: In OpenMM, eCut parameter is set to eCut_scaled, so we use eCut_scaled in the expression
    step_r_ge_r0 = jnp.where(r >= r0, 1.0, 0.0)
    step_r_lt_r0 = jnp.where(r < r0, 1.0, 0.0)
    step_cutoff_expr = jnp.where(r <= nbCutoffDist, 1.0, 0.0)  # cutoff in expression is nbCutoffDist
    
    term1 = lj * step_r_ge_r0 * step_cutoff_expr
    term2 = 0.5 * eCut_scaled * (1.0 + jnp.tanh((2.0 * lj / eCut_scaled) - 1.0)) * step_r_lt_r0
    
    U = term1 + term2
    return jnp.sum(jnp.where(within_cutoff, U, 0.0))

def energy_type_to_type_michrom(
    positions: jnp.ndarray,
    types: jnp.ndarray,
    interaction_matrix: jnp.ndarray,
    mu: float,
    rc: float,
    cutoff: float,
    lim: float,
    allowed_mask: Optional[jnp.ndarray] = None
) -> float:
    """
    Type-to-type interactions exactly as in OpenMiChroM.
    
    Energy: mapType(t1,t2) * 0.5 * (1. + tanh(mu*(rc - r))) * step(r-lim)
    
    Uses Discrete2DFunction mapping (upper triangle + transpose, flattened).
    """
    r = _pairwise_distances(positions)  # (N,N)
    N = r.shape[0]
    
    # Use upper triangle mask
    iu = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
    if allowed_mask is not None:
        iu = iu & allowed_mask
    
    within = (r <= cutoff) & (r >= lim) & iu
    
    # Tanh kernel: 0.5 * (1. + tanh(mu*(rc - r)))
    f = 0.5 * (1.0 + jnp.tanh(mu * (rc - r)))
    
    # Map types to interaction matrix
    # OpenMiChroM uses: lambdas = np.triu(interaction_matrix) + np.triu(interaction_matrix, k=1).T
    # Then flattened: lambdas_flat = list(np.ravel(lambdas))
    # For lookup: use interaction_matrix directly (symmetric)
    ti = types[:, None]  # (N,1)
    tj = types[None, :]  # (1,N)
    A = interaction_matrix[ti, tj]  # (N,N)
    
    # Energy: mapType(t1,t2) * 0.5 * (1. + tanh(mu*(rc - r))) * step(r-lim)
    # step(r-lim) means energy is only non-zero when r >= lim
    step_r_ge_lim = jnp.where(r >= lim, 1.0, 0.0)
    U = A * f * step_r_ge_lim
    return jnp.sum(jnp.where(within, U, 0.0))

def energy_ideal_chromosome_michrom(
    positions: jnp.ndarray,
    mu: float,
    rc: float,
    gamma1: float,
    gamma2: float,
    gamma3: float,
    d_init: int,
    d_end: int,
    cutoff: float,
    allowed_mask: Optional[jnp.ndarray] = None
) -> float:
    """
    Ideal chromosome force exactly as in OpenMiChroM.
    
    Energy: step(d-dinit)*(gamma1/log(d) + gamma2/d + gamma3/d^2)*step(dend-d)*f
    where f=0.5*(1. + tanh(mu*(rc - r))) and d=abs(idx1-idx2)
    """
    r = _pairwise_distances(positions)  # (N,N)
    N = r.shape[0]
    
    # Use upper triangle mask
    iu = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
    if allowed_mask is not None:
        iu = iu & allowed_mask
    
    within = (r <= cutoff) & iu
    
    # Genomic distance d = |i - j|
    i_indices = jnp.arange(N)[:, None]
    j_indices = jnp.arange(N)[None, :]
    d = jnp.abs(i_indices - j_indices).astype(jnp.float64)
    
    # Tanh kernel: f = 0.5*(1. + tanh(mu*(rc - r)))
    f = 0.5 * (1.0 + jnp.tanh(mu * (rc - r)))
    
    # Ideal chromosome term: (gamma1/log(d) + gamma2/d + gamma3/d^2)
    # Only for d >= d_init and d <= d_end
    # OpenMM uses log(d) directly - step(d-dinit) ensures d >= d_init
    # Exclusions handle d=1, so log(d) should be safe for d >= d_init
    ic_term = gamma1 / jnp.log(d) + gamma2 / d + gamma3 / (d * d)
    
    # Apply step functions exactly as in OpenMM: step(d-dinit) and step(dend-d)
    step_d_ge_dinit = jnp.where(d >= d_init, 1.0, 0.0)
    step_d_le_dend = jnp.where(d <= d_end, 1.0, 0.0)
    
    U = step_d_ge_dinit * ic_term * step_d_le_dend * f
    return jnp.sum(jnp.where(within, U, 0.0))


def energy_ideal_chromosome_force(
    positions: jnp.ndarray,
    mu: float,
    rc: float,
    lambda_IC: jnp.ndarray,
    d_init: int,
    d_end: int,
    cutoff: float,
    lim: float = 1.0,
    allowed_mask: Optional[jnp.ndarray] = None
) -> float:
    """
    Ideal chromosome force using per-distance lambdas (full phi).

    Energy: step(d-dinit)*IClist(d)*step(dend-d)*f*step(r-lim)
    where f=0.5*(1. + tanh(mu*(rc - r))) and d=abs(idx1-idx2).
    """
    r = _pairwise_distances(positions)  # (N,N)
    N = r.shape[0]

    # Use upper triangle mask
    iu = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
    if allowed_mask is not None:
        iu = iu & allowed_mask

    within = (r <= cutoff) & iu

    # Genomic distance d = |i - j|
    i_indices = jnp.arange(N)[:, None]
    j_indices = jnp.arange(N)[None, :]
    d = jnp.abs(i_indices - j_indices).astype(jnp.int32)

    # Tanh kernel: f = 0.5*(1. + tanh(mu*(rc - r)))
    f = 0.5 * (1.0 + jnp.tanh(mu * (rc - r)))

    # Tabulated IClist(d) with clamped indexing
    d_idx = jnp.clip(d, 0, lambda_IC.shape[0] - 1)
    ic_term = lambda_IC[d_idx]

    # Apply step functions and minimum distance cutoff
    step_d_ge_dinit = jnp.where(d >= d_init, 1.0, 0.0)
    step_d_le_dend = jnp.where(d <= d_end, 1.0, 0.0)
    step_r_ge_lim = jnp.where(r >= lim, 1.0, 0.0)

    U = step_d_ge_dinit * ic_term * step_d_le_dend * f * step_r_ge_lim
    return jnp.sum(jnp.where(within, U, 0.0))

def energy_loops_michrom(
    positions: jnp.ndarray,
    loop_pairs: jnp.ndarray,
    qsi: float,
    mu: float,
    rc: float
) -> float:
    """
    Loop interactions exactly as in OpenMiChroM.
    
    Energy: qsi * 0.5 * (1. + tanh(mu*(rc - r)))
    """
    if loop_pairs.size == 0:
        return jnp.array(0.0, dtype=positions.dtype)
    
    pi = positions[loop_pairs[:,0]]
    pj = positions[loop_pairs[:,1]]
    rij = jnp.linalg.norm(pj - pi, axis=-1)
    
    # Energy: qsi * 0.5 * (1. + tanh(mu*(rc - r)))
    f = 0.5 * (1.0 + jnp.tanh(mu * (rc - rij)))
    e = qsi * f
    return jnp.sum(e)

def energy_flat_bottom_harmonic(
    positions: jnp.ndarray,
    kR: float,
    nRad: float
) -> float:
    """
    Flat-bottom harmonic confinement exactly as in OpenMiChroM.
    
    Energy: step(r - rRes) * 0.5 * kR * (r - rRes)^2
    where r = sqrt(x^2 + y^2 + z^2)
    """
    # Distance from origin for each particle
    r = jnp.linalg.norm(positions, axis=-1)
    
    # Energy: step(r - rRes) * 0.5 * kR * (r - rRes)^2
    excess = r - nRad
    step_term = jnp.where(r > nRad, 1.0, 0.0)
    e = step_term * 0.5 * kR * excess * excess
    return jnp.sum(e)


#############################
# Per-frame energy: main entry
#############################

def energy_components_one_frame(positions: jnp.ndarray, params: ParamsMM, static: StaticMM, temperature: float = 120.3) -> Dict[str, jnp.ndarray]:
    """
    Calculate energy components for one frame using OpenMiChroM forces.
    
    Parameters
    ----------
    positions : jnp.ndarray
        (N,3) float64 particle positions in nanometers
    params : ParamsMM
        Force parameters (in reduced units, matching OpenMiChroM)
    static : StaticMM
        Static simulation parameters
    temperature : float, optional
        Temperature in Kelvin for unit conversion (default 120.3K, matching toy simulation)
        
    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary of energy components in kJ/mol units:
        - "fene_bonds": FENE bond energy
        - "angles": cosine-based angle energy
        - "repulsive_softcore": repulsive soft-core energy
        - "type_to_type": type-to-type interaction energy
        - "ideal_chromosome": ideal chromosome energy
        - "loops": loop interaction energy
        - "flat_bottom_harmonic": flat-bottom harmonic confinement energy
        - "total": sum of all components
    """
    # Unit conversion: OpenMiChroM uses reduced units
    # For comparison with OpenMM, we need to convert to kJ/mol
    # OpenMiChroM energies are in kT units (reduced units where kT=1)
    # OpenMM converts reduced unit energies to kJ/mol using kT = kB * T
    # Default temperature is 120.3K (matching the toy simulation)
    kB_kJ_per_mol_K = constants.k * constants.N_A / 1000.0  # kJ/(mol·K)
    kT_to_kJmol = kB_kJ_per_mol_K * temperature  # kJ/mol
    
    # Calculate energies in reduced units (kT)
    e_fene_kT = energy_fene_bonds(
        positions, static.bonds,
        params.kFb, params.fr0,
        params.epsilon, params.sigma
    )
    
    e_angle_kT = energy_angles_cosine(
        positions, static.triplets, params.kA
    )
    
    e_rep_kT = energy_repulsive_softcore(
        positions,
        params.eCut,
        params.epsilon, params.sigma,
        static.rep_cutoff,
        static.allowed_mask
    )
    
    e_type_kT = energy_type_to_type_michrom(
        positions, static.types, static.interaction_matrix,
        params.mu, params.rc,
        static.type_cutoff, static.lim,
        static.allowed_mask
    )
    
    if params.lambda_IC is not None:
        e_ic_kT = energy_ideal_chromosome_force(
            positions,
            params.mu, params.rc,
            params.lambda_IC,
            params.d_init, params.d_end,
            static.ic_cutoff,
            static.lim,
            static.allowed_mask
        )
    else:
        e_ic_kT = energy_ideal_chromosome_michrom(
            positions,
            params.mu, params.rc,
            params.gamma1, params.gamma2, params.gamma3,
            params.d_init, params.d_end,
            static.ic_cutoff,
            static.allowed_mask
        )
    
    e_loop_kT = energy_loops_michrom(
        positions, static.loop_pairs,
        params.qsi, params.mu, params.rc
    )
    
    e_conf_kT = energy_flat_bottom_harmonic(
        positions,
        params.kR, params.nRad
    )
    
    # Convert to kJ/mol
    e_fene = e_fene_kT * kT_to_kJmol
    e_angle = e_angle_kT * kT_to_kJmol
    e_rep = e_rep_kT * kT_to_kJmol
    e_type = e_type_kT * kT_to_kJmol
    e_ic = e_ic_kT * kT_to_kJmol
    e_loop = e_loop_kT * kT_to_kJmol
    e_conf = e_conf_kT * kT_to_kJmol
    
    total = e_fene + e_angle + e_rep + e_type + e_ic + e_loop + e_conf
    
    return {
        "fene_bonds": e_fene,
        "angles": e_angle,
        "repulsive_softcore": e_rep,
        "type_to_type": e_type,
        "ideal_chromosome": e_ic,
        "loops": e_loop,
        "flat_bottom_harmonic": e_conf,
        "total": total,
    }


# vmapped, jitted version over frames
# Note: temperature is a scalar, so we don't vmap over it
def _batched_components_wrapper(frames_chunk, params, static, temperature):
    """Wrapper to handle temperature parameter in vmap."""
    return jax.vmap(lambda pos: energy_components_one_frame(pos, params, static, temperature))(frames_chunk)

_batched_components = jax.jit(
    _batched_components_wrapper,
    static_argnames=("static", "temperature")
)

def compute_energies_chunk(frames_chunk: jnp.ndarray, params: ParamsMM, static: StaticMM, temperature: float = 120.3) -> Dict[str, jnp.ndarray]:
    """
    frames_chunk: (B, N, 3) float64 frames (B = frames * reps in this chunk)
    temperature: float, temperature in Kelvin for unit conversion
    Returns dict of arrays (B,) for each component and total.
    """
    comps = _batched_components(frames_chunk, params, static, temperature)
    return {k: v for k, v in comps.items()}


#############################
# End-to-end batching helpers
#############################

def compute_energies_all(frame_blocks: Sequence[jnp.ndarray],
                         params: ParamsMM,
                         static: StaticMM,
                         temperature: float = 120.3,
                         chunk_reps: int = 10) -> Dict[str, jnp.ndarray]:
    """
    frame_blocks: list of per-replicate arrays, each (N_frames, N, 3) float64
    temperature: float, temperature in Kelvin for unit conversion
    We process chunk_reps replicates at a time and return a flat vector over all replicates.
    """
    per_key = []
    for i in range(0, len(frame_blocks), chunk_reps):
        reps = frame_blocks[i:i+chunk_reps]
        chunk = jnp.concatenate(reps, axis=0)
        out = compute_energies_chunk(chunk, params, static, temperature=temperature)
        per_key.append(out)
    
    keys = per_key[0].keys()
    merged = {k: jnp.concatenate([d[k] for d in per_key], axis=0) for k in keys}
    return merged


#############################
# Builder functions for ParamsMM and StaticMM
#############################

def build_params_static_mm_from_inputs(
    monomer_types: np.ndarray,
    interaction_matrix: np.ndarray,
    chains: List[Tuple[int,int,bool]],
    loop_pairs: np.ndarray,  # (M_loops, 2) int32, 0-indexed
    force_kwargs: dict,
    N: int
) -> Tuple[ParamsMM, StaticMM]:
    """
    Build ParamsMM and StaticMM from OpenMiChroM-style inputs.
    
    Parameters
    ----------
    monomer_types : np.ndarray
        (N,) array of monomer type indices
    interaction_matrix : np.ndarray
        (T, T) symmetric interaction matrix
    chains : List[Tuple[int,int,bool]]
        List of (start, end, is_ring) tuples
    loop_pairs : np.ndarray
        (M_loops, 2) array of loop pairs (0-indexed)
    force_kwargs : dict
        Dictionary with force parameters:
        - "fene_bonds": {"kFb": float, ...}
        - "angles": {"kA": float, ...}
        - "repulsive_softcore": {"eCut": float, ...}
        - "type_to_type": {"mu": float, "rc": float, ...}
        - "ideal_chromosome": {"lambda_IC": array (optional), "gamma1": float, "gamma2": float, "gamma3": float, "d_init": int, "d_end": int, ...}
        - "loops": {"qsi": float, ...}
        - "flat_bottom_harmonic": {"kR": float, "nRad": float, ...}
    N : int
        Total number of particles
        
    Returns
    -------
    Tuple[ParamsMM, StaticMM]
    """
    # Generate topology
    bonds, triplets = make_bonds_triplets_from_chains(chains, N)
    
    # Create nonbonded mask (only exclude bonded pairs, d=1)
    allowed_mask = make_nonbonded_mask(N, bonds)
    
    # Extract force parameters with defaults
    fene_kwargs = force_kwargs.get("fene_bonds", {})
    angle_kwargs = force_kwargs.get("angles", {})
    rep_kwargs = force_kwargs.get("repulsive_softcore", {})
    type_kwargs = force_kwargs.get("type_to_type", {})
    ic_kwargs = force_kwargs.get("ideal_chromosome", {})
    loop_kwargs = force_kwargs.get("loops", {})
    conf_kwargs = force_kwargs.get("flat_bottom_harmonic", {})
    
    lambda_ic = ic_kwargs.get("lambda_IC", None)
    if lambda_ic is not None:
        d_init = int(ic_kwargs.get("d_init", 3))
        d_end = int(ic_kwargs.get("d_end", 500))
        lambda_ic = np.asarray(lambda_ic, dtype=float)
        expected_len = d_end - d_init
        if lambda_ic.shape[0] != expected_len:
            raise ValueError(f"lambda_IC length ({lambda_ic.shape[0]}) must equal dmax ({expected_len})")
        padded = np.zeros(d_end, dtype=float)
        padded[d_init:d_end] = lambda_ic
        lambda_ic = jnp.asarray(padded, dtype=jnp.float64)

    params = ParamsMM(
        kFb=jnp.array(fene_kwargs.get("kFb", 30.0)),
        fr0=jnp.array(fene_kwargs.get("fr0", 1.5)),
        epsilon=jnp.array(fene_kwargs.get("epsilon", 1.0)),
        sigma=jnp.array(fene_kwargs.get("sigma", 1.0)),
        kA=jnp.array(angle_kwargs.get("kA", 2.0)),
        eCut=jnp.array(rep_kwargs.get("eCut", 4.0)),
        mu=jnp.array(type_kwargs.get("mu", 3.22)),
        rc=jnp.array(type_kwargs.get("rc", 1.78)),
        gamma1=jnp.array(ic_kwargs.get("gamma1", -0.030)),
        gamma2=jnp.array(ic_kwargs.get("gamma2", -0.351)),
        gamma3=jnp.array(ic_kwargs.get("gamma3", -3.727)),
        lambda_IC=lambda_ic,
        d_init=jnp.array(ic_kwargs.get("d_init", 3)),
        d_end=jnp.array(ic_kwargs.get("d_end", 500)),
        qsi=jnp.array(loop_kwargs.get("qsi", -1.612990)),
        kR=jnp.array(conf_kwargs.get("kR", 5e-3)),
        nRad=jnp.array(conf_kwargs.get("nRad", 10.0)),
    )
    
    static = StaticMM(
        bonds=bonds,
        triplets=triplets,
        types=jnp.asarray(monomer_types.astype(np.int32)),
        interaction_matrix=jnp.asarray(interaction_matrix, dtype=jnp.float64),
        loop_pairs=jnp.asarray(loop_pairs, dtype=jnp.int32),
        rep_cutoff=rep_kwargs.get("cutoffDistance", 3.0),
        type_cutoff=type_kwargs.get("rCutoff", 3.0),
        ic_cutoff=ic_kwargs.get("rCutoff", 3.0),
        lim=type_kwargs.get("lim", 1.0),
        allowed_mask=allowed_mask,
    )
    
    return params, static
