# openmm_energy_benchmark.py
import numpy as np
import openmm as mm
from openmm import unit

from chunkchromatin.simulation import Simulation
from chunkchromatin.chromosome import Chromosome
from chunkchromatin.lamina import Lamina

def openmm_energy_benchmark(
    positions_batch: np.ndarray,             # (N_comparisons, N, 3) in reduced units (σ)
    chains,
    monomer_types: np.ndarray,               # (N,) int
    interaction_matrix: np.ndarray,          # (T, T) float (kT units)
    density: float = 0.30,                   # particles / σ^3  (for sphere radius)
    platform: str = "CUDA",                  # "CUDA" | "CPU" | "OpenCL" (if available)
    cuda_precision: str = "double",
    force_kwargs: dict | None = None,        # optional per-force overrides
) -> tuple[np.ndarray, list[str]]:
    """
    Returns:
      energies_kT: (N_comparisons, N_forces+1) array in kT
      columns:     list of column names in the same order
                   [bond, angle, conf, rep, tanh, total]
    Notes:
      - Uses no PBC.
      - Forces match your repo’s definitions; energy reported per force group in kT.
      - Builds context once; updates positions per frame for speed.
    """
    force_kwargs = force_kwargs or {}
    # Unpack optional overrides safely
    kw_bond   = force_kwargs.get("harmonic_bonds", {})
    kw_angle  = force_kwargs.get("angle_force", {})
    kw_conf   = force_kwargs.get("spherical_confinement", {"r": "density", "density": density})
    kw_rep    = force_kwargs.get("polynomial_repulsive", {})
    kw_tanh   = force_kwargs.get("tanh_type_force", {})

    # Basic checks
    assert positions_batch.ndim == 3 and positions_batch.shape[2] == 3, "positions_batch must be (M, N, 3)"
    M, N, _ = positions_batch.shape
    assert len(monomer_types) == N, "monomer_types length must match N"

    # --- Build a minimal Simulation (no PBC) in double precision ---
    sim = Simulation(
        integrator_type="variableLangevin",
        temperature=300.0,          # only used to scale to kJ/mol internally; we convert back to kT
        gamma=0.05,
        timestep=10,                # irrelevant for energy queries
        platform=platform,
        N=N,
        reporter=None,
        platform_properties={'CudaPrecision': cuda_precision} if platform.upper() == "CUDA" else None,
    )
    chrom = Chromosome(N, chains, sim)
    lam   = Lamina(N, chains, sim)

    # Add forces and assign deterministic force groups:
    # 0: bonds, 1: angles, 2: spherical conf, 3: polynomial repulsive, 4: tanh type
    f_bond = chrom.add_harmonic_bond(**kw_bond);            f_bond.setForceGroup(0); sim.add_force(f_bond)
    f_ang  = chrom.add_angle_force(**kw_angle);             f_ang.setForceGroup(1);  sim.add_force(f_ang)
    f_conf = lam.add_spherical_confinement(sim, **kw_conf); f_conf.setForceGroup(2); sim.add_force(f_conf)
    f_rep  = chrom.add_polynomial_repulsive(sim, **kw_rep); f_rep.setForceGroup(3);  sim.add_force(f_rep)
    f_tanh = chrom.add_tanh_type_force(sim, interaction_matrix, monomer_types, **kw_tanh)
    f_tanh.setForceGroup(4); sim.add_force(f_tanh)

    # Create context and set initial positions
    sim.create_context()
    # Positions are in reduced units (σ). Your Simulation class treats σ=1 nm (conlen),
    # so we hand them to OpenMM in nanometers.
    sim.set_positions(positions_batch[0].astype(np.float64))

    # Precompute conversion factor: kJ/mol -> kT
    kT_kJ_per_mol = sim.kT.value_in_unit(unit.kilojoule_per_mole)
    to_kT = 1.0 / kT_kJ_per_mol

    # Helper to read energy for a force group
    def energy_group_kT(group_id: int) -> float:
        state = sim.context.getState(getEnergy=True, groups={1 << group_id})
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) * to_kT

    # Allocate output: per-force + total
    columns = ["bond", "angle", "conf", "rep", "tanh", "total"]
    energies = np.zeros((M, len(columns)), dtype=np.float64)

    # Loop over comparison frames
    for m in range(M):
        sim.context.setPositions(positions_batch[m].astype(np.float64))
        # Query each group
        e_b = energy_group_kT(0)
        e_a = energy_group_kT(1)
        e_c = energy_group_kT(2)
        e_r = energy_group_kT(3)
        e_t = energy_group_kT(4)
        e_tot = e_b + e_a + e_c + e_r + e_t
        energies[m, :] = (e_b, e_a, e_c, e_r, e_t, e_tot)

    return energies, columns
