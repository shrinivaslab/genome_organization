"""
This script runs a single replicate with both type-type (TKL) and ideal chromosome (IC) force support.
Intended to be run via a SLURM job array using SLURM_ARRAY_TASK_ID.
"""
import argparse
    
import numpy as np
import os
import json
import time
import psutil
import openmm as mm
from chunkchromatin.simulation import Simulation, EKExceedsError
from chunkchromatin.chromosome import Chromosome
from chunkchromatin.lamina import Lamina
from chunkchromatin.binaryReporter import BinaryReporter
from chunkchromatin.initialization import grow_cubic_multi, create_multi_constrained_random_walk, init_multi_territory_rw, init_multi_territory_crumpled

parser = argparse.ArgumentParser()
parser.add_argument("--replicate_id", type=int, required=True)
args = parser.parse_args()

rep_idx = args.replicate_id
rep_name = f"rep{rep_idx+1:02d}"

# === CONFIGURATION ===
output_base = os.environ.get("MAXENT_REPL_OUT_BASE")
if not output_base:
    raise ValueError("MAXENT_REPL_OUT_BASE environment variable not set")

# Use environment variables for input paths
monomer_types_path = os.environ.get("MAXENT_MONOMER_TYPES")
if not monomer_types_path:
    raise ValueError("MAXENT_MONOMER_TYPES environment variable not set")

# Load epsilon (for tanh_type_force) and lambda_IC (for ideal_chromosome_force)
epsilon_path = os.environ.get("MAXENT_EPSILON_PATH")
if not epsilon_path:
    raise ValueError("MAXENT_EPSILON_PATH environment variable not set")

lambda_IC_path = os.environ.get("MAXENT_LAMBDA_IC_PATH")
if not lambda_IC_path:
    raise ValueError("MAXENT_LAMBDA_IC_PATH environment variable not set")

# IC parameters
d_init = int(os.environ.get("MAXENT_D_INIT", "3"))
d_end = int(os.environ.get("MAXENT_D_END", "300"))

# Read simulation parameters from environment variables
N = int(os.environ.get("MAXENT_N", "3725"))
density = float(os.environ.get("MAXENT_DENSITY", "0.33"))
chains_json = os.environ.get("MAXENT_CHAINS", "[[0, 1570, false], [1570, 2775, false], [2775, 3725, false]]")
chains = [(start, end, is_ring) for start, end, is_ring in json.loads(chains_json)]
initialization_method = os.environ.get("MAXENT_INITIALIZATION_METHOD", "random_walk")
monomer_types = np.load(monomer_types_path)
interaction_matrix = np.load(epsilon_path)  # Use epsilon for type-type interactions
lambda_IC = np.load(lambda_IC_path)  # Load lambda_IC for ideal chromosome force

# Default force kwargs overrides
DEFAULT_FORCE_KWARGS_OVERRIDES = {
    "harmonic_bonds": {
        "bondWiggleDistance": 0.10,
        "bondLength": 1.0
    },
    "angle_force": {
        "k": 2.0,
        "theta_0": np.pi
    },
    "spherical_confinement": {
        "r": "density",
        "density": 0.10,
        "k": 1.5,
        "center": [0.0, 0.0, 0.0],
        "invert": False,
        "particles": None,
        "name": "spherical_confinement"
    },
    "polynomial_repulsive": {
        "trunc": 5,
        "radiusMult": 1,
        "name": "polynomial_repulsive"
    },
    "tanh_type_force": {
        # Uses interaction_matrix (epsilon) passed in
    },
    "ideal_chromosome_force": {
        # Uses lambda_IC, d_init, d_end passed in
    }
}

# Read force kwargs from environment variable, or use defaults
force_kwargs_json = os.environ.get("MAXENT_FORCE_KWARGS")
if force_kwargs_json:
    force_kwargs_overrides = json.loads(force_kwargs_json)
else:
    force_kwargs_overrides = DEFAULT_FORCE_KWARGS_OVERRIDES
forces_list = ['harmonic_bonds','angle_force','spherical_confinement','tanh_type_force','polynomial_repulsive','ideal_chromosome_force']
box_length = (N/density) ** (1/3.)

out_dir = os.path.join(output_base, rep_name)
os.makedirs(out_dir, exist_ok=True)

# === SEED & LOGGING ===
rng = np.random.default_rng()
seed = rng.integers(1e9)
np.random.seed(seed)

# === METADATA ===
metadata = {
    "temperature": 300.0,
    "gamma": 0.05,
    "timestep_eq_fs": 5,
    "timestep_prod_fs": 10,
    "N": N,
    "chains": chains,
    "force_list": forces_list,
    "interaction_matrix": interaction_matrix.tolist(),
    "lambda_IC": lambda_IC.tolist(),
    "d_init": d_init,
    "d_end": d_end,
    "monomer_types_path": monomer_types_path,
    "random_seed": int(seed),
    "initialization_method": initialization_method
}

traj_path = os.path.join(out_dir, "trajectory.traj")
start_time = time.time()


# Initialize monomer positions based on the chosen method
if initialization_method == "random_walk":
    monomer_positions = create_multi_constrained_random_walk(chains,density,
    k_wall=force_kwargs_overrides['spherical_confinement']['k'], step_size=1, center=(0.0, 0.0, 0.0), 
    min_sep=0.5)
elif initialization_method == "grow_cubic":
    box_size = int(box_length)
    monomer_positions = grow_cubic_multi(chains, box_size, method='standard')
elif initialization_method == "territory_rw":
    monomer_positions = init_multi_territory_rw(chains, density,
    k_wall=force_kwargs_overrides['spherical_confinement']['k'], step_size=1, center=(0.0, 0.0, 0.0), 
    min_sep=0.1, territory_center_frac=0.7)
elif initialization_method == "territory_crumpled":
    monomer_positions = init_multi_territory_crumpled(chains, density,
    k_wall=force_kwargs_overrides['spherical_confinement']['k'], step_size=1, center=(0.0, 0.0, 0.0), 
    min_sep=0.1, block_size=20, bias_strength=0.6, territory_center_frac=0.7)
else:
    raise ValueError(f"Unknown initialization method: {initialization_method}")
with BinaryReporter(filename=traj_path, n_particles=N, mode='w', metadata=metadata) as reporter:

    # === EQUILIBRATION ===
    sim_eq = Simulation(
    integrator_type="variableLangevin",
    temperature=300.0,
    gamma=0.05,
    timestep=5,
    platform="CUDA",
    N=N,
    reporter=reporter
    )
    chromosome = Chromosome(N, chains, sim_eq)
    lamina = Lamina(N, chains, sim_eq)
    sim_eq.set_positions(monomer_positions)
    sim_eq.add_force(chromosome.add_harmonic_bond())
    sim_eq.add_force(chromosome.add_angle_force())
    sim_eq.add_force(lamina.add_spherical_confinement(sim_eq))
    sim_eq.add_force(chromosome.add_polynomial_repulsive(sim_eq))
    sim_eq.create_context()
    sim_eq.set_velocities()
    sim_eq.run_simulation_block(200000, save=False)
    monomer_positions_eq = sim_eq.get_positions()

    # === PRODUCTION ===
    sim = Simulation(
        integrator_type="variableLangevin",
        temperature=300.0,
        gamma=0.05,
        timestep=10,
        platform="CUDA",
        N=N,
        reporter=reporter
    )
    sim.set_positions(monomer_positions_eq)

    for force_name in forces_list:
        kwargs = force_kwargs_overrides.get(force_name, {})
        if force_name == "harmonic_bonds":
            force = chromosome.add_harmonic_bond(**kwargs)
        elif force_name == "angle_force":
            force = chromosome.add_angle_force(**kwargs)
        elif force_name == "spherical_confinement":
            force = lamina.add_spherical_confinement(sim, **kwargs)
        elif force_name == "polynomial_repulsive":
            force = chromosome.add_polynomial_repulsive(sim, **kwargs)
        elif force_name == "add_nonbonded_pair_potential":
            force = chromosome.add_nonbonded_pair_potential(sim, interaction_matrix, monomer_types, **kwargs)
        elif force_name == "tanh_type_force":
            force = chromosome.add_tanh_type_force(sim, interaction_matrix, monomer_types, **kwargs)
        elif force_name == "ideal_chromosome_force":
            # Add ideal chromosome force with lambda_IC
            force = chromosome.add_ideal_chromosome_force(
                sim,
                lambda_IC,
                d_init=d_init,
                d_end=d_end,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown force type: {force_name}")
        sim.add_force(force)

    sim.create_context()
    sim.set_velocities()
    sim.run_simulation_block(500000, save=False)
    
    # Load seeds from JSON if available
    frames = int(os.environ.get("MAXENT_FRAMES", "4000"))
    burnin = int(os.environ.get("MAXENT_BURNIN", "400"))
    save_frames = os.environ.get("MAXENT_SAVE_FRAMES", "1") == "1"
    
    seeds_json_path = os.environ.get("MAXENT_SEEDS_JSON")
    if seeds_json_path and os.path.exists(seeds_json_path):
        with open(seeds_json_path, "r") as f:
            seeds_dict = json.load(f)
        seed_key = str(rep_idx)  # Use rep_idx as key (0-based)
        if seed_key in seeds_dict:
            fixed_seed = seeds_dict[seed_key]
            np.random.seed(fixed_seed)
            # Reinitialize velocities with fixed seed for reproducibility
            sim.set_velocities()
    
    try:
        for i in range(frames):
            sim.run_simulation_block(1000, save=save_frames)
            if (i + 1) % 100 == 0:
                with open(f"{out_dir}/simulation_stats.txt", "a") as f:
                    stats = str(sim.print_stats())
                    f.write(stats + "\n")
    except EKExceedsError:
        with open(f"{out_dir}/simulation_failed.txt", "w") as f:
            f.write("EKExceedsError occurred.\n")

# === LOG STATS ===
runtime = time.time() - start_time
mem_usage_MB = psutil.Process().memory_info().rss / 1e6

summary = {
    "replicate": rep_name,
    "runtime_sec": runtime,
    "peak_memory_MB": mem_usage_MB,
    "random_seed": int(seed)
}
with open(os.path.join(out_dir, "manifest.json"), "w") as f:
    json.dump(summary, f, indent=2)

