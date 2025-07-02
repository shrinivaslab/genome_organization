import argparse
import numpy as np
import os
from chunkchromatin.simulation import Simulation
from chunkchromatin.chromosome import Chromosome
from chunkchromatin.lamina import Lamina
from chunkchromatin.hdf5_format import HDF5Reporter
from chunkchromatin.simulation import EKExceedsError
import openmm as mm
from polykit.polykit.generators.initial_conformations import create_random_walk

import json


def main(args):
    N = 3725
    density = args.density
    chains = [(0,1570,False), (1570,2775,False), (2775,3725,False)]

    monomer_types = np.load(args.monomer_types)

    interaction_matrix = np.array([
        [0.05, 0.05, 0.08],
        [0.05, 0.13, 0.17],
        [0.08, 0.17, 0.22]
    ])

    box_length = (N/density) ** (1/3.)
    monomer_positions = create_random_walk(step_size=1, N=N)

    out_dir = os.path.join(args.output_dir, '_'.join(args.forces))
    os.makedirs(out_dir, exist_ok=True)

    reporter = HDF5Reporter(folder=out_dir, max_data_length=500, overwrite=True)

    #make a different simulation object for equilibration
    sim_eq = Simulation(
        integrator_type="variableLangevin",
        temperature=300.0,  # in Kelvin
        gamma=0.05,         # in 1/ps
        timestep=5,    # in fs
        platform=args.device.upper(),
        N=N,
        reporter=reporter
    )
    chromosome = Chromosome(N, chains, sim_eq)
    lamina = Lamina(N, chains,sim_eq)

    sim_eq.set_positions(monomer_positions)

    harmonic_bond_force = chromosome.add_harmonic_bond()
    angle_force = chromosome.add_angle_force()
    spherical_confinement_force = lamina.add_spherical_confinement(sim_eq)
    polynomial_repulsive_force = chromosome.add_polynomial_repulsive(sim_eq)

    sim_eq.add_force(harmonic_bond_force)
    sim_eq.add_force(angle_force)
    sim_eq.add_force(spherical_confinement_force)
    sim_eq.add_force(polynomial_repulsive_force)

    sim_eq.create_context()
    sim_eq.set_velocities()


    sim_eq.run_simulation_block(200000)

    monomer_positions_eq = sim_eq.get_positions()

    #make a different simulation object for production
    sim = Simulation(
        integrator_type="variableLangevin",
        temperature=300.0,  # in Kelvin
        gamma=0.05,         # in 1/ps
        timestep=10,    # in fs
        platform=args.device.upper(),
        N=N,
        reporter=reporter
    )
    sim.set_positions(monomer_positions_eq)

    for force_name in args.forces:
        kwargs = args.force_kwargs.get(force_name, {})
        if force_name == "harmonic_bonds":
            force = chromosome.add_harmonic_bond(**kwargs)
        elif force_name == "angle_force":
            force = chromosome.add_angle_force(**kwargs)
        elif force_name == "spherical_confinement":
            force = lamina.add_spherical_confinement(sim_eq, **kwargs)
        elif force_name == "polynomial_repulsive":
            force = chromosome.add_polynomial_repulsive(sim_eq, **kwargs)
        elif force_name == "add_nonbonded_pair_potential":
            force = chromosome.add_nonbonded_pair_potential(sim_eq,interaction_matrix,monomer_types, **kwargs)
        elif force_name == "add_C_monomer_pinning":
            C_monomers = [i for i in range(N) if monomer_types[i] == 2]
            force = lamina.add_C_monomer_pinning(sim_eq,C_monomers, **kwargs)
        elif force_name == "add_B_monomer_lamina_attraction":
            B_monomers = [i for i in range(N) if monomer_types[i] == 1]
            force = lamina.add_B_monomer_lamina_attraction(sim_eq, B_monomers, **kwargs)
        else:
            raise ValueError(f"Unknown force type: {force_name}")
        sim.add_force(force)

    sim.create_context()
    sim.set_velocities()


    # if 'add_nonbonded_pair_potential' in args.forces:
    #     try:
    #         #sim.context.setParameter("custom_sticky_force_lambda_sticky", 0)
    #         sim.do_block(10000)
    #     except Simulation.EKExceedsError:
    #         with open(f"{out_dir}/simulation_failed_smooth_transition_0.txt", "w") as f:
    #             f.write("EKExceedsError occurred.\n")

    #     try:
    #         for lam in np.linspace(0, 1, num=1000):
    #             sim.context.setParameter("custom_sticky_force_lambda_sticky", lam)
    #             sim.do_block(100)
    #     except simulation.EKExceedsError:
    #         with open(f"{out_dir}/simulation_failed_smooth_transition.txt", "w") as f:
    #             f.write("EKExceedsError occurred.\n")
    #             f.write(f"Lambda: {lam}\n")
    #     sim.context.setParameter("custom_sticky_force_lambda_sticky", 1)
    try:
        for _ in range(100):
            sim.run_simulation_block(100000)
            with open(f"{out_dir}/simulation_stats.txt", "a") as f:
                stats = str(sim.print_stats())
                f.write(stats + "\n")
            reporter.dump_data()
    except EKExceedsError:
        with open(f"{out_dir}/simulation_failed.txt", "w") as f:
            f.write("EKExceedsError occurred.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monomer_types", type=str, required=True)
    parser.add_argument("--density", type=float, default=0.33)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--forces", nargs="+", required=True)
    parser.add_argument("--force_kwargs", type=str, default="{}",
                        help="JSON string mapping force names to kwargs, e.g. '{\"harmonic_bonds\": {\"bondLength\": 1.2}}'")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device to use for simulation: 'cuda' or 'cpu'")
    args = parser.parse_args()
    args.force_kwargs = json.loads(args.force_kwargs)
    main(args)