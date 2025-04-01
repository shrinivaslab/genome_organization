from polykit.polykit.generators.initial_conformations import grow_cubic, create_random_walk
from polychrom.hdf5_format import HDF5Reporter
import os
from polychrom import forcekits, forces, simulation
from polychrom.forces import heteropolymer_SSW
import numpy as np

def get_bonds(sim_object, chains=[(0, None, False)], extra_bonds=None):
    """
    Generates a list of bonds connecting polymer chains and returns them as an Mx2 int array.

    Parameters
    ----------
    sim_object : Simulation
        The simulation object containing the particles. Assumes sim_object.N is the number of particles.
    chains : list of tuples, optional
        The list of chains in the format [(start, end, isRing)]. The particle range is semi-open, 
        i.e., a chain (0, 3, False) links particles 0, 1, and 2. If isRing is True, then the first
        and last particles of the chain are linked into a ring.
        The default value links all particles of the system into one chain.
    extra_bonds : list of tuples, optional
        Additional bonds to add. Each tuple should be (particle1, particle2).

    Returns
    -------
    bonds : np.ndarray of shape (M, 2)
        An array of bonds, where each row is a bond represented as (particle1, particle2).
    """
    # Start with extra bonds if provided, otherwise an empty list
    bonds_list = [] if (extra_bonds is None or len(extra_bonds) == 0) else [tuple(b) for b in extra_bonds]

    for start, end, is_ring in chains:
        # If end is None, use total number of particles in sim_object
        end = sim_object.N if (end is None) else end

        # Add consecutive bonds within the chain
        bonds_list.extend([(j, j + 1) for j in range(start, end - 1)])

        # If the chain is a ring, add a bond connecting the last to the first particle
        if is_ring:
            bonds_list.append((start, end - 1))
    
    # Convert list of tuples to a NumPy array with integer type
    bonds = np.array(bonds_list, dtype=int)
    return bonds

def make_hp_sim_object(N, density, monomer_types, chains, interaction_matrix, force_list):
    """
    This function creates a polychrom simulation object given system parameters
    and a list of forces to apply to the simulation. This was made to troubleshoot
    EKExceeds errors by isolating the affect of each force on equilibriation

    forces are strings that indicate whether any force should be applied
    e.g., ['conf', 'hp', 'cp', 'bl']
    """
    box_length = (N/density) ** (1/3.)
    monomer_positions = create_random_walk(step_size=1,N=N)

    #want to save trajectories in a subdirectory named according to simulation size, forces applied, etc

    parent_dir_path = '/Users/kadendimarco/Desktop/Shrivinas_lab/genome_archetecture/chromosomes_openmm/trajectories/heteropolymer_benchmark/ekexceeds_fix'

    #make dir with all forces + parent dir
    all_forces = '_'.join(force_list)

    full_dir_path = os.path.join(parent_dir_path, all_forces)

    if not os.path.exists(full_dir_path):
        os.makedirs(full_dir_path)
    
    reporter = HDF5Reporter(folder=full_dir_path, max_data_length=500, overwrite=True)
    sim = simulation.Simulation(
    platform="CPU",
    integrator="variableLangevin",
    error_tol=0.0001,
    #GPU="1",
    collision_rate=0.05,
    N=N,
    save_decimals=2,
    PBCbox=False,
    reporters=[reporter],
    timestep=30 #femtoseconds
    )

    sim.set_data(monomer_positions, center=True)

    if 'conf' in force_list:
        sim.add_force(forces.spherical_confinement(sim, density=0.33, k=1))

    #these forces are always necessary
    polymer_chains_forces = forcekits.polymer_chains(
        sim,
        chains=chains,
        bond_force_func=forces.harmonic_bonds,
        bond_force_kwargs={
            "bondLength": 1.0,
            #"bondWiggleDistance": 0.05,
            "bondWiggleDistance": 0.05,
        },
        angle_force_func=forces.angle_force,
        angle_force_kwargs={
            "k": 1.5,
            #"k": 0.5,
        },
        nonbonded_force_func=forces.polynomial_repulsive,
        nonbonded_force_kwargs={
            "trunc": 3.0,
            #"trunc": 2.0,
        },
        except_bonds=True,
    )

    for force in polymer_chains_forces:
        sim.add_force(force)

    heteropolymer_force = heteropolymer_SSW(
    sim,
    interaction_matrix,
    monomer_types,
    extraHardParticlesIdxs=[],
    )
    if 'hp' in force_list:
        sim.add_force(heteropolymer_force)

    #add lamina attraction forces to b and c monomers
    B_monomers = [i for i in range(N) if monomer_types[i] == 1]
    C_monomers = [i for i in range(N) if monomer_types[i] == 2]

    # Add C monomer pinning
    if 'cp' in force_list:
        sim.add_force(forces.add_C_monomer_pinning(
            sim_object=sim,
            C_monomers=C_monomers,  # Changed from particles to C_monomers
            k=1,
            density=0.33
        ))

    if 'bl' in force_list:
        # Add B monomer lamina attraction
        sim.add_force(forces.add_B_monomer_lamina_attraction(
            sim_object=sim,
            B_monomers=B_monomers,
            BLam=1  # Adjust strength as needed
        ))

    # Create a set to track exclusions
    exclusions_set = set()

    # Define a function to add exclusions
    def add_exclusion(force, particle1, particle2):
        if (particle1, particle2) not in exclusions_set and (particle2, particle1) not in exclusions_set:
            force.addExclusion(particle1, particle2)
            exclusions_set.add((particle1, particle2))

    # Generate bonds and add exclusions
    bonds = get_bonds(sim, chains=chains)
    for bond in bonds:
        add_exclusion(heteropolymer_force, bond[0], bond[1])

    return sim, reporter, full_dir_path

