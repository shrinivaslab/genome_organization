import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import glob
import re
import argparse

from polychrom.hdf5_format import HDF5Reporter
from polychrom import forcekits, forces, simulation

from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize, to_rgb

from polykit.polykit.generators.initial_conformations import create_random_walk
import polykit as polykit


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

def get_triplets(sim_object, chains=[(0, None, False)], extra_triplets=None):
    """
    Generates a list of triplets connecting polymer chains and returns them as an Mx3 int array.
    e.g., if bonds are [(0,1), (1,2), (2,3), (3,4), (4,5)]
    then triplets are [(0,1,2), (1,2,3), (2,3,4), (3,4,5)]
    if it is a ring, then the last triplet is (4,5,0)

    Parameters
    ----------
    sim_object : Simulation
        The simulation object containing the particles. Assumes sim_object.N is the number of particles.
    chains : list of tuples, optional
        The list of chains in the format [(start, end, isRing)]. The particle range is semi-open, 
        i.e., a chain (0, 3, False) links particles 0, 1, and 2. If isRing is True, then the first
        and last particles of the chain are linked into a ring.
        The default value links all particles of the system into one chain.
    extra_triplets : list of tuples, optional
        Additional triplets to add. Each tuple should be (particle1, particle2, particle3).

    Returns
    -------
    triplets : np.ndarray of shape (M, 3)
        An array of triplets, where each row is a triplet represented as (particle1, particle2, particle3).
    """
    # Start with extra triplets if provided, otherwise an empty list
    triplets_list = [] if (extra_triplets is None or len(extra_triplets) == 0) else [tuple(t) for t in extra_triplets]

    for start, end, is_ring in chains:
        # If end is None, use total number of particles in sim_object
        end = sim_object.N if (end is None) else end    
        triplets_list += [(j - 1, j, j + 1) for j in range(start + 1, end - 1)]
        if is_ring:
            triplets_list.append((int(end - 2), int(end - 1), int(start)))
            triplets_list.append((int(end - 1), int(start), int(start + 1)))
    
    # Convert list of tuples to a NumPy array with integer type
    triplets = np.array(triplets_list, dtype=int)
    return triplets


# Define correct power law for squared distances
def power_law(x, A, nu):
    return A * x**nu

def plot_scaling_law(avg_spatial_distances, monomer_index_separations, trajectory_folder, first_100=True):
    from scipy.stats import linregress
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.array(monomer_index_separations)
    y = np.array(avg_spatial_distances)

    # Filter out very small separations to avoid log(0)
    mask = x > 1
    x = x[mask]
    y = y[mask]

    if first_100:
        x = x[:100]
        y = y[:100]
    # Log-transform
    log_x = np.log10(x)
    log_y = np.log10(y)

    # Linear regression on log-log data to find fitted ν
    slope, intercept, r_value, p_value, std_err = linregress(log_x, log_y)
    fit_y = 10**intercept * x**slope

    # Add a reference ν = 0.5 line for ideal Gaussian scaling
    ref_nu = 0.5
    ref_intercept = log_y[0] - ref_nu * log_x[0]  # match starting point
    ref_y = 10**ref_intercept * x**ref_nu

    # Prefactor A is 10**intercept
    A = 10**intercept

    # Plot
    plt.figure(figsize=(6, 4))
    plt.loglog(x, y, 'o', label=r'$\langle R_{i,j} \rangle$ (data)')
    plt.loglog(
        x, fit_y, '-', 
        label=fr'Fit: $\nu$ = {slope:.3f}, $A$ = {A:.3g}'
    )

    plt.xlabel(r'$|i - j|$ (monomers apart)', fontsize=12)
    plt.ylabel(r'$\langle R_{i,j} \rangle$', fontsize=12)
    plt.title(r'Scaling Fit: $\langle R_{i,j} \rangle \propto |i - j|^\nu$', fontsize=14)
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    plt.tight_layout()

    print(f"Fitted scaling exponent ν: {slope:.3f}")
    print(f"Prefactor A: {A:.3g}")
    print(f"R² of the fit: {r_value**2:.4f}")
    # Save the figure
    if first_100:
        plt.savefig(Path(trajectory_folder) / 'scaling_law_first100.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(Path(trajectory_folder) / 'scaling_law.png', dpi=300, bbox_inches='tight')

def run_homopolymer_sim(forcelist,trajectory_folder):
    """
    Run a homopolymer simulation with the given force list and force kwargs.
    Example forcelist:
    forcelist = [
    (forces.spherical_confinement, {"k": 3}),
    (forces.harmonic_bonds, {"bondLength": 1, "bondWiggleDistance": 0.05, "bonds": bonds})
]
    """
    Chromosome_sizes=[1000]
    chains = [(0,1000,False)]
    N=sum(Chromosome_sizes)
    density=0.33

    box_length = (N/density) ** (1/3.)
    monomer_positions = create_random_walk(step_size=1,N=N)

    #initialize simulation
    reporter = HDF5Reporter(folder=trajectory_folder, max_data_length=25, overwrite=True)



    sim = simulation.Simulation(
        platform="CPU",
        integrator="variableLangevin",
        error_tol=0.01,
        #GPU="1",
        collision_rate=0.06,
        N=N,
        save_decimals=2,
        PBCbox=False,
        reporters=[reporter],
        timestep=60,
        cpu_threads=16 #femtoseconds
    )

    sim.set_data(monomer_positions, center=True) #loads positions, set center of mass to origin
    bonds = get_bonds(sim, chains=chains, extra_bonds=None)
    bonds = [tuple(bond) for bond in bonds]
    triplets = get_triplets(sim, chains=chains, extra_triplets=None)
    triplets = [tuple(triplet) for triplet in triplets]

    for force_func, force_kwargs in forcelist:
        if force_func == forces.harmonic_bonds:
            force_kwargs = force_kwargs.copy()
            force_kwargs['bonds'] = bonds
        if force_func == forces.angle_force:
            force_kwargs['triplets'] = triplets
        sim.add_force(force_func(sim, **force_kwargs))
    


    sim.local_energy_minimization(tolerance=0.1, maxIterations=1000)
    for _ in range(100):  # Do 10 blocks
        sim.do_block(10000)  # Of 100 timesteps each. Data is saved automatically.
    sim.print_stats()  # In the end, print very simple statistics

    reporter.dump_data()

    block_files = glob.glob(str(Path(trajectory_folder) / "blocks_*.h5"))
    block_files.sort()

    def extract_start_index(filename):
        match = re.search(r'blocks_(\d+)-\d+\.h5', filename)
        return int(match.group(1)) if match else -1

# Sort by the extracted start index
    block_files_sorted = sorted(block_files, key=extract_start_index)

    spatial_distance_bins = {}
    for block_file in block_files_sorted:
        with h5py.File(block_file, 'r') as f:
            # Print the structure
            print("\nFile structure:")
            print("Groups:", list(f.keys()))
            
            for block in f.keys():

                pos = f[block]['pos'][:]
                N = pos.shape[0]
                for i in range(N):
                    for j in range(i+1, N):
                        monomer_index_separation = j-i
                        spatial_distance = np.linalg.norm(pos[i] - pos[j])
                        if monomer_index_separation not in spatial_distance_bins:
                            spatial_distance_bins[monomer_index_separation] = []
                        else:
                            spatial_distance_bins[monomer_index_separation].append(spatial_distance)            

    monomer_index_separations = list(spatial_distance_bins.keys())
    monomer_index_separations.sort()
    avg_spatial_distances = [np.mean(spatial_distance_bins[i]) for i in monomer_index_separations]

    plot_scaling_law(avg_spatial_distances,monomer_index_separations, trajectory_folder,first_100=True)
    plot_scaling_law(avg_spatial_distances,monomer_index_separations, trajectory_folder,first_100=False)

def main():
    parser = argparse.ArgumentParser(description='Run homopolymer simulation')
    parser.add_argument('--output', '-o', type=str, required=True,
                      help='Output directory for trajectory files')
    parser.add_argument('--force_list', '-f', type=str, required=True,
                        help='Force list to use for simulation')
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create force list (you might want to make this configurable too)

    # Run simulation
    run_homopolymer_sim(
        forcelist=eval(args.force_list),
        trajectory_folder=str(output_dir),
    )

if __name__ == "__main__":
    main()

