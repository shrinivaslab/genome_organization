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

def initialize_monomer_types(N, n_monomer_types, background_monomer_type, domain_length, distribution):
    """
    Initializes a 1XN array of monomer types (integers) where N = total number of monomers.

    Inputs:
    -------
    N : int
        The number of monomers.
    n_monomer_types: int
        The number of distinct monomer types.
    background_monomer_type: int
        The index in n_monomer_types corresponding to the background monomer.
        (basic chromosome monomer representation)
    domain_length: int
        Specifies how many non-background monomers should be next to each other.
    distribution: str
        Specifies the distribution of the non-background monomers.
        Options are 'uniform' or 'random'.

    Returns:
    --------
    monomer_types: np.ndarray
        1XN array of integers representing the monomer types.
    """
    # Initialize all monomers as background type
    monomer_types = np.full(N, background_monomer_type, dtype=int)

    if distribution == 'uniform':
        # Place non-background monomers at regular intervals
        positions = np.arange(0, N, domain_length)
        for pos in positions:
            if pos < N:
                monomer_types[pos:min(pos + domain_length, N)] = np.random.randint(1, n_monomer_types)

    elif distribution == 'random':
        # Randomly place non-background monomers
        positions = np.random.choice(N, size=N // domain_length, replace=False)
        for pos in positions:
            if pos < N:
                monomer_types[pos:min(pos + domain_length, N)] = np.random.randint(1, n_monomer_types)

    return monomer_types