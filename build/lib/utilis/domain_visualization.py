def domain_viewer(N, chains, monomer_types, colors):
    """
    Visualize chromatin domains per chromosome in 1D.
    
    Inputs:
    -------
    N : int
        The number of monomers.
    chains : list of tuples
        The list of chains in format [(start, end, isRing)].
    monomer_types: np.ndarray
        1D array of length N containing integers representing the monomer types.
    colors: np.ndarray
        Array of shape (N, 3) containing RGB colors.
    
    Outputs:
    -------
    matplotlib figure that shows one bar for each chromosome with each domain colored differently.
    """
    fig, ax = plt.subplots(figsize=(10, len(chains) * 0.7))  # Dynamically adjust height
    
    bar_height = 0.8  # Thickness of each chromosome bar
    y_positions = np.arange(len(chains))[::-1]  # Reverse order to stack from top to bottom

    # Iterate over chains to plot each chromosome
    for chain_idx, (start, end, isRing) in enumerate(chains):
        end = N if end is None else end  # Handle None case
        
        # Iterate over monomers in the chain to color each domain
        for i in range(start, end):
            ax.add_patch(
                mpatches.Rectangle(
                    (i, y_positions[chain_idx] - bar_height / 2),  # (x, y) position
                    1,  # Width (one monomer)
                    bar_height,  # Equal thickness
                    color=colors[i],  # Domain color
                    linewidth=0
                )
            )

    # Formatting the plot
    ax.set_xlim(0, N)
    ax.set_ylim(-0.5, len(chains) - 0.5)  
    ax.set_yticks(y_positions)  
    ax.set_yticklabels([f"Chromosome {i+1}" for i in range(len(chains))])  # Label chromosomes
    ax.set_xlabel("Monomer Index")
    ax.set_title("Chromatin Domains per Chromosome")

    # Remove box and ticks
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False, bottom=False)

    plt.show()


    
def compute_colors_tripolymer(positions, monomer_types):
    """
    Given positions and monomer types, returns an (N, 3) array of RGB colors.
    For heteropolymer visualization where:
    - Type 0 (A monomers) = red
    - Type 1 (B monomers) = blue 
    - Type 2 (C monomers) = green

    Parameters:
    -----------
    positions : np.ndarray
        Array of shape (N, 3) containing the positions.
    chains : list of tuples
        The list of chains in format [(start, end, isRing)].
    monomer_types: np.ndarray
        1D array of integers (shape (N,)) representing the monomer types.
        
    Returns:
    --------
    colors : np.ndarray
        Array of shape (N, 3) containing RGB colors.
    """
    N = positions.shape[0]
    colors = np.zeros((N, 3))  # Initialize colors array
    
    # Define fixed colors for each monomer type
    color_map = {
        0: [0.89, 0.10, 0.11],  # red for type A
        1: [0.22, 0.49, 0.72],  # blue for type B
        2: [0.30, 0.69, 0.29]   # green for type C
    }
    
    # Assign colors based on monomer type
    for monomer_type, color in color_map.items():
        mask = (monomer_types == monomer_type)
        colors[mask] = color
        
    return colors