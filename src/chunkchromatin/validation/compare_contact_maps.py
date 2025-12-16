#!/usr/bin/env python
"""
Compare simulated and experimental contact maps.

This script generates two plots:
1. Contact map overlay: simulated (lower triangle) vs experimental (upper triangle)
2. P(s) scaling curves: comparison of contact probability vs genomic separation

The script automatically trims the simulated contact map to match experimental
bin counts per chromosome, based on chain definitions in the config file.

Config file requirements:
- simulation.chains: List of [start, end, is_ring] tuples defining chains
- chromosomes: List of chromosome names matching chain order (e.g., ['chr4', 'chr13', 'chr17'])
- bin_size_bp: Resolution in basepairs (default: 100000)

The script assumes the simulated contact map is at monomer resolution (1 monomer = 1 bin).
If your contact map is at a different resolution, ensure chains and bin counts are
appropriately scaled.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
import cooler
from pathlib import Path
from typing import Sequence, Tuple


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_experimental_bin_counts(
    mcool_path: str,
    chroms: Sequence[str],
    resolution: int,
) -> dict:
    """
    Get bin counts per chromosome from experimental .mcool file.
    
    Parameters
    ----------
    mcool_path : str
        Path to experimental .mcool file
    chroms : Sequence[str]
        List of chromosome names (e.g., ['chr4', 'chr13', 'chr17'])
    resolution : int
        Resolution in basepairs (e.g., 100000)
    
    Returns
    -------
    dict
        Dictionary mapping chromosome name to bin count
    """
    clr = cooler.Cooler(f"{mcool_path}::resolutions/{resolution}")
    bins_df = clr.bins()[:]
    
    chrom_bin_counts = {}
    for chrom in chroms:
        chrom_mask = bins_df['chrom'] == chrom
        chrom_bin_counts[chrom] = int(chrom_mask.sum())
    
    return chrom_bin_counts


def trim_simulated_contact_map(
    sim_cmap: np.ndarray,
    chains: Sequence[Tuple[int, int, bool]],
    chroms: Sequence[str],
    exp_bin_counts: dict,
    verbose: bool = False,
) -> np.ndarray:
    """
    Trim simulated contact map to match experimental bin counts per chromosome.
    
    For each chain, if the simulated length exceeds the experimental bin count,
    trim excess monomers from the end of the chain.
    
    Parameters
    ----------
    sim_cmap : np.ndarray
        Full simulated contact map (N x N)
    chains : Sequence[Tuple[int, int, bool]]
        Chain definitions as [(start, end, is_ring), ...]
    chroms : Sequence[str]
        List of chromosome names matching chain order
    exp_bin_counts : dict
        Dictionary mapping chromosome name to experimental bin count
    verbose : bool
        Print trimming information
    
    Returns
    -------
    np.ndarray
        Trimmed contact map
    """
    if len(chains) != len(chroms):
        raise ValueError(
            f"Number of chains ({len(chains)}) must match number of chromosomes ({len(chroms)})"
        )
    
    # Determine which indices to keep
    keep_indices = []
    current_idx = 0
    
    for (start, end, is_ring), chrom in zip(chains, chroms):
        sim_length = end - start
        exp_bins = exp_bin_counts[chrom]
        
        if sim_length > exp_bins:
            # Trim from the end
            trim_count = sim_length - exp_bins
            new_end = end - trim_count
            if verbose:
                print(
                    f"Trimming {chrom}: simulated length={sim_length}, "
                    f"target bins={exp_bins}, trimming {trim_count} monomers "
                    f"(chain range: [{start}, {end}] -> [{start}, {new_end}])"
                )
            chain_indices = list(range(start, new_end))
        else:
            chain_indices = list(range(start, end))
            if sim_length < exp_bins and verbose:
                print(
                    f"Warning: {chrom}: simulated length={sim_length} < "
                    f"target bins={exp_bins} (no trimming needed)"
                )
        
        keep_indices.extend(chain_indices)
    
    keep_indices = np.array(keep_indices)
    
    # Extract trimmed contact map
    trimmed_cmap = sim_cmap[np.ix_(keep_indices, keep_indices)]
    
    if verbose:
        print(f"Original shape: {sim_cmap.shape}, Trimmed shape: {trimmed_cmap.shape}")
    
    return trimmed_cmap


def row_max_normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Normalize matrix by row maximum (excluding diagonal).
    
    Parameters
    ----------
    matrix : np.ndarray
        Input contact matrix
    
    Returns
    -------
    np.ndarray
        Row-max normalized matrix
    """
    matrix = matrix.copy().astype(float)
    
    # Fill diagonal with row maxes for visualization
    row_maxes = np.max(matrix, axis=1, keepdims=True)
    np.fill_diagonal(matrix, row_maxes.flatten())
    
    # Normalize
    row_maxes[row_maxes == 0] = 1  # Avoid division by zero
    normalized = matrix / row_maxes
    
    return normalized


def compute_ps(
    matrix: np.ndarray,
    min_s_bins: int = 1,
    max_s_bins: int = None,
    exclude_zeros: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute p(s): average contact vs genomic separation s (diagonal offset).
    
    Parameters
    ----------
    matrix : np.ndarray
        Contact matrix (square)
    min_s_bins : int
        Minimum diagonal offset to include
    max_s_bins : int
        Maximum diagonal offset to include (None = use all)
    exclude_zeros : bool
        Ignore zero entries when averaging
    
    Returns
    -------
    s_vals : np.ndarray
        Diagonal offsets (in bins)
    ps : np.ndarray
        Average contact probability per diagonal
    n_pairs : np.ndarray
        Number of pairs used per diagonal
    """
    assert matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1], "Matrix must be square"
    n = matrix.shape[0]
    
    if max_s_bins is None:
        max_s_bins = n - 1
    max_s_bins = min(max_s_bins, n - 1)
    
    s_vals = np.arange(min_s_bins, max_s_bins + 1, dtype=int)
    ps = np.empty_like(s_vals, dtype=float)
    n_pairs = np.zeros_like(s_vals, dtype=int)
    
    for i, s in enumerate(s_vals):
        d = np.diag(matrix, k=s).astype(float)
        if exclude_zeros:
            d = d[d > 0]
        d = d[np.isfinite(d)]
        n_pairs[i] = d.size
        ps[i] = np.nan if d.size == 0 else d.mean()
    
    # Remove NaNs-only tails
    good = np.isfinite(ps) & (n_pairs > 0)
    return s_vals[good], ps[good], n_pairs[good]


def fit_loglog_slope(
    s_bp: np.ndarray,
    ps: np.ndarray,
    fit_min_bp: int = None,
    fit_max_bp: int = None,
) -> Tuple[float, float]:
    """
    Fit slope of log10 p(s) vs log10 s over a selectable range.
    
    Parameters
    ----------
    s_bp : np.ndarray
        Genomic separations in basepairs
    ps : np.ndarray
        Contact probabilities
    fit_min_bp : int
        Minimum separation for fitting (in bp)
    fit_max_bp : int
        Maximum separation for fitting (in bp)
    
    Returns
    -------
    slope : float
        Fitted slope
    intercept : float
        Fitted intercept
    """
    s = np.asarray(s_bp, dtype=float)
    y = np.asarray(ps, dtype=float)
    m = np.isfinite(s) & np.isfinite(y) & (s > 0) & (y > 0)
    
    if fit_min_bp is not None:
        m &= (s >= fit_min_bp)
    if fit_max_bp is not None:
        m &= (s <= fit_max_bp)
    
    if m.sum() < 3:
        return np.nan, np.nan
    
    coef = np.polyfit(np.log10(s[m]), np.log10(y[m]), 1)
    return coef[0], coef[1]


def plot_contact_map_overlay(
    sim_cmap: np.ndarray,
    exp_cmap: np.ndarray,
    output_path: str,
    title: str = "Contact Map: Lower=Simulated, Upper=Experimental",
):
    """
    Plot contact map overlay with simulated (lower) and experimental (upper) triangles.
    
    Parameters
    ----------
    sim_cmap : np.ndarray
        Simulated contact map (normalized)
    exp_cmap : np.ndarray
        Experimental contact map (normalized)
    output_path : str
        Path to save the plot
    title : str
        Plot title
    """
    # Ensure same size
    n = min(sim_cmap.shape[0], exp_cmap.shape[0])
    sim = sim_cmap[:n, :n]
    exp = exp_cmap[:n, :n]
    
    # Shared LogNorm
    norm = mpl.colors.LogNorm(vmin=1e-5, vmax=1.0)
    
    # Mask opposite triangles
    sim_masked = np.ma.array(sim, mask=np.triu(np.ones_like(sim, dtype=bool), k=0))
    exp_masked = np.ma.array(exp, mask=np.tril(np.ones_like(exp, dtype=bool), k=-1))
    
    # Make masked pixels transparent
    cmap = plt.cm.Reds.copy()
    cmap.set_bad(alpha=0.0)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(sim_masked, norm=norm, cmap=cmap)
    ax.imshow(exp_masked, norm=norm, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Bin index")
    ax.set_ylabel("Bin index")
    plt.colorbar(ax.images[0], ax=ax, label="Normalized contact probability")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_ps_comparison(
    sim_cmap: np.ndarray,
    exp_cmap: np.ndarray,
    bin_size_bp: int,
    output_path: str,
    title: str = "P(s) Comparison: Experimental vs Simulation",
    fit_min_bp: int = 200_000,
    fit_max_bp: int = 20_000_000,
    min_s_bins: int = 1,
    max_s_bins: int = None,
    exclude_zeros: bool = True,
):
    """
    Plot p(s) scaling curves comparing experimental vs simulated contact maps.
    
    Parameters
    ----------
    sim_cmap : np.ndarray
        Simulated contact map (normalized)
    exp_cmap : np.ndarray
        Experimental contact map (normalized)
    bin_size_bp : int
        Bin size in basepairs
    output_path : str
        Path to save the plot
    title : str
        Plot title
    fit_min_bp : int
        Minimum separation for slope fitting (in bp)
    fit_max_bp : int
        Maximum separation for slope fitting (in bp)
    min_s_bins : int
        Minimum diagonal offset to include
    max_s_bins : int
        Maximum diagonal offset to include
    exclude_zeros : bool
        Ignore zero entries when averaging
    """
    # Ensure same size
    n = min(sim_cmap.shape[0], exp_cmap.shape[0])
    sim = sim_cmap[:n, :n]
    exp = exp_cmap[:n, :n]
    
    # Compute p(s)
    s_exp_bins, ps_exp, n_exp = compute_ps(exp, min_s_bins, max_s_bins, exclude_zeros)
    s_sim_bins, ps_sim, n_sim = compute_ps(sim, min_s_bins, max_s_bins, exclude_zeros)
    
    # Convert to bp
    s_exp_bp = s_exp_bins * bin_size_bp
    s_sim_bp = s_sim_bins * bin_size_bp
    
    # Fit slopes
    slope_exp, intercept_exp = fit_loglog_slope(s_exp_bp, ps_exp, fit_min_bp, fit_max_bp)
    slope_sim, intercept_sim = fit_loglog_slope(s_sim_bp, ps_sim, fit_min_bp, fit_max_bp)
    
    # Compute RMSE in log10 space over overlapping s
    common_bp = np.intersect1d(s_exp_bp, s_sim_bp)
    if common_bp.size:
        exp_map = {int(s): v for s, v in zip(s_exp_bp, ps_exp)}
        sim_map = {int(s): v for s, v in zip(s_sim_bp, ps_sim)}
        exp_vec = np.array([exp_map[int(s)] for s in common_bp], dtype=float)
        sim_vec = np.array([sim_map[int(s)] for s in common_bp], dtype=float)
        mask = (exp_vec > 0) & (sim_vec > 0) & np.isfinite(exp_vec) & np.isfinite(sim_vec)
        if mask.any():
            rmse_log10 = np.sqrt(np.mean((np.log10(exp_vec[mask]) - np.log10(sim_vec[mask]))**2))
        else:
            rmse_log10 = np.nan
    else:
        rmse_log10 = np.nan
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(s_exp_bp, ps_exp, 'o-', label=f'Experimental (slope ≈ {slope_exp:.2f})', markersize=4)
    ax.loglog(s_sim_bp, ps_sim, 'o-', label=f'Simulation (slope ≈ {slope_sim:.2f})', markersize=4)
    
    # Draw fitted lines
    for s_bp, ps, slope, intercept, color, name in [
        (s_exp_bp, ps_exp, slope_exp, intercept_exp, None, "exp"),
        (s_sim_bp, ps_sim, slope_sim, intercept_sim, None, "sim")
    ]:
        if np.isfinite(slope) and np.isfinite(intercept):
            if fit_min_bp is None:
                x0 = s_bp[0]
            else:
                x0 = max(s_bp[0], fit_min_bp)
            if fit_max_bp is None:
                x1 = s_bp[-1]
            else:
                x1 = min(s_bp[-1], fit_max_bp)
            xx = np.logspace(np.log10(x0), np.log10(x1), 100)
            yy = 10**(slope * np.log10(xx) + intercept)
            ax.loglog(xx, yy, '--', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Genomic separation s (bp)')
    ax.set_ylabel('Average contact p(s)')
    ax.set_title(f"{title}\nRMSE (log10): {rmse_log10:.3f}")
    ax.grid(True, which='both', alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def compare_contact_maps(
    sim_cmap_path: str,
    exp_cmap_path: str,
    exp_mcool_path: str,
    config_path: str,
    output_cmap_path: str,
    output_ps_path: str,
    verbose: bool = False,
):
    """
    Main function to compare simulated and experimental contact maps.
    
    Parameters
    ----------
    sim_cmap_path : str
        Path to simulated contact map (.npz with "contact_map" key or .npy)
    exp_cmap_path : str
        Path to experimental interpolated contact map (.npy)
    exp_mcool_path : str
        Path to experimental .mcool file (for getting bin counts)
    config_path : str
        Path to config YAML file
    output_cmap_path : str
        Path to save contact map overlay plot
    output_ps_path : str
        Path to save p(s) comparison plot
    verbose : bool
        Print progress information
    """
    # Load config
    if verbose:
        print(f"Loading config from {config_path}")
    config = load_config(config_path)
    
    # Extract parameters from config
    chains = [tuple(chain) for chain in config['simulation']['chains']]
    chroms = config.get('chromosomes', None)
    if chroms is None:
        raise ValueError(
            "Config must contain 'chromosomes' key with list of chromosome names.\n"
            "Example: chromosomes: ['chr4', 'chr13', 'chr17']\n"
            "The order must match the order of chains in simulation.chains."
        )
    
    bin_size_bp = config.get('bin_size_bp', 100000)
    if verbose:
        print(f"Chains: {len(chains)}, Chromosomes: {chroms}, Bin size: {bin_size_bp} bp")
    
    # Get experimental bin counts from .mcool
    if verbose:
        print(f"Loading experimental bin counts from {exp_mcool_path}")
    exp_bin_counts = get_experimental_bin_counts(exp_mcool_path, chroms, bin_size_bp)
    if verbose:
        print(f"Experimental bin counts: {exp_bin_counts}")
    
    # Load simulated contact map
    if verbose:
        print(f"Loading simulated contact map from {sim_cmap_path}")
    sim_cmap_file = Path(sim_cmap_path)
    if sim_cmap_file.suffix == '.npz':
        sim_cmap = np.load(sim_cmap_path)["contact_map"]
    else:
        sim_cmap = np.load(sim_cmap_path)
    if verbose:
        print(f"Simulated contact map shape: {sim_cmap.shape}")
    
    # Trim simulated contact map to match experimental bin counts
    if verbose:
        print("Trimming simulated contact map...")
    sim_cmap_trimmed = trim_simulated_contact_map(
        sim_cmap, chains, chroms, exp_bin_counts, verbose=verbose
    )
    
    # Load experimental contact map
    if verbose:
        print(f"Loading experimental contact map from {exp_cmap_path}")
    exp_cmap = np.load(exp_cmap_path)
    if verbose:
        print(f"Experimental contact map shape: {exp_cmap.shape}")
    
    # Normalize both matrices
    if verbose:
        print("Normalizing contact maps...")
    sim_normalized = row_max_normalize(sim_cmap_trimmed)
    exp_normalized = row_max_normalize(exp_cmap)
    
    # Generate plots
    if verbose:
        print(f"Generating contact map overlay plot: {output_cmap_path}")
    plot_contact_map_overlay(sim_normalized, exp_normalized, output_cmap_path)
    
    if verbose:
        print(f"Generating p(s) comparison plot: {output_ps_path}")
    plot_ps_comparison(
        sim_normalized,
        exp_normalized,
        bin_size_bp,
        output_ps_path,
    )
    
    if verbose:
        print("Done!")


if __name__ == "__main__":
    # Example usage (when called directly)
    import sys
    
    if len(sys.argv) < 7:
        print(
            "Usage: compare_contact_maps.py "
            "<sim_cmap_path> <exp_cmap_path> <exp_mcool_path> <config_path> "
            "<output_cmap_path> <output_ps_path> [--verbose]"
        )
        sys.exit(1)
    
    sim_cmap_path = sys.argv[1]
    exp_cmap_path = sys.argv[2]
    exp_mcool_path = sys.argv[3]
    config_path = sys.argv[4]
    output_cmap_path = sys.argv[5]
    output_ps_path = sys.argv[6]
    verbose = '--verbose' in sys.argv
    
    compare_contact_maps(
        sim_cmap_path,
        exp_cmap_path,
        exp_mcool_path,
        config_path,
        output_cmap_path,
        output_ps_path,
        verbose=verbose,
    )
