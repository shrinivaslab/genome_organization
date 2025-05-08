import numpy as np
import pandas as pd
import cooler
import h5py

# Load full matrix
contact_sum = np.load("/Users/kadendimarco/Desktop/Shrivinas_lab/genome_archetecture/chromosomes_openmm/polychrom_base/benchmarking/hp_traj_cones_bm/contact_sum_48000.npy")
bin_size_bp = 40_000

# Constants
copies_per_chrom = 4
chain_len = 6000
total_len = chain_len * 2  # 12000 for final matrix

# Create consensus matrix (12000 x 12000)
consensus_matrix = np.zeros((total_len, total_len), dtype=np.float32)

# Sum chr1-chr1 interactions (top-left quadrant)
for i in range(4):
    for j in range(4):
        block = contact_sum[i*chain_len:(i+1)*chain_len, j*chain_len:(j+1)*chain_len]
        consensus_matrix[:chain_len, :chain_len] += block

# Sum chr2-chr2 interactions (bottom-right quadrant)
for i in range(4):
    for j in range(4):
        block = contact_sum[(i+4)*chain_len:(i+5)*chain_len, (j+4)*chain_len:(j+5)*chain_len]
        consensus_matrix[chain_len:, chain_len:] += block

# Sum chr1-chr2 interactions (top-right and bottom-left quadrants)
for i in range(4):
    for j in range(4):
        # Top-right quadrant
        block = contact_sum[i*chain_len:(i+1)*chain_len, (j+4)*chain_len:(j+5)*chain_len]
        consensus_matrix[:chain_len, chain_len:] += block
        # Bottom-left quadrant (transpose of top-right)
        consensus_matrix[chain_len:, :chain_len] += block.T

# Function to convert matrix to pixel format
def matrix_to_pixels(matrix, tril=False):
    i, j = np.triu_indices(matrix.shape[0]) if not tril else np.tril_indices(matrix.shape[0])
    counts = matrix[i, j]
    mask = counts > 0
    return pd.DataFrame({
        'bin1_id': i[mask],
        'bin2_id': j[mask],
        'count': counts[mask]
    })

# Create bins for 40kb resolution
bins_chr1 = pd.DataFrame({
    'chrom': ['sim_chr1'] * chain_len,
    'start': np.arange(chain_len) * bin_size_bp,
    'end': (np.arange(chain_len) + 1) * bin_size_bp
})
bins_chr2 = pd.DataFrame({
    'chrom': ['sim_chr2'] * chain_len,
    'start': np.arange(chain_len) * bin_size_bp,
    'end': (np.arange(chain_len) + 1) * bin_size_bp
})
all_bins_df = pd.concat([bins_chr1, bins_chr2], ignore_index=True)

# Convert to pixels for 40kb resolution
pixels_df = matrix_to_pixels(consensus_matrix)

# Create base cooler at 40kb resolution
cooler.create_cooler(
    cool_uri="hp_sim_cones_consensus_40kb.cool",
    bins=all_bins_df,
    pixels=pixels_df,
    dtypes={'count': 'float32'},
    symmetric_upper=True
)

# Create 200kb resolution by summing 5x5 blocks
block_size = 5
new_chain_len = chain_len // block_size
new_total_len = total_len // block_size

# Create new matrix for 200kb resolution
coarse_matrix = np.zeros((new_total_len, new_total_len), dtype=np.float32)

# Sum blocks for each quadrant
for i in range(new_total_len):
    for j in range(new_total_len):
        block = consensus_matrix[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
        coarse_matrix[i, j] = np.sum(block)

# Create bins for 200kb resolution
bins_chr1_coarse = pd.DataFrame({
    'chrom': ['sim_chr1'] * new_chain_len,
    'start': np.arange(new_chain_len) * (bin_size_bp * block_size),
    'end': (np.arange(new_chain_len) + 1) * (bin_size_bp * block_size)
})
bins_chr2_coarse = pd.DataFrame({
    'chrom': ['sim_chr2'] * new_chain_len,
    'start': np.arange(new_chain_len) * (bin_size_bp * block_size),
    'end': (np.arange(new_chain_len) + 1) * (bin_size_bp * block_size)
})
all_bins_coarse_df = pd.concat([bins_chr1_coarse, bins_chr2_coarse], ignore_index=True)

# Convert to pixels for 200kb resolution
pixels_coarse_df = matrix_to_pixels(coarse_matrix)

# Create 200kb resolution cooler
cooler.create_cooler(
    cool_uri="hp_sim_cones_consensus_200kb.cool",
    bins=all_bins_coarse_df,
    pixels=pixels_coarse_df,
    dtypes={'count': 'float32'},
    symmetric_upper=True
)

# Create multi-resolution cooler
cooler.zoomify_cooler(
    ["hp_sim_cones_consensus_40kb.cool"],
    "hp_sim_cones_consensus.mcool",
    resolutions=[40000, 200000],
    nproc=1,
    chunksize=10000000
)

print("Multi-resolution cooler file created: hp_sim_cones_consensus.mcool") 