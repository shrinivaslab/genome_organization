# Standalone Simulation Batches

This guide explains how to run a batch of 50 simulations with a specific epsilon (interaction matrix) without running the full MaxEnt iterative loop.

## Quick Start

```bash
# Run 50 simulations with your epsilon matrix
python bin/standalone_sim_batch.py \
    --epsilon /path/to/your/epsilon.npy \
    --output-dir /path/to/results/my_batch_run \
    --name my_batch_001
```

## What This Does

The `standalone_sim_batch.py` script will:

1. **Set up directory structure** in your specified output directory
2. **Copy your epsilon matrix** as the interaction matrix for simulations
3. **Generate seeds** for reproducible replicates
4. **Create a SLURM array job** to run 50 simulation replicates
5. **Store all results** in your specified location (separate from MaxEnt loop runs)

## Arguments

- `--epsilon`: Path to your KxK epsilon interaction matrix (.npy file)
- `--output-dir`: Directory where you want to store results
- `--name`: Short name for this batch (used in SLURM job names)
- `--config`: Path to config.yaml (optional, defaults to config.yaml in this directory)
- `--n-replicates`: Number of replicates (optional, defaults to 50)

## Output Structure

After running, your output directory will contain:

```
my_batch_run/
├── run_manifest.json          # Metadata about this batch
├── interaction_matrix.npy     # Copy of your epsilon matrix
├── seeds.json                 # Seeds for reproducible runs
├── exp_targets/               # Copied from config
│   ├── kernel.json
│   └── T_type_kl.npy
├── logs/                      # SLURM output logs
│   ├── sim_12345_0.out
│   ├── sim_12345_1.out
│   └── ...
├── sims/                      # Simulation results
│   ├── rep01/
│   │   ├── trajectory.traj
│   │   ├── manifest.json
│   │   └── simulation_stats.txt
│   ├── rep02/
│   └── ...
└── submit_sim_batch.sh        # Generated SBATCH script
```

## Example Workflow

1. **Prepare your epsilon matrix**:
   ```python
   import numpy as np
   
   # Your optimized epsilon values
   epsilon = np.array([
       [0.0, -1.2, -0.8, 0.5, 1.0],
       [-1.2, 0.0, -0.5, 0.3, 0.8],
       [-0.8, -0.5, 0.0, -0.2, 0.4],
       [0.5, 0.3, -0.2, 0.0, -0.6],
       [1.0, 0.8, 0.4, -0.6, 0.0]
   ])
   np.save('/path/to/my_epsilon.npy', epsilon)
   ```

2. **Run the batch**:
   ```bash
   python bin/standalone_sim_batch.py \
       --epsilon /path/to/my_epsilon.npy \
       --output-dir /gpfs/home/USER/sim_batches/epsilon_test_001 \
       --name eps_test_001
   ```

3. **Submit to SLURM** (the script will ask if you want to submit immediately):
   ```bash
   # Or submit manually later:
   sbatch /gpfs/home/USER/sim_batches/epsilon_test_001/submit_sim_batch.sh
   ```

4. **Monitor the job**:
   ```bash
   squeue -u $USER
   ```

## Differences from MaxEnt Loop

- **No iteration**: Runs simulations once with your fixed epsilon
- **No processing**: Only generates trajectory files, no observable computation
- **No updates**: No parameter optimization or convergence checking
- **Custom location**: Results stored wherever you specify
- **Independent**: Doesn't interfere with ongoing MaxEnt loop runs

## Resource Configuration

The script uses the same SLURM resources configured in `config.yaml` under `resources.simulation`. You can modify these settings in the config file if needed:

```yaml
resources:
  simulation:
    array_len: 5              # Number of SLURM array tasks
    per_task_replicates: 10   # Replicates per task (5×10 = 50 total)
    time_limit: "4:00:00"     # Time limit per task
    cpus_per_task: 1          # CPUs per task
    mem: "5G"                 # Memory per task
    partition: "gengpu"       # GPU partition
    gres: "gpu:a100:1"        # GPU allocation
```

## Processing Results

After simulations complete, you can process the trajectories using your existing analysis tools. The trajectory files are stored in the same format as regular MaxEnt loop simulations.
