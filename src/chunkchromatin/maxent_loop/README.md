# MaxEnt Iterative Loop (SLURM)

This package orchestrates a Maximum Entropy fitting loop over chromatin type–type interaction parameters:

**simulate → process observables → update ε → repeat until convergence**

## One-time setup

1. Place your pinned kernel and experimental target vector here (or update `config.yaml` to point elsewhere):
   - `exp_targets/kernel.json` — must contain JSON like: `{"mu": 11.465, "rc": 1.904, "rcut": 3.0}`
   - `exp_targets/T_type_kl.npy` — a 1D numpy array containing the experimental ⟨Φ⟩ vector in **upper-triangle order**

2. Ensure your project scripts exist (default paths in `config.yaml`):
   - `paths.run_replicates_array` — per-replicate MD runner (invoked via SLURM array)
   - `paths.submit_process_obs`   — reduces frames to write `phi_mean.npy` and `phi_cov_diag.npy` into `iter_XXX/obs/`

   These scripts should read configuration from the following **environment variables** (set for you automatically):
   - `MAXENT_ITER_DIR` `MAXENT_EPS_PATH` `MAXENT_SEEDS_JSON` `MAXENT_FRAMES` `MAXENT_BURNIN` `MAXENT_SAVE_FRAMES`
   - `MAXENT_N_REPLICATES` `MAXENT_KERNEL_JSON` `MAXENT_TARGETS_NPY` `MAXENT_OBS_DIR` `MAXENT_N_TYPES`

   And they should write at minimum:
   - `{iter_dir}/obs/phi_mean.npy` (1D vector in upper-triangle order)
   - `{iter_dir}/obs/phi_cov_diag.npy` (1D vector of variances; NaNs acceptable if not available)
   - optionally `{iter_dir}/obs/qc.json`

3. Edit `config.yaml` for your SLURM account/partition/resources and your system size (`simulation.n_types`).

## How to start a run

From a login node on your cluster:

```bash
python /maxent_loop/bin/maxent_loop.py   --run-root /gpfs/home/USER/maxent_runs/run_001   --initial-epsilon /gpfs/home/USER/maxent_inputs/epsilon_init.npy   --name run_001
```

This will create the run folder, seed schedule, and submit **iteration 000**. Each iteration submits:
- a SLURM **array** for 50 replicates
- a **process** job dependent on the array completion
- an **update+continue** job to compute Δε with BB adaptation and either (a) submit the next iteration or (b) stop

### Storage policy
At any time, only the most recently completed iteration’s frames are retained. When iteration *i* submits *i+1*, it deletes frames from *i-1*.

### Convergence
Defaults (edit in `config.yaml`): stop when both
- `max_abs_residual ≤ 1e-3` and `l2_residual ≤ 5e-3`, or
- `max_param_step ≤ 1e-4`

for **two consecutive** iterations.

### Vectorization order (critical)
Upper-triangle order over type pairs (k≤l), row-major over K×K where K = `simulation.n_types`.
This order must be consistent across your processor and `exp_targets/T_type_kl.npy`.

### Reproducibility
- Seeds are fixed across iterations (`seeds.json`) and mapped `replicate_id → seed` deterministically.
- Full parameter history is stored under each `iter_XXX/params/epsilon.npy` and deltas in `iter_XXX/update/`.

