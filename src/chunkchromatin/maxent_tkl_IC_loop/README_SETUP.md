# Pre-Run Checklist for MaxEnt TKL+IC Loop

Before starting your optimization run, ensure the following:

## 1. Configuration File
- [ ] Create/edit your config YAML file (see `config_tkl_IC_example.yaml`)
- [ ] Set all required paths:
  - [ ] `exp_targets.T_type_kl_npy` - TKL experimental targets
  - [ ] `exp_targets.phi_exp_IC_npy` - IC experimental targets
  - [ ] `processing_inputs.monomer_types` - monomer type assignments
  - [ ] `processing_inputs.exp_tkl` - TKL experimental data path
  - [ ] `processing_inputs.exp_phi_IC` - IC experimental data path
  - [ ] `processing_inputs.epsilon_init` - initial K×K epsilon matrix
- [ ] Set `ideal_chromosome.d_init` and `d_end` (dmax = d_end - d_init)
- [ ] Configure `update.max_lambda_step_size` (default: 0.5 if not specified)
- [ ] Adjust SLURM resource settings for your cluster

## 2. Required Input Files
- [ ] `T_type_kl.npy` or `.npz` - TKL experimental targets (K×K matrix or vectorized)
- [ ] `phi_exp_IC.npy` - IC experimental targets (dmax vector)
- [ ] `ME_bed_types.npy` - monomer type assignments (N-length array)
- [ ] `epsilon_init.npy` - initial epsilon matrix (K×K, symmetric)
- [ ] (Optional) `lambda_IC_init.npy` - initial lambda_IC (dmax vector, or starts from zeros)

## 3. File Compatibility Checks
- [ ] Verify epsilon_init shape is K×K (matches config `simulation.n_types`)
- [ ] Verify phi_exp_IC length is dmax (matches config `ideal_chromosome.d_end - d_init`)
- [ ] Verify monomer_types length is N (matches config `simulation.N`)
- [ ] Verify T_type_kl dimensions match expected (K×K matrix or K*(K+1)/2 vector)

## 4. Environment Setup
- [ ] Python environment activated (chunkchromatin or your environment)
- [ ] All required packages installed (numpy, scipy, etc.)
- [ ] Module imports work: `python -c "from chunkchromatin.maxent_tkl_IC_loop.bin.utils import load_config"`

## 5. SLURM Access
- [ ] SLURM account configured correctly in config
- [ ] Partitions accessible (check `gengpu` partition for simulations if using GPU)
- [ ] Sufficient quota/resources available

## 6. Directory Structure
- [ ] Choose/create a `--run-root` directory (where results will be stored)
- [ ] Ensure you have write permissions to run-root
- [ ] Consider disk space (simulations can generate large trajectory files)

## 7. Initial Run Command
Once everything is ready, initialize with:

```bash
python -m chunkchromatin.maxent_tkl_IC_loop.bin.maxent_tkl_IC_loop \
  --run-root /path/to/run_directory \
  --name my_run_name \
  --config /path/to/config_tkl_IC.yaml
```

This will:
- Copy experimental targets to `run_root/exp_targets/`
- Initialize iteration 0 with epsilon and lambda_IC
- Submit the first iteration driver job

## 8. Monitoring
- [ ] Check `run_root/logs/` for SLURM output
- [ ] Monitor `run_root/last_update_summary.json` for convergence status
- [ ] Track `run_root/convergence_track.json` for convergence streaks

## Troubleshooting
- If initialization fails: check file paths in config are absolute and files exist
- If simulation fails: check GPU partition access and resource limits
- If processing fails: verify experimental target shapes match expectations
- If convergence issues: adjust `convergence` thresholds or `max_lambda_step_size`

