#!/usr/bin/env python3
"""
Standalone simulation batch runner for MaxEnt IC loop system.

This script allows you to run a single iteration of simulations with specific
epsilon and lambda_IC values from the config, compute observables and loss,
without running the full iterative optimization loop.

Usage:
    python standalone_sim_batch.py --output-dir /path/to/output --config /path/to/config.yaml --name my_batch_run

The script will:
1. Create the necessary directory structure
2. Load epsilon and lambda_IC_init from config
3. Generate and submit a SLURM array job for simulations
4. Run processing workers, reduce, and update steps to compute observables and loss
5. Store results in your specified output directory
"""

import argparse
import os
import shutil
import json
from pathlib import Path
import numpy as np
from chunkchromatin.maxent_IC_loop.bin.utils import (
    ensure_dir, write_json, load_config, prepare_seeds, human_time, 
    format_iter, sbatch_submit, make_executable
)

def main():
    ap = argparse.ArgumentParser(description="Run standalone simulation batch with epsilon and lambda_IC from config")
    ap.add_argument("--output-dir", required=True, help="Directory to store simulation results")
    ap.add_argument("--config", required=True, help="Path to config.yaml file")
    ap.add_argument("--name", required=True, help="Name for this batch run (used in job names)")
    args = ap.parse_args()

    # Resolve paths
    proj_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    
    # Load configuration
    cfg = load_config(config_path)
    
    # Validate config file exists
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1
    
    # Create output directory structure
    print(f"Setting up standalone simulation batch in: {output_dir}")
    ensure_dir(output_dir)
    ensure_dir(output_dir / "exp_targets")
    ensure_dir(output_dir / "logs")
    
    # Copy experimental targets from config
    phi_exp_IC_src = Path(cfg["exp_targets"]["phi_exp_IC_npy"]).resolve()
    phi_exp_IC_dst = output_dir / "exp_targets" / "phi_exp_IC.npy"
    
    if not phi_exp_IC_src.exists():
        print(f"ERROR: Experimental targets file not found: {phi_exp_IC_src}")
        return 1
        
    shutil.copy2(phi_exp_IC_src, phi_exp_IC_dst)
    
    # Generate seeds
    seeds = prepare_seeds(cfg["simulation"]["n_replicates"], cfg["simulation"]["seeds_base"])
    with open(output_dir / "seeds.json", "w") as f:
        json.dump(seeds, f, indent=2, sort_keys=True)
    
    # Compute dmax
    dmax = cfg["ideal_chromosome"]["d_end"] - cfg["ideal_chromosome"]["d_init"]
    
    # Verify epsilon path exists (fixed, from config)
    # Check for both 'epsilon' (maxent_IC_loop) and 'epsilon_init' (maxent_tkl_IC_loop)
    epsilon_path = cfg["processing_inputs"].get("epsilon") or cfg["processing_inputs"].get("epsilon_init")
    if epsilon_path is None:
        raise ValueError("config.processing_inputs.epsilon or config.processing_inputs.epsilon_init must be set to the KxK epsilon .npy path")
    eps_path = Path(epsilon_path).resolve()
    if not eps_path.exists():
        raise FileNotFoundError(f"Epsilon file not found at config path: {eps_path}")
    
    # Initialize lambda_IC: load from config if provided, otherwise zeros
    lambda_IC_init_path = cfg["processing_inputs"].get("lambda_IC_init")
    if lambda_IC_init_path is not None and lambda_IC_init_path != "":
        lambda_IC_init_path = Path(lambda_IC_init_path).resolve()
        if not lambda_IC_init_path.exists():
            raise FileNotFoundError(f"lambda_IC_init file not found: {lambda_IC_init_path}")
        lambda_IC_0 = np.load(lambda_IC_init_path)
        if lambda_IC_0.shape != (dmax,):
            raise ValueError(f"lambda_IC_init has wrong shape {lambda_IC_0.shape}, expected ({dmax},)")
        print(f"[SETUP] Loaded initial lambda_IC from {lambda_IC_init_path}")
    else:
        lambda_IC_0 = np.zeros(dmax, dtype=float)
        print(f"[SETUP] Initialized lambda_IC to zeros (dmax={dmax})")
    
    # Create run manifest
    manifest = {
        "name": args.name,
        "created_at": human_time(),
        "type": "standalone_simulation_batch_IC",
        "config_path": str(config_path),
        "phi_exp_IC_npy": str(phi_exp_IC_dst),
        "epsilon_source": str(eps_path),
        "lambda_IC_init_source": str(lambda_IC_init_path) if lambda_IC_init_path else "zeros",
        "n_replicates": cfg["simulation"]["n_replicates"],
        "frames": cfg["simulation"]["frames"],
        "burnin_frames": cfg["simulation"]["burnin_frames"],
        "save_frames": cfg["simulation"]["save_frames"],
        "d_init": cfg["ideal_chromosome"]["d_init"],
        "d_end": cfg["ideal_chromosome"]["d_end"],
        "dmax": dmax,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    
    # Set up iteration 0 directory structure
    iter0 = output_dir / format_iter(0)
    ensure_dir(iter0 / "params")
    ensure_dir(iter0 / "sims")
    ensure_dir(iter0 / "obs")
    ensure_dir(iter0 / "update")
    
    # Save lambda_IC for iteration 0
    lambda_IC_dst = iter0 / "params" / "lambda_IC.npy"
    np.save(lambda_IC_dst, lambda_IC_0)
    
    # Also copy as lambda_IC_tk_0.npy for the Newton update versioning
    lambda_IC_0_dst = iter0 / "update" / "lambda_IC_tk_0.npy"
    np.save(lambda_IC_0_dst, lambda_IC_0)
    
    # Submit iteration 0 driver (which will run sims, workers, reduce, and update)
    driver = Path(__file__).resolve().parent / "iteration_driver_IC_standalone.py"
    import subprocess
    cmd = ["sbatch",
       "--job-name", f"{args.name}_iter000_driver",
       "--account", cfg["slurm"]["account"],
       "--partition", cfg["slurm"]["partition"],
       "--time", "00:10:00",
       "--cpus-per-task", "1",
       "--mem", "1G",
       "--output", str((output_dir / "logs" / "driver_%j.out")),
       "--error",  str((output_dir / "logs" / "driver_%j.err")),
       str(driver),
       "--run-root", str(output_dir),
       "--iter", "0",
       "--config", str(config_path),
       "--name", args.name,
       "--proj-root", str(proj_root),
            ]
    # Optional constraints
    if cfg["slurm"].get("constraint"):
        cmd[1:1] = ["--constraint", cfg["slurm"]["constraint"]]
    if cfg["slurm"].get("qos"):
        cmd[1:1] = ["--qos", cfg["slurm"]["qos"]]

    print("Submitting iteration 0 driver:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    
    print(f"\nCreated standalone simulation batch setup:")
    print(f"  Output directory: {output_dir}")
    print(f"  Epsilon matrix: {eps_path}")
    print(f"  Lambda_IC: {lambda_IC_init_path if lambda_IC_init_path else 'zeros'}")
    print(f"  Number of replicates: {cfg['simulation']['n_replicates']}")
    print(f"\nThe iteration driver will submit:")
    print(f"  1. Simulation array jobs")
    print(f"  2. Processing workers")
    print(f"  3. Reduce step (computes observables)")
    print(f"  4. Update step (computes loss and creates state.json)")
    print(f"\nResults will be stored in:")
    print(f"  {iter0}/sims/ - simulation outputs")
    print(f"  {iter0}/obs/ - observables and processed data")
    print(f"  {iter0}/update/state.json - loss and diagnostics")
    
    return 0

if __name__ == "__main__":
    exit(main())

