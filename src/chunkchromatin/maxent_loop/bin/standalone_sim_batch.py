#!/usr/bin/env python3
"""
Standalone simulation batch runner for MaxEnt loop system.

This script allows you to run a batch of 50 simulations with a specific epsilon
matrix without running the full MaxEnt iterative loop. Results are stored in
a separate location you specify.

Usage:
    python standalone_sim_batch.py --epsilon /path/to/epsilon.npy --output-dir /path/to/output --name my_batch_run

The script will:
1. Create the necessary directory structure
2. Copy your epsilon matrix as the interaction matrix
3. Generate and submit a SLURM array job for 50 replicates
4. Store results in your specified output directory
"""

import argparse
import os
import shutil
import json
from pathlib import Path
import numpy as np
from utils import ensure_dir, write_json, load_config, prepare_seeds, human_time, sbatch_submit

def main():
    ap = argparse.ArgumentParser(description="Run standalone simulation batch with specific epsilon")
    ap.add_argument("--epsilon", required=True, help="Path to epsilon interaction matrix (.npy file)")
    ap.add_argument("--output-dir", required=True, help="Directory to store simulation results")
    ap.add_argument("--name", required=True, help="Name for this batch run (used in job names)")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"),
                    help="Path to config.yaml file")
    ap.add_argument("--n-replicates", type=int, default=50, help="Number of replicates to run (default: 50)")
    args = ap.parse_args()

    # Resolve paths
    proj_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir).resolve()
    epsilon_path = Path(args.epsilon).resolve()
    
    # Load configuration
    cfg = load_config(Path(args.config))
    
    # Validate epsilon file exists
    if not epsilon_path.exists():
        print(f"ERROR: Epsilon file not found: {epsilon_path}")
        return 1
    
    # Create output directory structure
    print(f"Setting up simulation batch in: {output_dir}")
    ensure_dir(output_dir)
    ensure_dir(output_dir / "exp_targets")
    ensure_dir(output_dir / "logs")
    ensure_dir(output_dir / "sims")
    
    # Copy experimental targets from config
    kernel_src = Path(cfg["exp_targets"]["kernel_json"]).resolve()
    targets_src = Path(cfg["exp_targets"]["T_type_kl_npy"]).resolve()
    kernel_dst = output_dir / "exp_targets" / "kernel.json"
    targets_dst = output_dir / "exp_targets" / "T_type_kl.npy"
    
    if not kernel_src.exists():
        print(f"ERROR: Kernel file not found: {kernel_src}")
        return 1
    if not targets_src.exists():
        print(f"ERROR: Targets file not found: {targets_src}")
        return 1
        
    shutil.copy2(kernel_src, kernel_dst)
    shutil.copy2(targets_src, targets_dst)
    
    # Copy epsilon matrix as interaction matrix
    interaction_dst = output_dir / "interaction_matrix.npy"
    shutil.copy2(epsilon_path, interaction_dst)
    
    # Generate seeds
    seeds = prepare_seeds(args.n_replicates, cfg["simulation"]["seeds_base"])
    with open(output_dir / "seeds.json", "w") as f:
        json.dump(seeds, f, indent=2, sort_keys=True)
    
    # Create run manifest
    manifest = {
        "name": args.name,
        "created_at": human_time(),
        "type": "standalone_simulation_batch",
        "epsilon_source": str(epsilon_path),
        "config_path": str(Path(args.config).resolve()),
        "kernel_json": str(kernel_dst),
        "targets_npy": str(targets_dst),
        "interaction_matrix": str(interaction_dst),
        "n_replicates": args.n_replicates,
        "frames": cfg["simulation"]["frames"],
        "burnin_frames": cfg["simulation"]["burnin_frames"],
        "save_frames": cfg["simulation"]["save_frames"],
        "n_types": cfg["simulation"]["n_types"],
    }
    write_json(output_dir / "run_manifest.json", manifest)
    
    # Generate SBATCH script for simulation array
    sim_res = cfg["resources"]["simulation"]
    per_task = int(sim_res.get("per_task_replicates", 10))
    array_len = int(sim_res["array_len"])
    
    # Ensure array covers all replicates
    if array_len * per_task < args.n_replicates:
        array_len = (args.n_replicates + per_task - 1) // per_task
        print(f"Adjusted array_len to {array_len} to cover {args.n_replicates} replicates")
    
    tpl_dir = proj_root / "templates"
    bin_dir = proj_root / "bin"
    
    # Read and format the simulation template
    sim_script_tpl = tpl_dir / "sbatch_sim_array.sh"
    sim_script_text = sim_script_tpl.read_text()
    
    # Format the template
    sim_script_formatted = sim_script_text.format(
        job_name=f"{args.name}_sim",
        account=cfg["slurm"]["account"],
        partition=sim_res.get("partition", cfg["slurm"]["partition"]),
        time_limit=sim_res.get("time_limit", cfg["slurm"]["time_limit"]),
        cpus_per_task=sim_res.get("cpus_per_task", cfg["slurm"]["cpus_per_task"]),
        mem=sim_res.get("mem", cfg["slurm"]["mem"]),
        gres=sim_res.get("gres", ""),
        array_max=array_len - 1,
        constraint_line=f"#SBATCH --constraint={cfg['slurm']['constraint']}" if cfg["slurm"]["constraint"] else "",
        qos_line=f"#SBATCH --qos={cfg['slurm']['qos']}" if cfg["slurm"]["qos"] else "",
        log_dir=output_dir / "logs",
        iter_dir=output_dir,  # Use output_dir as the "iteration" directory
        eps_path=interaction_dst,  # Point to our interaction matrix
        seeds_json=output_dir / "seeds.json",
        frames=cfg["simulation"]["frames"],
        burnin=cfg["simulation"]["burnin_frames"],
        save_frames=str(cfg["simulation"]["save_frames"]).lower(),
        n_reps=args.n_replicates,
        kernel_json=kernel_dst,
        targets_npy=targets_dst,
        obs_dir=output_dir / "obs",  # Not used for standalone sim, but required by template
        n_types=cfg["simulation"]["n_types"],
        N=cfg["simulation"]["N"],
        density=cfg["simulation"]["density"],
        chains=json.dumps(cfg["simulation"]["chains"]),
        monomer_types=cfg["processing_inputs"]["monomer_types"],
        interaction_matrix=interaction_dst,
        per_task_reps=per_task,
        series_runner=bin_dir / "series_runner.py",
        run_replicates_array=bin_dir / "run_replicates_array.py"
    )
    
    # Write the SBATCH script
    sim_script_path = output_dir / "submit_sim_batch.sh"
    with open(sim_script_path, "w") as f:
        f.write(sim_script_formatted)
    
    # Make it executable
    os.chmod(sim_script_path, 0o755)
    
    print(f"Created simulation batch setup:")
    print(f"  Output directory: {output_dir}")
    print(f"  Epsilon matrix: {epsilon_path} -> {interaction_dst}")
    print(f"  Number of replicates: {args.n_replicates}")
    print(f"  SLURM array tasks: {array_len} (each running {per_task} replicates)")
    print(f"  SBATCH script: {sim_script_path}")
    print()
    print("To submit the simulation batch:")
    print(f"  sbatch {sim_script_path}")
    print()
    print("Results will be stored in:")
    print(f"  {output_dir}/sims/rep01/, {output_dir}/sims/rep02/, ...")
    
    # Optionally submit immediately
    submit_now = input("Submit the job now? [y/N]: ").strip().lower()
    if submit_now in ['y', 'yes']:
        try:
            jobid = sbatch_submit(str(sim_script_path))
            print(f"Submitted job {jobid}")
            write_json(output_dir / "submit.json", {"jobid": jobid, "script": str(sim_script_path)})
        except Exception as e:
            print(f"Failed to submit job: {e}")
            print(f"You can submit manually with: sbatch {sim_script_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())
