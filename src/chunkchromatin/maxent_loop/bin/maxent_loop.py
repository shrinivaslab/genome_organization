
import argparse, os, shutil
from pathlib import Path
import numpy as np
from chunkchromatin.maxent_loop.bin.utils import ensure_dir, write_json, load_config, prepare_seeds, human_time, format_iter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, help="Root directory for this run")
    ap.add_argument("--initial-epsilon", help="Path to KxK epsilon matrix .npy")
    ap.add_argument("--name", required=True, help="Short run name used in job names")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    ap.add_argument("--resume-iter", type=int, help="Resume from specific iteration (will skip initial setup)")
    ap.add_argument("--resume-step", choices=["sim", "process", "update"], help="Resume from specific step within iteration")
    args = ap.parse_args()
    

    proj_root = Path(__file__).resolve().parent.parent
    run_root = Path(args.run_root).resolve()
    cfg = load_config(Path(args.config))

    # Validate arguments based on mode
    if args.resume_iter is not None:
        # Resume mode - initial-epsilon not required
        pass
    else:
        # New run mode - initial-epsilon is required
        if not args.initial_epsilon:
            print("ERROR: --initial-epsilon is required for new runs (not resuming)")
            return

    # Handle resume mode
    if args.resume_iter is not None:
        print(f"Resuming run at iteration {args.resume_iter}")
        if args.resume_step == "process":
            # Special handling for process reduce step resume
            # Clean up the alpha files and then use iteration_driver to handle proper job chaining
            iter_dir = run_root / format_iter(args.resume_iter)
            if not (iter_dir / "update").exists():
                ensure_dir(iter_dir / "update")
            
            # Check epsilon file situation for any iteration
            update_dir = iter_dir / "update"
            existing_epsilons = list(update_dir.glob("epsilon_tk_*.npy"))
            
            if len(existing_epsilons) == 0:
                # No epsilon files exist - need to create the appropriate one
                if args.resume_iter == 0:
                    # For iteration 0, create epsilon_tk_0.npy from initial epsilon
                    epsilon_path = update_dir / "epsilon_tk_0.npy"
                    eps_path = iter_dir / "params" / "epsilon.npy"
                    if eps_path.exists():
                        shutil.copy2(eps_path, epsilon_path)
                        print(f"Created {epsilon_path} from {eps_path}")
                    else:
                        if not args.initial_epsilon:
                            print("ERROR: For iteration 0 resume without existing epsilon.npy, --initial-epsilon is required")
                            return
                        eps0_src = Path(args.initial_epsilon).resolve()
                        shutil.copy2(eps0_src, epsilon_path)
                        print(f"Created {epsilon_path} from {eps0_src}")
                else:
                    # For later iterations, should copy from previous iteration
                    prev_iter = args.resume_iter - 1
                    prev_iter_dir = run_root / format_iter(prev_iter)
                    prev_update_dir = prev_iter_dir / "update"
                    
                    # Find the latest epsilon file from previous iteration
                    prev_epsilons = list(prev_update_dir.glob("epsilon_tk_*.npy"))
                    if prev_epsilons:
                        # Find the highest numbered epsilon file
                        import re
                        max_n = -1
                        latest_epsilon = None
                        for epsilon_file in prev_epsilons:
                            m = re.match(r"epsilon_tk_(\d+)\.npy", epsilon_file.name)
                            if m:
                                n = int(m.group(1))
                                if n > max_n:
                                    max_n = n
                                    latest_epsilon = epsilon_file
                        
                        if latest_epsilon:
                            # Copy as epsilon_tk_{max_n}.npy (same number as source)
                            epsilon_path = update_dir / latest_epsilon.name
                            shutil.copy2(latest_epsilon, epsilon_path)
                            print(f"Copied {latest_epsilon} to {epsilon_path}")
                        else:
                            print(f"ERROR: Could not find valid epsilon files in {prev_update_dir}")
                            return
                    else:
                        print(f"ERROR: No epsilon files found in previous iteration {prev_update_dir}")
                        return
            elif len(existing_epsilons) == 1:
                # Exactly one epsilon file exists - should be fine
                print(f"Found existing epsilon file: {existing_epsilons[0].name}")
            else:
                # Multiple epsilon files - warn user but continue (might be expected)
                print("Found multiple epsilon files:")
                for epsilon_file in existing_epsilons:
                    print(f"  - {epsilon_file.name}")
                print("The process reduce step will use the highest numbered file.")
                print("If this is not intended, please clean up manually before resuming.")
            
            print("Now running process reduce step directly...")
            
            # Verify that obs directory exists and has the necessary files
            obs_dir = iter_dir / "obs"
            if not obs_dir.exists():
                print(f"ERROR: obs directory not found: {obs_dir}")
                print("The simulation step must complete before running the process reduce step.")
                return
            
            # Check for rep files (simulation outputs)
            rep_files = list(obs_dir.glob("rep??_upper_grad_hess.npz"))
            if len(rep_files) == 0:
                print(f"ERROR: No replicate files found in {obs_dir}")
                print("The simulation processing must complete before running the reduce step.")
                print("Expected files like: rep01_upper_grad_hess.npz, rep02_upper_grad_hess.npz, etc.")
                return
            
            print(f"Found {len(rep_files)} replicate files for processing")
            
            # Run process reduce step directly (no SLURM)
            import subprocess
            reduce_cmd = [
                "python", str(proj_root / "bin" / "process_tkl_update.py"), "reduce",
                "--output-dir", str(iter_dir / "obs"),
                "--epsilon-dir", str(iter_dir / "update")
            ]
            print("Running:", " ".join(reduce_cmd))
            result = subprocess.run(reduce_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print("Process reduce step failed:")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return
            
            print("Process reduce step completed successfully!")
            print("STDOUT:", result.stdout)
            
            # Verify epsilon_next.npy was created by the reduce step
            epsilon_next_path = obs_dir / "epsilon_next.npy"
            if not epsilon_next_path.exists():
                print(f"ERROR: epsilon_next.npy not found at {epsilon_next_path}")
                print("The Newton update should have created this file.")
                return
            
            # Verify phi_mean.npy exists for the update step
            phi_mean_path = obs_dir / "phi_mean.npy"
            if not phi_mean_path.exists():
                print(f"ERROR: phi_mean.npy not found at {phi_mean_path}")
                print("This file should be created by the simulation processing step.")
                print("Make sure the simulation and processing workers completed successfully.")
                return
            
            # Now run the update step to complete the iteration and set up the next one
            print("\nRunning update step to complete the iteration...")
            update_cmd = [
                "python", str(proj_root / "bin" / "update_step.py"),
                "--run-root", str(run_root),
                "--iter", str(args.resume_iter),
                "--config", str(Path(args.config).resolve())
            ]
            print("Running:", " ".join(update_cmd))
            result = subprocess.run(update_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print("Update step failed:")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return
            
            print("Update step completed successfully!")
            print("STDOUT:", result.stdout)
            print("\n✅ Iteration", args.resume_iter, "completed successfully!")
            print("The next iteration has been set up and should continue automatically via SLURM.")
            return
        elif args.resume_step == "update":
            # Special handling for update step resume
            iter_dir = run_root / format_iter(args.resume_iter)
            if not (iter_dir / "update").exists():
                ensure_dir(iter_dir / "update")
            
            # Submit just the update step
            import subprocess
            from chunkchromatin.maxent_loop.bin.utils import sbatch_submit
            tpl_dir = proj_root / "templates"
            bin_dir = proj_root / "bin"
            logd = ensure_dir(run_root / "logs")
            
            upd_script_tpl = (tpl_dir / "sbatch_update.sh")
            upd_script_text = upd_script_tpl.read_text().format(
                job_name=f"{args.name}_update_{args.resume_iter:03d}",
                account=cfg["slurm"]["account"],
                partition=cfg["slurm"]["partition"],
                time_limit="00:20:00",
                cpus_per_task=2,
                mem="4G",
                constraint_line=(f"#SBATCH --constraint={cfg['slurm']['constraint']}\n" if cfg["slurm"].get("constraint") else ""),
                qos_line=(f"#SBATCH --qos={cfg['slurm']['qos']}\n" if cfg["slurm"].get("qos") else ""),
                log_dir=str(logd),
                iter_dir=str(iter_dir),
                update_step=str((bin_dir / "update_step.py").resolve()),
                run_root=str(run_root),
                iter_idx=args.resume_iter,
                config_yaml=str(Path(args.config).resolve()),
            )
            upd_sbatch = iter_dir / "update" / "submit_update.sh"
            upd_sbatch.write_text(upd_script_text)
            upd_sbatch.chmod(0o755)
            upd_jobid = sbatch_submit(upd_sbatch)
            write_json(iter_dir / "update" / "submit.json", {"jobid": upd_jobid})
            print(f"Submitted update job {upd_jobid} for iteration {args.resume_iter}")
            return
        else:
            print(f"Resume functionality for step '{args.resume_step}' not implemented yet")
            return

    # Normal initialization mode
    # Prepare run structure
    ensure_dir(run_root / "exp_targets")
    ensure_dir(run_root / "logs")
    seeds = prepare_seeds(cfg["simulation"]["n_replicates"], cfg["simulation"]["seeds_base"])

    # Copy/resolve exp targets and kernel into run_root
    kernel_src = Path(cfg["exp_targets"]["kernel_json"]).resolve()
    targets_src = Path(cfg["exp_targets"]["T_type_kl_npy"]).resolve()
    kernel_dst = run_root / "exp_targets" / "kernel.json"
    targets_dst = run_root / "exp_targets" / "T_type_kl.npy"
    shutil.copy2(kernel_src, kernel_dst)
    shutil.copy2(targets_src, targets_dst)

    # Save seeds
    import json
    with open(run_root / "seeds.json", "w") as f:
        json.dump(seeds, f, indent=2, sort_keys=True)

    # Manifest
    manifest = {
        "name": args.name,
        "created_at": human_time(),
        "config_path": str(Path(args.config).resolve()),
        "kernel_json": str(kernel_dst),
        "targets_npy": str(targets_dst),
        "n_replicates": cfg["simulation"]["n_replicates"],
        "frames": cfg["simulation"]["frames"],
        "burnin_frames": cfg["simulation"]["burnin_frames"],
        "save_frames": cfg["simulation"]["save_frames"],
        "n_types": cfg["simulation"]["n_types"],
    }
    write_json(run_root / "run_manifest.json", manifest)

    # Iteration 0
    iter0 = run_root / format_iter(0)
    ensure_dir(iter0 / "params")
    ensure_dir(iter0 / "sims")
    ensure_dir(iter0 / "obs")
    ensure_dir(iter0 / "update")

    # Copy initial interaction matrix as epsilon
    eps0_src = Path(args.initial_epsilon).resolve()
    eps0_dst = iter0 / "params" / "epsilon.npy"
    shutil.copy2(eps0_src, eps0_dst)
    
    # Also copy as epsilon_tk_0.npy for the Newton update versioning
    epsilon0_dst = iter0 / "update" / "epsilon_tk_0.npy"
    shutil.copy2(eps0_src, epsilon0_dst)

    # Submit iteration 0 driver
    driver = Path(__file__).resolve().parent / "iteration_driver.py"
    import subprocess
    cmd = ["sbatch",
       "--job-name", f"{args.name}_iter000_driver",
       "--account", cfg["slurm"]["account"],
       "--partition", cfg["slurm"]["partition"],
       "--time", "00:10:00",
       "--cpus-per-task", "1",
       "--mem", "1G",
       "--output", str((run_root / "logs" / "driver_%j.out")),
       "--error",  str((run_root / "logs" / "driver_%j.err")),
       str(driver),
       "--run-root", str(run_root),
       "--iter", "0",
       "--config", str(Path(args.config).resolve()),
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

if __name__ == "__main__":
    main()
