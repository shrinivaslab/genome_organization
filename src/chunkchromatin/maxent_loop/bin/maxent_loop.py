
import argparse, os, shutil
from pathlib import Path
import numpy as np
from chunkchromatin.maxent_loop.bin.utils import ensure_dir, write_json, load_config, prepare_seeds, human_time, format_iter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, help="Root directory for this run")
    ap.add_argument("--initial-epsilon", required=True, help="Path to KxK epsilon matrix .npy")
    ap.add_argument("--name", required=True, help="Short run name used in job names")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yaml"))
    ap.add_argument("--resume-iter", type=int, help="Resume from specific iteration (will skip initial setup)")
    ap.add_argument("--resume-step", choices=["sim", "process", "update"], help="Resume from specific step within iteration")
    args = ap.parse_args()
    

    proj_root = Path(__file__).resolve().parent.parent
    run_root = Path(args.run_root).resolve()
    cfg = load_config(Path(args.config))

    # Handle resume mode
    if args.resume_iter is not None:
        print(f"Resuming run at iteration {args.resume_iter}")
        if args.resume_step == "process":
            # Special handling for process reduce step resume
            iter_dir = run_root / format_iter(args.resume_iter)
            if not (iter_dir / "update").exists():
                ensure_dir(iter_dir / "update")
            
            # If this is iter_000 and no alpha_tk_0.npy exists, create it
            if args.resume_iter == 0:
                alpha0_path = iter_dir / "update" / "alpha_tk_0.npy"
                if not alpha0_path.exists():
                    eps_path = iter_dir / "params" / "epsilon.npy"
                    if eps_path.exists():
                        shutil.copy2(eps_path, alpha0_path)
                        print(f"Created {alpha0_path} from {eps_path}")
                    else:
                        eps0_src = Path(args.initial_epsilon).resolve()
                        shutil.copy2(eps0_src, alpha0_path)
                        print(f"Created {alpha0_path} from {eps0_src}")
            
            # Submit the process reduce step
            import subprocess
            from chunkchromatin.maxent_loop.bin.utils import sbatch_submit
            tpl_dir = proj_root / "templates"
            bin_dir = proj_root / "bin"
            logd = ensure_dir(run_root / "logs")
            
            procr = cfg["resources"]["processing"]["reduce"]
            alpha_dir = procr.get("alpha_dir") or str(iter_dir / "update")
            procr_tpl = (tpl_dir / "sbatch_process_reduce.sh")
            procr_text = procr_tpl.read_text().format(
                job_name=f"{args.name}_pred_{args.resume_iter:03d}",
                account=cfg["slurm"]["account"],
                partition=cfg["slurm"]["partition"],
                time_limit=procr["time_limit"],
                cpus_per_task=procr["cpus_per_task"],
                mem=procr["mem"],
                constraint_line=(f"#SBATCH --constraint={cfg['slurm']['constraint']}\n" if cfg["slurm"].get("constraint") else ""),
                qos_line=(f"#SBATCH --qos={cfg['slurm']['qos']}\n" if cfg["slurm"].get("qos") else ""),
                log_dir=str(logd),
                iter_dir=str(iter_dir),
                obs_dir=str(iter_dir / "obs"),
                alpha_dir=alpha_dir,
                process_tkl_update=str((bin_dir / "process_tkl_update.py").resolve()),
            )
            procr_sbatch = iter_dir / "obs" / "submit_process_reduce.sh"
            procr_sbatch.write_text(procr_text)
            procr_sbatch.chmod(0o755)
            procr_jobid = sbatch_submit(procr_sbatch)
            write_json(iter_dir / "obs" / "submit_reduce.json", {"jobid": procr_jobid})
            print(f"Submitted process reduce job {procr_jobid} for iteration {args.resume_iter}")
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

    # Copy initial epsilon
    eps0_src = Path(args.initial_epsilon).resolve()
    eps0_dst = iter0 / "params" / "epsilon.npy"
    shutil.copy2(eps0_src, eps0_dst)
    
    # Also copy initial epsilon as alpha_tk_0.npy for the first update step
    # This prevents FileNotFoundError in process_tkl_update.py reduce mode
    alpha0_dst = iter0 / "update" / "alpha_tk_0.npy"
    shutil.copy2(eps0_src, alpha0_dst)

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
