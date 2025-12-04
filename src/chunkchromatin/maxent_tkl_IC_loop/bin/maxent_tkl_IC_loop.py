
import argparse, os, shutil
from pathlib import Path
import numpy as np
from chunkchromatin.maxent_tkl_IC_loop.bin.utils import ensure_dir, write_json, load_config, prepare_seeds, human_time, format_iter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, help="Root directory for this run")
    ap.add_argument("--name", required=True, help="Short run name used in job names")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config_tkl_IC.yaml"))
    args = ap.parse_args()
    

    proj_root = Path(__file__).resolve().parent.parent
    run_root = Path(args.run_root).resolve()
    cfg = load_config(Path(args.config))

    # Normal initialization mode
    # Prepare run structure
    ensure_dir(run_root / "exp_targets")
    ensure_dir(run_root / "logs")
    seeds = prepare_seeds(cfg["simulation"]["n_replicates"], cfg["simulation"]["seeds_base"])

    # Copy/resolve exp targets for TKL
    tkl_targets_src = Path(cfg["exp_targets"]["T_type_kl_npy"]).resolve()
    tkl_targets_dst = run_root / "exp_targets" / "T_type_kl.npy"
    shutil.copy2(tkl_targets_src, tkl_targets_dst)

    # Copy/resolve exp targets for IC
    phi_exp_IC_src = Path(cfg["exp_targets"]["phi_exp_IC_npy"]).resolve()
    phi_exp_IC_dst = run_root / "exp_targets" / "phi_exp_IC.npy"
    shutil.copy2(phi_exp_IC_src, phi_exp_IC_dst)

    # Save seeds
    import json
    with open(run_root / "seeds.json", "w") as f:
        json.dump(seeds, f, indent=2, sort_keys=True)

    # Manifest
    dmax = cfg["ideal_chromosome"]["d_end"] - cfg["ideal_chromosome"]["d_init"]
    manifest = {
        "name": args.name,
        "created_at": human_time(),
        "config_path": str(Path(args.config).resolve()),
        "T_type_kl_npy": str(tkl_targets_dst),
        "phi_exp_IC_npy": str(phi_exp_IC_dst),
        "n_replicates": cfg["simulation"]["n_replicates"],
        "frames": cfg["simulation"]["frames"],
        "burnin_frames": cfg["simulation"]["burnin_frames"],
        "save_frames": cfg["simulation"]["save_frames"],
        "n_types": cfg["simulation"]["n_types"],
        "d_init": cfg["ideal_chromosome"]["d_init"],
        "d_end": cfg["ideal_chromosome"]["d_end"],
        "dmax": dmax,
    }
    write_json(run_root / "run_manifest.json", manifest)

    # Iteration 0
    iter0 = run_root / format_iter(0)
    ensure_dir(iter0 / "params")
    ensure_dir(iter0 / "sims")
    ensure_dir(iter0 / "obs")
    ensure_dir(iter0 / "update")

    # Initialize epsilon (for TKL optimization)
    epsilon_init_path = cfg["processing_inputs"].get("epsilon_init")
    if epsilon_init_path is None or epsilon_init_path == "":
        raise ValueError("config.processing_inputs.epsilon_init must be set to the initial KxK epsilon .npy path")
    eps_init_path = Path(epsilon_init_path).resolve()
    if not eps_init_path.exists():
        raise FileNotFoundError(f"Initial epsilon file not found: {eps_init_path}")
    
    eps0_dst = iter0 / "params" / "epsilon.npy"
    shutil.copy2(eps_init_path, eps0_dst)
    
    # Also copy as epsilon_tk_0.npy for the Newton update versioning
    epsilon0_dst = iter0 / "update" / "epsilon_tk_0.npy"
    shutil.copy2(eps_init_path, epsilon0_dst)
    print(f"[SETUP] Initialized epsilon from {eps_init_path}")

    # Initialize lambda_IC (for IC optimization)
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
    
    lambda_IC_dst = iter0 / "params" / "lambda_IC.npy"
    np.save(lambda_IC_dst, lambda_IC_0)
    
    # Also copy as lambda_IC_tk_0.npy for the Newton update versioning
    lambda_IC_0_dst = iter0 / "update" / "lambda_IC_tk_0.npy"
    np.save(lambda_IC_0_dst, lambda_IC_0)

    # Submit iteration 0 driver
    driver = Path(__file__).resolve().parent / "iteration_driver_tkl_IC.py"
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

