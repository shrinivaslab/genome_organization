
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
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    cfg = load_config(Path(args.config))

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
