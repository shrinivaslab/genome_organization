from diffTre.tests.workflows.common import load_json, ensure_dir, write_json, load_slurm_profile, submit_sbatch_script
import pathlib
from pathlib import Path
import json
import math
from math import ceil
import argparse

def create_obs_slurm_script(config_path: Path, run_root: Path, config: dict, iter_num: int):
    """Create SLURM script to submit simulations for a given iteration"""
    
    iter_dir = run_root / f"iter_{iter_num:03d}"
    
    # Setup directories
    ensure_dir(run_root / "logs")
    ensure_dir(iter_dir / "sims")
    ensure_dir(iter_dir / "configs")

    # read config
    with open(config_path) as f:
        cfg = load_json(f)
    slurm_cfg_path = Path(cfg.get("slurm_profile", {}))
    slurm_cfg = load_json(slurm_cfg_path)

    # Get our flexible script paths
    script_dir = Path(__file__).parent
    obs_worker_script = script_dir / "energy_observable_worker.py"
    
    print(f"Using obs worker script: {obs_worker_script}")
    
    # Get SLURM configuration
    sim_cfg = cfg["simulation"]
    n_reps = int(sim_cfg["n_replicates"])
    per_task = int(slurm_cfg.get("per_task_replicates", 1))
    
    array_len = int(ceil(n_reps / max(1, per_task)))
    
    obs_res = slurm_cfg.get("processing", slurm_cfg)
    
    env_name = cfg.get("orchestration", {}).get("env_name", "chunkchromatin")
    project_root = Path("/projects/p32733/ME")
    
    # Create simulation script
    obs_script_path = iter_dir / "submit_energy_observables.sh"
    
    obs_script_content = f"""#!/usr/bin/env bash
#SBATCH --job-name=difftre_flexible_fit_update_{iter_num:03d}
#SBATCH --account={obs_res["account"]}
#SBATCH --partition={obs_res["partition"]}
#SBATCH --time={obs_res["time"]}
#SBATCH --cpus-per-task={obs_res["cpus"]}
#SBATCH --mem={obs_res["mem"]}
#SBATCH --output={run_root}/logs/update_{iter_num:03d}_%j.out
#SBATCH --error={run_root}/logs/update_{iter_num:03d}_%j.err

set -euo pipefail
module purge || true
eval "$(/home/pkv4601/.local/bin/micromamba shell hook --shell bash)"
set +u; micromamba activate {env_name}; set -u
cd {project_root}
export PYTHONPATH={project_root}:${{PYTHONPATH:-}}
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export JAX_ENABLE_X64=true
export CUDA_VISIBLE_DEVICES=""

python "{obs_worker_script}" \
  --config "{config_path}" \
  --run-root "{run_root}" \
  --iter {iter_num}
"""
    
    obs_script_path.write_text(obs_script_content)
    obs_script_path.chmod(0o755)
    
    return obs_script_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to simulation configuration JSON file")
    parser.add_argument("--run-root", required=True, help="Root directory for simulation run")
    parser.add_argument("--iter", required=True, type=int, help="Iteration number")
    parser.add_argument("--dependency", default=None, help="SLURM job dependency")
    args = parser.parse_args()

    config_path = Path(args.config)
    run_root = Path(args.run_root)
    iter_num = args.iter

    # Load config
    with open(config_path) as f:
        cfg = load_json(f)
    
    # Submit observables job
    print(f"\nSubmitting observables job for iteration {iter_num}...")
    obs_script = create_obs_slurm_script(config_path, run_root, cfg, iter_num)
    obs_job_id = submit_sbatch_script(obs_script, dependency=args.dependency)
    print(f"Submitted observables job: {obs_job_id}")
    
    # Update manifest
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        iter_key = f"iter_{iter_num:03d}"
        if "iterations" not in manifest:
            manifest["iterations"] = {}
        if iter_key not in manifest["iterations"]:
            manifest["iterations"][iter_key] = {}
        
        if "obs" not in manifest["iterations"][iter_key]:
            manifest["iterations"][iter_key]["obs"] = {}
        manifest["iterations"][iter_key]["obs"]["job_id"] = obs_job_id
        manifest["iterations"][iter_key]["obs"]["state"] = "pending"
        
        if "reweight" not in manifest["iterations"][iter_key]:
            manifest["iterations"][iter_key]["reweight"] = {}
        manifest["iterations"][iter_key]["reweight"]["state"] = "pending"
        
        manifest_path.write_text(json.dumps(manifest, indent=2))
    
    # Submit reweighting job (depends on obs job)
    print(f"\nSubmitting reweighting job for iteration {iter_num}...")
    from submit_reweight import create_reweight_slurm_script
    reweight_script = create_reweight_slurm_script(config_path, run_root, cfg, iter_num)
    reweight_job_id = submit_sbatch_script(reweight_script, dependency=obs_job_id)
    print(f"Submitted reweighting job: {reweight_job_id}")

    # Update manifest with reweight job_id
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        iter_key = f"iter_{iter_num:03d}"
        if "reweight" not in manifest["iterations"][iter_key]:
            manifest["iterations"][iter_key]["reweight"] = {}
        manifest["iterations"][iter_key]["reweight"]["job_id"] = reweight_job_id
        manifest_path.write_text(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()











