from diffTre.tests.workflows.common import load_json, ensure_dir, write_json, load_slurm_profile, submit_sbatch_script
import pathlib
from pathlib import Path
import json
import math
from math import ceil
import argparse

def create_reweight_slurm_script(config_path: Path, run_root: Path, config: dict, iter_num: int):
    """Create SLURM script to submit reweighting job for a given iteration"""
    
    iter_dir = run_root / f"iter_{iter_num:03d}"
    
    # Setup directories
    ensure_dir(run_root / "logs")
    ensure_dir(iter_dir / "update")
    
    # read config
    with open(config_path) as f:
        cfg = load_json(f)
    slurm_cfg_path = Path(cfg.get("slurm_profile", {}))
    slurm_cfg = load_json(slurm_cfg_path)
    
    # Get our flexible script paths
    script_dir = Path(__file__).parent
    reweight_worker_script = script_dir / "reweight_worker.py"
    
    print(f"Using reweight worker script: {reweight_worker_script}")
    
    reweight_res = slurm_cfg.get("reweight", slurm_cfg)
    
    env_name = cfg.get("orchestration", {}).get("env_name", "chunkchromatin")
    project_root = Path("/projects/p32733/ME")
    
    # Create reweight script
    reweight_script_path = iter_dir / "submit_reweight.sh"
    
    reweight_script_content = f"""#!/usr/bin/env bash
#SBATCH --job-name=difftre_flexible_fit_reweight_{iter_num:03d}
#SBATCH --account={reweight_res["account"]}
#SBATCH --partition={reweight_res["partition"]}
#SBATCH --time={reweight_res["time"]}
#SBATCH --cpus-per-task={reweight_res["cpus"]}
#SBATCH --mem={reweight_res["mem"]}
#SBATCH --output={run_root}/logs/reweight_{iter_num:03d}_%j.out
#SBATCH --error={run_root}/logs/reweight_{iter_num:03d}_%j.err

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

python "{reweight_worker_script}" \\
  --config "{config_path}" \\
  --run-root "{run_root}" \\
  --iter {iter_num}
"""
    
    reweight_script_path.write_text(reweight_script_content)
    reweight_script_path.chmod(0o755)
    
    return reweight_script_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to simulation configuration JSON file")
    parser.add_argument("--run-root", required=True, help="Root directory for simulation run")
    parser.add_argument("--iter", required=True, type=int, help="Iteration number")
    parser.add_argument("--dependency", default=None, help="SLURM job dependency (e.g., '12345' or 'afterok:12345')")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    run_root = Path(args.run_root)
    iter_num = args.iter
    
    # Load config for the function
    with open(config_path) as f:
        cfg = load_json(f)
    
    print(f"\nSubmitting reweighting job for iteration {iter_num}...")
    reweight_script = create_reweight_slurm_script(config_path, run_root, cfg, iter_num)
    reweight_job_id = submit_sbatch_script(reweight_script, dependency=args.dependency)
    print(f"Submitted reweighting job: {reweight_job_id}")
    
    # Update manifest
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        iter_key = f"iter_{iter_num:03d}"
        if "iterations" not in manifest:
            manifest["iterations"] = {}
        if iter_key not in manifest["iterations"]:
            manifest["iterations"][iter_key] = {}
        
        if "reweight" not in manifest["iterations"][iter_key]:
            manifest["iterations"][iter_key]["reweight"] = {}
        manifest["iterations"][iter_key]["reweight"]["job_id"] = reweight_job_id
        manifest["iterations"][iter_key]["reweight"]["state"] = "pending"
        manifest_path.write_text(json.dumps(manifest, indent=2))
    
    return reweight_job_id

if __name__ == "__main__":
    main()