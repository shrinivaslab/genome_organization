from diffTre.tests.workflows.common import load_json, ensure_dir, write_json, load_slurm_profile, submit_sbatch_script
import pathlib
from pathlib import Path
import json
import math
from math import ceil
import argparse

def create_sim_slurm_script(config_path: Path, run_root: Path, config: dict, iter_num: int):
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

    # Get flexible script paths
    script_dir = Path(__file__).parent
    sim_worker_script = script_dir / "simulation_worker.py"
    
    print(f"Using sim worker script: {sim_worker_script}")
    
    # Get SLURM configuration
    sim_cfg = cfg["simulation"]
    n_reps = int(sim_cfg["n_replicates"])
    per_task = int(slurm_cfg.get("per_task_replicates", 1))
    
    array_len = int(ceil(n_reps / max(1, per_task)))
    
    sim_res = slurm_cfg.get("sim", slurm_cfg)
    
    env_name = cfg.get("orchestration", {}).get("env_name", "chunkchromatin")
    project_root = Path("/projects/p32733/ME")
    
    # Create simulation script
    sim_script_path = iter_dir / "submit_simulations.sh"
    
    sim_script_content = f"""#!/usr/bin/env bash
#SBATCH --job-name=rep_{iter_num:03d}
#SBATCH --account={sim_res["account"]}
#SBATCH --partition={sim_res["partition"]}
#SBATCH --time={sim_res["time"]}
#SBATCH --cpus-per-task={sim_res["cpus"]}
#SBATCH --mem={sim_res["mem"]}
#SBATCH --gres={sim_res["gres"]}
#SBATCH --array=0-{array_len-1}
#SBATCH --output={run_root}/logs/sim_{iter_num:03d}_%A_%a.out
#SBATCH --error={run_root}/logs/sim_{iter_num:03d}_%A_%a.err

set -euo pipefail
module purge || true
eval "$(/home/pkv4601/.local/bin/micromamba shell hook --shell bash)"
set +u; micromamba activate {env_name}; set -u
cd {project_root}
export PYTHONPATH={project_root}:${{PYTHONPATH:-}}

TASK_ID=${{SLURM_ARRAY_TASK_ID}}
START=$(( TASK_ID * {per_task} + 1 ))
END=$(( START + {per_task} - 1 ))
if [ $END -gt {n_reps} ]; then END={n_reps}; fi

for REP in $(seq $START $END); do
  python "{sim_worker_script}" \
    --config "{config_path}" \
    --run-root "{run_root}" \
    --iter {iter_num} \
    --replicate $REP
done
"""
    
    sim_script_path.write_text(sim_script_content)
    sim_script_path.chmod(0o755)
    
    
    
    return sim_script_path





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
    
    # Submit simulation job
    print(f"\nSubmitting simulation job for iteration {iter_num}...")
    sim_script = create_sim_slurm_script(config_path, run_root, cfg, iter_num)
    sim_job_id = submit_sbatch_script(sim_script, dependency=args.dependency)
    print(f"Submitted simulation job: {sim_job_id}")
    
    # Update manifest: mark sim and obs as pending, store sim job_id
    manifest_path = run_root / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        iter_key = f"iter_{iter_num:03d}"
        if "iterations" not in manifest:
            manifest["iterations"] = {}
        if iter_key not in manifest["iterations"]:
            manifest["iterations"][iter_key] = {}
        
        manifest["iterations"][iter_key]["sim"] = {
            "job_id": sim_job_id,
            "state": "pending"
        }
        manifest["iterations"][iter_key]["obs"] = {
            "state": "pending"
        }
        manifest["iterations"][iter_key]["reweight"] = {
            "state": "pending"
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
    
    # Submit observables job (depends on sim job)
    print(f"\nSubmitting observables job for iteration {iter_num}...")
    from sumbit_energy_observables import create_obs_slurm_script
    obs_script = create_obs_slurm_script(config_path, run_root, cfg, iter_num)
    obs_job_id = submit_sbatch_script(obs_script, dependency=sim_job_id)
    print(f"Submitted observables job: {obs_job_id}")
    
    # Update manifest with obs job_id
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        iter_key = f"iter_{iter_num:03d}"
        if "obs" not in manifest["iterations"][iter_key]:
            manifest["iterations"][iter_key]["obs"] = {}
        manifest["iterations"][iter_key]["obs"]["job_id"] = obs_job_id
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











