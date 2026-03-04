"""
Submit the reference simulation SLURM job array and, as a dependent job,
the observable calculation step.

Usage:
    python submit_simulations.py \
        --config /path/to/config.json \
        --run-root /path/to/run_root

The SLURM scripts are written to run_root/submit_simulations.sh and
run_root/submit_observables.sh.
"""
from __future__ import annotations

import argparse
import json
from math import ceil
import sys
from pathlib import Path

# Make the project root importable regardless of PYTHONPATH
PROJECT_ROOT = Path("/projects/p32733/ME")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffTre.tests.workflows.common import (
    ensure_dir,
    load_json,
    submit_sbatch_script,
)


# ============================================================
# SCRIPT BUILDERS
# ============================================================

def create_sim_slurm_script(config_path: Path, run_root: Path, cfg: dict) -> Path:
    """Create and return the path to the simulation SLURM array script."""
    ensure_dir(run_root / "logs")
    ensure_dir(run_root / "sims")

    slurm_cfg_path = Path(cfg["slurm_profile"])
    slurm_cfg      = load_json(slurm_cfg_path)

    script_dir        = Path(__file__).parent
    sim_worker_script = script_dir / "reference_simulation_worker.py"

    sim_cfg    = cfg["simulation"]
    n_reps     = int(sim_cfg["n_replicates"])
    per_task   = int(slurm_cfg.get("per_task_replicates", 1))
    array_len  = int(ceil(n_reps / max(1, per_task)))

    sim_res      = slurm_cfg.get("sim", slurm_cfg)
    env_name     = cfg.get("orchestration", {}).get("env_name", "chunkchromatin")
    project_root = Path("/projects/p32733/ME")

    gres_line = f"#SBATCH --gres={sim_res['gres']}" if sim_res.get("gres") else ""

    script_content = f"""#!/usr/bin/env bash
#SBATCH --job-name=ref_sim
#SBATCH --account={sim_res["account"]}
#SBATCH --partition={sim_res["partition"]}
#SBATCH --time={sim_res["time"]}
#SBATCH --cpus-per-task={sim_res["cpus"]}
#SBATCH --mem={sim_res["mem"]}
{gres_line}
#SBATCH --array=0-{array_len - 1}
#SBATCH --output={run_root}/logs/sim_%A_%a.out
#SBATCH --error={run_root}/logs/sim_%A_%a.err

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
  python "{sim_worker_script}" \\
    --config  "{config_path}" \\
    --run-root "{run_root}" \\
    --replicate $REP
done
"""

    script_path = run_root / "submit_simulations.sh"
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    print(f"Wrote simulation script: {script_path}")
    return script_path


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Submit reference simulation jobs and dependent observable calculation."
    )
    parser.add_argument("--config",     required=True, help="Path to config JSON")
    parser.add_argument("--dependency", default=None,  help="Optional SLURM job dependency")
    args = parser.parse_args()

    config_path = Path(args.config)

    

    with open(config_path) as f:
        cfg = json.load(f)
    
    run_root = Path(cfg["run"]["output_dir"])
    run_root.mkdir(parents=True, exist_ok=True)
    # --- Submit simulation array ---
    print("Submitting reference simulation job array...")
    sim_script = create_sim_slurm_script(config_path, run_root, cfg)
    sim_job_id = submit_sbatch_script(sim_script, dependency=args.dependency)
    print(f"Submitted simulation job: {sim_job_id}")

    # --- Submit observable calculation as dependency ---
    print("\nSubmitting reference observable calculation job (depends on sim)...")
    from submit_observables import create_obs_slurm_script
    obs_script = create_obs_slurm_script(config_path, run_root, cfg)
    obs_job_id = submit_sbatch_script(obs_script, dependency=sim_job_id)
    print(f"Submitted observable calculation job: {obs_job_id}")


if __name__ == "__main__":
    main()
