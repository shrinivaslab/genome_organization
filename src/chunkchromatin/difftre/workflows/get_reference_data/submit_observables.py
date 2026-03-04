"""
Submit the reference observable calculation SLURM job.

Can be called standalone or imported by submit_simulations.py to chain the
observable step as a dependency of the simulation array job.

Usage:
    python submit_observables.py \
        --config /path/to/config.json \
        --run-root /path/to/run_root [--dependency <job_id>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from diffTre.tests.workflows.common import (
    ensure_dir,
    load_json,
    submit_sbatch_script,
)


# ============================================================
# SCRIPT BUILDER
# ============================================================

def create_obs_slurm_script(config_path: Path, run_root: Path, cfg: dict) -> Path:
    """Create and return the path to the observable calculation SLURM script."""
    ensure_dir(run_root / "logs")

    slurm_cfg_path = Path(cfg["slurm_profile"])
    slurm_cfg      = load_json(slurm_cfg_path)

    script_dir       = Path(__file__).parent
    obs_worker_script = script_dir / "reference_observable_calculation.py"

    obs_res      = slurm_cfg.get("processing", slurm_cfg)
    env_name     = cfg.get("orchestration", {}).get("env_name", "chunkchromatin")
    project_root = Path("/projects/p32733/ME")

    script_content = f"""#!/usr/bin/env bash
#SBATCH --job-name=ref_obs
#SBATCH --account={obs_res["account"]}
#SBATCH --partition={obs_res["partition"]}
#SBATCH --time={obs_res["time"]}
#SBATCH --cpus-per-task={obs_res["cpus"]}
#SBATCH --mem={obs_res["mem"]}
#SBATCH --gres={obs_res["gres"]}
#SBATCH --output={run_root}/logs/obs_%j.out
#SBATCH --error={run_root}/logs/obs_%j.err

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

python "{obs_worker_script}" \\
  --config   "{config_path}" \\
  --run-root "{run_root}"
"""

    script_path = run_root / "submit_observables.sh"
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    print(f"Wrote observable script: {script_path}")
    return script_path


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Submit the reference observable calculation job."
    )
    parser.add_argument("--config",     required=True, help="Path to config JSON")
    parser.add_argument("--run-root",   required=True, help="Reference run root directory")
    parser.add_argument("--dependency", default=None,  help="Optional SLURM job dependency")
    args = parser.parse_args()

    config_path = Path(args.config)
    run_root    = Path(args.run_root)

    with open(config_path) as f:
        cfg = json.load(f)

    print("Submitting reference observable calculation job...")
    obs_script = create_obs_slurm_script(config_path, run_root, cfg)
    obs_job_id = submit_sbatch_script(obs_script, dependency=args.dependency)
    print(f"Submitted observable calculation job: {obs_job_id}")


if __name__ == "__main__":
    main()
