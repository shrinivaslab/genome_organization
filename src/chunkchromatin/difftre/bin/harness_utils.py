from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from diffTre.bin.slurm_utils import ensure_dir, render_preamble, render_sbatch_header, sbatch_submit, write_script


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2))


def submit_python_dispatch(
    *,
    script_path: Path,
    command: str,
    slurm_cfg: dict,
    job_name: str,
    output_path: Path,
    error_path: Path,
    deps: list[str] | None = None,
    env_name: str = "chunkchromatin",
) -> str:
    header = render_sbatch_header(
        job_name=job_name,
        slurm_cfg=slurm_cfg,
        output=output_path,
        error=error_path,
        array=None,
    )
    body = [render_preamble()]
    body.append(f"set +u; micromamba activate {env_name}; set -u")
    body.append("cd /projects/p32733/ME")
    body.append("export PYTHONPATH=/projects/p32733/ME:${PYTHONPATH:-}")
    body.append(command)
    write_script(script_path, header + "\n" + "\n".join(body) + "\n")
    return sbatch_submit(script_path, deps=deps)
