from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _read_template(name: str) -> str:
    path = Path(__file__).resolve().parent / "templates" / name
    return path.read_text()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_script(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | 0o111)


def render_sbatch_header(
    job_name: str,
    slurm_cfg: dict,
    output: Path,
    error: Path,
    array: str | None = None,
) -> str:
    lines = ["#!/usr/bin/env bash"]
    lines.append(f"#SBATCH --job-name={job_name}")
    lines.append(f"#SBATCH --account={slurm_cfg['account']}")
    lines.append(f"#SBATCH --partition={slurm_cfg['partition']}")
    if slurm_cfg.get("qos"):
        lines.append(f"#SBATCH --qos={slurm_cfg['qos']}")
    if slurm_cfg.get("constraint"):
        lines.append(f"#SBATCH --constraint={slurm_cfg['constraint']}")
    if array:
        lines.append(f"#SBATCH --array={array}")
    lines.append(f"#SBATCH --time={slurm_cfg['time']}")
    lines.append(f"#SBATCH --cpus-per-task={slurm_cfg['cpus']}")
    lines.append(f"#SBATCH --mem={slurm_cfg['mem']}")
    if slurm_cfg.get("gres"):
        lines.append(f"#SBATCH --gres={slurm_cfg['gres']}")
    lines.append(f"#SBATCH --output={output}")
    lines.append(f"#SBATCH --error={error}")
    return "\n".join(lines) + "\n"


def render_preamble() -> str:
    return _read_template("sbatch_preamble.sh") + "\n"


@dataclass
class SlurmSubmission:
    job_id: str
    raw_stdout: str


def parse_sbatch_job_id(stdout: str) -> str:
    parts = stdout.strip().split()
    if not parts:
        raise RuntimeError(f"Could not parse sbatch output: {stdout!r}")
    return parts[-1]


def render_dependency_arg(deps: Iterable[str] | None = None, *, dep_type: str = "afterok") -> str | None:
    dep_list = [str(d) for d in (deps or []) if str(d).strip()]
    if not dep_list:
        return None
    if dep_type not in {"afterok", "afterany", "afternotok"}:
        raise ValueError(f"Unsupported dependency type: {dep_type}")
    return f"--dependency={dep_type}:{':'.join(dep_list)}"


def sbatch_submit(
    script_path: Path,
    deps: Iterable[str] | None = None,
    *,
    dep_type: str = "afterok",
    extra_args: Iterable[str] | None = None,
    return_details: bool = False,
) -> str | SlurmSubmission:
    cmd = ["sbatch"]
    dep_arg = render_dependency_arg(deps, dep_type=dep_type)
    if dep_arg:
        cmd.append(dep_arg)
    if extra_args:
        cmd.extend([str(a) for a in extra_args])
    cmd.append(str(script_path))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed: {proc.stderr.strip()}")
    job_id = parse_sbatch_job_id(proc.stdout)
    if return_details:
        return SlurmSubmission(job_id=job_id, raw_stdout=proc.stdout.strip())
    return job_id


def sbatch_submit_many(
    script_paths: Iterable[Path],
    deps: Iterable[str] | None = None,
    *,
    dep_type: str = "afterok",
    extra_args: Iterable[str] | None = None,
) -> list[str]:
    job_ids: list[str] = []
    for script_path in script_paths:
        jid = sbatch_submit(script_path, deps=deps, dep_type=dep_type, extra_args=extra_args)
        job_ids.append(str(jid))
    return job_ids
