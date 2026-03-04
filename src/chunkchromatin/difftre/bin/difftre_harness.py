#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from diffTre.bin.harness_config import (
    load_harness_config,
    materialize_fit_config,
    materialize_reference_config,
    validate_harness_config,
    wire_fit_to_reference,
)
from diffTre.bin.harness_utils import timestamp_slug, write_json
from diffTre.bin.slurm_utils import ensure_dir


def _run_cmd(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified DiffTRE harness for reference+fit workflows.")
    ap.add_argument("--config", required=True, help="Path to harness JSON config.")
    ap.add_argument("--run-root", required=False, help="Override output run root.")
    ap.add_argument(
        "--submit-only",
        action="store_true",
        help="Submit SLURM jobs and return immediately.",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = load_harness_config(cfg_path)
    validate_harness_config(cfg)

    workflow_mode = cfg.get("workflow", {}).get("mode", "both")
    output_root = Path(args.run_root or cfg["run"]["output_root"]).resolve()
    run_root = output_root / f"{cfg['run']['name']}_{timestamp_slug()}"
    ensure_dir(run_root)
    ensure_dir(run_root / "configs")
    ensure_dir(run_root / "logs")

    ref_cfg_path = None
    fit_cfg_path = None
    ref_root = run_root / "reference"
    fit_root = run_root / "fit"

    if workflow_mode in {"reference", "both"}:
        ref_cfg = materialize_reference_config(cfg, run_root)
        ref_cfg_path = run_root / "configs" / "reference_config.json"
        write_json(ref_cfg_path, ref_cfg)

    if workflow_mode in {"fit", "both"}:
        fit_cfg = materialize_fit_config(cfg, run_root)
        fit_cfg_path = run_root / "configs" / "fit_config.json"
        # Wire to local reference outputs if we run both workflows.
        if workflow_mode == "both":
            fit_cfg = wire_fit_to_reference(fit_cfg, ref_root)
        write_json(fit_cfg_path, fit_cfg)

    summary: dict[str, object] = {
        "harness_config": str(cfg_path),
        "workflow_mode": workflow_mode,
        "run_root": str(run_root),
    }

    if workflow_mode in {"reference", "both"}:
        ref_job_ids_path = run_root / "reference_job_ids.json"
        _run_cmd(
            [
                "python",
                "-m",
                "diffTre.bin.difftre_reference_driver",
                "--run-root",
                str(ref_root),
                "--config",
                str(ref_cfg_path),
                "--job-ids-out",
                str(ref_job_ids_path),
            ]
        )
        summary["reference_job_ids_path"] = str(ref_job_ids_path)
        if ref_job_ids_path.exists():
            summary["reference_job_ids"] = json.loads(ref_job_ids_path.read_text())

    if workflow_mode in {"fit", "both"}:
        if workflow_mode == "fit":
            # fit-only expects explicit reference inputs in config.
            fit_job_ids_path = run_root / "fit_job_ids.json"
            _run_cmd(
                [
                    "python",
                    "-m",
                    "diffTre.workflows.fit.fit_loop",
                    "--config",
                    str(fit_cfg_path),
                    "--run-root",
                    str(fit_root),
                    "--job-ids-out",
                    str(fit_job_ids_path),
                ]
            )
            if fit_job_ids_path.exists():
                summary["fit_job_ids"] = json.loads(fit_job_ids_path.read_text())
        else:
            # For both mode, submit fit dispatch that depends on reference observable completion.
            from diffTre.bin.harness_utils import submit_python_dispatch

            ref_job_ids = json.loads((run_root / "reference_job_ids.json").read_text())
            ref_obs_job = ref_job_ids["obs_job_id"]
            dispatch_cfg = cfg.get("orchestration", {}).get("fit_dispatch", cfg.get("slurm", {}).get("driver"))
            if not dispatch_cfg:
                raise ValueError("Missing orchestration.fit_dispatch (or slurm.driver) for both-mode workflow.")
            fit_dispatch_job = submit_python_dispatch(
                script_path=run_root / "submit_fit_dispatch.sh",
                command=f'python -m diffTre.workflows.fit.fit_loop --config "{fit_cfg_path}" --run-root "{fit_root}"',
                slurm_cfg=dispatch_cfg,
                job_name=f"{cfg['run']['name']}_fit_dispatch",
                output_path=run_root / "logs" / "fit_dispatch_%j.out",
                error_path=run_root / "logs" / "fit_dispatch_%j.err",
                deps=[ref_obs_job],
                env_name=cfg.get("orchestration", {}).get("env_name", "chunkchromatin"),
            )
            summary["fit_dispatch_job_id"] = fit_dispatch_job

    write_json(run_root / "submission_summary.json", summary)
    print(str(run_root / "submission_summary.json"))


if __name__ == "__main__":
    main()
