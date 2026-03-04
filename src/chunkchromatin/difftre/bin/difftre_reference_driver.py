#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from math import ceil
from pathlib import Path

from diffTre.bin.slurm_utils import ensure_dir, render_preamble, render_sbatch_header, sbatch_submit, write_script


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def _sim_array_bounds(n_reps: int, per_task: int) -> int:
    return int(ceil(n_reps / max(1, per_task)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--job-ids-out", required=False, default=None)
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    cfg = load_config(Path(args.config))
    sim_cfg = cfg["simulation"]
    slurm = cfg["slurm"]
    mode = slurm.get("mode", "array")

    log_dir = ensure_dir(run_root / "logs")
    sims_dir = ensure_dir(run_root / "sims")

    n_reps = int(sim_cfg["n_replicates"])
    per_task = int(slurm.get("per_task_replicates", 1))
    array_len = int(slurm.get("array_len", _sim_array_bounds(n_reps, per_task)))

    sim_res = slurm.get("sim", slurm)
    cpu_res = slurm.get("cpu", slurm)

    runner = Path(__file__).resolve().parents[1] / "workflows" / "reference" / "run_reference_replicate.py"
    obs_runner = Path(__file__).resolve().parents[1] / "workflows" / "reference" / "compute_reference_observables.py"

    sim_script = run_root / "submit_reference_sim.sh"
    array_clause = None
    if mode == "array":
        array_clause = f"0-{array_len - 1}"

    sim_header = render_sbatch_header(
        job_name=f"{cfg['run'].get('name','difftre_reference')}_sim",
        slurm_cfg=sim_res,
        output=log_dir / "sim_%A_%a.out",
        error=log_dir / "sim_%A_%a.err",
        array=array_clause,
    )

    sim_body = [render_preamble()]
    sim_body.append(f"export DIFFTRE_RUN_ROOT=\"{run_root}\"")
    sim_body.append(f"export DIFFTRE_CONFIG=\"{Path(args.config).resolve()}\"")
    sim_body.append(f"export DIFFTRE_N_REPS=\"{n_reps}\"")
    sim_body.append(f"export DIFFTRE_PER_TASK=\"{per_task}\"")

    if mode == "array":
        sim_body.append("TASK_ID=${SLURM_ARRAY_TASK_ID}")
        sim_body.append("START=$(( TASK_ID * DIFFTRE_PER_TASK + 1 ))")
        sim_body.append("END=$(( START + DIFFTRE_PER_TASK - 1 ))")
        sim_body.append("if [ $END -gt $DIFFTRE_N_REPS ]; then END=$DIFFTRE_N_REPS; fi")
        sim_body.append("for REP in $(seq $START $END); do")
        sim_body.append(f"  python \"{runner}\" --config \"$DIFFTRE_CONFIG\" --run-root \"$DIFFTRE_RUN_ROOT\" --replicate $REP")
        sim_body.append("done")
    else:
        sim_body.append("REP=${DIFFTRE_REP}")
        sim_body.append(f"python \"{runner}\" --config \"$DIFFTRE_CONFIG\" --run-root \"$DIFFTRE_RUN_ROOT\" --replicate $REP")

    write_script(sim_script, sim_header + "\n" + "\n".join(sim_body) + "\n")

    sim_job_ids = []
    if mode == "array":
        sim_job_ids.append(sbatch_submit(sim_script))
    else:
        for rep in range(1, n_reps + 1):
            indiv = run_root / f"submit_reference_sim_rep{rep:02d}.sh"
            header = render_sbatch_header(
                job_name=f"{cfg['run'].get('name','difftre_reference')}_sim_{rep:02d}",
                slurm_cfg=sim_res,
                output=log_dir / f"sim_{rep:02d}_%j.out",
                error=log_dir / f"sim_{rep:02d}_%j.err",
                array=None,
            )
            body = [render_preamble()]
            body.append(f"export DIFFTRE_RUN_ROOT=\"{run_root}\"")
            body.append(f"export DIFFTRE_CONFIG=\"{Path(args.config).resolve()}\"")
            body.append(f"export DIFFTRE_REP={rep}")
            body.append(f"python \"{runner}\" --config \"$DIFFTRE_CONFIG\" --run-root \"$DIFFTRE_RUN_ROOT\" --replicate $DIFFTRE_REP")
            write_script(indiv, header + "\n" + "\n".join(body) + "\n")
            sim_job_ids.append(sbatch_submit(indiv))

    obs_script = run_root / "submit_reference_obs.sh"
    obs_header = render_sbatch_header(
        job_name=f"{cfg['run'].get('name','difftre_reference')}_obs",
        slurm_cfg=cpu_res,
        output=log_dir / "obs_%j.out",
        error=log_dir / "obs_%j.err",
        array=None,
    )
    obs_body = [render_preamble()]
    obs_body.append(f"python \"{obs_runner}\" --config \"{Path(args.config).resolve()}\" --run-root \"{run_root}\"")
    write_script(obs_script, obs_header + "\n" + "\n".join(obs_body) + "\n")

    obs_job_id = sbatch_submit(obs_script, deps=sim_job_ids)

    job_ids = {
        "mode": mode,
        "sim_job_ids": sim_job_ids,
        "obs_job_id": obs_job_id,
    }
    if args.job_ids_out:
        Path(args.job_ids_out).write_text(json.dumps(job_ids, indent=2))

    print(json.dumps(job_ids))


if __name__ == "__main__":
    main()
