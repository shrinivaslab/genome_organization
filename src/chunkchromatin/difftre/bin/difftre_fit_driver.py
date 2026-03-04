#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--iter", required=True, type=int)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    cfg = load_config(Path(args.config))
    sim_cfg = cfg["simulation"]
    slurm = cfg["slurm"]
    mode = slurm.get("mode", "array")

    it = int(args.iter)
    it_dir = run_root / f"iter_{it:03d}"
    log_dir = ensure_dir(run_root / "logs")
    ensure_dir(it_dir / "sims")
    ensure_dir(it_dir / "obs")
    ensure_dir(it_dir / "update")
    ensure_dir(it_dir / "params")

    n_reps = int(sim_cfg["n_replicates"])
    per_task = int(slurm.get("per_task_replicates", 1))
    array_len = int(slurm.get("array_len", _sim_array_bounds(n_reps, per_task)))

    sim_res = slurm.get("sim", slurm)
    cpu_res = slurm.get("cpu", slurm)

    runner = Path(__file__).resolve().parents[1] / "workflows" / "fit" / "run_fit_replicate.py"
    update_runner = Path(__file__).resolve().parents[1] / "workflows" / "fit" / "fit_update_reweight.py"

    sim_script = it_dir / "submit_fit_sim.sh"
    array_clause = None
    if mode == "array":
        array_clause = f"0-{array_len - 1}"

    sim_header = render_sbatch_header(
        job_name=f"{cfg['run'].get('name','difftre_fit')}_sim_{it:03d}",
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
    sim_body.append(f"export DIFFTRE_ITER={it}")

    if mode == "array":
        sim_body.append("TASK_ID=${SLURM_ARRAY_TASK_ID}")
        sim_body.append("START=$(( TASK_ID * DIFFTRE_PER_TASK + 1 ))")
        sim_body.append("END=$(( START + DIFFTRE_PER_TASK - 1 ))")
        sim_body.append("if [ $END -gt $DIFFTRE_N_REPS ]; then END=$DIFFTRE_N_REPS; fi")
        sim_body.append("for REP in $(seq $START $END); do")
        sim_body.append(
            f"  python \"{runner}\" --config \"$DIFFTRE_CONFIG\" --run-root \"$DIFFTRE_RUN_ROOT\" --iter $DIFFTRE_ITER --replicate $REP"
        )
        sim_body.append("done")
    else:
        sim_body.append("REP=${DIFFTRE_REP}")
        sim_body.append(
            f"python \"{runner}\" --config \"$DIFFTRE_CONFIG\" --run-root \"$DIFFTRE_RUN_ROOT\" --iter $DIFFTRE_ITER --replicate $REP"
        )

    write_script(sim_script, sim_header + "\n" + "\n".join(sim_body) + "\n")

    sim_job_ids = []
    if mode == "array":
        sim_job_ids.append(sbatch_submit(sim_script))
    else:
        for rep in range(1, n_reps + 1):
            indiv = it_dir / f"submit_fit_sim_rep{rep:02d}.sh"
            header = render_sbatch_header(
                job_name=f"{cfg['run'].get('name','difftre_fit')}_sim_{it:03d}_{rep:02d}",
                slurm_cfg=sim_res,
                output=log_dir / f"sim_{it:03d}_{rep:02d}_%j.out",
                error=log_dir / f"sim_{it:03d}_{rep:02d}_%j.err",
                array=None,
            )
            body = [render_preamble()]
            body.append(f"export DIFFTRE_RUN_ROOT=\"{run_root}\"")
            body.append(f"export DIFFTRE_CONFIG=\"{Path(args.config).resolve()}\"")
            body.append(f"export DIFFTRE_ITER={it}")
            body.append(f"export DIFFTRE_REP={rep}")
            body.append(
                f"python \"{runner}\" --config \"$DIFFTRE_CONFIG\" --run-root \"$DIFFTRE_RUN_ROOT\" --iter $DIFFTRE_ITER --replicate $DIFFTRE_REP"
            )
            write_script(indiv, header + "\n" + "\n".join(body) + "\n")
            sim_job_ids.append(sbatch_submit(indiv))

    update_script = it_dir / "submit_fit_update.sh"
    update_header = render_sbatch_header(
        job_name=f"{cfg['run'].get('name','difftre_fit')}_update_{it:03d}",
        slurm_cfg=cpu_res,
        output=log_dir / f"update_{it:03d}_%j.out",
        error=log_dir / f"update_{it:03d}_%j.err",
        array=None,
    )
    update_body = [render_preamble()]
    update_body.append(
        f"python \"{update_runner}\" --config \"{Path(args.config).resolve()}\" --run-root \"{run_root}\" --iter {it}"
    )
    write_script(update_script, update_header + "\n" + "\n".join(update_body) + "\n")

    sbatch_submit(update_script, deps=sim_job_ids)


if __name__ == "__main__":
    main()
