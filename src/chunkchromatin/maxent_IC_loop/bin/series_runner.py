#!/usr/bin/env python3
"""
Run a series of replicates in one SLURM array task.

For each replicate r in [start, end], we call your existing runner
while injecting SLURM_ARRAY_TASK_ID=r into the subprocess environment
(so legacy scripts that key off that ID continue to work unmodified).
"""

import argparse, os, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True, help="Path to run_replicates_array.py")
    ap.add_argument("--start",  type=int, required=True, help="Inclusive replicate start index")
    ap.add_argument("--end",    type=int, required=True, help="Inclusive replicate end index (clamped to n_reps-1)")
    args = ap.parse_args()

    runner = str(Path(args.runner).resolve())
    for r in range(args.start, args.end + 1):
        env = os.environ.copy()
        # Make legacy code happy by pretending this sub-run is its own SLURM array task
        env["SLURM_ARRAY_TASK_ID"] = str(r)
        env["MAXENT_REPLICATE_ID"] = str(r)  # also provide explicit ID
        print(f"[series_runner] launching replicate {r} with {runner}", flush=True)
        proc = subprocess.run([sys.executable, runner, "--replicate_id", str(r)], env=env)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
        # Optional: mark success under per-replicate outdir if the runner didn't already
        outdir = env.get("MAXENT_REPLICATE_OUTDIR")
        if outdir:
            Path(outdir).mkdir(parents=True, exist_ok=True)
            (Path(outdir)/"status.ok").touch()

if __name__ == "__main__":
    main()
