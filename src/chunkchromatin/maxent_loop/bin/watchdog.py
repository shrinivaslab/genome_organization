#!/usr/bin/env python3
"""
Watchdog for a MaxEnt loop run.

Usage:
    python watchdog.py --run-root /path/to/run_001
"""

import argparse, json
from pathlib import Path

def load_state(run_root: Path):
    last = run_root / "last_update_summary.json"
    if not last.exists():
        return None, None
    data = json.loads(last.read_text())
    return data.get("iteration"), data

def load_iter_state(run_root: Path, it: int):
    p = run_root / f"iter_{it:03d}" / "update" / "state.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    args = ap.parse_args()
    run_root = Path(args.run_root)

    it, summary = load_state(run_root)
    if it is None:
        print("No iterations have completed yet.")
        return

    print(f"=== Run: {run_root.name} ===")
    print(f" Last completed iteration: {it}")
    print(f" Converged this iter? {summary['converged_this_iter']}")
    print(f" Consecutive streak: {summary['consecutive_streak']}")
    print(f" Done? {summary['done']}")
    print("")

    st = load_iter_state(run_root, it)
    if st:
        print(" Residuals:")
        print(f"   max |g| = {st['max_abs_residual']:.3e}")
        print(f"   L2   ||g|| = {st['l2_residual']:.3e}")
        print(" Step sizes:")
        print(f"   max Δε = {st['max_param_step']:.3e}")
        print(f"   η = {st['eta']:.3e}")
    else:
        print(" No detailed state.json for this iteration yet.")

if __name__ == "__main__":
    main()
