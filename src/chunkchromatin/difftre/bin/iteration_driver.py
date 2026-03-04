#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from diffTre.bin.difftre_fit_driver import main as fit_driver_main


def main() -> None:
    ap = argparse.ArgumentParser(description="Compatibility wrapper around difftre_fit_driver for a single iteration.")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--iter", required=True, type=int)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    # Reuse existing fit driver implementation to avoid behavior drift.
    import sys

    sys.argv = [
        str(Path(__file__).resolve()),
        "--run-root",
        args.run_root,
        "--iter",
        str(args.iter),
        "--config",
        args.config,
    ]
    fit_driver_main()


if __name__ == "__main__":
    main()
