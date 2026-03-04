#!/usr/bin/env python3
"""
Computes per-frame observables (TKL, phi/ideal-chromosome, loop) from all
reference simulation trajectories and saves both per-frame arrays and their
mean to run_root/observables/.

Usage:
    python reference_observable_calculation.py \
        --config /path/to/config.json \
        --run-root /path/to/run_root
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from diffTre.bin.io_utils import load_all_replicates_jax
from diffTre.bin.contact_map import calculate_trajectory_chunk_size
from diffTre.bin.observables_contactmap import compute_observables_per_frame


# ============================================================
# HELPERS
# ============================================================
def load_loop_pairs(looplist_path: Path, one_indexed: bool = True) -> np.ndarray:
    """Read a loop-list text file and return (N,2) int32 array of pair indices."""
    if not looplist_path.exists():
        return np.zeros((0, 2), dtype=np.int32)
    pairs = []
    for line in looplist_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        i, j = int(parts[0]), int(parts[1])
        if one_indexed:
            i -= 1
            j -= 1
        pairs.append([i, j])
    return np.array(pairs, dtype=np.int32)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Compute observables from reference simulation trajectories."
    )
    parser.add_argument("--config",   required=True, help="Path to config JSON")
    parser.add_argument("--run-root", required=True, help="Reference run root directory")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        cfg = json.load(f)

    run_root = Path(args.run_root).resolve()

    # --- SLURM profile (used for available_memory_gb) ---
    slurm_cfg_path = Path(cfg["slurm_profile"])
    slurm_cfg      = json.loads(slurm_cfg_path.read_text())

    # --- Monomer types & loop pairs ---
    monomer_types = np.load(Path(cfg["monomer_types"]["types_path"]), allow_pickle=True)

    loops_section = cfg.get("loops")
    if loops_section and cfg["reference_forces"].get("add_loop", False):
        loop_pairs = load_loop_pairs(Path(loops_section["looplist_path"]), one_indexed=True)
    else:
        loop_pairs = np.zeros((0, 2), dtype=np.int32)

    # --- Chunking parameters ---
    sim_cfg          = cfg["simulation"]
    n_particles      = int(sim_cfg["n_particles"])
    n_frames_per_rep = int(sim_cfg["prod_steps"]) // int(sim_cfg["save_every"])
    avail_mem_gb     = float(slurm_cfg["processing"]["mem"].rstrip("G"))

    traj_chunk_size = calculate_trajectory_chunk_size(
        n_particles=n_particles,
        n_frames_per_traj=n_frames_per_rep,
        dtype_bytes=8,           # float64
        available_memory_gb=avail_mem_gb,
        headroom_frac=0.5,
        max_chunk=None,
    )

    # --- Observable parameters from config ---
    dk_cfg  = cfg["distance_kernel"]
    ref_cfg = cfg["reference_forces"]
    ic_cfg  = ref_cfg["ideal_chromosome"]

    obs_base_kwargs = dict(
        monomer_types=monomer_types,
        loop_pairs=loop_pairs,
        chains=sim_cfg["chains"],
        mu=float(dk_cfg["mu"]),
        rc=float(dk_cfg["rc"]),
        d_init=int(ic_cfg["d_init"]),
        d_end=int(ic_cfg["d_end"]),
        use_fp32=False,
        available_memory_gb=avail_mem_gb,
        headroom_frac=0.7,
    )

    # --- Find trajectories ---
    traj_paths = sorted((run_root / "sims").glob("rep*/trajectory.traj"))
    if not traj_paths:
        raise FileNotFoundError(f"No trajectories found under {run_root / 'sims'}")
    print(f"Found {len(traj_paths)} trajectories.")

    # --- Process in chunks ---
    tkl_frames_list  = []
    phi_frames_list  = []
    loop_frames_list = []

    for start_idx in range(0, len(traj_paths), traj_chunk_size):
        end_idx   = min(start_idx + traj_chunk_size, len(traj_paths))
        chunk     = traj_paths[start_idx:end_idx]
        print(f"  Processing trajectories {start_idx}–{end_idx-1} ({len(chunk)} files)...")

        positions_list = load_all_replicates_jax(chunk, discard_initial=1)
        positions      = jnp.concatenate(positions_list, axis=0)  # (total_frames, N, 3)

        obs_chunk = compute_observables_per_frame(positions=positions, **obs_base_kwargs)

        tkl_frames_list.append(obs_chunk["tkl_frames"])
        phi_frames_list.append(obs_chunk["phi_frames"])
        loop_frames_list.append(obs_chunk["loop_frames"])

        del positions, positions_list

    # --- Concatenate and convert ---
    tkl_frames  = np.asarray(jnp.concatenate(tkl_frames_list,  axis=0))
    phi_frames  = np.asarray(jnp.concatenate(phi_frames_list,  axis=0))
    loop_frames = np.asarray(jnp.concatenate(loop_frames_list, axis=0))

    # --- Save ---
    obs_dir = run_root / "observables"
    obs_dir.mkdir(parents=True, exist_ok=True)

    # Per-frame arrays
    np.save(obs_dir / "tkl_frames.npy",  tkl_frames)
    np.save(obs_dir / "phi_frames.npy",  phi_frames)
    np.save(obs_dir / "loop_frames.npy", loop_frames)

    np.savetxt(obs_dir / "tkl_frames.txt",  tkl_frames.reshape(tkl_frames.shape[0], -1))
    np.savetxt(obs_dir / "phi_frames.txt",  phi_frames)
    np.savetxt(obs_dir / "loop_frames.txt", loop_frames)

    # Mean observables (used as targets in DiffTRE fitting)
    np.save(obs_dir / "tkl_mean.npy",  tkl_frames.mean(axis=0))
    np.save(obs_dir / "phi_mean.npy",  phi_frames.mean(axis=0))
    np.save(obs_dir / "loop_mean.npy", loop_frames.mean(axis=0))

    np.savetxt(obs_dir / "tkl_mean.txt",  tkl_frames.mean(axis=0).reshape(1, -1))
    np.savetxt(obs_dir / "phi_mean.txt",  phi_frames.mean(axis=0).reshape(1, -1))
    np.savetxt(obs_dir / "loop_mean.txt", loop_frames.mean(axis=0).reshape(1, -1))

    print(f"\nSaved observables to {obs_dir}")
    print(f"  tkl_frames  : {tkl_frames.shape}")
    print(f"  phi_frames  : {phi_frames.shape}")
    print(f"  loop_frames : {loop_frames.shape}")
    print(f"  tkl_mean    : {tkl_frames.mean(axis=0).shape}")


if __name__ == "__main__":
    main()
