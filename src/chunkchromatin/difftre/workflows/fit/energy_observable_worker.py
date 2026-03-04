import argparse
from pathlib import Path
import json
import numpy as np
import jax.numpy as jnp

from diffTre.bin.io_utils import load_all_replicates_jax
from diffTre.bin.contact_map import calculate_trajectory_chunk_size
from diffTre.bin.observables_contactmap import compute_observables_per_frame
from diffTre.bin.jax_U_calc_mm_forces import build_params_static_mm_from_inputs, compute_energies_chunk
from diffTre.bin.difftre_pipeline import DiffTREPipeline
from resource_profiler import profile_job

# === FUNCTIONS ===
def load_loop_pairs(looplist_path: Path, one_indexed: bool = True) -> np.ndarray:
    if not looplist_path.exists():
        return np.zeros((0, 2), dtype=np.int32)
    pairs = []
    for line in looplist_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        i = int(parts[0])
        j = int(parts[1])
        if one_indexed:
            i -= 1
            j -= 1
        pairs.append([i, j])
    return np.array(pairs, dtype=np.int32)

# ================================================
def main():
    # === ARGUMENT PARSING ===
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to simulation configuration JSON file")
    parser.add_argument("--run-root", required=True, help="Root directory for simulation run")
    parser.add_argument("--iter", required=True, type=int, help="Iteration number")
    args = parser.parse_args()

    # === CONFIGURATION ===
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    slurm_cfg_path = Path(cfg.get("slurm_profile", {}))
    slurm_cfg = json.loads(slurm_cfg_path.read_text())

    monomer_types_path = Path(cfg["monomer_types"]["types_path"])
    monomer_types = np.load(monomer_types_path, allow_pickle=True)

    loop_pairs = load_loop_pairs(Path(cfg["loops"]["looplist_path"]), one_indexed=True)
    # ================================================

    # === CALCULATE TRAJECTORYCHUNK SIZE ===
    n_particles = int(cfg["simulation"]["n_particles"])
    n_frames_per_traj = int(cfg["simulation"]["prod_steps"] / cfg["simulation"]["save_every"])
    available_memory_gb = float(slurm_cfg['processing']['mem'].rstrip('G'))

    #calculate chunk size
    trajectory_chunk_size = calculate_trajectory_chunk_size(
        n_particles=n_particles,
        n_frames_per_traj=n_frames_per_traj,
        dtype_bytes=8,  # float64
        available_memory_gb=available_memory_gb,
        headroom_frac=0.5,
        max_chunk=None,
    )

    # ================================================

    # === CALCULATE ENERGIES AND OBSERVABLES ===
    # Start resource profiling
    run_root = Path(args.run_root)
    iter_num = args.iter
    
    with profile_job("observables", run_root, iter_num):
        iter_dir = run_root / f"iter_{args.iter:03d}"
        params_dir = iter_dir / "params"
        params = {
            "interaction_matrix": np.load(params_dir / "interaction_matrix.npy"),
            "loop_X": float(np.load(params_dir / "loop_X.npy").reshape(-1)[0]),
            "lambda_IC": np.load(params_dir / "lambda_IC.npy"),
        }
        # Use DiffTREPipeline to build force_kwargs (handles config parsing nicely)
        pipeline = DiffTREPipeline("", str(config_path))  # Empty glob, we'll use it just for config
        pipeline.load_monomer_types()
        force_kwargs = pipeline.build_force_kwargs(params)

        traj_paths = sorted((iter_dir / "sims").glob("rep*/trajectory.traj"))

        tkl_frames_list = []
        phi_frames_list = []
        loop_frames_list = []
        energies_list = []

        # Process trajectories in chunks - load, compute, write, then move to next chunk
        for start_idx in range(0, len(traj_paths), trajectory_chunk_size):
            end_idx = min(start_idx + trajectory_chunk_size, len(traj_paths))
            chunk_traj_paths = traj_paths[start_idx:end_idx]
            
            print(f"Processing trajectory chunk {start_idx}-{end_idx} ({len(chunk_traj_paths)} files)...")
            
            # Load this chunk of trajectories into memory
            positions_list = load_all_replicates_jax(chunk_traj_paths, discard_initial=1)
            
            # Concatenate all replicates in this chunk for processing
            positions = jnp.concatenate(positions_list, axis=0)  # Shape: (total_frames_in_chunk, n_particles, 3)

            # Build params and static for energy computation
            params_mm, static_mm = build_params_static_mm_from_inputs(
                monomer_types=monomer_types,
                interaction_matrix=params["interaction_matrix"],
                chains=[tuple(c) for c in cfg["simulation"]["chains"]],
                loop_pairs=loop_pairs,
                force_kwargs=force_kwargs,
                N=n_particles,
            )

            energies_chunk = compute_energies_chunk(
                positions=positions,
                params_mm=params_mm,
                static_mm=static_mm,
                temperature=float(cfg["simulation"]["temperature"]),
            )
            energies_list.append(energies_chunk["total"])

            obs_kwargs = {
                "positions": positions,
                "monomer_types": monomer_types,
                "loop_pairs": loop_pairs,
                "chains": cfg["simulation"]["chains"],
                "mu": float(cfg["distance_kernel"]["mu"]),
                "rc": float(cfg["distance_kernel"]["rc"]),
                "use_fp32": False,
                "available_memory_gb": available_memory_gb,
                "headroom_frac": 0.7,  # Slightly more headroom for inner chunking
            }

            obs_kwargs["d_init"] = int(cfg["learned_forces"]["ideal_chromosome"]["d_init"])
            obs_kwargs["d_end"] = int(cfg["learned_forces"]["ideal_chromosome"]["d_end"])

            obs_chunk = compute_observables_per_frame(**obs_kwargs)

            tkl_frames_list.append(obs_chunk["tkl_frames"])
            phi_frames_list.append(obs_chunk["phi_frames"])
            loop_frames_list.append(obs_chunk["loop_frames"])

            del positions, positions_list, energies_chunk

        tkl_frames_jax = jnp.concatenate(tkl_frames_list, axis=0)
        phi_frames_jax = jnp.concatenate(phi_frames_list, axis=0)
        loop_frames_jax = jnp.concatenate(loop_frames_list, axis=0)
        energies_jax = jnp.concatenate(energies_list, axis=0)

        # Convert to numpy for saving
        tkl_frames = np.asarray(tkl_frames_jax)
        phi_frames = np.asarray(phi_frames_jax)
        loop_frames = np.asarray(loop_frames_jax)
        energies = np.asarray(energies_jax)

        # === SAVE OBSERVABLES AND ENERGIES===
        obs_dir = iter_dir / "observables"
        obs_dir.mkdir(parents=True, exist_ok=True)
        np.save(obs_dir / "tkl_frames.npy", tkl_frames)
        np.save(obs_dir / "phi_frames.npy", phi_frames)
        np.save(obs_dir / "loop_frames.npy", loop_frames)

        np.savetxt(obs_dir / "tkl_frames.txt", tkl_frames.reshape(tkl_frames.shape[0], -1))
        np.savetxt(obs_dir / "phi_frames.txt", phi_frames)
        np.savetxt(obs_dir / "loop_frames.txt", loop_frames)

        np.save(obs_dir / "energies.npy", energies)
        np.savetxt(obs_dir / "energies.txt", energies)

if __name__ == "__main__":
    main()



