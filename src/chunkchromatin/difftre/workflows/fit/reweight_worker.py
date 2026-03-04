import argparse
from pathlib import Path
import json
import os
import sys
import subprocess
import numpy as np
import jax.numpy as jnp

from diffTre.bin.difftre_pipeline import DiffTREPipeline
from diffTre.bin.reweight import compute_weights, effective_sample_size
from diffTre.bin.update import tkl_newton_step, ic_newton_step_fullphi
from diffTre.tests.workflows.common import ensure_dir, write_json

from diffTre.bin.io_utils import load_all_replicates_jax
from diffTre.bin.contact_map import calculate_trajectory_chunk_size
from diffTre.bin.jax_U_calc_mm_forces import build_params_static_mm_from_inputs, compute_energies_chunk
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

def _flatten_upper(mat: np.ndarray) -> np.ndarray:
    """Flatten upper triangular matrix (including diagonal)."""
    iu = np.triu_indices(mat.shape[0])
    return mat[iu]

def weighted_covariance(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute weighted covariance matrix."""
    w = np.asarray(w, dtype=float)
    w = w / np.sum(w)
    mean = np.sum(x * w[:, None], axis=0)
    xc = x - mean
    cov = (xc.T * w) @ xc
    return cov

def _load_params(iter_dir: Path) -> dict:
    """Load parameters from iteration directory."""
    params_dir = iter_dir / "params"
    params = {
        "interaction_matrix": np.load(params_dir / "interaction_matrix.npy"),
        "loop_X": float(np.load(params_dir / "loop_X.npy").reshape(-1)[0]),
    }
    # Load lambda_IC
    lambda_ic_path = params_dir / "lambda_IC.npy"
    params["lambda_IC"] = np.load(lambda_ic_path)

    return params

def _save_params(iter_dir: Path, params: dict, d_init: int, d_end: int) -> None:
    """Save parameters to iteration directory."""
    params_dir = iter_dir / "params"
    ensure_dir(params_dir)
    np.save(params_dir / "interaction_matrix.npy", params["interaction_matrix"])
    np.save(params_dir / "loop_X.npy", np.array(params["loop_X"]))
    np.save(params_dir / "lambda_IC.npy", params["lambda_IC"])



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

    run_root = Path(args.run_root).resolve()
    it = int(args.iter)
    it_dir = run_root / f"iter_{it:03d}"
    obs_dir = it_dir / "observables"
    update_dir = it_dir / "update"
    params_dir = it_dir / "params"
    logs_dir = ensure_dir(run_root / "logs")
    ensure_dir(update_dir)

    # Get observable flags from config
    fit_cfg = cfg["fit"]
    fit_tkl = fit_cfg.get("fit_tkl")
    fit_ic = fit_cfg.get("fit_ic")
    fit_loop = fit_cfg.get("fit_loop")
    
    # Get d_init and d_end from observables section
    d_init = int(cfg["learned_forces"]["ideal_chromosome"]["d_init"])
    d_end = int(cfg["learned_forces"]["ideal_chromosome"]["d_end"])

    # Load current parameters
    params = _load_params(it_dir)

    # Load reference targets
    ref_dir = Path(cfg["reference"]["targets_dir"]).resolve()
    tkl_target = None
    phi_target = None
    loop_target = None
    
    tkl_target = np.load(ref_dir / "T_type_kl.npy")
    phi_target = np.load(ref_dir / "phi_exp_IC.npy")
    loop_target = float(np.load(ref_dir / "loop_target.npy").reshape(-1)[0])

    monomer_types = np.load(Path(cfg["monomer_types"]["types_path"]), allow_pickle=True)
    loop_pairs = load_loop_pairs(Path(cfg["loops"]["looplist_path"]), one_indexed=True)

    # chunk size: mirror energy_observable_worker.py logic
    slurm_cfg_path = Path(cfg.get("slurm_profile", {}))
    slurm_cfg = json.loads(slurm_cfg_path.read_text())

    n_particles = int(cfg["simulation"]["n_particles"])
    n_frames_per_traj = int(cfg["simulation"]["prod_steps"] / cfg["simulation"]["save_every"])

    available_memory_gb = float(slurm_cfg["reweight"]["mem"].rstrip("G"))

    trajectory_chunk_size = calculate_trajectory_chunk_size(
        n_particles=n_particles,
        n_frames_per_traj=n_frames_per_traj,
        dtype_bytes=8,          # float64 positions
        available_memory_gb=available_memory_gb,
        headroom_frac=0.5,
        max_chunk=None,
    )

    # precompute traj ordering once so energies + observables stay aligned
    traj_paths = sorted((it_dir / "sims").glob("rep*/trajectory.traj"))

    # === HELPER FUNCTIONS ===
    def compute_current_energy_streaming(params: dict) -> np.ndarray:
        """Compute energies for *all* frames in the same order as obs_frames."""
        energies_list = []

        # build force_kwargs consistently (same as obs worker)
        pipeline_local = DiffTREPipeline("", str(config_path))
        pipeline_local.load_monomer_types()
        force_kwargs = pipeline_local.build_force_kwargs(params)

        for start_idx in range(0, len(traj_paths), trajectory_chunk_size):
            end_idx = min(start_idx + trajectory_chunk_size, len(traj_paths))
            chunk_traj_paths = traj_paths[start_idx:end_idx]

            positions_list = load_all_replicates_jax(chunk_traj_paths, discard_initial=1)
            positions = jnp.concatenate(positions_list, axis=0)  # (frames, N, 3)

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

            energies_list.append(np.asarray(energies_chunk["total"]))

            del positions, positions_list, energies_chunk

        return np.concatenate(energies_list, axis=0)

    # ================================================

    # Start resource profiling
    with profile_job("reweight", run_root, it):
        # Load pre-computed observables and energies from disk
        print("Loading pre-computed observables and energies...")
        tkl_frames = np.load(obs_dir / "tkl_frames.npy")
        phi_frames = np.load(obs_dir / "phi_frames.npy")
        loop_frames = np.load(obs_dir / "loop_frames.npy")
        
        # TODO: verify that energies are loaded in the same order as tkl_frames, phi_frames, and loop_frames.
        #you can do this by checking energy.npy and comparing elementwise against reference_energy

        obs_frames = {
            "tkl_frames": tkl_frames,
            "phi_frames": phi_frames,
            "loop_frames": loop_frames,
        }
        
        # Load reference energies - always compute both and validate
        print("Loading/computing reference energies...")
        pipeline = DiffTREPipeline("", str(config_path))  # Empty glob, just for config
        pipeline.load_monomer_types()
        
        traj_glob = str(it_dir / "sims" / "rep*" / "trajectory.traj")
        pipeline.data_glob_path = traj_glob
        
        # Load positions for energy computation
        # TODO: Stream positions in chunks if necessary to avoid memory issues.
        from diffTre.bin.io_utils import load_all_replicates
        positions_list = load_all_replicates(traj_glob, discard_initial=1)
        pipeline.positions_list = positions_list

        # TODO: verify that loop pairs are one-indexed.
        loop_pairs = load_loop_pairs(Path(cfg["loops"]["looplist_path"]), one_indexed=True)
        computed_ref_energy = pipeline.compute_current_energy(params, loop_pairs)
        
        # === COMPARE ENERGIES ===
        # Try to load from trajectory ENRG blocks
        traj_enrg_energy = None
        try:
            traj_enrg_energy = pipeline.load_reference_energies()
            print(f"Successfully loaded {len(traj_enrg_energy)} energies from trajectory ENRG blocks")
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Warning: Could not load reference energies from trajectory ENRG blocks: {e}")
            print("Will use computed energies as reference")
        
        # Validate energies match if both are available
        if traj_enrg_energy is not None:
            if len(traj_enrg_energy) != len(computed_ref_energy):
                raise ValueError(
                    f"Energy length mismatch: ENRG blocks={len(traj_enrg_energy)}, "
                    f"computed={len(computed_ref_energy)}"
                )
            
            # Compare energies with tolerance
            energy_tolerance = float(cfg["fit"].get("energy_tolerance", 1e-3))  # Default 1e-3 kJ/mol
            energy_diff = np.abs(traj_enrg_energy - computed_ref_energy)
            max_diff = np.max(energy_diff)
            mean_diff = np.mean(energy_diff)
            
            print(f"Energy comparison: max_diff={max_diff:.6f} kJ/mol, mean_diff={mean_diff:.6f} kJ/mol")
            
            if max_diff > energy_tolerance:
                np.save(update_dir / "ref_energy_traj_enrg.npy", traj_enrg_energy if traj_enrg_energy is not None else np.array([]))
                np.save(update_dir / "ref_energy_computed.npy", computed_ref_energy)
                error_msg = (
                    f"Energy mismatch exceeds tolerance ({energy_tolerance} kJ/mol):\n"
                    f"  max_diff={max_diff:.6f} kJ/mol\n"
                    f"  mean_diff={mean_diff:.6f} kJ/mol\n"
                    f"  This may indicate a problem with force field parameters or energy computation."
                )
                (update_dir / "energy_mismatch_warning.txt").write_text(error_msg + "\n")
                raise RuntimeError(f"ERROR: {error_msg}")
            else:
                print(f"✓ Energies match within tolerance ({energy_tolerance} kJ/mol)")
            
            # Use jax energy as reference
            ref_energy = computed_ref_energy
        else:
            # Trajectories should have ENRG blocks, so this should never happen.
            raise RuntimeError("ERROR: Could not load reference energies from trajectory ENRG blocks")
        

        # ================================================

        # Get reweighting configuration
        max_reweight_steps = cfg["fit"].get("max_reweight_steps")
        neff_threshold_frac = float(cfg["fit"].get("neff_threshold_frac"))
        max_resamples = int(cfg["fit"].get("max_resamples"))
        temperature = float(cfg["simulation"]["temperature"])
        kB_kJ_per_mol_K = 0.008314462618
        beta = 1.0 / (kB_kJ_per_mol_K * temperature) # which is 1

        # Load resample state
        state_path = run_root / "resample_state.json"
        if state_path.exists():
            res_state = json.loads(state_path.read_text())
        else:
            res_state = {"resamples_done": 0}

        # Reweighting loop
        rw = 0
        resample_requested = False
        
        while True:
            rw_log = logs_dir / f"iter_{it:03d}_rw{rw:02d}.log"
            
            # Compute current energies (already loaded)
            current_energy = compute_current_energy_streaming(params)
            
            # Compute weights
            delta_energy = current_energy - ref_energy
            weights = compute_weights(delta_energy, beta=beta)
            weights = np.asarray(weights)

            # Reweight observables
            tkl_weighted = np.sum(obs_frames["tkl_frames"] * weights[:, None, None], axis=0)
            phi_weighted = np.sum(obs_frames["phi_frames"] * weights[:, None], axis=0)
            loop_weighted = float(np.sum(obs_frames["loop_frames"] * weights))
            
            
            # Compute residuals
            residuals_list = []
            
            if fit_tkl:
                np.save(obs_dir / f"tkl_weighted_rw{rw:02d}.npy", tkl_weighted)
            tkl_weighted_flat = _flatten_upper(tkl_weighted)
            tkl_resid = tkl_weighted_flat - tkl_target
            residuals_list.append(tkl_resid)
            tkl_mare = float(np.mean(np.abs(tkl_resid) / (np.abs(tkl_target) + 1e-10)))
            tkl_are = (np.abs(tkl_resid) / (np.abs(tkl_target) + 1e-10)).tolist()
            tkl_sare = float(np.sum(np.abs(tkl_resid) / (np.abs(tkl_target) + 1e-10)))

            
            if fit_ic:
                np.save(obs_dir / f"phi_weighted_rw{rw:02d}.npy", phi_weighted)
            phi_resid = phi_weighted - phi_target
            residuals_list.append(phi_resid)
            phi_mare = float(np.mean(np.abs(phi_resid) / (np.abs(phi_target) + 1e-10)))
            phi_are = (np.abs(phi_resid) / (np.abs(phi_target) + 1e-10)).tolist()
            phi_sare = float(np.sum(np.abs(phi_resid) / (np.abs(phi_target) + 1e-10)))

            
            if fit_loop:
                np.save(obs_dir / f"loop_weighted_rw{rw:02d}.npy", np.array(loop_weighted))
            loop_resid = loop_weighted - loop_target
            residuals_list.append(np.array([loop_resid]))
            loop_mare = float(np.mean(np.abs(loop_resid) / (np.abs(loop_target) + 1e-10)))
            loop_are = float(np.abs(loop_resid) / (np.abs(loop_target) + 1e-10))
            loop_sare = float(np.sum(np.abs(loop_resid) / (np.abs(loop_target) + 1e-10)))
            
            np.save(obs_dir / f"weights_rw{rw:02d}.npy", weights)

            # Compute statistics
            neff = float(effective_sample_size(weights))
            loss = float(np.mean(np.concatenate(residuals_list) ** 2)) if residuals_list else 0.0
            
            # Build state dict
            state = {
                "iteration": it,
                "reweight_step": rw,
                "loss": loss,
                "params": {
                    "loop_X": float(params["loop_X"]),
                    "lambda_ic_l2": float(np.linalg.norm(params["lambda_IC"])) if "lambda_IC" in params else 0.0,
                    "lambda_ic_mean": float(np.mean(params["lambda_IC"])) if "lambda_IC" in params and params["lambda_IC"].size else 0.0,
                    "interaction_matrix": params["interaction_matrix"].tolist() if "interaction_matrix" in params else None,
                },
                "residuals": {},
                "weights": {
                    "min": float(np.min(weights)),
                    "max": float(np.max(weights)),
                    "neff": neff,
                },
            }
            
            #store statistics for all observables regardless of what is being fit
            state["residuals"]["tkl_l2"] = float(np.linalg.norm(tkl_resid))
            state["residuals"]["tkl_are"] = tkl_are
            state["residuals"]["tkl_sare"] = tkl_sare
            state["residuals"]["tkl_mare"] = tkl_mare

            state["residuals"]["phi_l2"] = float(np.linalg.norm(phi_resid))
            state["residuals"]["phi_are"] = phi_are
            state["residuals"]["phi_sare"] = phi_sare
            state["residuals"]["phi_mare"] = phi_mare

            state["residuals"]["loop_abs"] = abs(loop_resid)
            state["residuals"]["loop_mare"] = loop_mare
            state["residuals"]["loop_are"] = loop_are
            state["residuals"]["loop_sare"] = loop_sare
            state["loop"] = {
                "loop_X": float(params["loop_X"]),
                "loop_mean_weighted": loop_weighted,
                "loop_target": loop_target,
            }

            # Save state
            (update_dir / f"state_rw{rw:02d}.json").write_text(json.dumps(state, indent=2))
            rw_log.write_text(json.dumps(state, indent=2) + "\n")

            # Update parameters
            if fit_tkl and cfg["fit"].get("train_tkl", True):
                damp_tkl = float(cfg["fit"].get("damp_tkl"))
                tkl_step_bounds = cfg["fit"].get("tkl_step_bounds")
                tkl_frames_flat = obs_frames["tkl_frames"]
                iu = np.triu_indices(tkl_frames_flat.shape[1])
                tkl_vecs = tkl_frames_flat[:, iu[0], iu[1]]
                pi_pj_mean = weighted_covariance(tkl_vecs, weights) + np.outer(tkl_weighted_flat, tkl_weighted_flat)
                new_mat, tkl_step = tkl_newton_step(
                    params["interaction_matrix"],
                    tkl_weighted_flat,
                    tkl_target,
                    pi_pj_mean,
                    damp=damp_tkl,
                    step_bounds=tkl_step_bounds,
                )
                params["interaction_matrix"] = np.asarray(new_mat)
                np.save(update_dir / f"tkl_update_rw{rw:02d}.npy", tkl_step)
                np.save(params_dir / f"interaction_matrix_reweight_rw{rw:02d}.npy", params["interaction_matrix"])

            if fit_ic and cfg["fit"].get("train_ic", True):
                if not bool(cfg["fit"].get("ic_use_full_phi", False)):
                    raise RuntimeError("IC update requires full-phi (fit.ic_use_full_phi=true).")
                damp_ic = float(cfg["fit"].get("damp_ic", 3e-7))
                ic_step_bounds = cfg["fit"].get("ic_step_bounds", None)
                phi_sim = phi_weighted
                phi_exp = phi_target
                pi_pj_mean = weighted_covariance(obs_frames["phi_frames"], weights) + np.outer(phi_sim, phi_sim)
                lambda_next, ic_step = ic_newton_step_fullphi(
                    params["lambda_IC"],
                    phi_sim,
                    phi_exp,
                    pi_pj_mean,
                    damp=damp_ic,
                    step_bounds=ic_step_bounds,
                )
                params["lambda_IC"] = np.asarray(lambda_next)
                np.save(update_dir / f"lambda_ic_update_rw{rw:02d}.npy", ic_step)
                np.save(params_dir / f"lambda_ic_reweight_rw{rw:02d}.npy", params["lambda_IC"])

            if fit_loop and cfg["fit"].get("train_loop", True):
                from diffTre.bin.update import loop_gradient_step
                eta = float(cfg["fit"].get("eta", 0.01))
                max_step = cfg["fit"].get("max_step", None)
                loop_next = loop_gradient_step(params["loop_X"], loop_weighted, loop_target, eta=eta, max_step=max_step)
                params["loop_X"] = float(loop_next)
                np.save(update_dir / f"loop_X_tk_{it+1}_rw{rw:02d}.npy", np.array(params["loop_X"]))
                np.save(params_dir / f"loop_X_reweight_rw{rw:02d}.npy", params["loop_X"])

            # Check resampling condition
            neff_threshold = neff_threshold_frac * weights.shape[0]
            print(f"Reweighting step {rw}: neff={neff:.2f}, threshold={neff_threshold:.2f} (neff_threshold_frac={neff_threshold_frac}, total_frames={weights.shape[0]})")
            
            if neff < neff_threshold:
                print(f"  -> neff ({neff:.2f}) < threshold ({neff_threshold:.2f}), requesting resampling")
                resample_requested = True
                res_state["resamples_done"] += 1
                state["resample_requested"] = True
                state["neff_threshold"] = float(neff_threshold)
                state["neff_threshold_frac"] = float(neff_threshold_frac)
                state["total_frames"] = int(weights.shape[0])
                (update_dir / f"state_rw{rw:02d}.json").write_text(json.dumps(state, indent=2))
                rw_log.write_text(json.dumps(state, indent=2) + "\n")
                
                if res_state["resamples_done"] > max_resamples:
                    (update_dir / "stop_reason.txt").write_text(
                        f"Stopped: resamples_done={res_state['resamples_done']} exceeded max_resamples={max_resamples}\n"
                    )
                    return
                break

            rw += 1
            if max_reweight_steps is not None and rw >= int(max_reweight_steps):
                (update_dir / "stop_reason.txt").write_text(
                    f"Stopped: max_reweight_steps={max_reweight_steps} reached without resample trigger\n"
                )
                break

        # Save resample state
        state_path.write_text(json.dumps(res_state, indent=2))

        # Save updated parameters
        _save_params(it_dir, params, d_init=d_init, d_end=d_end)

        # Update manifest: mark reweight as completed
        manifest_path = run_root / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            iter_key = f"iter_{it:03d}"
            if "iterations" not in manifest:
                manifest["iterations"] = {}
            if iter_key not in manifest["iterations"]:
                manifest["iterations"][iter_key] = {}
            if "reweight" not in manifest["iterations"][iter_key]:
                manifest["iterations"][iter_key]["reweight"] = {}
            
            manifest["iterations"][iter_key]["reweight"]["state"] = "completed"
            manifest["iterations"][iter_key]["reweight"]["resample_requested"] = resample_requested
            manifest_path.write_text(json.dumps(manifest, indent=2))

        # Check if we should continue to next iteration
        next_iter = it + 1
        n_iters = int(cfg["fit"]["n_iters"])
        if next_iter >= n_iters:
            print(f"Fit workflow completed: reached maximum iterations ({n_iters})")
            return

        # Prepare next iteration directory and copy parameters
        next_dir = run_root / f"iter_{next_iter:03d}"
        ensure_dir(next_dir / "sims")
        ensure_dir(next_dir / "observables")
        ensure_dir(next_dir / "update")
        _save_params(next_dir, params, d_init=d_init, d_end=d_end)

        print(f"Reweighting complete. Updated parameters saved to {it_dir}/params/")
        print(f"Next iteration ({next_iter}) parameters initialized in {next_dir}/params/")
        
        if resample_requested:
            print(f"Resampling requested for iteration {it}. Submitting next iteration...")
            
            # Update manifest: initialize next iteration steps as pending
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                next_iter_key = f"iter_{next_iter:03d}"
                if "iterations" not in manifest:
                    manifest["iterations"] = {}
                if next_iter_key not in manifest["iterations"]:
                    manifest["iterations"][next_iter_key] = {}
                
                manifest["iterations"][next_iter_key]["sim"] = {"state": "pending"}
                manifest["iterations"][next_iter_key]["obs"] = {"state": "pending"}
                manifest["iterations"][next_iter_key]["reweight"] = {"state": "pending"}
                manifest_path.write_text(json.dumps(manifest, indent=2))
            
            # Submit next iteration's simulation job
            script_dir = Path(__file__).parent
            submit_sim_script = script_dir / "submit_simulations.py"
            
            current_job_id = os.environ.get("SLURM_JOB_ID", "")
            dep_str = f"afterok:{current_job_id}" if current_job_id else None
            
            cmd = [
                "python", str(submit_sim_script),
                "--config", str(config_path),
                "--run-root", str(run_root),
                "--iter", str(next_iter),
            ]
            if dep_str:
                cmd.extend(["--dependency", dep_str])
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Successfully submitted next iteration ({next_iter}) simulation job")
                # Parse job_id from output and update manifest
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if "Submitted simulation job:" in line:
                        sim_job_id = line.split("job:")[-1].strip()
                        if manifest_path.exists():
                            manifest = json.loads(manifest_path.read_text())
                            next_iter_key = f"iter_{next_iter:03d}"
                            if "sim" not in manifest["iterations"][next_iter_key]:
                                manifest["iterations"][next_iter_key]["sim"] = {}
                            manifest["iterations"][next_iter_key]["sim"]["job_id"] = sim_job_id
                            manifest_path.write_text(json.dumps(manifest, indent=2))
                        break
            else:
                print(f"ERROR: Failed to submit next iteration: {result.stderr}")
        else:
            print(f"Fit workflow completed: no resampling requested for iteration {it}")

if __name__ == "__main__":
    main()