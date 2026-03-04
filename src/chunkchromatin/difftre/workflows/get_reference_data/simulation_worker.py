#!/usr/bin/env python3
"""
This script runs a single replicate of a flexible fit simulation for a given iteration,
intended to be run via a SLURM job array using SLURM_ARRAY_TASK_ID.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import psutil

from chunkchromatin.binaryReporter import BinaryReporter
from chunkchromatin.initialization import create_multi_constrained_random_walk
from chunkchromatin.simulation import EKExceedsError, Simulation

from diffTre.bin.chromosome_michrom import ChromosomeMichroM

# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to simulation configuration JSON file")
parser.add_argument("--run-root", required=True, help="Root directory for simulation run")
parser.add_argument("--iter", required=True, type=int, help="Iteration number")
parser.add_argument("--replicate", required=True, type=int, help="1-based replicate index")
args = parser.parse_args()

iter_num = args.iter
rep_idx = args.replicate
rep_name = f"rep{rep_idx:02d}"

# === CONFIGURATION ===
config_path = Path(args.config)
if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

with open(config_path) as f:
    cfg = json.load(f)

# Get base seed for parameter generation (before parameter setup)
base_seed = int(cfg.get("random_seed", 0))

run_root = Path(args.run_root).resolve()
iter_dir = run_root / f"iter_{iter_num:03d}"
rep_dir = iter_dir / "sims" / rep_name
rep_dir.mkdir(parents=True, exist_ok=True)

log_path = run_root / "logs" / f"iter_{iter_num:03d}_rep{rep_idx:02d}.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

# Load monomer types
types_path = Path(cfg["monomer_types"]["types_path"]).resolve()
if not types_path.exists():
    raise FileNotFoundError(f"Monomer types not found at {types_path}")
monomer_types = np.load(types_path, allow_pickle=True)

# Simulation configuration
sim_cfg = cfg["simulation"]
n_particles = int(sim_cfg["n_particles"])
chains = [tuple(c) for c in sim_cfg["chains"]]
density = float(sim_cfg["density"])
temperature = float(sim_cfg["temperature"])
gamma = float(sim_cfg["gamma"])
timestep_fs = float(sim_cfg["timestep_fs"])
save_eq_frames = bool(sim_cfg["save_equilibration_frames"])
eq1_save_every = int(sim_cfg["eq_phase1_save_every"])
eq2_save_every = int(sim_cfg["eq_phase2_save_every"])

# Resolve platform from run config and slurm profile
slurm_cfg_path = Path(cfg.get("slurm_profile", {}))
slurm_cfg = json.loads(slurm_cfg_path.read_text())
sim_slurm = slurm_cfg.get("sim", slurm_cfg)
gres = str(sim_slurm.get("gres") or "").lower()
partition = str(sim_slurm.get("partition") or "").lower()
slurm_requests_gpu = "gpu" in gres or "gpu" in partition
platform = str(sim_slurm.get("platform"))


# Homopolymer-term Force configuration
hp_term_cfg = cfg["homopolymer_term_forces"]
mu = float(hp_term_cfg["distance_kernel"]["mu"])
rc = float(hp_term_cfg["distance_kernel"]["rc"])

# === LEARNED PARAMETER SETUP ===
# Load learned parameters from iteration directory
params_dir = iter_dir / "params"
if not params_dir.exists():
    raise FileNotFoundError(f"Parameters directory not found at {params_dir}")

interaction_matrix = np.load(params_dir / "interaction_matrix.npy")
loop_x = float(np.load(params_dir / "loop_X.npy").reshape(-1)[0])
lambda_IC = np.load(params_dir / "initial_lambda_IC.npy")

# === SEED & LOGGING ===
seed = base_seed + iter_num * 1000 + rep_idx
np.random.seed(seed)

# === METADATA ===
metadata = {
    "replicate": rep_name,
    "iteration": iter_num,
    "random_seed": int(seed),
    "temperature": temperature,
    "gamma": gamma,
    "timestep_fs": timestep_fs,
    "n_particles": n_particles,
    "platform": platform,
    "chains": chains,
    "density": density,
}

traj_path = rep_dir / "trajectory.traj"
start_time = time.time()

# === LOGGING SETUP ===
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


@contextlib.contextmanager
def tee_console(log_path: Path):
    with log_path.open("w") as f:
        out = _Tee(sys.stdout, f)
        err = _Tee(sys.stderr, f)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            yield

# === INITIALIZATION ===
with tee_console(log_path):
    print(f"Using OpenMM platform: {platform}")
    print(f"Iteration: {iter_num}, Replicate: {rep_name}, Seed: {seed}")
    

    # Retry initialization up to 50 times if bead placement fails
    max_init_retries = 50
    positions = None
    for init_attempt in range(1, max_init_retries + 1):
        try:
            positions = create_multi_constrained_random_walk(
                chains=chains,
                density=density,
            )
            break
        except RuntimeError as e:
            if "Failed to place bead" in str(e) and init_attempt < max_init_retries:
                print(f"Init attempt {init_attempt}/{max_init_retries} failed: {e}")
                continue
            raise

    if positions is None:
        raise RuntimeError(f"Failed to initialize after {max_init_retries} attempts")

    # === EQUILIBRATION PHASE 1 ===
    #fene, hardcore, angle, softcore, rg bias
    print("Starting equilibration phase 1...")
    eq1_traj_path = (rep_dir / "trajectory_eq1.traj") if save_eq_frames else (rep_dir / "trajectory_eq1.tmp.traj")
    eq1_metadata = {**metadata, "phase": "eq1"}
    
    try:
        with BinaryReporter(filename=str(eq1_traj_path), n_particles=n_particles, mode="w", metadata=eq1_metadata) as reporter:
            sim_eq1 = Simulation(
                N=n_particles,
                temperature=temperature,
                gamma=gamma,
                timestep=timestep_fs,
                platform=platform,
                reporter=reporter,
            )
            chrom = ChromosomeMichroM(N=n_particles, chains=chains, sim_object=sim_eq1)
            
            # Build forces

            #FENE: contains hc repulsion
            fene = chrom.add_fene_bonds(kFb=float(hp_term_cfg["fene_bonds"]["kFb"]))
            sim_eq1.add_force(fene, name="fene_bonds")
            print("✓ Added fene_bonds force")

            #ANGLE
            angles = chrom.add_angles(kA=float(hp_term_cfg["angles"]["kA"]))
            sim_eq1.add_force(angles, name="angles")
            print("✓ Added angles force")
            
            #HARDCORE not obviously in michrom codebase

            #SOFTCORE
            rep = chrom.add_repulsive_softcore(
                sim_object=sim_eq1,
                eCut=float(hp_term_cfg["repulsive_softcore"]["eCut"]),
                cutoffDistance=float(hp_term_cfg["repulsive_softcore"]["cutoffDistance"]),
            )
            sim_eq1.add_force(rep, name="repulsive_softcore")
            print("✓ Added repulsive_softcore force")
            
            #RG BIAS not obviously in michrom codebase so will skip

            sim_eq1.set_positions(positions)
            sim_eq1.create_context()
            sim_eq1.set_velocities()
            
            eq1_steps = int(sim_cfg["eq_steps_phase1"])
            if eq1_steps <= 0:
                positions = np.asarray(sim_eq1.get_positions())
            elif not save_eq_frames:
                sim_eq1.run_simulation_block(eq1_steps, save=False)
                positions = np.asarray(sim_eq1.get_positions())
            else:
                n_blocks, rem_steps = divmod(eq1_steps, eq1_save_every)
                for _ in range(n_blocks):
                    sim_eq1.run_simulation_block(eq1_save_every, save=True, get_energies=False)
                if rem_steps > 0:
                    sim_eq1.run_simulation_block(rem_steps, save=True, get_energies=False)
                positions = np.asarray(sim_eq1.get_positions())
        
        # === EQUILIBRATION PHASE 2 ===
        # FENE, hardcore, angle, softcore, spherical confinement
        print("Starting equilibration phase 2...")
        eq2_traj_path = (rep_dir / "trajectory_eq2.traj") if save_eq_frames else (rep_dir / "trajectory_eq2.tmp.traj")
        eq2_metadata = {**metadata, "phase": "eq2"}
        
        with BinaryReporter(filename=str(eq2_traj_path), n_particles=n_particles, mode="w", metadata=eq2_metadata) as reporter:
            sim_eq2 = Simulation(
                N=n_particles,
                temperature=temperature,
                gamma=gamma,
                timestep=timestep_fs,
                platform=platform,
                reporter=reporter,
            )
            chrom = ChromosomeMichroM(N=n_particles, chains=chains, sim_object=sim_eq2)
            
            # Build forces (same as phase 1)
            #FENE
            fene = chrom.add_fene_bonds(kFb=float(hp_term_cfg["fene_bonds"]["kFb"]))
            sim_eq2.add_force(fene, name="fene_bonds")
            
            #ANGLE
            angles = chrom.add_angles(kA=float(hp_term_cfg["angles"]["kA"]))
            sim_eq2.add_force(angles, name="angles")
            
            #SOFTCORE
            rep = chrom.add_repulsive_softcore(
                sim_object=sim_eq2,
                eCut=float(hp_term_cfg["repulsive_softcore"]["eCut"]),
                cutoffDistance=float(hp_term_cfg["repulsive_softcore"]["cutoffDistance"]),
            )
            sim_eq2.add_force(rep, name="repulsive_softcore")
            
            #FLAT BOTTOM CONFINEMENT
            fb = chrom.add_flat_bottom_harmonic(
                sim_object=sim_eq2,
                kR=float(hp_term_cfg["flat_bottom_harmonic"]["kR"]),
                nRad=float(hp_term_cfg["flat_bottom_harmonic"]["nRad"]),
            )
            sim_eq2.add_force(fb, name="flat_bottom_harmonic")
            
            sim_eq2.set_positions(positions)
            sim_eq2.create_context()
            sim_eq2.set_velocities()
            sim_eq2.save_initial_state()
            
            eq2_steps = int(sim_cfg["eq_steps_phase2"])
            if eq2_steps <= 0:
                positions = np.asarray(sim_eq2.get_positions())
            elif not save_eq_frames:
                sim_eq2.run_simulation_block(eq2_steps, save=False)
                positions = np.asarray(sim_eq2.get_positions())
            else:
                n_blocks, rem_steps = divmod(eq2_steps, eq2_save_every)
                for _ in range(n_blocks):
                    sim_eq2.run_simulation_block(eq2_save_every, save=True, get_energies=False)
                if rem_steps > 0:
                    sim_eq2.run_simulation_block(rem_steps, save=True, get_energies=False)
                positions = np.asarray(sim_eq2.get_positions())
    
    except EKExceedsError:
        raise RuntimeError(f"Equilibration failed for {rep_dir}")
    finally:
        # Clean up temporary trajectory files if not saving equilibration frames
        if not save_eq_frames:
            for tmp in (rep_dir / "trajectory_eq1.tmp.traj", rep_dir / "trajectory_eq2.tmp.traj"):
                if tmp.exists():
                    tmp.unlink()

    # === PRODUCTION ===
    print("Starting production phase...")
    prod_metadata = {**metadata, "phase": "prod"}
    
    with BinaryReporter(filename=str(traj_path), n_particles=n_particles, mode="w", metadata=prod_metadata) as reporter:
        sim = Simulation(
            N=n_particles,
            temperature=temperature,
            gamma=gamma,
            timestep=timestep_fs,
            platform=platform,
            reporter=reporter,
        )
        chrom = ChromosomeMichroM(N=n_particles, chains=chains, sim_object=sim)
        
        # Add homopolymer term forces
        #FENE
        fene = chrom.add_fene_bonds(kFb=float(hp_term_cfg["fene_bonds"]["kFb"]))
        sim.add_force(fene, name="fene_bonds")
        
        #ANGLE
        angles = chrom.add_angles(kA=float(hp_term_cfg["angles"]["kA"]))
        sim.add_force(angles, name="angles")
        
        #SOFTCORE
        rep = chrom.add_repulsive_softcore(
            sim_object=sim,
            eCut=float(hp_term_cfg["repulsive_softcore"]["eCut"]),
            cutoffDistance=float(hp_term_cfg["repulsive_softcore"]["cutoffDistance"]),
        )
        sim.add_force(rep, name="repulsive_softcore")
        
        #FLAT BOTTOM CONFINEMENT
        fb = chrom.add_flat_bottom_harmonic(
            sim_object=sim,
            kR=float(hp_term_cfg["flat_bottom_harmonic"]["kR"]),
            nRad=float(hp_term_cfg["flat_bottom_harmonic"]["nRad"]),
        )
        sim.add_force(fb, name="flat_bottom_harmonic")
        
        ### LEARNED FORCES
        if cfg["fit"]["fit_tkl"]:
            t2t = chrom.add_type_to_type_michrom(
                sim_object=sim,
                interaction_matrix=interaction_matrix,
                monomer_types=monomer_types,
                mu=mu,
                rc=rc,
                rCutoff=float(cfg["learned_forces"]["type_to_type"]["rCutoff"]),
            )
            sim.add_force(t2t, name="type_to_type_michrom")
        
        if cfg["fit"]["fit_ic"]:
            ic = chrom.add_ideal_chromosome_force(
                sim_object=sim,
                lambda_IC=lambda_IC,
                d_init=int(cfg["learned_forces"]["ideal_chromosome"]["d_init"]),
                d_end=int(cfg["learned_forces"]["ideal_chromosome"]["d_end"]),
                mu=mu,
                rc=rc,
                rCutoff=float(cfg["learned_forces"]["ideal_chromosome"]["rCutoff"]),
            )
            # else:
            #     ic = chrom.add_ideal_chromosome_michrom(
            #         sim_object=sim,
            #         gamma1=params["gamma1"],
            #         gamma2=params["gamma2"],
            #         gamma3=params["gamma3"],
            #         d_init=int(f_cfg["ideal_chromosome"]["d_init"]),
            #         d_end=int(f_cfg["ideal_chromosome"]["d_end"]),
            #         mu=mu,
            #         rc=rc,
            #         rCutoff=float(f_cfg["ideal_chromosome"]["rCutoff"]),
            #     )
            sim.add_force(ic, name="ideal_chromosome_force")
        
        if cfg["fit"]["fit_loop"]:
            loops = chrom.add_loops_michrom(
                sim_object=sim,
                looplists=cfg["loops"]["looplist_path"],
                mu=mu,
                rc=rc,
                X=loop_x,
            )
            sim.add_force(loops, name="loops_michrom")
        
        sim.set_positions(positions)
        sim.create_context()
        sim.set_velocities()
        sim.save_initial_state()
        
        prod_steps = int(sim_cfg["prod_steps"])
        save_every = int(sim_cfg["save_every"])
        n_blocks, rem_steps = divmod(prod_steps, save_every)
        for _ in range(n_blocks):
            sim.run_simulation_block(save_every, save=True, get_energies=True)
        if rem_steps > 0:
            sim.run_simulation_block(rem_steps, save=True, get_energies=True)
    
    if save_eq_frames:
        shutil.copyfile(traj_path, rep_dir / "trajectory_prod.traj")

# === LOG STATS ===
runtime = time.time() - start_time
mem_usage_MB = psutil.Process().memory_info().rss / 1e6

summary = {
    "iteration": iter_num,
    "replicate": rep_name,
    "runtime_sec": runtime,
    "peak_memory_MB": mem_usage_MB,
    "random_seed": int(seed),
}

manifest_path = rep_dir / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump(summary, f, indent=2)
