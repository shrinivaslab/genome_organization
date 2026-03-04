#!/usr/bin/env python3
"""
Runs a single replicate of a reference simulation, intended to be run via a
SLURM job array using SLURM_ARRAY_TASK_ID.

Forces are read directly from the config (reference_forces). When add_ic is true, lambda_IC is loaded
from the numpy file path given in reference_forces.ideal_chromosome.lambda_IC_path.
"""
from __future__ import annotations

import argparse
import contextlib
import json
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
parser.add_argument("--config",    required=True, help="Path to simulation configuration JSON file")
parser.add_argument("--run-root",  required=True, help="Root directory for the reference run")
parser.add_argument("--replicate", required=True, type=int, help="1-based replicate index")
args = parser.parse_args()

rep_idx  = args.replicate
rep_name = f"rep{rep_idx:02d}"

# === CONFIGURATION ===
config_path = Path(args.config)
if not config_path.exists():
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

with open(config_path) as f:
    cfg = json.load(f)

base_seed = int(cfg.get("random_seed", 0))

run_root = Path(args.run_root).resolve()
rep_dir  = run_root / "sims" / rep_name
rep_dir.mkdir(parents=True, exist_ok=True)

log_path = run_root / "logs" / f"rep{rep_idx:02d}.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

# === MONOMER TYPES ===
types_path = Path(cfg["monomer_types"]["types_path"]).resolve()
if not types_path.exists():
    raise FileNotFoundError(f"Monomer types not found at {types_path}")
monomer_types = np.load(types_path, allow_pickle=True)
n_types = int(cfg["monomer_types"]["n_types"])

# === SIMULATION CONFIG ===
sim_cfg       = cfg["simulation"]
n_particles   = int(sim_cfg["n_particles"])
chains        = [tuple(c) for c in sim_cfg["chains"]]
density       = float(sim_cfg["density"])
temperature   = float(sim_cfg["temperature"])
gamma         = float(sim_cfg["gamma"])
timestep_fs   = float(sim_cfg["timestep_fs"])
save_eq_frames = bool(sim_cfg.get("save_equilibration_frames", False))
save_energies  = bool(sim_cfg.get("save_energies", True))

# === PLATFORM ===
slurm_cfg_path = Path(cfg["slurm_profile"])
slurm_cfg      = json.loads(slurm_cfg_path.read_text())
sim_slurm      = slurm_cfg.get("sim", slurm_cfg)
platform       = str(sim_slurm.get("platform", "CPU"))

# === HOMOPOLYMER TERM FORCES ===
hp_term_cfg = cfg["homopolymer_term_forces"]

# === DISTANCE KERNEL ===
dk_cfg = cfg["distance_kernel"]
mu = float(dk_cfg["mu"])
rc = float(dk_cfg["rc"])

# === REFERENCE FORCES ===
ref_cfg  = cfg["reference_forces"]
add_tkl  = bool(ref_cfg.get("add_tkl"))
add_ic   = bool(ref_cfg.get("add_ic"))
add_loop = bool(ref_cfg.get("add_loop"))

if add_tkl:
    raw_matrix = np.array(ref_cfg["type_to_type"]["interaction_matrix"], dtype=float)
    interaction_matrix = raw_matrix.reshape(n_types, n_types)

if add_ic:
    lambda_IC_path = Path(ref_cfg["ideal_chromosome"]["lambda_IC_path"])
    if not lambda_IC_path.exists():
        raise FileNotFoundError(f"lambda_IC numpy file not found: {lambda_IC_path}")
    lambda_IC = np.load(lambda_IC_path)

if add_loop:
    loop_x = float(ref_cfg["loop"]["X"])

# === SEED ===
seed = base_seed + rep_idx
np.random.seed(seed)

# === METADATA ===
metadata = {
    "replicate":    rep_name,
    "random_seed":  int(seed),
    "temperature":  temperature,
    "gamma":        gamma,
    "timestep_fs":  timestep_fs,
    "n_particles":  n_particles,
    "platform":     platform,
    "chains":       chains,
    "density":      density,
}

traj_path  = rep_dir / "trajectory.traj"
start_time = time.time()


# === LOGGING HELPERS ===
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


# ============================================================
# MAIN SIMULATION
# ============================================================
with tee_console(log_path):
    print(f"Platform:  {platform}")
    print(f"Replicate: {rep_name}  Seed: {seed}")
    print(f"add_tkl={add_tkl}  add_ic={add_ic}  add_loop={add_loop}")

    # --- Initialization ---
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
        raise RuntimeError(f"Failed to initialize positions after {max_init_retries} attempts")

    # -------------------------------------------------------
    # EQUILIBRATION PHASE 1  (FENE + angles + softcore)
    # -------------------------------------------------------
    print("Starting equilibration phase 1...")
    eq1_tmp   = not save_eq_frames
    eq1_tpath = (rep_dir / "trajectory_eq1.traj") if save_eq_frames \
                else (rep_dir / "trajectory_eq1.tmp.traj")

    try:
        with BinaryReporter(str(eq1_tpath), n_particles=n_particles,
                            mode="w", metadata={**metadata, "phase": "eq1"}) as rep1:
            sim_eq1 = Simulation(N=n_particles, temperature=temperature, gamma=gamma,
                                 timestep=timestep_fs, platform=platform, reporter=rep1)
            chrom = ChromosomeMichroM(N=n_particles, chains=chains, sim_object=sim_eq1)

            sim_eq1.add_force(chrom.add_fene_bonds(kFb=float(hp_term_cfg["fene_bonds"]["kFb"])),
                              name="fene_bonds")
            sim_eq1.add_force(chrom.add_angles(kA=float(hp_term_cfg["angles"]["kA"])),
                              name="angles")
            sim_eq1.add_force(chrom.add_repulsive_softcore(
                sim_object=sim_eq1,
                eCut=float(hp_term_cfg["repulsive_softcore"]["eCut"]),
                cutoffDistance=float(hp_term_cfg["repulsive_softcore"]["cutoffDistance"])),
                name="repulsive_softcore")
            print("✓ eq1 forces added")

            sim_eq1.set_positions(positions)
            sim_eq1.create_context()
            sim_eq1.set_velocities()

            eq1_steps = int(sim_cfg["eq_steps_phase1"])
            if eq1_steps <= 0:
                positions = np.asarray(sim_eq1.get_positions())
            elif eq1_tmp:
                sim_eq1.run_simulation_block(eq1_steps, save=False)
                positions = np.asarray(sim_eq1.get_positions())
            else:
                save_every_eq1 = int(sim_cfg.get("eq_phase1_save_every", eq1_steps))
                for _ in range(eq1_steps // save_every_eq1):
                    sim_eq1.run_simulation_block(save_every_eq1, save=True, get_energies=False)
                rem = eq1_steps % save_every_eq1
                if rem:
                    sim_eq1.run_simulation_block(rem, save=True, get_energies=False)
                positions = np.asarray(sim_eq1.get_positions())

        # -------------------------------------------------------
        # EQUILIBRATION PHASE 2  (+ flat-bottom confinement)
        # -------------------------------------------------------
        print("Starting equilibration phase 2...")
        eq2_tmp   = not save_eq_frames
        eq2_tpath = (rep_dir / "trajectory_eq2.traj") if save_eq_frames \
                    else (rep_dir / "trajectory_eq2.tmp.traj")

        with BinaryReporter(str(eq2_tpath), n_particles=n_particles,
                            mode="w", metadata={**metadata, "phase": "eq2"}) as rep2:
            sim_eq2 = Simulation(N=n_particles, temperature=temperature, gamma=gamma,
                                 timestep=timestep_fs, platform=platform, reporter=rep2)
            chrom = ChromosomeMichroM(N=n_particles, chains=chains, sim_object=sim_eq2)

            sim_eq2.add_force(chrom.add_fene_bonds(kFb=float(hp_term_cfg["fene_bonds"]["kFb"])),
                              name="fene_bonds")
            sim_eq2.add_force(chrom.add_angles(kA=float(hp_term_cfg["angles"]["kA"])),
                              name="angles")
            sim_eq2.add_force(chrom.add_repulsive_softcore(
                sim_object=sim_eq2,
                eCut=float(hp_term_cfg["repulsive_softcore"]["eCut"]),
                cutoffDistance=float(hp_term_cfg["repulsive_softcore"]["cutoffDistance"])),
                name="repulsive_softcore")
            sim_eq2.add_force(chrom.add_flat_bottom_harmonic(
                sim_object=sim_eq2,
                kR=float(hp_term_cfg["flat_bottom_harmonic"]["kR"]),
                nRad=float(hp_term_cfg["flat_bottom_harmonic"]["nRad"])),
                name="flat_bottom_harmonic")
            print("✓ eq2 forces added")

            sim_eq2.set_positions(positions)
            sim_eq2.create_context()
            sim_eq2.set_velocities()

            eq2_steps = int(sim_cfg["eq_steps_phase2"])
            if eq2_steps <= 0:
                positions = np.asarray(sim_eq2.get_positions())
            elif eq2_tmp:
                sim_eq2.run_simulation_block(eq2_steps, save=False)
                positions = np.asarray(sim_eq2.get_positions())
            else:
                save_every_eq2 = int(sim_cfg.get("eq_phase2_save_every", eq2_steps))
                for _ in range(eq2_steps // save_every_eq2):
                    sim_eq2.run_simulation_block(save_every_eq2, save=True, get_energies=False)
                rem = eq2_steps % save_every_eq2
                if rem:
                    sim_eq2.run_simulation_block(rem, save=True, get_energies=False)
                positions = np.asarray(sim_eq2.get_positions())

    except EKExceedsError:
        raise RuntimeError(f"Equilibration failed for {rep_dir}")
    finally:
        if not save_eq_frames:
            for tmp_path in (rep_dir / "trajectory_eq1.tmp.traj",
                             rep_dir / "trajectory_eq2.tmp.traj"):
                if tmp_path.exists():
                    tmp_path.unlink()

    # -------------------------------------------------------
    # PRODUCTION
    # -------------------------------------------------------
    print("Starting production phase...")
    with BinaryReporter(str(traj_path), n_particles=n_particles,
                        mode="w", metadata={**metadata, "phase": "prod"}) as rep_prod:
        sim = Simulation(N=n_particles, temperature=temperature, gamma=gamma,
                         timestep=timestep_fs, platform=platform, reporter=rep_prod)
        chrom = ChromosomeMichroM(N=n_particles, chains=chains, sim_object=sim)

        # --- homopolymer term forces ---
        sim.add_force(chrom.add_fene_bonds(kFb=float(hp_term_cfg["fene_bonds"]["kFb"])),
                      name="fene_bonds")
        sim.add_force(chrom.add_angles(kA=float(hp_term_cfg["angles"]["kA"])),
                      name="angles")
        sim.add_force(chrom.add_repulsive_softcore(
            sim_object=sim,
            eCut=float(hp_term_cfg["repulsive_softcore"]["eCut"]),
            cutoffDistance=float(hp_term_cfg["repulsive_softcore"]["cutoffDistance"])),
            name="repulsive_softcore")
        sim.add_force(chrom.add_flat_bottom_harmonic(
            sim_object=sim,
            kR=float(hp_term_cfg["flat_bottom_harmonic"]["kR"]),
            nRad=float(hp_term_cfg["flat_bottom_harmonic"]["nRad"])),
            name="flat_bottom_harmonic")
        print("✓ Homopolymer term forces added")

        # --- reference forces ---
        if add_tkl:
            t2t = chrom.add_type_to_type_michrom(
                sim_object=sim,
                interaction_matrix=interaction_matrix,
                monomer_types=monomer_types,
                mu=mu,
                rc=rc,
                rCutoff=float(ref_cfg["type_to_type"]["rCutoff"]),
            )
            sim.add_force(t2t, name="type_to_type_michrom")
            print("✓ Added type_to_type_michrom (TKL)")

        if add_ic:
            ic = chrom.add_ideal_chromosome_force(
                sim_object=sim,
                lambda_IC=lambda_IC,
                d_init=int(ref_cfg["ideal_chromosome"]["d_init"]),
                d_end=int(ref_cfg["ideal_chromosome"]["d_end"]),
                mu=mu,
                rc=rc,
                rCutoff=float(ref_cfg["ideal_chromosome"]["rCutoff"]),
            )
            sim.add_force(ic, name="ideal_chromosome_force")
            print("✓ Added ideal_chromosome_force (IC)")

        if add_loop:
            loops = chrom.add_loops_michrom(
                sim_object=sim,
                looplists=cfg["loops"]["looplist_path"],
                mu=mu,
                rc=rc,
                X=loop_x,
            )
            sim.add_force(loops, name="loops_michrom")
            print("✓ Added loops_michrom")

        sim.set_positions(positions)
        sim.create_context()
        sim.set_velocities()

        prod_steps = int(sim_cfg["prod_steps"])
        save_every = int(sim_cfg["save_every"])
        n_blocks, rem_steps = divmod(prod_steps, save_every)
        for _ in range(n_blocks):
            sim.run_simulation_block(save_every, save=True, get_energies=save_energies)
        if rem_steps > 0:
            sim.run_simulation_block(rem_steps, save=True, get_energies=save_energies)

    if save_eq_frames:
        shutil.copyfile(traj_path, rep_dir / "trajectory_prod.traj")

# === LOG STATS ===
runtime        = time.time() - start_time
mem_usage_MB   = psutil.Process().memory_info().rss / 1e6

summary = {
    "replicate":      rep_name,
    "runtime_sec":    runtime,
    "peak_memory_MB": mem_usage_MB,
    "random_seed":    int(seed),
}

with open(rep_dir / "manifest.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"Done. Runtime: {runtime:.1f}s  Memory: {mem_usage_MB:.0f} MB")
