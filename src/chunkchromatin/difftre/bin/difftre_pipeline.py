from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import jax.numpy as jnp

from .io_utils import load_all_replicates, read_traj_energy_block
from .observables_contactmap import compute_observables_per_frame
from .reweight import compute_weights, effective_sample_size
from .update import (
    loop_gradient_step,
    tkl_gradient_step,
    ic_gradient_step,
    tkl_newton_step,
    ic_newton_step,
    ic_newton_step_fullphi,
    ic_newton_step_projected,
)
from .jax_U_calc_mm_forces import build_params_static_mm_from_inputs, compute_energies_all


class DiffTREPipeline:
    def __init__(self, data_glob_path: str, config_path: str):
        self.data_glob_path = data_glob_path
        self.config_path = config_path
        self.config = self._load_config()

        beta_cfg = self._get_cfg("observables.kernel.beta", default=None)
        if beta_cfg is None:
            temp = float(self._get_cfg("simulation.temperature", default=120.3))
            kB_kJ_per_mol_K = 0.008314462618  # kJ/(mol·K)
            self.beta = 1.0 / (kB_kJ_per_mol_K * temp)
        else:
            self.beta = float(beta_cfg)
        self.chains = self._get_cfg("simulation.chains", default=[])
        self.interaction_matrix = self._get_cfg("simulation.interaction_matrix", default=None)
        self.density = self._get_cfg("simulation.density", default=None)
        self.monomer_types_path = self._get_cfg("simulation.monomer_types_path", default=None)
        if self.monomer_types_path is None:
            self.monomer_types_path = self._get_cfg("monomer_types.types_path", default=None)
        self.tkl_exp_path = self._get_cfg("simulation.tkl_exp_path", default=None)

        self.mu = float(self._get_cfg("distance_kernel.mu", default=3.22))
        self.rc = float(self._get_cfg("distance_kernel.rc", default=1.78))
        self.rcut = float(self._get_cfg("distance_kernel.rcut", default=0.0))

        self.theta_ref = self._get_cfg("learned_forces.free_param_initialization", default=None)
        self.theta = self.theta_ref
        self.loop_x = self._get_cfg("learned_forces.loop_X", default=None)
        self.loop_target_path = self._get_cfg("observables.loop_target_path", default=None)

        self.monomer_types = None
        self.positions_list: List[np.ndarray] = []
        self.exp_obs = None
        self.reference_energy = None
        self.current_energy = None
        self.weights = None

    def _load_config(self) -> dict:
        with open(self.config_path, "r") as f:
            return json.load(f)

    def _get_cfg(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        current = self.config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def load_monomer_types(self):
        if self.monomer_types is None:
            if not self.monomer_types_path:
                self.monomer_types_path = self._get_cfg("monomer_types.types_path", default=None)
            if self.monomer_types_path:
                self.monomer_types = np.load(self.monomer_types_path)

    def load_positions(self, discard_initial: int = 0):
        self.positions_list = load_all_replicates(self.data_glob_path, discard_initial=discard_initial)
        return self.positions_list

    def load_reference_energies(self):
        if not self.positions_list:
            raise ValueError("positions_list not loaded")
        energies = []
        for traj_path in sorted(self.data_glob_path if isinstance(self.data_glob_path, list) else []):
            _ = traj_path
        if isinstance(self.data_glob_path, str):
            import glob
            paths = sorted(glob.glob(self.data_glob_path))
        else:
            paths = list(self.data_glob_path)
        skipped = 0
        for path in paths:
            block = read_traj_energy_block(path)
            if block is None:
                raise RuntimeError(f"No ENRG block found in {path}")
            frames = block.get("frames", [])
            if not frames:
                raise RuntimeError(f"Empty ENRG frames in {path}")
            for idx, frame in enumerate(frames):
                if frame is None:
                    skipped += 1
                    continue
                energies.append(float(frame["potential_total_kJmol"]))
        if skipped:
            self.config.setdefault("diagnostics", {})
            self.config["diagnostics"]["energy_frames_skipped"] = int(skipped)
        self.reference_energy = np.array(energies, dtype=float)
        return self.reference_energy

    def load_experimental_observables(self):
        if self.tkl_exp_path:
            tkl_exp = np.load(self.tkl_exp_path)
            self.exp_obs = tkl_exp.flatten()
        if self.loop_target_path:
            loop_target = np.load(self.loop_target_path)
            self.loop_target = float(np.asarray(loop_target).reshape(-1)[0])

    def compute_reference_energy_from_traj(self):
        """
        Read per-frame energy metadata (ENRG block) from each trajectory.
        """
        energies = []
        for traj_path in sorted(self.positions_list):
            _ = traj_path
        # Placeholder: use io_utils.read_traj_energy_block() per traj file when wiring file paths.
        self.reference_energy = None

    def compute_observables(self, loop_pairs: np.ndarray, d_init: int, d_end: int, use_fp32: bool = False):
        self.load_monomer_types()
        if not self.positions_list:
            self.load_positions()
        # flatten replicates into single array of frames
        positions = np.concatenate(self.positions_list, axis=0)
        obs = compute_observables_per_frame(
            positions=positions,
            monomer_types=self.monomer_types,
            loop_pairs=loop_pairs,
            chains=self.chains,
            d_init=d_init,
            d_end=d_end,
            mu=self.mu,
            rc=self.rc,
            use_fp32=use_fp32,
        )
        return obs

    def compute_frame_observables(self, loop_pairs: np.ndarray, d_init: int, d_end: int, use_fp32: bool = False):
        return self.compute_observables(loop_pairs, d_init, d_end, use_fp32=use_fp32)

    def build_force_kwargs(self, params: dict) -> dict:
        # Start with homopolymer_term_forces (fene_bonds, angles, repulsive_softcore, flat_bottom_harmonic)
        f_cfg = json.loads(json.dumps(self._get_cfg("homopolymer_term_forces", default={})))
        
        # Get distance kernel parameters
        mu = float(self._get_cfg("distance_kernel.mu", default=3.22))
        rc = float(self._get_cfg("distance_kernel.rc", default=1.78))
        
        # Add type_to_type force from learned_forces
        learned_tt = self._get_cfg("learned_forces.type_to_type", default={})
        f_cfg["type_to_type"] = {
            "mu": mu,
            "rc": rc,
            "rCutoff": float(learned_tt.get("rCutoff", 3.0)),
            "lim": float(learned_tt.get("lim", 1.0)),
        }
        
        # Add ideal_chromosome force
        f_cfg.setdefault("ideal_chromosome", {})
        if "lambda_IC" in params:
            f_cfg["ideal_chromosome"]["lambda_IC"] = np.asarray(params["lambda_IC"], dtype=float)
        # Get d_init/d_end from observables section
        f_cfg["ideal_chromosome"]["d_init"] = int(self._get_cfg("observables.d_init", default=3))
        f_cfg["ideal_chromosome"]["d_end"] = int(self._get_cfg("observables.d_end", default=300))
        f_cfg["ideal_chromosome"]["mu"] = mu
        f_cfg["ideal_chromosome"]["rc"] = rc
        f_cfg["ideal_chromosome"]["rCutoff"] = float(learned_tt.get("rCutoff", 3.0))  # Use same as type_to_type
        
        # Add loops force
        f_cfg.setdefault("loops", {})
        f_cfg["loops"]["qsi"] = float(params["loop_X"])
        f_cfg["loops"]["mu"] = mu
        f_cfg["loops"]["rc"] = rc
        
        return f_cfg

    def compute_current_energy(self, params: dict, loop_pairs: np.ndarray):
        if not self.positions_list:
            raise ValueError("positions_list not loaded")
        self.load_monomer_types()
        force_kwargs = self.build_force_kwargs(params)
        params_mm, static_mm = build_params_static_mm_from_inputs(
            monomer_types=self.monomer_types,
            interaction_matrix=params["interaction_matrix"],
            chains=[tuple(c) for c in self.chains],
            loop_pairs=loop_pairs,
            force_kwargs=force_kwargs,
            N=int(self._get_cfg("simulation.n_particles", default=len(self.monomer_types))),
        )
        energies = compute_energies_all(
            self.positions_list,
            params_mm,
            static_mm,
            temperature=float(self._get_cfg("simulation.temperature", default=120.3)),
            chunk_reps=2,
        )
        self.current_energy = np.asarray(energies["total"])
        return self.current_energy

    def reweight_observables(self, obs_frames: dict, weights: np.ndarray) -> dict:
        tkl_weighted = np.sum(obs_frames["tkl_frames"] * weights[:, None, None], axis=0)
        phi_weighted = np.sum(obs_frames["phi_frames"] * weights[:, None], axis=0)
        loop_weighted = float(np.sum(obs_frames["loop_frames"] * weights))
        return {
            "tkl_weighted": tkl_weighted,
            "tkl_weighted_flat": tkl_weighted[np.triu_indices(tkl_weighted.shape[0])],
            "phi_weighted": phi_weighted,
            "loop_weighted": loop_weighted,
        }

    def update_loop(self, loop_x, loop_mean, loop_target, eta, max_step=None):
        return float(loop_gradient_step(loop_x, loop_mean, loop_target, eta=eta, max_step=max_step))

    def update_tkl(self, interaction_matrix, tkl_resid_flat, eta, max_step=None):
        new_mat, step = tkl_gradient_step(interaction_matrix, tkl_resid_flat, eta=eta, max_step=max_step)
        return np.asarray(new_mat), np.asarray(step)

    def update_ic(self, lambda_ic, phi_resid, eta, max_step=None):
        return ic_gradient_step(lambda_ic, phi_resid, eta=eta, max_step=max_step)

    def update_tkl_newton(self, interaction_matrix, phi_sim_vec, phi_exp_vec, pi_pj_mean, damp, step_bounds=None):
        new_mat, step = tkl_newton_step(
            interaction_matrix, phi_sim_vec, phi_exp_vec, pi_pj_mean, damp=damp, step_bounds=step_bounds
        )
        return np.asarray(new_mat), np.asarray(step)

    def update_ic_newton(self, params: dict, phi_sim, phi_exp, pi_pj_mean, damp, step_bounds=None):
        return ic_newton_step(params, phi_sim, phi_exp, pi_pj_mean, damp=damp, step_bounds=step_bounds)

    def update_ic_newton_fullphi(self, lambda_ic, phi_sim, phi_exp, pi_pj_mean, damp, step_bounds=None):
        return ic_newton_step_fullphi(lambda_ic, phi_sim, phi_exp, pi_pj_mean, damp=damp, step_bounds=step_bounds)

    def update_ic_newton_projected(
        self,
        params: dict,
        phi_sim,
        phi_exp,
        pi_pj_mean,
        d_init,
        damp,
        step_bounds=None,
    ):
        return ic_newton_step_projected(
            params,
            phi_sim,
            phi_exp,
            pi_pj_mean,
            d_init=d_init,
            damp=damp,
            step_bounds=step_bounds,
        )

    def compute_loop_mean(self, obs: Dict[str, np.ndarray], weights: np.ndarray | None = None) -> float:
        loop_frames = obs["loop_frames"]
        if weights is None:
            return float(np.mean(loop_frames))
        if weights.shape[0] != loop_frames.shape[0]:
            raise ValueError("weights length must match number of loop_frames")
        return float(jnp.sum(jnp.asarray(weights) * jnp.asarray(loop_frames)))

    def update_loop_param(
        self,
        loop_mean: float,
        loop_target: float | None = None,
        loop_x: float | None = None,
        eta: float | None = None,
        max_step: float | None = None,
    ) -> float:
        """
        Apply loop gradient step and update self.loop_x.
        """
        if loop_target is None:
            if not hasattr(self, "loop_target"):
                raise ValueError("loop_target not provided and no loop_target loaded.")
            loop_target = self.loop_target
        if loop_x is None:
            if self.loop_x is None:
                raise ValueError("loop_x not provided and no loop_X configured.")
            loop_x = float(self.loop_x)
        if eta is None:
            eta = float(self._get_cfg("update.loop_eta", default=self._get_cfg("update.eta_init", default=1.0e-3)))
        max_step_cfg = self._get_cfg("update.loop_max_step_size", default=self._get_cfg("update.max_lambda_step_size", default=None))
        if max_step is None:
            max_step = max_step_cfg
        loop_next = loop_gradient_step(loop_x, loop_mean, loop_target, eta=eta, max_step=max_step)
        self.loop_x = float(loop_next)
        return float(loop_next)

    def compute_weights(self):
        if self.reference_energy is None or self.current_energy is None:
            raise ValueError("reference_energy and current_energy must be computed before weights.")
        delta_energy = self.current_energy - self.reference_energy
        self.weights = compute_weights(delta_energy, beta=self.beta)
        return self.weights

    def should_resample(self, threshold_frac: float = 0.7) -> bool:
        if self.weights is None:
            raise ValueError("weights not computed")
        n = self.weights.shape[0]
        n_eff_bar = threshold_frac * n
        n_eff = effective_sample_size(self.weights)
        return bool(n_eff < n_eff_bar)
