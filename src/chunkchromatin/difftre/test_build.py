##### Test implementation of differentiable trajectory reweighting #####
import numpy as np
import jax as jp
import jax.numpy as jnp
import struct
import os
import glob
from typing import List, Any
import json
from chunkchromatin.difftre.jax_O_calc import compute_observables_all


class DiffTRE:
    def __init__(self, data_glob_path: str, config_path: str):
        self.data_glob_path = data_glob_path
        self.config_path = config_path
        self.beta = 1
        self.chains = self.load_config_var("simulation.chains")
        self.interaction_matrix = self.load_config_var("simulation.interaction_matrix")  
        self.density = self.load_config_var("simulation.density")
        self.monomer_types_path = self.load_config_var("simulation.monomer_types_path")
        self.monomer_types = None  # will be loaded on first use
        self.mu = self.load_config_var("distance_kernel.mu")
        self.rc = self.load_config_var("distance_kernel.rc")
        self.rcut = self.load_config_var("distance_kernel.rcut")
        self.tkl_exp_path = self.load_config_var("simulation.tkl_exp_path")

        # all force parameters in one dict, e.g., forces.harmonic_bonds, etc.
        self.force_kwargs = self.load_config_var("forces")


    def load_config_var(self, key_path: str) -> Any:
        """
        Load a nested variable from a JSON configuration file using dot notation.

        Example:
            load_config_var("config.json", "simulation.forces.angle.k")

        Parameters
        ----------
        key_path : str
            Dot-delimited key path (e.g., "simulation.forces.angle.k").

        Returns
        -------
        Any
            The value stored at the given key path.

        Raises
        ------
        FileNotFoundError
            If the config file does not exist.
        KeyError
            If any key in the path is missing.
        ValueError
            If the file cannot be parsed as JSON.
        """
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON config file: {self.config_path}\n{e}")

        keys = key_path.split(".")
        current = config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                raise KeyError(f"Key '{k}' not found while traversing '{key_path}' in {self.config_path}")

        return current

    def ensure_monomer_types_loaded(self):
        """
        Loads in monomer types from numpy file
        according to config file
        """
        if self.monomer_types is None:
            self.monomer_types = np.load(self.monomer_types_path)
    
    @staticmethod
    def load_all_positions_jax(filename: str) -> jnp.ndarray:
        """
        Load all particle positions from a single .traj file and return a JAX array.

        Returns
        -------
        jnp.ndarray
            (n_frames, n_particles, 3) float64
        """
        HEADER_FORMAT = "<4sBHII16s"
        HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
        with open(filename, "rb") as f:
            header = f.read(HEADER_SIZE)
            magic, version, n_particles, frame_size, n_frames, _ = struct.unpack(HEADER_FORMAT, header)
            assert magic == b"CHRM", f"Bad magic in {filename}"
            # Metadata length (bytes), then skip metadata
            metadata_len = struct.unpack("<I", f.read(4))[0]
            f.seek(HEADER_SIZE + 4 + metadata_len)

            # Raw float32 payload → reshape → float64 (no extra copy) → JAX array
            data = np.frombuffer(f.read(), dtype=np.float32)
            arr = data.reshape((n_frames, n_particles, 3)).astype(np.float64, copy=False)
            return jnp.asarray(arr)

    @staticmethod
    def load_all_replicates_jax(glob_path: str, discard_initial: int = 500) -> List[jnp.ndarray]:
        """
        Load positions from all replicate .traj files matching a glob pattern.

        Parameters
        ----------
        glob_path : str
            Glob pattern (e.g., "/path/to/runs/*.traj").
        discard_initial : int, default 500
            Number of initial frames to drop from each replicate.

        Returns
        -------
        List[jnp.ndarray]
            A list where each element is (n_sampled_frames, n_particles, 3) float64
            for a single replicate, stored as a JAX array.
        """
        files = sorted(glob.glob(glob_path))
        if not files:
            raise FileNotFoundError(f"No files matched glob: {glob_path}")

        out: List[jnp.ndarray] = []
        for fp in files:
            arr = DiffTRE.load_all_positions_jax(fp)  # (F, N, 3) float64 in JAX
            F = arr.shape[0]
            start = min(discard_initial, F)  # avoid negative length if short
            sampled = arr[start:]            # (F - start, N, 3)
            out.append(sampled)

        return out

    def calculate_reference_observable(self, positions):
        """
        Calculates the reference observable from the reference trajectory.
        """
        #make sure monomer types are loaded
        self.ensure_monomer_types_loaded()

        #load in positions
        positions = DiffTRE.load_all_replicates_jax(self.data_glob_path)

        obs_list = compute_observables_all(positions, self.monomer_types, self.mu, self.rc, rcut=self.rcut, max_cell_particles=96, rep_chunk_size=10)
        return obs_list

    def load_experimental_observable(self):
        """
        Loads in experimental observables from numpy file
        according to config file
        tkl_exp is the flattened upper triangle of the experimental Tkl matrix
        which contains sums of average HiC contact probabilities between each type-type pair

        HiC contact probabilities are calculated by KR-normalizing the HiC contact counts
        and then dividing by the max contact count for each row.
        """

        tkl_exp = np.load(self.tkl_exp_path)
        tkl_exp = tkl_exp.flatten()
        return tkl_exp
    
    def calculate_energy(self, positions):
        """
        Calculates the energy of the system.
        Parameters
        ----------
        positions: list of arrays of shape (n_frames, n_particles, 3)
        Returns
        -------
        energy: array of shape (n_frames*n_replicates,)
        """

        params, static = build_params_static_from_inputs(
            monomer_types=self.monomer_types,
            interaction_matrix=self.interaction_matrix,
            chains=self.chains,
            force_kwargs=self.force_kwargs,
            density=self.density
        )

        #process in chunks of 10 replicates at a time
        chunk_reps = 10
        current_energy = compute_energies_all(positions, params, static, chunk_reps, return_components=False)

        return current_energy
    

    def calculate_traj_weights(self):
        """
        Weight each frame based on the probability of the frame being sampled
        from the reference distribution given the current energy.
        Parameters
        ----------
        delta_energy: array of delta energies for each frame

        Returns
        -------
        w: array of weights for each frame
        """

        w = jnp.exp(-self.beta * delta_energy)
        w /= jnp.sum(w)
        return w
        


    def loss(self, theta, data):
        """
        Loss function for the differentiable trajectory reweighting.
        MSE of the weighted observable and the experimental observable.
        """

        w = self.calculate_traj_weights()
        return -jnp.sum(w * data)









### OLD CODE ###
    def calculate_reference_observable_old(self, positions, monomer_types, mu=MU_DEFAULT, rc=RC_DEFAULT, rcut=None):
        """
        Calculates the reference observable from the reference trajectory.

        Parameters
        ----------
        positions: array of shape (n_frames, n_particles, 3)
        monomer_types: array of shape (n_particles,)
        mu: float
        rc: float
        rcut: float
        """

        F, N, _ = positions.shape
        type_labels, inv = np.unique(monomer_types, return_inverse=True)
        K = len(type_labels)
        acc = UpperTriOnlineCov(K)
        if rcut is None:
            rcut = rc + 4.0 / mu

        iuK = np.triu_indices(K)
        for f in range(F):
            X = positions[f]
            tree = cKDTree(X, leafsize=40)
            pairs = tree.query_pairs(rcut, output_type='ndarray')

            T_up = np.zeros((K, K), float)
            if pairs.size != 0:
                i = pairs[:, 0]; j = pairs[:, 1]
                rij = np.linalg.norm(X[i] - X[j], axis=1)
                fij = f_switch(rij, mu=mu, rc=rc)
                ti = inv[i]; tj = inv[j]
                k = np.minimum(ti, tj)
                l = np.maximum(ti, tj)
                flat = k * K + l
                sums = np.bincount(flat, weights=fij, minlength=K*K).reshape(K, K)
                T_up[iuK] = sums[iuK]
            acc.add_frame_from_upper_mat(T_up)

        return acc.finalize(beta=1.0) + (type_labels,)

