import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
import struct
import os
import contextlib
from krbalancing import kr_balancing

class MaxEnt_sim_obs:
    """
    A class to process MD trajectory files for type-type contact observables
    under Maximum Entropy modeling assumptions.
    """

    def __init__(self, n_types=5, cutoff=2.5, leafsize=40):
        """
        Parameters
        ----------
        n_types : int
            Number of monomer types.
        cutoff : float
            Cutoff distance to define a contact (in reduced units).
        leafsize : int
            Leaf size for KDTree contact search.
        """
        self.n_types = n_types
        self.cutoff = cutoff
        self.leafsize = leafsize

    @staticmethod
    def load_all_positions(filename):
        """
        Load all particle positions from a binary .traj file.

        Parameters
        ----------
        filename : str
            Path to .traj file.

        Returns
        -------
        np.ndarray
            Array of shape (n_frames, n_particles, 3)
        """
        HEADER_FORMAT = "<4sBHII16s"
        HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

        with open(filename, 'rb') as f:
            header = f.read(HEADER_SIZE)
            magic, version, n_particles, frame_size, n_frames, _ = struct.unpack(HEADER_FORMAT, header)
            assert magic == b'CHRM'

            metadata_len = struct.unpack("<I", f.read(4))[0]
            f.seek(HEADER_SIZE + 4 + metadata_len)

            data = np.frombuffer(f.read(), dtype=np.float32)
            return data.reshape((n_frames, n_particles, 3))

    def compute_contact_map_frame(self, pos):
        """
        Compute binary contact map for a single frame using cKDTree.

        Parameters
        ----------
        pos : np.ndarray
            Array of shape (N, 3) with particle positions.

        Returns
        -------
        np.ndarray
            Binary contact map of shape (N, N).
        """
        N = pos.shape[0]
        contact_map = np.zeros((N, N), dtype=np.uint8)
        pairs = cKDTree(pos, leafsize=self.leafsize).query_pairs(self.cutoff, output_type="ndarray")
        if pairs.size > 0:
            i, j = pairs[:, 0], pairs[:, 1]
            contact_map[i, j] = 1
            contact_map[j, i] = 1  # symmetry
        return contact_map

    @contextlib.contextmanager
    def suppress_stdout_stderr(self):
        """
        Suppress stdout and stderr for cleaner logs (used during KR balancing).
        """
        with open(os.devnull, 'w') as devnull:
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            try:
                os.dup2(devnull.fileno(), 1)
                os.dup2(devnull.fileno(), 2)
                yield
            finally:
                os.dup2(old_stdout_fd, 1)
                os.dup2(old_stderr_fd, 2)
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)

    def kr_balance_hic_matrix_upper(self, hic_matrix, rescale=True):
        """
        Apply Knight-Ruiz matrix balancing and return upper triangle of result.

        Parameters
        ----------
        hic_matrix : np.ndarray
            Binary contact map matrix.
        rescale : bool
            Whether to rescale the balanced matrix.

        Returns
        -------
        tuple of (values, global_i, global_j, shape, correction_vector, valid_indices)
        """
        row_nz = np.where(hic_matrix.sum(axis=1).flatten() > 0)[0]
        col_nz = np.where(hic_matrix.sum(axis=0).flatten() > 0)[0]
        valid_indices = np.intersect1d(row_nz, col_nz)
        N = hic_matrix.shape[0]

        if len(valid_indices) == 0:
            raise ValueError("No valid bins found.")

        submatrix = hic_matrix[np.ix_(valid_indices, valid_indices)].astype(np.float64)
        sparse_matrix = csr_matrix(submatrix)

        with self.suppress_stdout_stderr():
            kr = kr_balancing(
                sparse_matrix.shape[0],
                sparse_matrix.shape[1],
                sparse_matrix.nnz,
                sparse_matrix.indptr.astype(np.int64, copy=False),
                sparse_matrix.indices.astype(np.int64, copy=False),
                sparse_matrix.data.astype(np.float64, copy=False)
            )
            kr.computeKR()
            balanced = np.array(kr.get_normalised_matrix(rescale).todense(), dtype=np.float64)
            correction = np.array(kr.get_normalisation_vector(rescale).todense(), dtype=np.float64).flatten()

        tri_i_local, tri_j_local = np.triu_indices(len(valid_indices))
        values = balanced[tri_i_local, tri_j_local]
        global_i = valid_indices[tri_i_local]
        global_j = valid_indices[tri_j_local]

        return values, global_i, global_j, (N, N), correction, valid_indices

    def compute_type_type_vectorized(self, upper_values, tri_i, tri_j, matrix_shape, monomer_types):
        """
        Compute type-type observable matrix from upper triangle values.

        Parameters
        ----------
        upper_values : np.ndarray
            Balanced contact strengths.
        tri_i, tri_j : np.ndarray
            Global indices corresponding to upper triangle.
        matrix_shape : tuple
            Shape of the original full matrix (N, N).
        monomer_types : np.ndarray
            Monomer type labels of shape (N,).

        Returns
        -------
        np.ndarray
            Type-type matrix of shape (n_types, n_types)
        """
        valid_mask = ~np.isnan(upper_values)
        vals = upper_values[valid_mask]
        i_vals = tri_i[valid_mask]
        j_vals = tri_j[valid_mask]

        ti = monomer_types[i_vals].astype(int)
        tj = monomer_types[j_vals].astype(int)

        contact_sums = np.zeros((self.n_types, self.n_types), dtype=float)
        np.add.at(contact_sums, (ti, tj), vals)

        off_diag = i_vals != j_vals
        np.add.at(contact_sums, (tj[off_diag], ti[off_diag]), vals[off_diag])

        total = contact_sums.sum()
        if total > 0:
            contact_sums /= total

        return contact_sums

    # --- PATCHED MaxEnt_sim_obs method ---
    def compute_sim_type_type_observables_streaming(self, traj_file=None, monomer_types=None, positions=None, verbose=True):
        """
        Compute per-frame type-type observables.

        You must provide EITHER `positions` (preferred when throttling I/O) OR `traj_file`.
        If `positions` is provided, the method will NOT read from disk.

        Parameters
        ----------
        traj_file : str or None
            Path to .traj file (used only if positions is None).
        monomer_types : np.ndarray
            Monomer type labels of shape (N,).
        positions : np.ndarray or None
            Preloaded positions of shape (n_frames, N, 3). If provided, skip disk I/O.
        verbose : bool

        Returns
        -------
        np.ndarray
            Array (n_frames, n_types, n_types)
        """
        if positions is None:
            assert traj_file is not None, "Provide traj_file or positions."
            positions = self.load_all_positions(traj_file)
            src_label = traj_file
        else:
            src_label = "in-memory positions"

        n_frames, N, _ = positions.shape
        assert monomer_types is not None, "monomer_types is required."
        assert len(monomer_types) == N, "monomer_types must match number of particles"

        output = np.zeros((n_frames, self.n_types, self.n_types), dtype=float)
        if verbose:
            print(f"Processing {src_label} ({n_frames} frames, N={N})")

        from tqdm import tqdm
        with tqdm(total=n_frames, desc="Frames") as pbar:
            for f in range(n_frames):
                pos = positions[f]
                contact_map = self.compute_contact_map_frame(pos)
                try:
                    upper_values, tri_i, tri_j, shape, correction, valid_idx = self.kr_balance_hic_matrix_upper(
                        contact_map, rescale=True
                    )
                    obs = self.compute_type_type_vectorized(
                        upper_values, tri_i, tri_j, shape, monomer_types
                    )
                except ValueError:
                    obs = np.full((self.n_types, self.n_types), np.nan)
                output[f] = obs
                pbar.update(1)

        return output

