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

############################## sim_obs V2 ####################################
import numpy as np
from scipy.spatial import cKDTree
import struct
import os
import contextlib
from krbalancing import kr_balancing

# --- optional: use numba if present for the tiny accumulator ---
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
    def njit(*args, **kwargs):
        def _wrap(f): return f
        return _wrap


@njit
def _accumulate_type_type_edges_numba(iu, ju, d, monomer_types, n_types):
    """
    Upper-tri edges only (i<j). Adds symmetric contributions.
    """
    out = np.zeros((n_types, n_types), np.float64)
    for k in range(iu.shape[0]):
        ii = iu[k]
        jj = ju[k]
        val = d[ii] * d[jj]
        ti = int(monomer_types[ii])
        tj = int(monomer_types[jj])
        out[ti, tj] += val
        out[tj, ti] += val  # symmetry
    s = out.sum()
    if s > 0.0:
        out /= s
    return out


def _accumulate_type_type_edges_py(iu, ju, d, monomer_types, n_types):
    out = np.zeros((n_types, n_types), dtype=np.float64)
    ti = monomer_types[iu].astype(np.int64, copy=False)
    tj = monomer_types[ju].astype(np.int64, copy=False)
    vals = d[iu] * d[ju]
    # add upper
    np.add.at(out, (ti, tj), vals)
    # and mirror for symmetry
    np.add.at(out, (tj, ti), vals)
    s = out.sum()
    if s > 0.0:
        out /= s
    return out


def _pairs_to_csr_from_upper(N, iu, ju):
    """
    Build CSR adjacency from upper-triangle pairs (i<j) by symmetrizing.
    Returns (indptr, indices, data) where data is float64 ones.
    """
    if iu.size == 0:
        # empty matrix
        indptr = np.zeros(N + 1, dtype=np.int64)
        indices = np.zeros(0, dtype=np.int64)
        data = np.zeros(0, dtype=np.float64)
        return indptr, indices, data

    # Symmetrize edges: (i,j) and (j,i)
    rows = np.concatenate([iu, ju]).astype(np.int64, copy=False)
    cols = np.concatenate([ju, iu]).astype(np.int64, copy=False)

    # Stable sort by row to pack CSR
    order = np.argsort(rows, kind='mergesort')
    rows = rows[order]
    cols = cols[order]

    # Build indptr via bincount
    indptr = np.zeros(N + 1, dtype=np.int64)
    counts = np.bincount(rows, minlength=N)
    np.cumsum(counts, out=indptr[1:])

    indices = cols  # already aligned with 'rows' order
    data = np.ones_like(indices, dtype=np.float64)
    return indptr, indices, data


class MaxEnt_sim_obs_v2:
    """
    Faster version of MaxEnt_sim_obs that:
      - avoids building dense contact maps
      - runs KR on a sparse CSR built directly from KDTree pairs
      - uses only the KR diagonal vector 'd' to accumulate type–type over edges
      - never densifies the balanced matrix

    API kept similar to your original for easy swap-in.
    """

    def __init__(self, n_types=5, cutoff=2.5, leafsize=40, kr_rescale_vector=False):
        """
        Parameters
        ----------
        n_types : int
            Number of monomer types.
        cutoff : float
            Cutoff distance to define a contact (in reduced units).
        leafsize : int
            Leaf size for KDTree contact search.
        kr_rescale_vector : bool
            If True, call KR's rescale pass on the normalization vector (extra O(nnz) work).
            Usually not needed because we normalize the final type–type matrix anyway.
        """
        self.n_types = n_types
        self.cutoff = cutoff
        self.leafsize = leafsize
        self.kr_rescale_vector = kr_rescale_vector

        # pick the accumulator impl
        self._accum_fn = _accumulate_type_type_edges_numba if _HAVE_NUMBA else _accumulate_type_type_edges_py

    # ---------- I/O ----------

    @staticmethod
    def load_all_positions(filename):
        """
        Load all particle positions from a binary .traj file.
        Returns array (n_frames, n_particles, 3)
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

    # ---------- geometry → edges ----------

    def compute_pairs_frame(self, pos):
        """
        Return upper-triangle neighbor pairs (i<j) within cutoff using KDTree.
        pos: (N,3)
        returns iu, ju as int64 arrays (same length), each i<j
        """
        pairs = cKDTree(pos, leafsize=self.leafsize).query_pairs(self.cutoff, output_type="ndarray")
        if pairs.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        iu = pairs[:, 0].astype(np.int64, copy=False)
        ju = pairs[:, 1].astype(np.int64, copy=False)
        return iu, ju

    # ---------- KR on sparse adjacency ----------

    @contextlib.contextmanager
    def suppress_stdout_stderr(self):
        # keep your suppression helper for KR verbosity
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

    def kr_vector_from_pairs(self, N, iu, ju):
        """
        Build CSR from upper-tri pairs and run KR to get the diagonal vector d.
        Returns a dense np.ndarray 'd' of length N (float64).
        """
        # Build CSR for full N×N (symmetric)
        indptr, indices, data = _pairs_to_csr_from_upper(N, iu, ju)

        # Hand to KR (constructor adds a tiny diagonal; zero-rows become nonzero)
        with self.suppress_stdout_stderr():
            kr = kr_balancing(
                int(N), int(N), int(data.size),
                indptr.astype(np.int64, copy=False),
                indices.astype(np.int64, copy=False),
                data.astype(np.float64, copy=False),
            )
            kr.computeKR()
            # prefer not to rescale; we normalize at the end
            rescale = bool(self.kr_rescale_vector)
            x_sparse = kr.get_normalisation_vector(rescale)
            # x is returned as an Eigen sparse column vector; convert once to dense
            # The previous code used .todense(); keep that for safety.
            d = np.array(x_sparse.todense(), dtype=np.float64).ravel()
            # If KR returned a subvector (shouldn't, given identity added), pad if needed
            if d.shape[0] != N:
                dd = np.zeros(N, dtype=np.float64)
                dd[:d.shape[0]] = d
                d = dd
        return d

    # ---------- final observable per frame ----------

    def type_type_from_edges_and_d(self, N, iu, ju, d, monomer_types):
        """
        Produce (n_types, n_types) from edges (upper) and KR vector d.
        """
        if iu.size == 0:
            return np.full((self.n_types, self.n_types), np.nan, dtype=np.float64)
        # Safety checks
        assert monomer_types.shape[0] == N
        return self._accum_fn(iu, ju, d, monomer_types, self.n_types)

    # ---------- main driver ----------

    def compute_sim_type_type_observables_streaming(self, traj_file=None, monomer_types=None, positions=None, verbose=True):
        """
        Compute per-frame type–type observables.
        You can pass positions directly (n_frames,N,3) or a traj file path.

        Returns: array (n_frames, n_types, n_types)
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
        monomer_types = np.asarray(monomer_types)

        out = np.zeros((n_frames, self.n_types, self.n_types), dtype=np.float64)
        if verbose:
            print(f"[v2] Processing {src_label} ({n_frames} frames, N={N})  |  cutoff={self.cutoff}")

        # lazy import tqdm only if verbose
        pbar = range(n_frames)
        if verbose:
            try:
                from tqdm import tqdm  # local import
                pbar = tqdm(pbar, desc="Frames(v2)")
            except Exception:
                pass

        for f in pbar:
            pos = positions[f]

            # 1) geometry → upper-tri neighbor pairs
            iu, ju = self.compute_pairs_frame(pos)

            if iu.size == 0:
                out[f] = np.full((self.n_types, self.n_types), np.nan, dtype=np.float64)
                continue

            # 2) KR diagonal vector from sparse adjacency
            d = self.kr_vector_from_pairs(N, iu, ju)

            # 3) accumulate type–type using edges and d (O(E), no dense N×N)
            out[f] = self.type_type_from_edges_and_d(N, iu, ju, d, monomer_types)

        return out

import numpy as np
from typing import Iterable, List, Optional, Tuple
from scipy.spatial.distance import cdist


class MaxEntTypesObs:
    """
    Utilities to compute type–type block-mean observables for simulation frames (from positions)
    and for experimental Hi-C maps.

    Conventions
    -----------
    - Diagonal (i == j) is excluded everywhere.
    - Optional near-diagonal genomic neighbor removal: |i-j| <= band_width -> 0.
    - SAME-TYPE blocks (A==B): unordered pairs i<j, denominator C(|A|,2).
      DIFFERENT-TYPE blocks (A!=B): rectangle A x B once, denominator |A||B|.
    - Tiled over the upper triangle of bead indices to avoid dense N×N storage.
    - Outputs are symmetric (upper mirrored to lower) by default.
    """

    # -------------------- lifecycle --------------------

    def __init__(
        self,
        monomer_types: np.ndarray,     # (N,) integer labels; will be captured as self.monomer_types
        n_types: int = 5,
        mode: str = "binary",            # 'soft' or 'binary'
        mu: float = 1.0,               # soft only
        r_cut: float = 3.0,            # soft+binary (geom cutoff for sim)
        p_min: float = 1.0,            # soft only: entries < p_min -> 0
        band_width: int = 3,           # |i-j| <= band_width -> 0
        tile_size: int = 4096,
        return_symmetric: bool = True,
        dtype=np.float64,
    ):
        self.monomer_types = np.asarray(monomer_types)
        self.n_types = int(n_types)
        self.mode = str(mode)
        self.mu = float(mu)
        self.r_cut = float(r_cut)
        self.p_min = float(p_min)
        self.band_width = int(band_width)
        self.tile_size = int(tile_size)
        self.return_symmetric = bool(return_symmetric)
        self.dtype = dtype

        # Precompute type index lists (sorted) for fast slicing
        self._idx_by_type = self._prepare_type_indices(self.monomer_types, self.n_types)

    # -------------------- SIMULATION (keep your docstrings/behavior) --------------------

    def sim_type_type_block_means_frame(
        self,
        pos: np.ndarray,                 # (N,3)
        mode: Optional[str] = None,
        mu: Optional[float] = None,
        r_cut: Optional[float] = None,
        p_min: Optional[float] = None,
        band_width: Optional[int] = None,
        tile_size: Optional[int] = None,
        return_symmetric: Optional[bool] = None,
        dtype=None,
    ) -> np.ndarray:
        """
        Compute a single-frame (n_types x n_types) observable from positions.
        """
        mode = self.mode if mode is None else mode
        mu = self.mu if mu is None else mu
        r_cut = self.r_cut if r_cut is None else r_cut
        p_min = self.p_min if p_min is None else p_min
        band_width = self.band_width if band_width is None else band_width
        tile_size = self.tile_size if tile_size is None else tile_size
        return_symmetric = self.return_symmetric if return_symmetric is None else return_symmetric
        dtype = self.dtype if dtype is None else dtype

        pos = np.asarray(pos, dtype=dtype, order="C")
        N = pos.shape[0]

        num = np.zeros((self.n_types, self.n_types), dtype=dtype)
        den = np.zeros((self.n_types, self.n_types), dtype=dtype)

        for I, J in self._tile_pairs(N, tile_size):
            D = cdist(pos[I], pos[J])  # (len(I), len(J))

            # exclude diagonal of self-tile
            if I.start == J.start and I.stop == J.stop:
                np.fill_diagonal(D, np.inf)

            # distances -> P
            if mode == "binary":
                P = (D < r_cut).astype(dtype, copy=False)
            elif mode == "soft":
                P = 0.5 * (1.0 + np.tanh(mu * (r_cut - D)))
                if p_min < 1.0:
                    P[P < p_min] = 0.0
                if I.start == J.start and I.stop == J.stop:
                    np.fill_diagonal(P, 0.0)
            else:
                raise ValueError("mode must be 'binary' or 'soft'")

            # genomic neighbor removal
            if band_width > 0:
                self._apply_band_mask_inplace(P, I, J, band_width)

            # accumulate block means
            self._accumulate_type_blocks_tile(P, I, J, self._idx_by_type, num, den)

        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.divide(num, den, out=np.zeros_like(num), where=(den > 0))

        if return_symmetric:
            iu, ju = np.triu_indices(self.n_types, k=1)
            out[ju, iu] = out[iu, ju]
        return out

    def sim_type_type_block_means_batch(
        self,
        positions: np.ndarray,           # (n_frames, N, 3)
        **kwargs                         # same overrides as frame()
    ) -> np.ndarray:
        """
        Compute per-frame observables for a whole replicate.
        Returns (n_frames, n_types, n_types).
        """
        dtype = kwargs.get("dtype", self.dtype)
        positions = np.asarray(positions, dtype=dtype)
        n_frames = positions.shape[0]
        out = np.zeros((n_frames, self.n_types, self.n_types), dtype=dtype)
        for f in range(n_frames):
            out[f] = self.sim_type_type_block_means_frame(positions[f], **kwargs)
        return out

    # -------------------- EXPERIMENT: POST-KR (binary only per your request) --------------------

    def exp_type_type_block_means_from_KR(
        self,
        C_KR,                             # (N,N) dense ndarray OR array-like supporting slicing
        mode: Optional[str] = None,       # 'binary' or 'soft' (soft disabled)
        p_min: Optional[float] = None,    # ignored in binary path
        band_width: Optional[int] = None,
        tile_size: Optional[int] = None,
        return_symmetric: Optional[bool] = None,
        dtype=None,
        soft_scaler: Optional[callable] = None,  # ignored (no soft KR method)
    ) -> np.ndarray:
        """
        Compute experimental observable from a KR-balanced map using the same
        block-mean logic.
        """
        # enforce binary-only behavior for KR paths
        mode = "binary"

        band_width = self.band_width if band_width is None else band_width
        tile_size = self.tile_size if tile_size is None else tile_size
        return_symmetric = self.return_symmetric if return_symmetric is None else return_symmetric
        dtype = self.dtype if dtype is None else dtype

        N = C_KR.shape[0]
        num = np.zeros((self.n_types, self.n_types), dtype=dtype)
        den = np.zeros((self.n_types, self.n_types), dtype=dtype)

        for I, J in self._tile_pairs(N, tile_size):
            V = np.array(C_KR[I, J], dtype=dtype, copy=False)

            if I.start == J.start and I.stop == J.stop:
                np.fill_diagonal(V, 0.0)

            # binary only
            P = (V > 0.0).astype(dtype, copy=False)

            if band_width > 0:
                self._apply_band_mask_inplace(P, I, J, band_width)

            self._accumulate_type_blocks_tile(P, I, J, self._idx_by_type, num, den)

        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.divide(num, den, out=np.zeros_like(num), where=(den > 0))
        if return_symmetric:
            iu, ju = np.triu_indices(self.n_types, k=1)
            out[ju, iu] = out[iu, ju]
        return out

    # -------------------- EXPERIMENT: RAW COUNTS (runs KR internally; binary only) --------------------

    def kr_vector_from_counts(
        self,
        counts: np.ndarray,
        rescale: bool = True,
        verbose: bool = False,
        dtype=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run KR on a raw Hi-C counts matrix (dense), return:
          - d_valid: normalization vector for valid bins
          - valid_indices: indices retained (non-zero rows & cols)
        """
        from scipy.sparse import csr_matrix
        from krbalancing import kr_balancing

        dtype = self.dtype if dtype is None else dtype
        counts = np.asarray(counts, dtype=dtype)
        N = counts.shape[0]
        if verbose:
            print("Starting KR balancing...")
            print(f"Input matrix shape: {counts.shape}")

        row_nz = np.where(counts.sum(axis=1).reshape(-1) > 0)[0]
        col_nz = np.where(counts.sum(axis=0).reshape(-1) > 0)[0]
        valid_indices = np.intersect1d(row_nz, col_nz)
        if verbose:
            print(f"Valid bins: {len(valid_indices)} / {N}")
        if valid_indices.size == 0:
            raise ValueError("No valid bins found for KR.")

        sub = counts[np.ix_(valid_indices, valid_indices)].astype(dtype, copy=False)
        A = csr_matrix(sub)

        kr = kr_balancing(
            A.shape[0], A.shape[1], A.nnz,
            A.indptr.astype(np.int64, copy=False),
            A.indices.astype(np.int64, copy=False),
            A.data.astype(dtype, copy=False),
        )
        if verbose:
            print("Computing KR...")
        kr.computeKR()
        d_valid = np.array(kr.get_normalisation_vector(bool(rescale)).todense(), dtype=dtype).ravel()
        if verbose:
            print("KR balancing completed.")
        return d_valid, valid_indices

    def exp_type_type_block_means_from_raw_counts(
        self,
        counts: np.ndarray,               # (N,N) raw counts, symmetric, dense
        mode: Optional[str] = None,       # 'binary' or 'soft' (soft disabled)
        p_min: Optional[float] = None,    # ignored
        band_width: Optional[int] = None,
        tile_size: Optional[int] = None,
        return_symmetric: Optional[bool] = None,
        dtype=None,
        rescale: bool = True,
        verbose: bool = False,
        soft_scaler: Optional[callable] = None,  # ignored (no soft KR method)
    ) -> np.ndarray:
        """
        Full experimental pipeline from RAW counts:
          1) Identify valid bins (non-zero rows/cols), subset counts and types.
          2) KR on the sparse weighted submatrix -> d.
          3) Tile over valid (I,J): V_bal = counts[Iglob,Jglob] * d[I] * d[J].
          4) Build P in binary mode with diagonal + band removal.
          5) Accumulate type–type block means (unordered pairs for A==B).
        """
        # enforce binary-only behavior for KR paths
        mode = "binary"

        band_width = self.band_width if band_width is None else band_width
        tile_size = self.tile_size if tile_size is None else tile_size
        return_symmetric = self.return_symmetric if return_symmetric is None else return_symmetric
        dtype = self.dtype if dtype is None else dtype

        counts = np.asarray(counts, dtype=dtype)

        # 1–2) KR to get d over valid bins
        d_valid, valid = self.kr_vector_from_counts(counts, rescale=rescale, verbose=verbose, dtype=dtype)
        sub_types = self.monomer_types[valid]
        Nv = valid.size
        idx_by_type = self._prepare_type_indices(sub_types, self.n_types)

        num = np.zeros((self.n_types, self.n_types), dtype=dtype)
        den = np.zeros((self.n_types, self.n_types), dtype=dtype)

        # 3–5) Tiled over valid coordinates
        for I, J in self._tile_pairs(Nv, tile_size):
            # map valid coords -> arrays of global indices
            row_idx = valid[I.start:I.stop]
            col_idx = valid[J.start:J.stop]

            # slice counts with fancy indexing (handles non‑contiguity safely)
            V = counts[np.ix_(row_idx, col_idx)].astype(dtype, copy=False)

            # balance on the fly: d[I] * counts * d[J]
            dI = d_valid[I.start:I.stop][:, None]
            dJ = d_valid[J.start:J.stop][None, :]
            V_bal = V * dI * dJ

            if I.start == J.start and I.stop == J.stop:
                np.fill_diagonal(V_bal, 0.0)

            # binary only
            P = (V_bal > 0.0).astype(dtype, copy=False)

            if band_width > 0:
                self._apply_band_mask_inplace(P, I, J, band_width)

            self._accumulate_type_blocks_tile(P, I, J, idx_by_type, num, den)

        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.divide(num, den, out=np.zeros_like(num), where=(den > 0))
        if return_symmetric:
            iu, ju = np.triu_indices(self.n_types, k=1)
            out[ju, iu] = out[iu, ju]
        return out

    # -------------------- helpers (keep style) --------------------

    @staticmethod
    def _prepare_type_indices(monomer_types: np.ndarray, n_types: int) -> List[np.ndarray]:
        mt = np.asarray(monomer_types)
        idx_by_type = []
        for t in range(n_types):
            idx = np.flatnonzero(mt == t)
            idx_by_type.append(idx.astype(np.int64))
        return idx_by_type

    @staticmethod
    def _tile_pairs(N: int, tile_size: int) -> Iterable[Tuple[slice, slice]]:
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        edges = list(range(0, N, tile_size)) + [N]
        for a in range(len(edges) - 1):
            I = slice(edges[a], edges[a+1])
            for b in range(a, len(edges) - 1):
                J = slice(edges[b], edges[b+1])
                yield I, J

    @staticmethod
    def _apply_band_mask_inplace(P_tile: np.ndarray, I: slice, J: slice, band_width: int) -> None:
        if band_width <= 0:
            return
        rows = np.arange(I.start, I.stop)[:, None]
        cols = np.arange(J.start, J.stop)[None, :]
        P_tile[np.abs(rows - cols) <= band_width] = 0.0

    @staticmethod
    def _accumulate_type_blocks_tile(
        P: np.ndarray,
        I: slice, J: slice,
        idx_by_type: List[np.ndarray],
        num: np.ndarray,
        den: np.ndarray,
    ) -> None:
        """
        Add tile contributions into (num, den) for each type block.
        Handles (A!=B), (A==B and I!=J), and (A==B and I==J with i<j).
        """
        same_tile = (I.start == J.start and I.stop == J.stop)
        T = num.shape[0]

        for A in range(T):
            I_A = MaxEntTypesObs._intersect_contiguous(idx_by_type[A], I.start, I.stop)
            if I_A.size == 0:
                continue
            for B in range(A, T):
                J_B = MaxEntTypesObs._intersect_contiguous(idx_by_type[B], J.start, J.stop)
                if J_B.size == 0:
                    continue
                if A != B:
                    block_sum = P[np.ix_(I_A, J_B)].sum(dtype=num.dtype)
                    num[A, B] += block_sum
                    den[A, B] += (I_A.size * J_B.size)
                else:
                    if not same_tile:
                        block_sum = P[np.ix_(I_A, J_B)].sum(dtype=num.dtype)
                        num[A, B] += block_sum
                        den[A, B] += (I_A.size * J_B.size)
                    else:
                        if I_A.size >= 2:
                            sub = P[np.ix_(I_A, I_A)]
                            iu, ju = np.triu_indices(I_A.size, k=1)
                            num[A, B] += sub[iu, ju].sum(dtype=num.dtype)
                            den[A, B] += (I_A.size * (I_A.size - 1) // 2)

    @staticmethod
    def _intersect_contiguous(indexes_of_type: np.ndarray, start: int, stop: int) -> np.ndarray:
        """
        Intersect a sorted array of indices with a contiguous slice [start:stop),
        and return LOCAL positions within that slice (0..len-1).
        """
        lo = np.searchsorted(indexes_of_type, start, side='left')
        hi = np.searchsorted(indexes_of_type, stop, side='left')
        if hi <= lo:
            return np.empty(0, dtype=np.int64)
        return (indexes_of_type[lo:hi] - start).astype(np.int64)


