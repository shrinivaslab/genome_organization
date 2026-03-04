from __future__ import annotations

import glob
import json
import struct
from typing import List, Optional, Union
from pathlib import Path
import jax.numpy as jnp

import numpy as np


_TRAJ_HEADER_FORMAT = "<4sBHII16s"
_TRAJ_HEADER_SIZE = struct.calcsize(_TRAJ_HEADER_FORMAT)
_ENERGY_TAG = b"ENRG"


def load_all_positions(filename: str) -> np.ndarray:
    """
    Load all particle positions from a binary .traj file.
    Returns (n_frames, n_particles, 3) float64.
    """
    with open(filename, "rb") as f:
        header = f.read(_TRAJ_HEADER_SIZE)
        magic, version, n_particles, frame_size, n_frames, _ = struct.unpack(
            _TRAJ_HEADER_FORMAT, header
        )
        if magic != b"CHRM":
            raise ValueError(f"Bad magic in {filename}")
        metadata_len = struct.unpack("<I", f.read(4))[0]
        data_start = _TRAJ_HEADER_SIZE + 4 + metadata_len
        f.seek(data_start)
        nbytes = n_frames * frame_size
        data = np.frombuffer(f.read(nbytes), dtype=np.float32)
        arr = data.reshape((n_frames, n_particles, 3)).astype(np.float64, copy=False)
        return arr

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

            # Read EXACTLY the number of bytes for positions (ignore any appended energy blocks)
            nbytes = n_frames * frame_size
            raw = f.read(nbytes)
            data = np.frombuffer(raw, dtype=np.float32)
            arr = data.reshape((n_frames, n_particles, 3)).astype(np.float64, copy=False)
            return jnp.asarray(arr)


def load_all_replicates(
    glob_path_or_files: Union[str, List[str], List[Path]], 
    discard_initial: int = 0
) -> List[np.ndarray]:
    """
    Load positions from multiple trajectory files.
    
    Args:
        glob_path_or_files: Either a glob pattern string (e.g., "rep*/trajectory.traj")
                           or a list of file paths (strings or Path objects)
        discard_initial: Number of initial frames to discard from each file
    
    Returns:
        List of numpy arrays, one per file, each with shape (n_frames, n_particles, 3)
    """
    # Handle both glob pattern and list of files
    if isinstance(glob_path_or_files, str):
        files = sorted(glob.glob(glob_path_or_files))
        if not files:
            raise FileNotFoundError(f"No files matched glob: {glob_path_or_files}")
    elif isinstance(glob_path_or_files, list):
        # Convert Path objects to strings if needed
        files = [str(fp) if isinstance(fp, Path) else fp for fp in glob_path_or_files]
        if not files:
            raise ValueError("Empty list of files provided")
    else:
        raise TypeError(
            f"Expected str or List[str/Path], got {type(glob_path_or_files).__name__}"
        )
    
    out: List[np.ndarray] = []
    for fp in files:
        arr = load_all_positions(fp)
        start = min(discard_initial, arr.shape[0])
        out.append(arr[start:])
    return out

def load_all_replicates_jax(
    glob_path_or_files: Union[str, List[str], List[Path]], 
    discard_initial: int = 0
) -> List[np.ndarray]:
    """
    Load positions from multiple trajectory files.
    
    Args:
        glob_path_or_files: Either a glob pattern string (e.g., "rep*/trajectory.traj")
                           or a list of file paths (strings or Path objects)
        discard_initial: Number of initial frames to discard from each file
    
    Returns:
        List of numpy arrays, one per file, each with shape (n_frames, n_particles, 3)
    """
    # Handle both glob pattern and list of files
    if isinstance(glob_path_or_files, str):
        files = sorted(glob.glob(glob_path_or_files))
        if not files:
            raise FileNotFoundError(f"No files matched glob: {glob_path_or_files}")
    elif isinstance(glob_path_or_files, list):
        # Convert Path objects to strings if needed
        files = [str(fp) if isinstance(fp, Path) else fp for fp in glob_path_or_files]
        if not files:
            raise ValueError("Empty list of files provided")
    else:
        raise TypeError(
            f"Expected str or List[str/Path], got {type(glob_path_or_files).__name__}"
        )
    
    out: List[np.ndarray] = []
    for fp in files:
        arr = load_all_positions_jax(fp)
        start = min(discard_initial, arr.shape[0])
        out.append(arr[start:])
    return out


def read_traj_energy_block(filename: str) -> Optional[dict]:
    """
    Read optional ENRG block appended to .traj. Returns dict or None.
    """
    with open(filename, "rb") as f:
        header = f.read(_TRAJ_HEADER_SIZE)
        magic, version, n_particles, frame_size, n_frames, _ = struct.unpack(
            _TRAJ_HEADER_FORMAT, header
        )
        if magic != b"CHRM":
            raise ValueError(f"Bad magic in {filename}")
        metadata_len = struct.unpack("<I", f.read(4))[0]
        positions_end = _TRAJ_HEADER_SIZE + 4 + metadata_len + n_frames * frame_size
        f.seek(positions_end)
        tag = f.read(4)
        if tag != _ENERGY_TAG:
            return None
        payload_len = struct.unpack("<I", f.read(4))[0]
        payload = f.read(payload_len)
        return json.loads(payload.decode("utf-8"))
