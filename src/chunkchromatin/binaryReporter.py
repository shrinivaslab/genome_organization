# binaryReporter.py
# -----------------------------------------------------------------------------
# This file contains the BinaryReporter class, which is used to write the trajectory data to a binary file.
# -----------------------------------------------------------------------------

import numpy as np
import struct
import json

class BinaryReporter:
    HEADER_FORMAT = "<4sBHII16s"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    MAGIC = b"CHRM"
    VERSION = 1
    RESERVED = b"\x00" * 16

    # tagged appendix for aux data (energies, etc.)
    ENERGY_TAG = b"ENRG"   # 4 bytes

    def __init__(self, filename, n_particles=1570, mode='w', metadata=None):
        self.filename = filename
        self.n_particles = n_particles
        self.frame_size = n_particles * 3 * 4   # float32 xyz per particle
        self.frame_count = 0
        self.metadata = metadata or {}
        self.metadata_bytes = json.dumps(self.metadata).encode('utf-8')
        self.metadata_len = len(self.metadata_bytes)

        if mode not in ('w',):
            raise ValueError("Only mode='w' is supported")

        # buffer for optional per-frame energies; keep same length as frames
        # each entry is either None or a dict like:
        # {"time_ns": ..., "potential_total_kJmol": ..., "kinetic_total_kJmol": ..., "potential_by_force_kJmol": {...}}
        self._energy_frames = []

        self.file = open(filename, 'wb')

        # Write placeholder header and metadata (frame count patched on close)
        header = struct.pack(
            self.HEADER_FORMAT,
            self.MAGIC,
            self.VERSION,
            self.n_particles,
            self.frame_size,
            0,  # placeholder for frame count
            self.RESERVED
        )
        self.file.write(header)
        self.file.write(struct.pack("<I", self.metadata_len))
        self.file.write(self.metadata_bytes)

    def _write_frame_positions(self, positions: np.ndarray):
        if positions.shape != (self.n_particles, 3):
            raise ValueError(f"Expected shape ({self.n_particles}, 3), got {positions.shape}")
        self.file.write(positions.astype(np.float32).tobytes())
        self.frame_count += 1

    def _record_energy(self, energy_payload):
        # Keep list aligned with frames; always append one entry per call
        self._energy_frames.append(energy_payload if energy_payload is not None else None)

    def report(self, *args):
        """
        Supported calls:
          - report(positions: np.ndarray)
          - report(result_dict) where result_dict has keys:
                'pos' (required), and optionally 'energy_breakdown', 'time', etc.
          - report(name, data): tolerated; if data is dict, handled as above; if array, treated as positions.
        """
        if not hasattr(self, 'file') or self.file is None or self.file.closed:
            raise RuntimeError("Cannot write to closed BinaryReporter")

        # Normalize inputs
        positions = None
        energy_payload = None

        if len(args) == 1:
            data = args[0]
            if isinstance(data, dict):
                # dict style: expect 'pos'
                if 'pos' not in data:
                    raise ValueError("dict passed to BinaryReporter.report must contain key 'pos'")
                positions = np.asarray(data['pos'])
                # collect energies if present
                if 'energy_breakdown' in data:
                    # allow attaching a minimal per-frame record; include time if available
                    eb = data['energy_breakdown']
                    energy_payload = {
                        "time_ns": data.get("time", None),
                        "potential_total_kJmol": eb.get("potential_total_kJmol"),
                        "kinetic_total_kJmol": eb.get("kinetic_total_kJmol"),
                        "potential_by_force_kJmol": eb.get("potential_by_force_kJmol"),
                    }
                else:
                    energy_payload = None
            else:
                # legacy array style
                positions = np.asarray(data)
                energy_payload = None

        elif len(args) == 2:
            _name, data = args
            if isinstance(data, dict):
                if 'pos' in data:
                    positions = np.asarray(data['pos'])
                    if 'energy_breakdown' in data:
                        eb = data['energy_breakdown']
                        energy_payload = {
                            "time_ns": data.get("time", None),
                            "potential_total_kJmol": eb.get("potential_total_kJmol"),
                            "kinetic_total_kJmol": eb.get("kinetic_total_kJmol"),
                            "potential_by_force_kJmol": eb.get("potential_by_force_kJmol"),
                        }
                    else:
                        energy_payload = None
                else:
                    # Named, non-frame metadata (e.g., "initArgs"): ignore gracefully
                    return
            else:
                # Treat as legacy "(name, positions)" form
                positions = np.asarray(data)
                energy_payload = None
        else:
            raise TypeError("BinaryReporter.report expects 1 or 2 arguments")

        if positions is None:
            raise ValueError("BinaryReporter.report could not find positions to write")
        self._write_frame_positions(positions)
        self._record_energy(energy_payload)


    def _append_energy_block(self):
        """
        Append a tagged ENRG block containing a JSON object:
          {
            "schema": "per_frame_energy_v1",
            "n_frames": <int>,
            "frames": [ null | {time_ns:..., potential_total_kJmol:..., kinetic_total_kJmol:..., potential_by_force_kJmol:{...}}, ... ]
          }
        """
        # Do not write an ENRG block if no energies were provided (default behavior)
        if not any(frame is not None for frame in self._energy_frames):
            return

        payload = {
            "schema": "per_frame_energy_v1",
            "n_frames": self.frame_count,
            "frames": self._energy_frames,
        }
        payload_bytes = json.dumps(payload).encode("utf-8")
        self.file.write(self.ENERGY_TAG)
        self.file.write(struct.pack("<I", len(payload_bytes)))
        self.file.write(payload_bytes)

    def close(self):
        if hasattr(self, 'file') and self.file and not self.file.closed:
            # Overwrite the header with the correct frame count
            self.file.seek(0)
            header = struct.pack(
                self.HEADER_FORMAT,
                self.MAGIC,
                self.VERSION,
                self.n_particles,
                self.frame_size,
                self.frame_count,
                self.RESERVED
            )
            self.file.write(header)
            # Rewrite original metadata (length stays identical)
            self.file.seek(self.HEADER_SIZE)
            self.file.write(struct.pack("<I", self.metadata_len))
            self.file.write(self.metadata_bytes)

            # Seek to end of positions region and append energy block
            # (positions occupy HEADER_SIZE + 4 + metadata_len + frame_count*frame_size bytes)
            positions_end = self.HEADER_SIZE + 4 + self.metadata_len + self.frame_count * self.frame_size
            self.file.seek(positions_end)
            self._append_energy_block()

            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()




# #how to get positions:
# def load_all_positions(filename):
#     """
#     Load all particle positions from a binary .traj file.
#     Returns (n_frames, n_particles, 3) float32 -> float64
#     Ignores any tagged auxiliary blocks appended after the positions.
#     """
#     import numpy as np
#     import struct

#     HEADER_FORMAT = "<4sBHII16s"
#     HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
#     with open(filename, 'rb') as f:
#         header = f.read(HEADER_SIZE)
#         magic, version, n_particles, frame_size, n_frames, _ = struct.unpack(HEADER_FORMAT, header)
#         assert magic == b'CHRM'
#         metadata_len = struct.unpack("<I", f.read(4))[0]
#         # Positions stream begins after header+metadata
#         data_start = HEADER_SIZE + 4 + metadata_len
#         f.seek(data_start)
#         nbytes = n_frames * frame_size
#         raw = f.read(nbytes)  # read EXACTLY positions bytes
#         data = np.frombuffer(raw, dtype=np.float32)
#         return data.reshape((n_frames, n_particles, 3)).astype(np.float64, copy=False)


# load energies
# def load_all_energies(filename):
#     """
#     Load all per-frame energy data from a binary .traj file written by BinaryReporter.

#     Returns:
#         dict with keys:
#             - "schema": str
#             - "n_frames": int
#             - "frames": list of dicts (or None where no energy was recorded)

#     Raises:
#         RuntimeError if no ENRG block is found in the file.
#     """
#     import struct, json

#     ENERGY_TAG = b"ENRG"
#     HEADER_FORMAT = "<4sBHII16s"
#     HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

#     with open(filename, "rb") as f:
#         # --- read and skip header + metadata ---
#         header = f.read(HEADER_SIZE)
#         if len(header) < HEADER_SIZE:
#             raise RuntimeError("File too small or corrupted header.")
#         magic, version, n_particles, frame_size, n_frames, _ = struct.unpack(HEADER_FORMAT, header)
#         if magic != b"CHRM":
#             raise RuntimeError("Invalid .traj file (bad magic number).")

#         metadata_len = struct.unpack("<I", f.read(4))[0]
#         f.seek(HEADER_SIZE + 4 + metadata_len + n_frames * frame_size)

#         # --- scan for the ENRG tag ---
#         tag = f.read(4)
#         if tag != ENERGY_TAG:
#             raise RuntimeError(f"No energy block found in file '{filename}'. (Tag {tag!r})")

#         length_bytes = f.read(4)
#         if len(length_bytes) != 4:
#             raise RuntimeError("Corrupted energy block header (missing length).")
#         payload_len = struct.unpack("<I", length_bytes)[0]

#         payload_bytes = f.read(payload_len)
#         if len(payload_bytes) != payload_len:
#             raise RuntimeError("Corrupted energy block (incomplete payload).")

#         try:
#             payload = json.loads(payload_bytes.decode("utf-8"))
#         except Exception as e:
#             raise RuntimeError(f"Failed to parse energy JSON payload: {e}")

#         return payload

