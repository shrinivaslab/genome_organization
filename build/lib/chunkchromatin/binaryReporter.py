import numpy as np
import struct
import json

class BinaryReporter:
    HEADER_FORMAT = "<4sBHII16s"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    MAGIC = b"CHRM"
    VERSION = 1
    RESERVED = b"\x00" * 16

    def __init__(self, filename, n_particles=1570, mode='w', metadata=None):
        self.filename = filename
        self.n_particles = n_particles
        self.frame_size = n_particles * 3 * 4
        self.frame_count = 0
        self.metadata = metadata or {}
        self.metadata_bytes = json.dumps(self.metadata).encode('utf-8')
        self.metadata_len = len(self.metadata_bytes)
        self.header_and_meta_offset = self.HEADER_SIZE + 4 + self.metadata_len

        if mode not in ('w',):
            raise ValueError("Only mode='w' is supported")

        self.file = open(filename, 'wb')

        # Write placeholder header and metadata
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

    def report(self, positions: np.ndarray):
        if not hasattr(self, 'file') or self.file is None or self.file.closed:
            raise RuntimeError("Cannot write to closed BinaryReporter")
        if positions.shape != (self.n_particles, 3):
            raise ValueError(f"Expected shape ({self.n_particles}, 3), got {positions.shape}")
        self.file.write(positions.astype(np.float32).tobytes())
        self.frame_count += 1

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
            # Rewrite metadata to ensure consistency
            self.file.seek(self.HEADER_SIZE)
            self.file.write(struct.pack("<I", self.metadata_len))
            self.file.write(self.metadata_bytes)
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
