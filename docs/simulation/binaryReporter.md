# BinaryReporter Developer Guide

## Overview

`BinaryReporter` is a custom binary file writer for storing per-frame particle positions in molecular dynamics simulations. It stores:

- Particle positions as float32 arrays of shape (N_particles, 3)
- Simulation metadata in JSON format
- All data in a compact, versioned binary format

## File Format

### Structure

```
[Header (44 bytes)]
[Metadata length (4 bytes)]
[Metadata (JSON)]
[Frame 0: float32[N_particles, 3]]
[Frame 1]
...
```

### Header Layout

Format string: `<4sBHII16s`

| Field        | Type     | Size | Description                         |
|--------------|----------|------|-------------------------------------|
| `magic`      | `4s`     | 4    | File identifier (e.g., `b'CHRM'`)   |
| `version`    | `B`      | 1    | Format version (currently 1)        |
| `n_particles`| `H`      | 2    | Number of particles per frame       |
| `frame_size` | `I`      | 4    | Bytes per frame                     |
| `n_frames`   | `I`      | 4    | Number of frames (updated on close) |
| `reserved`   | `16s`    | 16   | Padding for future use              |

### Metadata

- Length prefix: 4-byte unsigned int
- UTF-8 encoded JSON string with simulation parameters

Example:
```json
{
  "temperature": 300,
  "dt_fs": 10,
  "N": 1570
}
```

## Writing Trajectories

```python
from binary_reporter import BinaryReporter
import numpy as np

metadata = {
    "temperature": 300,
    "dt_fs": 10,
    "N": 1570
}
filename = "rep01.traj"

with BinaryReporter(filename=filename, n_particles=1570, metadata=metadata) as reporter:
    for _ in range(4000):
        positions = np.random.rand(1570, 3).astype(np.float32)
        reporter.report(positions)
```

## Reading Trajectories

### Load Single Frame

```python
def load_single_frame(filename, frame_index):
    import struct, numpy as np
    HEADER_FORMAT = "<4sBHII16s"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

    with open(filename, 'rb') as f:
        header = f.read(HEADER_SIZE)
        magic, version, n_particles, frame_size, n_frames, _ = struct.unpack(HEADER_FORMAT, header)
        assert magic == b'CHRM'

        meta_len = struct.unpack("<I", f.read(4))[0]
        f.seek(HEADER_SIZE + 4 + meta_len + frame_index * frame_size)
        raw = f.read(frame_size)
        return np.frombuffer(raw, dtype=np.float32).reshape((n_particles, 3))
```

### Load Metadata

```python
def load_metadata(filename):
    import struct, json
    HEADER_SIZE = struct.calcsize("<4sBHII16s")
    with open(filename, 'rb') as f:
        f.seek(HEADER_SIZE)
        meta_len = struct.unpack("<I", f.read(4))[0]
        metadata = f.read(meta_len)
        return json.loads(metadata.decode('utf-8'))
```

## Limitations

- Appending not supported
- No compression
- No timestamps or auxiliary observables

## Design Principles

- Minimal overhead
- Binary-compatible versioning
- Self-contained files with metadata

## Future Extensions

- Append mode
- Compression support
- Frame-level observables
- Memory-mapped readers
