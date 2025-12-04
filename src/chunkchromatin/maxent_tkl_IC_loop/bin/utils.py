
import os, json, subprocess, shutil, time, math
from pathlib import Path
import numpy as np

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)

def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)

def parse_sbatch_jobid(output: str) -> str:
    # Typical: "Submitted batch job 123456"
    for tok in output.strip().split():
        if tok.isdigit():
            return tok
    # Fallback: try to find last integer
    ints = [s for s in output.split() if s.isdigit()]
    if ints:
        return ints[-1]
    raise RuntimeError(f"Could not parse job ID from sbatch output: {output}")

def sbatch_submit(script_path: Path, extra_args=None, env=None) -> str:
    cmd = ["sbatch"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(script_path))
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env or os.environ)
    if proc.returncode != 0:
        raise RuntimeError(f"sbatch failed: {proc.stderr.strip()}")
    return parse_sbatch_jobid(proc.stdout)

def vectorize_upper_tri(mat: np.ndarray) -> np.ndarray:
    K = mat.shape[0]
    iu = np.triu_indices(K)
    return mat[iu]

def devectorize_upper_tri(vec: np.ndarray, K: int) -> np.ndarray:
    mat = np.zeros((K, K), dtype=float)
    iu = np.triu_indices(K)
    mat[iu] = vec
    mat[(iu[1], iu[0])] = vec  # mirror
    return mat

def load_config(config_path: Path):
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def prepare_seeds(n_replicates: int, base: int) -> dict:
    # Deterministic, fixed across iterations
    rng = np.random.default_rng(base)
    seeds = rng.integers(low=1, high=2**31-1, size=n_replicates, dtype=np.int64).tolist()
    return {int(i): int(s) for i, s in enumerate(seeds)}

def human_time():
    import datetime as dt
    return dt.datetime.now().isoformat(timespec="seconds")

def delete_dir_if_exists(p: Path):
    if p.exists():
        shutil.rmtree(p)

def make_executable(p: Path):
    p.chmod(p.stat().st_mode | 0o111)

def format_iter(i: int) -> str:
    return f"iter_{i:03d}"

