# submit_jobs.py
import os
from datetime import date
import json
from collections import defaultdict
import argparse

output_base = "/gpfs/home/pkv4601/genome_architecture/chunkchromatin/heteropolymer_benchmark_output/"  # or scratch
monomer_types_path = "/gpfs/home/pkv4601/genome_architecture/monomer_types_6000monomer.npy"
forces_list = [
    #['heteropolymer_SSW'],
    #['harmonic_bonds','angle_force','spherical_confinement','polynomial_repulsive'],
    ['harmonic_bonds','angle_force','spherical_confinement','add_nonbonded_pair_potential','add_C_monomer_pinning','add_B_monomer_lamina_attraction'],
    ['harmonic_bonds','angle_force','spherical_confinement','add_nonbonded_pair_potential','add_C_monomer_pinning','add_B_monomer_lamina_attraction'],
    ['harmonic_bonds','angle_force','spherical_confinement','add_nonbonded_pair_potential','add_C_monomer_pinning','add_B_monomer_lamina_attraction'],
    ['harmonic_bonds','angle_force','spherical_confinement','add_nonbonded_pair_potential','add_C_monomer_pinning','add_B_monomer_lamina_attraction']
    #['custom_sticky_force','conf','cp','bl'],

    # ['conf', 'hp', 'cp', 'bl'],
    # ['hp', 'cp', 'bl'],
    # ['conf', 'cp', 'bl'],
    # ['conf', 'hp', 'bl'],
    # ['conf', 'hp', 'cp'],
    # ['conf', 'hp']
]

# Define kwargs for each force (edit as needed)
force_kwargs_dict = {
    "harmonic_bonds": {"bondLength": 1.0, "bondWiggleDistance": 0.05},
    "angle_force": {"k": 1.5},
    "spherical_confinement": {},
    "polynomial_repulsive": {},
    "add_nonbonded_pair_potential": {},
    "add_C_monomer_pinning": {},
    "add_B_monomer_lamina_attraction": {},
}

today = date.today().strftime('%Y%m%d')

# Keep track of how many times each force combination has been used
force_counts = defaultdict(int)

def write_and_submit_job(forces, mode, device):
    print(f"Preparing job for forces: {forces} (mode: {mode}, device: {device})")
    force_counts[tuple(forces)] += 1
    count = force_counts[tuple(forces)]

    label = "_".join(forces)
    job_name = f"{label}_{today}" if count == 1 else f"{label}_{today}_{count}"
    job_dir = os.path.join(output_base, job_name)
    os.makedirs(job_dir, exist_ok=True)

    # Build force_kwargs for this job
    job_force_kwargs = {f: force_kwargs_dict.get(f, {}) for f in forces if force_kwargs_dict.get(f, {})}
    force_kwargs_json = json.dumps(job_force_kwargs)

    python_cmd = (
        f"python /gpfs/home/pkv4601/genome_architecture/chunkchromatin/run_heteropolymer_benchmarks.py "
        f"--monomer_types {monomer_types_path} "
        f"--output_dir {job_dir} "
        f"--forces {' '.join(forces)} "
        f"--force_kwargs '{force_kwargs_json}' "
        f"--device {device}"
    )

    if mode == "local":
        print(f"[LOCAL] Running: {python_cmd}")
        os.system(python_cmd)
    elif mode == "sbatch":
        sbatch_path = os.path.join(job_dir, "run.sh")
        with open(sbatch_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --job-name={job_name}\n")
            f.write("#SBATCH --account=p32733\n")
            if device == "cuda":
                f.write("#SBATCH --partition=gengpu\n")
                f.write("#SBATCH --gres=gpu:a100:1\n")
            elif device == "cpu":
                f.write("#SBATCH --partition=short\n")
            else:
                raise ValueError(f"Unknown device: {device}")
            f.write("#SBATCH -n 1\n")
            f.write("#SBATCH -N 1\n")
            f.write("#SBATCH -t 4:00:00\n")
            f.write("#SBATCH --mem=1GB\n")
            f.write(f"#SBATCH -o {job_dir}/%x_%j.out\n")
            f.write(f"#SBATCH -e {job_dir}/%x_%j.err\n")

            f.write("module purge all\n")
            f.write('eval "$(/gpfs/home/pkv4601/micromamba/bin/micromamba shell hook --shell bash)"\n')
            f.write("/gpfs/home/pkv4601/micromamba/bin/micromamba activate /gpfs/home/pkv4601/micromamba/envs/polychrom\n")
            f.write("\n")
            f.write(python_cmd + "\n")
        print(f"[SBATCH] Submitting: {sbatch_path} (device: {device})")
        os.system(f"sbatch {sbatch_path}")
    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="sbatch", choices=["sbatch", "local"], help="Run mode: sbatch (default) or local")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", None], help="Device: cuda or cpu. If not set, will be inferred from mode.")
    args = parser.parse_args()

    # Device selection logic
    # If mode is sbatch, default to cuda; if local, default to cpu
    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if args.mode == "sbatch" else "cpu"

    try:
        for forces in forces_list:
            print(f"Submitting job for: {forces}")
            write_and_submit_job(forces, args.mode, device)
    except Exception as e:
        print(f"Error during submission: {e}")
