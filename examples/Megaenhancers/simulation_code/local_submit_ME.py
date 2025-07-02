import os
from datetime import date
import json
from collections import defaultdict
import argparse

output_base = "/Users/kadendimarco/Desktop/Shrivinas_lab/genome_archetecture/Megaenhancers/simulation_code/sim_output"  # or scratch
monomer_types_path = "/Users/kadendimarco/Desktop/Shrivinas_lab/genome_archetecture/Megaenhancers/simulation_code/ME_monomer_types.npy"
forces_list = [
    #['heteropolymer_SSW'],
    #['harmonic_bonds','angle_force','spherical_confinement','polynomial_repulsive'],
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

def write_and_submit_job(forces, device):
    print(f"Preparing job for forces: {forces} (device: {device})")
    force_counts[tuple(forces)] += 1
    count = force_counts[tuple(forces)]

    #label = "_".join(forces)
    label = "All_forces_ME_sim"
    job_name = f"{label}_{today}" if count == 1 else f"{label}_{today}_{count}"
    job_dir = os.path.join(output_base, job_name)
    os.makedirs(job_dir, exist_ok=True)

    # Build force_kwargs for this job
    job_force_kwargs = {f: force_kwargs_dict.get(f, {}) for f in forces if force_kwargs_dict.get(f, {})}
    force_kwargs_json = json.dumps(job_force_kwargs)

    python_cmd = (
        f"python /Users/kadendimarco/Desktop/Shrivinas_lab/genome_archetecture/Megaenhancers/simulation_code/run_ME_sim.py "
        f"--monomer_types {monomer_types_path} "
        f"--output_dir {job_dir} "
        f"--forces {' '.join(forces)} "
        f"--force_kwargs '{force_kwargs_json}' "
        f"--device {device}"
    )

    print(f"Running: {python_cmd}")
    os.system(python_cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"], help="Device: cuda or cpu")
    args = parser.parse_args()

    try:
        for forces in forces_list:
            print(f"Submitting job for: {forces}")
            write_and_submit_job(forces, args.device)
    except Exception as e:
        print(f"Error during submission: {e}")