import json
from pathlib import Path
import numpy as np
import argparse
import subprocess
import sys

def setup_initial_directory_structure(config):

    run_root = Path(config['run']['output_dir'])
    #create run root directory
    run_root.mkdir(parents=True, exist_ok=True)

    #initialize forces depending on which terms should be fit
    n_types = config['monomer_types']['n_types']

    #copy initial parameters into initial params directory
    initial_params_dir = run_root / "initial_params"
    initial_params_dir.mkdir(parents=True, exist_ok=True)

    # initialize parameters as all zeros
    # only saved initial params for the fit parameters in /initial_params
    # save all in iter_000/params

    initial_interaction_matrix = np.zeros((n_types, n_types))
    initial_loop_X = 0.0
    IC_shape = config["learned_forces"]["ideal_chromosome"]["d_end"] - config["learned_forces"]["ideal_chromosome"]["d_init"]
    initial_lambda_IC = np.zeros(IC_shape)

    if config["fit"]["fit_tkl"]:
        np.save(initial_params_dir / "initial_interaction_matrix.npy", initial_interaction_matrix)
    if config["fit"]["fit_loop"]:
        np.save(initial_params_dir / "initial_loop_X.npy", initial_loop_X)
    if config["fit"]["fit_ic"]:
        np.save(initial_params_dir / "initial_lambda_IC.npy", initial_lambda_IC)
    

    #create iteration directory
    iter_dir = run_root / "iter_000"
    iter_dir.mkdir(parents=True, exist_ok=True)
    # Create /params directory in iter_dir and save initial parameters
    params_dir = iter_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    np.save(params_dir / "interaction_matrix.npy", initial_interaction_matrix)
    np.save(params_dir / "loop_X.npy", initial_loop_X)
    np.save(params_dir / "lambda_IC.npy", initial_lambda_IC)

    #create simulation directory
    sim_dir = iter_dir / "sims"
    sim_dir.mkdir(parents=True, exist_ok=True)

    #create logs directory
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

def is_run_root_empty(run_root):
    """Check if run_root is effectively empty (only has logs or empty dirs)."""
    run_root = Path(run_root)
    if not run_root.exists():
        return True
    
    # Check for meaningful content
    has_iter_dirs = any((run_root / f"iter_{i:03d}").exists() for i in range(100))
    has_manifest = (run_root / "run_manifest.json").exists()
    has_initial_params = (run_root / "initial_params").exists()
    
    # If any of these exist, it's not empty
    return not (has_iter_dirs or has_manifest or has_initial_params)

def update_manifest_job_id(manifest_path, iter_num, step, job_id):
    """Update manifest with job_id for a step."""
    manifest = json.loads(manifest_path.read_text())
    iter_key = f"iter_{iter_num:03d}"
    
    if "iterations" not in manifest:
        manifest["iterations"] = {}
    if iter_key not in manifest["iterations"]:
        manifest["iterations"][iter_key] = {}
    if step not in manifest["iterations"][iter_key]:
        manifest["iterations"][iter_key][step] = {}
    
    manifest["iterations"][iter_key][step]["job_id"] = job_id
    manifest_path.write_text(json.dumps(manifest, indent=2))

def submit_step(config_path, run_root, iter_num, step):
    """Submit a step by calling the appropriate submit script."""
    script_dir = Path(__file__).parent
    
    if step == "sim":
        submit_script = script_dir / "submit_simulations.py"
        cmd = [
            "python", str(submit_script),
            "--config", str(config_path),
            "--run-root", str(run_root),
            "--iter", str(iter_num)
        ]
    elif step == "obs":
        submit_script = script_dir / "sumbit_energy_observables.py"
        cmd = [
            "python", str(submit_script),
            "--config", str(config_path),
            "--run-root", str(run_root),
            "--iter", str(iter_num)
        ]
    elif step == "reweight":
        submit_script = script_dir / "submit_reweight.py"
        cmd = [
            "python", str(submit_script),
            "--config", str(config_path),
            "--run-root", str(run_root),
            "--iter", str(iter_num)
        ]
    else:
        raise ValueError(f"Unknown step: {step}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: Failed to submit {step} job:")
        print(result.stderr)
        sys.exit(1)
    
    # Parse job_id from output
    output_lines = result.stdout.split('\n')
    job_id = None
    for line in output_lines:
        if "Submitted" in line and "job:" in line:
            job_id = line.split("job:")[-1].strip()
            break
    
    if job_id:
        update_manifest_job_id(run_root / "run_manifest.json", iter_num, step, job_id)
        print(f"Updated manifest: {step} job_id = {job_id}")
    else:
        print(f"WARNING: Could not parse job_id from output")

def main():
    parser = argparse.ArgumentParser(
        description="Driver for DiffTRE fit workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Start new run
        python driver.py --config /path/to/config.json
        
        # Resume from specific step
        python driver.py --config /path/to/config.json --resume iter000,sim
        python driver.py --config /path/to/config.json --resume iter001,obs
        python driver.py --config /path/to/config.json --resume iter001,reweight
        """
    )
    parser.add_argument("--config", required=True, help="Path to simulation configuration JSON file")
    parser.add_argument("--resume", help="Resume from specific step: iter###,step (e.g., iter000,sim)")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = json.loads(config_path.read_text())
    run_root = Path(config['run']['output_dir'])
    run_manifest_path = run_root / 'run_manifest.json'
    
    # Determine which step to submit
    iter_num = None
    step = None
    
    if not run_manifest_path.exists():
        # New run: initialize and start at iter000
        if not is_run_root_empty(run_root):
            raise RuntimeError(
                f"Run root directory is not empty: {run_root}\n"
                f"Use --resume to continue an existing run, or clear the directory to start fresh."
            )
        
        run_manifest = {
            'run_name': config['run']['name'],
            'config_path': str(config_path),
            'run_root': str(run_root),
            'n_iters': config['fit']['n_iters'],
            'iterations': {
                'iter_000': {
                    'sim': {'state': 'pending'},
                    'obs': {'state': 'pending'},
                    'reweight': {'state': 'pending'}
                }
            }
        }
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2))
        setup_initial_directory_structure(config)
        
        # Start at iter000, sim step
        iter_num = 0
        step = "sim"
        
    else:
        # Existing run
        run_manifest = json.loads(run_manifest_path.read_text())
        
        if args.resume:
            # Parse resume argument: iter###,step
            try:
                iter_str, step = args.resume.split(',')
                iter_num = int(iter_str.replace('iter', ''))
                if step not in ['sim', 'obs', 'reweight']:
                    raise ValueError(f"Invalid step: {step}. Must be sim, obs, or reweight")
            except ValueError as e:
                print(f"ERROR: Invalid --resume format: {args.resume}")
                print("Expected format: iter###,step (e.g., iter000,sim)")
                sys.exit(1)
        else:
            # No --resume flag: error if run_root is not empty
            if not is_run_root_empty(run_root):
                raise RuntimeError(
                    f"Run root directory is not empty: {run_root}\n"
                    f"Use --resume iter###,step to continue an existing run."
                )
            # If empty, treat as new run
            run_manifest = {
                'run_name': config['run']['name'],
                'config_path': str(config_path),
                'run_root': str(run_root),
                'n_iters': config['fit']['n_iters'],
                'iterations': {
                    'iter_000': {
                        'sim': {'state': 'pending'},
                        'obs': {'state': 'pending'},
                        'reweight': {'state': 'pending'}
                    }
                }
            }
            run_manifest_path.write_text(json.dumps(run_manifest, indent=2))
            setup_initial_directory_structure(config)
            iter_num = 0
            step = "sim"
    
    # Submit the step
    print(f"Submitting {step} job for iteration {iter_num}...")
    s(config_path, run_root, iter_num, step)
    print(f"Driver complete. {step} job submitted for iteration {iter_num}.")





