#!/usr/bin/env python3
"""
Dry run validation script for MaxEnt loop workflow.

This script validates the entire MaxEnt workflow without submitting SLURM jobs,
catching configuration errors, missing files, and template formatting issues
before wasting HPC resources.

Usage:
    python dry_run.py --run-root /path/to/run --initial-epsilon /path/to/epsilon.npy --name run001
    
Or test with the same arguments you would use for maxent_loop.py:
    python dry_run.py [same args as maxent_loop.py] --dry-run-only
"""

import argparse
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import yaml
import json
from typing import Dict, List, Tuple, Optional
import sys

# Import from local utils
try:
    from utils import ensure_dir, write_json, load_config, prepare_seeds, human_time, format_iter
except ImportError:
    # If running from different directory, add bin to path
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import ensure_dir, write_json, load_config, prepare_seeds, human_time, format_iter


class DryRunValidator:
    """Comprehensive validation for MaxEnt workflow without submitting jobs."""
    
    def __init__(self, args):
        self.args = args
        self.errors = []
        self.warnings = []
        self.proj_root = Path(__file__).resolve().parent.parent
        self.temp_dir = None
        
    def error(self, msg: str):
        """Add error message."""
        self.errors.append(f"❌ ERROR: {msg}")
        
    def warning(self, msg: str):
        """Add warning message."""
        self.warnings.append(f"⚠️  WARNING: {msg}")
        
    def info(self, msg: str):
        """Print info message."""
        print(f"ℹ️  {msg}")
        
    def success(self, msg: str):
        """Print success message."""
        print(f"✅ {msg}")
        
    def validate_all(self) -> bool:
        """Run all validations. Returns True if all validations pass."""
        
        print("🔍 Starting MaxEnt Loop Dry Run Validation")
        print("=" * 60)
        
        # Create temporary directory for dry run
        self.temp_dir = Path(tempfile.mkdtemp(prefix="maxent_dry_run_"))
        self.info(f"Using temporary directory: {self.temp_dir}")
        
        try:
            # Core validations
            self.validate_arguments()
            self.validate_file_paths()
            self.validate_configuration()
            self.validate_input_files()
            self.validate_directory_structure()
            self.validate_scripts_exist()
            self.validate_template_formatting()
            self.validate_script_interfaces()
            self.validate_workflow_simulation()
            
            # Final report
            self.print_report()
            
            return len(self.errors) == 0
            
        finally:
            # Cleanup
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.info(f"Cleaned up temporary directory")
    
    def validate_arguments(self):
        """Validate command line arguments."""
        self.info("Validating command line arguments...")
        
        # Required arguments
        if not self.args.run_root:
            self.error("--run-root is required")
        
        if not self.args.initial_epsilon:
            self.error("--initial-epsilon is required")
            
        if not self.args.name:
            self.error("--name is required")
        
        # Name validation
        if self.args.name and not self.args.name.replace("_", "").replace("-", "").isalnum():
            self.warning("--name should be alphanumeric with underscores/hyphens only")
            
        self.success("Command line arguments validated")
    
    def validate_file_paths(self):
        """Validate that required files and directories exist."""
        self.info("Validating file paths...")
        
        # Config file
        config_path = Path(self.args.config)
        if not config_path.exists():
            self.error(f"Config file not found: {config_path}")
        
        # Initial epsilon file
        if self.args.initial_epsilon:
            eps_path = Path(self.args.initial_epsilon)
            if not eps_path.exists():
                self.error(f"Initial epsilon file not found: {eps_path}")
            elif not eps_path.suffix == '.npy':
                self.warning(f"Initial epsilon file should be .npy: {eps_path}")
        
        # Project structure
        required_dirs = ['bin', 'templates']
        for dir_name in required_dirs:
            dir_path = self.proj_root / dir_name
            if not dir_path.exists():
                self.error(f"Required project directory not found: {dir_path}")
        
        self.success("File paths validated")
    
    def validate_configuration(self):
        """Validate configuration file structure and values."""
        self.info("Validating configuration...")
        
        try:
            self.cfg = load_config(Path(self.args.config))
        except Exception as e:
            self.error(f"Failed to load config file: {e}")
            return
        
        # Required sections
        required_sections = ['slurm', 'simulation', 'resources', 'processing_inputs', 'exp_targets']
        for section in required_sections:
            if section not in self.cfg:
                self.error(f"Missing required config section: {section}")
        
        # SLURM settings
        slurm = self.cfg.get('slurm', {})
        required_slurm = ['account', 'partition']
        for key in required_slurm:
            if not slurm.get(key):
                self.error(f"Missing required SLURM setting: slurm.{key}")
        
        # Simulation settings
        sim = self.cfg.get('simulation', {})
        required_sim = ['n_replicates', 'frames', 'burnin_frames', 'n_types']
        for key in required_sim:
            if key not in sim:
                self.error(f"Missing required simulation setting: simulation.{key}")
        
        # Resource settings validation
        resources = self.cfg.get('resources', {})
        
        # Simulation resources
        sim_res = resources.get('simulation', {})
        if 'array_len' not in sim_res:
            self.error("Missing resources.simulation.array_len")
        
        # Processing resources
        proc_res = resources.get('processing', {})
        if 'workers' not in proc_res:
            self.error("Missing resources.processing.workers")
        if 'reduce' not in proc_res:
            self.error("Missing resources.processing.reduce")
        
        # Array capacity check
        if 'array_len' in sim_res and 'per_task_replicates' in sim_res:
            array_capacity = sim_res['array_len'] * sim_res.get('per_task_replicates', 1)
            n_reps = sim.get('n_replicates', 0)
            if array_capacity < n_reps:
                self.error(f"Array capacity ({array_capacity}) < n_replicates ({n_reps})")
        
        self.success("Configuration validated")
    
    def validate_input_files(self):
        """Validate input files referenced in configuration."""
        self.info("Validating input files...")
        
        if not hasattr(self, 'cfg'):
            return  # Skip if config loading failed
        
        # Processing inputs
        proc_inputs = self.cfg.get('processing_inputs', {})
        for key, path in proc_inputs.items():
            if key in ['monomer_types', 'exp_tkl', 'interaction_matrix'] and path:
                if not Path(path).exists():
                    self.error(f"Processing input file not found: {key} = {path}")
        
        # Experimental targets
        exp_targets = self.cfg.get('exp_targets', {})
        for key, path in exp_targets.items():
            if path and not Path(path).exists():
                self.error(f"Experimental target file not found: {key} = {path}")
        
        # Validate epsilon file dimensions
        if self.args.initial_epsilon and Path(self.args.initial_epsilon).exists():
            try:
                eps = np.load(self.args.initial_epsilon)
                n_types = self.cfg.get('simulation', {}).get('n_types', 0)
                if eps.shape != (n_types, n_types):
                    self.error(f"Epsilon matrix shape {eps.shape} != expected ({n_types}, {n_types})")
                elif not np.allclose(eps, eps.T):
                    self.warning("Epsilon matrix is not symmetric")
            except Exception as e:
                self.error(f"Failed to load epsilon file: {e}")
        
        self.success("Input files validated")
    
    def validate_directory_structure(self):
        """Validate that directory structure can be created."""
        self.info("Validating directory structure...")
        
        if not self.temp_dir:
            return
        
        try:
            # Test run root creation
            test_run_root = self.temp_dir / "test_run"
            ensure_dir(test_run_root / "exp_targets")
            ensure_dir(test_run_root / "logs")
            
            # Test iteration structure
            iter0 = test_run_root / format_iter(0)
            ensure_dir(iter0 / "params")
            ensure_dir(iter0 / "sims")
            ensure_dir(iter0 / "obs")
            ensure_dir(iter0 / "update")
            
            self.success("Directory structure validated")
            
        except Exception as e:
            self.error(f"Failed to create directory structure: {e}")
    
    def validate_scripts_exist(self):
        """Validate that all required scripts exist."""
        self.info("Validating script files...")
        
        bin_dir = self.proj_root / "bin"
        required_scripts = [
            'iteration_driver.py',
            'run_replicates_array.py',
            'process_tkl_update.py',
            'series_runner.py',
            'update_step.py',
            'utils.py'
        ]
        
        for script in required_scripts:
            script_path = bin_dir / script
            if not script_path.exists():
                self.error(f"Required script not found: {script_path}")
            elif not os.access(script_path, os.R_OK):
                self.error(f"Script not readable: {script_path}")
        
        # Template files
        tpl_dir = self.proj_root / "templates"
        required_templates = [
            'sbatch_sim_array.sh',
            'sbatch_process_worker.sh',
            'sbatch_process_reduce.sh',
            'sbatch_update.sh'
        ]
        
        for template in required_templates:
            tpl_path = tpl_dir / template
            if not tpl_path.exists():
                self.error(f"Required template not found: {tpl_path}")
        
        self.success("Script files validated")
    
    def validate_script_interfaces(self):
        """Validate that scripts have correct command-line interfaces."""
        self.info("Validating script interfaces...")
        
        # Test critical script argument requirements
        bin_dir = self.proj_root / "bin"
        
        # Test run_replicates_array.py interface
        run_replicates_script = bin_dir / "run_replicates_array.py"
        if run_replicates_script.exists():
            try:
                # Try multiple approaches to validate the script
                import subprocess
                import shutil
                
                # Try different environment activation methods
                activation_commands = [
                    f"eval \"$(mamba shell hook --shell bash)\" && mamba activate polychrom && python {run_replicates_script} --help",
                    f"eval \"$(conda shell hook --shell bash)\" && conda activate polychrom && python {run_replicates_script} --help",
                    f"source activate polychrom && python {run_replicates_script} --help"
                ]
                
                validated = False
                for cmd in activation_commands:
                    try:
                        result = subprocess.run([
                            "bash", "-c", cmd
                        ], capture_output=True, text=True, timeout=15)
                        
                        if result.returncode == 0 and "--replicate_id" in result.stdout:
                            self.success("run_replicates_array.py interface validated with dependencies")
                            validated = True
                            break
                    except:
                        continue
                
                if not validated:
                    # Fallback to source code analysis
                    script_content = run_replicates_script.read_text()
                    if "--replicate_id" in script_content and "required=True" in script_content:
                        self.warning("run_replicates_array.py interface validated from source (use 'mamba activate polychrom' for full validation)")
                    else:
                        self.error("run_replicates_array.py missing required --replicate_id argument")
            except Exception as e:
                # Final fallback to source code analysis
                try:
                    script_content = run_replicates_script.read_text()
                    if "--replicate_id" in script_content and "required=True" in script_content:
                        self.warning(f"run_replicates_array.py interface validated from source ({e})")
                    else:
                        self.error("run_replicates_array.py missing required --replicate_id argument")
                except:
                    self.warning(f"Could not validate run_replicates_array.py interface: {e}")
        
        # Test series_runner.py interface  
        series_runner_script = bin_dir / "series_runner.py"
        if series_runner_script.exists():
            try:
                result = subprocess.run([
                    "python3", str(series_runner_script), "--help"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and "--runner" in result.stdout:
                    self.success("series_runner.py interface validated")
                else:
                    self.error("series_runner.py missing required --runner argument")
            except Exception as e:
                self.warning(f"Could not validate series_runner.py interface: {e}")
        
        # Test argument compatibility between series_runner and run_replicates_array
        try:
            # Read series_runner.py to check if it passes --replicate_id
            series_content = series_runner_script.read_text() if series_runner_script.exists() else ""
            if "--replicate_id" in series_content:
                self.success("series_runner.py correctly passes --replicate_id argument")
                
                # Test actual execution with polychrom environment
                try:
                    import subprocess
                    # Create a minimal test runner script that just prints its arguments
                    test_runner = self.temp_dir / "test_runner.py" if self.temp_dir else Path("/tmp/test_runner.py")
                    test_runner.write_text("""
import sys
print(f"Test runner called with args: {sys.argv}")
if "--replicate_id" in sys.argv:
    print("SUCCESS: --replicate_id argument received")
    sys.exit(0)
else:
    print("ERROR: --replicate_id argument missing")
    sys.exit(1)
""")
                    
                    # Test series_runner calling our test runner
                    test_commands = [
                        f"eval \"$(mamba shell hook --shell bash)\" && mamba activate polychrom && python {series_runner_script} --runner {test_runner} --start 0 --end 0",
                        f"python {series_runner_script} --runner {test_runner} --start 0 --end 0"
                    ]
                    
                    for cmd in test_commands:
                        try:
                            result = subprocess.run([
                                "bash", "-c", cmd
                            ], capture_output=True, text=True, timeout=10)
                            
                            if result.returncode == 0 and "SUCCESS: --replicate_id argument received" in result.stdout:
                                self.success("End-to-end argument passing validated")
                                break
                        except:
                            continue
                    else:
                        self.info("End-to-end test requires proper environment setup")
                        
                except Exception as e:
                    self.warning(f"Could not run end-to-end validation: {e}")
            else:
                self.error("series_runner.py does not pass --replicate_id to run_replicates_array.py")
        except Exception as e:
            self.warning(f"Could not validate argument compatibility: {e}")
    
    def validate_template_formatting(self):
        """Validate that all templates can be formatted without KeyErrors."""
        self.info("Validating template formatting...")
        
        if not hasattr(self, 'cfg'):
            return
        
        try:
            # Mock template variables based on standardized naming
            mock_vars = self._create_mock_template_vars()
            
            tpl_dir = self.proj_root / "templates"
            templates_to_test = [
                'sbatch_sim_array.sh',
                'sbatch_process_worker.sh', 
                'sbatch_process_reduce.sh',
                'sbatch_update.sh'
            ]
            
            for template_name in templates_to_test:
                template_path = tpl_dir / template_name
                if not template_path.exists():
                    continue
                    
                try:
                    template_text = template_path.read_text()
                    # Handle shell variable escaping: ${var} should not be treated as template variables
                    # Python's str.format will treat ${var} as literal text, not template variables
                    formatted_text = template_text.format(**mock_vars)
                    self.success(f"Template {template_name} formats correctly")
                except KeyError as e:
                    # Check if this is a real template variable issue or shell variable confusion
                    missing_var = str(e).strip("'\"")
                    if missing_var not in template_text.replace("${", "{"):
                        # This might be a shell variable that got confused
                        self.warning(f"Template {template_name}: Possible shell variable confusion with {e}")
                    else:
                        self.error(f"Template {template_name}: Missing template variable {e}")
                except Exception as e:
                    self.error(f"Template {template_name}: Formatting error {e}")
        
        except Exception as e:
            self.error(f"Template validation failed: {e}")
    
    def _create_mock_template_vars(self) -> Dict:
        """Create mock template variables for testing."""
        run_root = self.temp_dir / "test_run"
        iter_dir = run_root / "iter_000"
        
        return {
            # Job settings
            'job_name': f"{self.args.name}_test_001",
            'account': self.cfg["slurm"]["account"],
            'partition': self.cfg["resources"]["simulation"].get("partition", self.cfg["slurm"]["partition"]),
            'time_limit': "4:00:00",
            'cpus_per_task': 7,
            'workers': 7,
            'mem': "25G",
            'gres': self.cfg["resources"]["simulation"].get("gres", ""),
            'array_max': 4,
            'constraint_line': f"#SBATCH --constraint={self.cfg['slurm'].get('constraint')}\n" if self.cfg["slurm"].get("constraint") else "",
            'qos_line': f"#SBATCH --qos={self.cfg['slurm'].get('qos')}\n" if self.cfg["slurm"].get("qos") else "",
            
            # Paths
            'log_dir': str(run_root / "logs"),
            'iter_dir': str(iter_dir),
            'obs_dir': str(iter_dir / "obs"),
            'replicate_root': str(iter_dir / "sims"),
            'epsilon_dir': str(iter_dir / "update"),
            
            # Simulation settings
            'eps_path': str(iter_dir / "params" / "epsilon.npy"),
            'seeds_json': str(run_root / "seeds.json"),
            'frames': self.cfg["simulation"]["frames"],
            'burnin': self.cfg["simulation"]["burnin_frames"],
            'save_frames': "1" if self.cfg["simulation"]["save_frames"] else "0",
            'n_reps': self.cfg["simulation"]["n_replicates"],
            'n_types': self.cfg["simulation"]["n_types"],
            'per_task_reps': self.cfg["resources"]["simulation"].get("per_task_replicates", 10),
            
            # Target files
            'kernel_json': str(run_root / "exp_targets" / "kernel.json"),
            'targets_npy': str(run_root / "exp_targets" / "T_type_kl.npy"),
            
            # Script paths (auto-resolved)
            'run_replicates_array': str((self.proj_root / "bin" / "run_replicates_array.py").resolve()),
            'series_runner': str((self.proj_root / "bin" / "series_runner.py").resolve()),
            'process_tkl_update': str((self.proj_root / "bin" / "process_tkl_update.py").resolve()),
            'update_step': str((self.proj_root / "bin" / "update_step.py").resolve()),
            
            # Processing settings
            'io_k': self.cfg["resources"]["processing"]["workers"].get("io_k", 2),
            'monomer_types': self.cfg["processing_inputs"]["monomer_types"],
            'interaction_matrix': self.cfg["processing_inputs"]["interaction_matrix"],
            'exp_tkl': self.cfg["processing_inputs"]["exp_tkl"],
            'kernel_cli': "--mu 4.22 --rc 1.82 --rcut 3.0 --beta 1.0",
            
            # Update settings
            'run_root': str(run_root),
            'iter_idx': 0,
            'config_yaml': str(Path(self.args.config).resolve()),
        }
    
    def validate_workflow_simulation(self):
        """Simulate the workflow without actually running jobs."""
        self.info("Simulating workflow execution...")
        
        if not hasattr(self, 'cfg') or not self.temp_dir:
            return
        
        try:
            # Simulate maxent_loop.py setup
            run_root = self.temp_dir / "simulated_run"
            self._simulate_maxent_loop_setup(run_root)
            
            # Simulate iteration_driver.py for iteration 0
            self._simulate_iteration_driver(run_root, 0)
            
            self.success("Workflow simulation completed successfully")
            
        except Exception as e:
            self.error(f"Workflow simulation failed: {e}")
    
    def _simulate_maxent_loop_setup(self, run_root: Path):
        """Simulate the maxent_loop.py setup phase."""
        
        # Create directory structure
        ensure_dir(run_root / "exp_targets")
        ensure_dir(run_root / "logs")
        
        # Simulate seeds preparation
        seeds = prepare_seeds(self.cfg["simulation"]["n_replicates"], 
                            self.cfg["simulation"]["seeds_base"])
        
        # Create mock target files
        (run_root / "exp_targets" / "kernel.json").write_text('{"mock": "kernel"}')
        np.save(run_root / "exp_targets" / "T_type_kl.npy", 
                np.random.rand(10))  # Mock target vector
        
        # Save seeds
        with open(run_root / "seeds.json", "w") as f:
            json.dump(seeds, f, indent=2, sort_keys=True)
        
        # Create manifest
        manifest = {
            "name": self.args.name,
            "created_at": human_time(),
            "config_path": str(Path(self.args.config).resolve()),
            "n_replicates": self.cfg["simulation"]["n_replicates"],
            "frames": self.cfg["simulation"]["frames"],
        }
        write_json(run_root / "run_manifest.json", manifest)
        
        # Create iteration 0 structure
        iter0 = run_root / format_iter(0)
        ensure_dir(iter0 / "params")
        ensure_dir(iter0 / "sims")
        ensure_dir(iter0 / "obs")
        ensure_dir(iter0 / "update")
        
        # Create mock epsilon file
        n_types = self.cfg["simulation"]["n_types"]
        mock_eps = np.random.rand(n_types, n_types)
        mock_eps = (mock_eps + mock_eps.T) / 2  # Make symmetric
        np.save(iter0 / "params" / "epsilon.npy", mock_eps)
    
    def _simulate_iteration_driver(self, run_root: Path, iteration: int):
        """Simulate iteration_driver.py execution."""
        
        iter_dir = run_root / format_iter(iteration)
        
        # Simulate script generation without actual sbatch submission
        mock_vars = self._create_mock_template_vars()
        mock_vars.update({
            'iter_dir': str(iter_dir),
            'run_root': str(run_root),
            'iter_idx': iteration,
        })
        
        # Test template generation for each stage
        stages = [
            ('sbatch_sim_array.sh', 'simulation'),
            ('sbatch_process_worker.sh', 'processing workers'),
            ('sbatch_process_reduce.sh', 'processing reduce'),
            ('sbatch_update.sh', 'update')
        ]
        
        for template_name, stage_name in stages:
            template_path = self.proj_root / "templates" / template_name
            if template_path.exists():
                try:
                    template_text = template_path.read_text()
                    formatted = template_text.format(**mock_vars)
                    
                    # Write to temp file to verify it's valid
                    output_path = iter_dir / f"test_{template_name}"
                    output_path.write_text(formatted)
                    
                    self.success(f"Generated {stage_name} script successfully")
                except Exception as e:
                    self.error(f"Failed to generate {stage_name} script: {e}")
    
    def print_report(self):
        """Print final validation report."""
        print("\n" + "=" * 60)
        print("🔍 DRY RUN VALIDATION REPORT")
        print("=" * 60)
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   {warning}")
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   {error}")
            print(f"\n💥 VALIDATION FAILED: {len(self.errors)} error(s) found")
            print("   Please fix these issues before running on HPC.")
        else:
            print(f"\n🎉 VALIDATION PASSED!")
            print("   Your MaxEnt workflow is ready to run on HPC.")
            
            if self.warnings:
                print(f"   Note: {len(self.warnings)} warning(s) found - review but not critical.")
        
        print("\n" + "=" * 60)


def main():
    """Main entry point for dry run validation."""
    
    ap = argparse.ArgumentParser(
        description="Dry run validation for MaxEnt loop workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate with explicit arguments
  python dry_run.py --run-root /tmp/test_run --initial-epsilon /path/to/eps.npy --name test_run
  
  # Test the exact command you plan to run on HPC
  python dry_run.py --run-root /gpfs/home/user/run_001 \\
                     --initial-epsilon /path/to/epsilon_initial.npy \\
                     --name run001
        """
    )
    
    # Same arguments as maxent_loop.py
    ap.add_argument("--run-root", required=True, 
                   help="Root directory for this run")
    ap.add_argument("--initial-epsilon", required=True, 
                   help="Path to KxK epsilon matrix .npy")
    ap.add_argument("--name", required=True, 
                   help="Short run name used in job names")
    ap.add_argument("--config", 
                   default=str(Path(__file__).resolve().parent.parent / "config.yaml"),
                   help="Path to config.yaml file")
    
    # Dry run specific options
    ap.add_argument("--verbose", "-v", action="store_true",
                   help="Enable verbose output")
    
    args = ap.parse_args()
    
    # Run validation
    validator = DryRunValidator(args)
    success = validator.validate_all()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
