# MaxEnt Loop Dry Run Validation

The `bin/dry_run.py` script provides comprehensive validation of your MaxEnt workflow **before** submitting expensive SLURM jobs on HPC. This saves time and resources by catching configuration errors, missing files, and template formatting issues early.

## Quick Start

Test your exact HPC command locally:

```bash
# Instead of running this on HPC:
python /gpfs/home/pkv4601/genome_architecture/github/genome_organization/src/chunkchromatin/maxent_loop/bin/maxent_loop.py \
  --run-root /gpfs/home/pkv4601/genome_architecture/chunkchromatin/Megaenhancers/MaxEnt_runs/run_001 \
  --initial-epsilon /home/pkv4601/genome_architecture/chunkchromatin/Megaenhancers/sim_params/maxent_interaction_parameters/alpha_tk_1.npy \
  --name run001

# Run this locally first:
python bin/dry_run.py \
  --run-root /gpfs/home/pkv4601/genome_architecture/chunkchromatin/Megaenhancers/MaxEnt_runs/run_001 \
  --initial-epsilon /home/pkv4601/genome_architecture/chunkchromatin/Megaenhancers/sim_params/maxent_interaction_parameters/alpha_tk_1.npy \
  --name run001
```

## What It Validates

### ✅ **Configuration Validation**
- YAML syntax and structure
- Required sections and parameters
- Resource allocation sanity checks
- Array capacity vs number of replicates

### ✅ **File Path Validation**
- Config file exists and is readable
- Initial epsilon file exists and has correct shape
- Processing input files (monomer_types, exp_tkl)
- Experimental target files (kernel.json, T_type_kl.npy)
- All required scripts in bin/ directory
- All required templates

### ✅ **Template Validation**
- All SLURM templates format without KeyError
- Variable naming consistency
- Shell variable escaping correctness

### ✅ **Workflow Simulation**
- Directory structure creation
- File copying and preparation
- Script generation for each stage:
  - Simulation array jobs
  - Processing worker jobs  
  - Processing reduce job
  - Update job

### ✅ **Dependency Validation**
- Python imports work
- NumPy operations on epsilon matrix
- YAML configuration loading

## Expected vs Actual Errors

When testing locally, you'll see "errors" for files that exist on HPC but not locally. This is normal:

### 🆗 **Expected Local Errors** (safe to ignore):
```
❌ ERROR: Processing input file not found: monomer_types = /gpfs/home/.../ME_bed_types.npy
❌ ERROR: Processing input file not found: exp_tkl = /gpfs/home/.../Tkl_exp.npy  
❌ ERROR: Experimental target file not found: kernel_json = /gpfs/home/.../kernel.json
```

### ⚠️ **Real Errors** (must fix before HPC):
```
❌ ERROR: Config file not found: config.yaml
❌ ERROR: Missing required config section: slurm
❌ ERROR: Template sbatch_sim_array.sh: Missing variable time_limit
❌ ERROR: Array capacity (25) < n_replicates (50)
❌ ERROR: Epsilon matrix shape (3, 3) != expected (5, 5)
```

## Usage Examples

### Basic Validation
```bash
python bin/dry_run.py \
  --run-root /tmp/test_run \
  --initial-epsilon /path/to/epsilon.npy \
  --name test_run
```

### Validate with Custom Config
```bash
python bin/dry_run.py \
  --run-root /tmp/test_run \
  --initial-epsilon /path/to/epsilon.npy \
  --name test_run \
  --config /path/to/custom_config.yaml
```

### Verbose Output
```bash
python bin/dry_run.py \
  --run-root /tmp/test_run \
  --initial-epsilon /path/to/epsilon.npy \
  --name test_run \
  --verbose
```

## Integration with HPC Workflow

### Recommended Workflow:

1. **Develop Locally**: Edit configs and scripts on your local machine
2. **Validate Locally**: Run `dry_run.py` to catch issues early
3. **Transfer to HPC**: Copy validated code to HPC
4. **Final Check**: Run `dry_run.py` on HPC login node to verify file paths
5. **Submit Job**: Run `maxent_loop.py` with confidence

### Example HPC Login Node Check:
```bash
# On HPC login node, validate file paths exist
cd /gpfs/home/pkv4601/genome_architecture/github/genome_organization/src/chunkchromatin/maxent_loop
python bin/dry_run.py \
  --run-root /gpfs/home/pkv4601/genome_architecture/chunkchromatin/Megaenhancers/MaxEnt_runs/run_001 \
  --initial-epsilon /home/pkv4601/genome_architecture/chunkchromatin/Megaenhancers/sim_params/maxent_interaction_parameters/alpha_tk_1.npy \
  --name run001

# If validation passes, submit the real job
python bin/maxent_loop.py \
  --run-root /gpfs/home/pkv4601/genome_architecture/chunkchromatin/Megaenhancers/MaxEnt_runs/run_001 \
  --initial-epsilon /home/pkv4601/genome_architecture/chunkchromatin/Megaenhancers/sim_params/maxent_interaction_parameters/alpha_tk_1.npy \
  --name run001
```

## Output Interpretation

### ✅ Success Output:
```
🎉 VALIDATION PASSED!
   Your MaxEnt workflow is ready to run on HPC.
```

### ❌ Failure Output:
```
💥 VALIDATION FAILED: 3 error(s) found
   Please fix these issues before running on HPC.
```

### ⚠️ Warnings:
Warnings indicate potential issues but won't prevent execution:
```
⚠️ WARNING: --name should be alphanumeric with underscores/hyphens only
⚠️ WARNING: Epsilon matrix is not symmetric  
```

## Benefits

- **Save HPC Resources**: Catch errors before expensive job submission
- **Faster Development**: Quick local validation cycle
- **Reduce Failed Jobs**: Prevent queue failures from configuration errors  
- **Template Safety**: Ensure all SLURM scripts will generate correctly
- **Path Validation**: Verify all file dependencies exist

Run `python bin/dry_run.py --help` for full usage details.
