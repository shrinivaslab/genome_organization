#!/bin/bash
# Resume MaxEnt optimization from iteration 7 with spectral conditioning
# This script will redo iteration 8 with the new numerical stability improvements

# Configuration - UPDATE THESE PATHS FOR YOUR SYSTEM
RUN_ROOT="/gpfs/home/pkv4601/genome_architecture/chunkchromatin/Megaenhancers/MaxEnt_runs/run_002_interp"
CONFIG_PATH="/home/pkv4601/genome_architecture/github/genome_organization/src/chunkchromatin/maxent_loop/config.yaml"
PROJ_ROOT="/home/pkv4601/genome_architecture/github/genome_organization/src/chunkchromatin/maxent_loop"
RUN_NAME="run_002"

echo "=== Resuming MaxEnt Optimization from Iteration 7 ==="
echo "Run root: $RUN_ROOT"
echo "Config: $CONFIG_PATH"
echo "Project root: $PROJ_ROOT"
echo ""

# Step 1: Clean up iterations 8+ (they had ill-conditioned Hessian issues)
echo "Step 1: Cleaning up problematic iterations 8-18..."
for iter in {8..25}; do
    iter_dir="$RUN_ROOT/iter_$(printf "%03d" $iter)"
    if [ -d "$iter_dir" ]; then
        echo "  Removing $iter_dir"
        rm -rf "$iter_dir"
    fi
done

# Step 2: Verify iteration 7 exists and looks good
ITER7_DIR="$RUN_ROOT/iter_007"
if [ ! -d "$ITER7_DIR" ]; then
    echo "ERROR: Iteration 7 directory not found at $ITER7_DIR"
    exit 1
fi

if [ ! -f "$ITER7_DIR/params/epsilon.npy" ]; then
    echo "ERROR: epsilon.npy not found in iteration 7"
    exit 1
fi

echo "Step 2: Verified iteration 7 exists with parameters"

# Step 3: Reset convergence tracking
echo "Step 3: Resetting convergence tracking..."
TRACK_FILE="$RUN_ROOT/convergence_track.json"
cat > "$TRACK_FILE" << EOF
{
  "streak": 0,
  "last_iter": 7
}
EOF

# Step 4: Create iteration 8 directory structure
echo "Step 4: Setting up iteration 8..."
ITER8_DIR="$RUN_ROOT/iter_008"
mkdir -p "$ITER8_DIR"/{params,sims,obs,update}

# Copy epsilon from iteration 7 to iteration 8
cp "$ITER7_DIR/params/epsilon.npy" "$ITER8_DIR/params/epsilon.npy"

# Also create the epsilon_tk file that the Newton update expects
cp "$ITER7_DIR/params/epsilon.npy" "$ITER8_DIR/update/epsilon_tk_8.npy"

echo "  Created iteration 8 structure and copied parameters"

# Step 5: Submit iteration 8 with spectral conditioning
echo "Step 5: Submitting iteration 8 with spectral conditioning..."
echo ""
echo "Command to run:"
echo "python $PROJ_ROOT/bin/iteration_driver.py \\"
echo "  --run-root $RUN_ROOT \\"
echo "  --iter 8 \\"
echo "  --config $CONFIG_PATH \\"
echo "  --name $RUN_NAME \\"
echo "  --proj-root $PROJ_ROOT"
echo ""

# Actually submit the job
python "$PROJ_ROOT/bin/iteration_driver.py" \
  --run-root "$RUN_ROOT" \
  --iter 8 \
  --config "$CONFIG_PATH" \
  --name "$RUN_NAME" \
  --proj-root "$PROJ_ROOT"

if [ $? -eq 0 ]; then
    echo ""
    echo "=== SUCCESS ==="
    echo "Iteration 8 submitted successfully with spectral conditioning!"
    echo ""
    echo "Monitor progress with:"
    echo "  tail -f $RUN_ROOT/logs/*.out"
    echo ""
    echo "Check for spectral conditioning logs:"
    echo "  grep 'SPECTRAL' $RUN_ROOT/logs/*.out"
    echo ""
    echo "Expected to see:"
    echo "  [SPECTRAL] κ_raw: ~1e7, λ_reg: ~1e4, κ_after: ~1e4"
    echo "  This should prevent the residual spike you saw before."
else
    echo "ERROR: Failed to submit iteration 8"
    exit 1
fi

