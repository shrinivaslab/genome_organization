#About polymer simulation examples

Given the specified forces and parameters, Homopolymer_scaling_bulk.ipynb writes commands to run homopolymer simulations using run_homopolymer_sim.py. The output text file is intended to be piped to GNU parallel as follows:
cat simulation_homopolymer_commands.txt | parallel -j NUM_THREADS

The collision rate notebook shows the EKExceeds error for N=10 unless collision rate is high.
