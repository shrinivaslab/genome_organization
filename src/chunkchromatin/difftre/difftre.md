# Differentiable trajectory reweighting pseudocode

Differentiable trajctory reweighting (DiffTre) uses thermodynamic perturbation 
theory to reweight MD observables according to the probability of observing
states with perturbed energies against the reference ensemble. Instead of 
backpropagating through the MD model, you can backpropagate through these 
weights to update parameters, saving the reference states until the perturbed 
states become too improbable.


## Optimization pseudocode

Initialize particle types from HiC data

Choose priors based on the literature
- Harmonic bond strength
- Angle potential slope
- Spherical confinement strength
- Particle density in confining sphere

Build posteriors by optimizing simulated observables against simulated 
observables
- Type-type interaction strengths are free - fit with DiffTre
- We could reduce prior assumptions and fit the homopolymer potential
parameters
- Define convergence criteria
- Evaluate converged states against experimental data beyond optimization
objectives
    - p(s) scaling
    - spearman correlation coefficients between simulated and experimental 
    contact maps
    - qualitative evaluation of chromosome territories


## DiffTre optimization loop
- Naively initialize type-type interaction energies
- Run 50 replicate simulations with sbatch template
    -Calculate reference sim observable, 50 replicates with 4000 frames each (only process last 3500 frames)
        -load 4000 frames for each of the 50 replicates and discard the first 500 frames
        -for each frame create contact map representing the probability row and column particles are contacting
        -make a list of particles involved in each of the 15 type-type interactions
        -sum up probabilities for each of these 15 bins (this is simulated observable)
        -average over each frame and replicate (could pool all used frames from all replicates and average)
        -return this
    -Calculate reference energies for each frame
        -Pass in simulator object to DiffTre class and use some get energy method from the sim object
    -Calculate loss (MSE) between simulated and experimental observables
        -compare average (ensemble) reference sim observable vs experimental observable (already processed)
    -Update parameters
        - use jax to get gradient and use adam optimization
    -Calculate updated frame energies and corresponding weights
        - Set positions for the sim object, change energy forces w sim method, get energy
    -Use weights to calculate updated observables
    -Recalculate loss
    -update parameters
    -repeat until weights are too small
    -run another simulation





