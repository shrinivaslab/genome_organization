# Fitting type-type observable

The type-type energy function is a nonbonded pair potential that takes the form of a smooth square well

This potential is the product of a type-type interaction matrix and a distance kernel.

Before fitting the interaction matrix with MaxEnt, I will fit the parameters of the distance kernel, g(r)
- These parameters include center, width, cutoff, and scale so that g(r) maps geometric distance to “contact probability” on the same numerical scale as experimental Hi-C probabilities at short genomic separations