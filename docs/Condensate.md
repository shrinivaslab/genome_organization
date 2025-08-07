# Condensate Class Developer Guide

## Overview

`condensate.py` defines the `Condensate` class, which introduces interactions between condensate particles and between condensates and chromatin. It is designed for use in coarse-grained molecular dynamics simulations, and builds OpenMM force objects to describe condensate behavior.

Both condensate–condensate and condensate–chromatin interactions are implemented as Lennard-Jones-style potentials, with parameters supplied as NumPy arrays.

## Inputs

The `Condensate` class requires:

- `N` *(int)*: Total number of particles in the system.
- `chains` *(list of lists)*: Each sublist contains the particle indices for one chromosome chain.
- `simulation` *(OpenMM Simulation object)*: The target simulation to which forces will be added.

Particles not included in any chain will be automatically assigned as condensates.

## Initialization

Condensate particles should be initialized with positions uniformly sampled within the simulation box. This step must be done externally before dynamics begin, as the class does not initialize positions.

## Forces

The class will define and add the following forces:

- **Condensate–condensate attraction** via `CustomNonbondedForce`
- **Condensate–chromatin attraction/repulsion** via `CustomNonbondedForce`

Force parameters are provided as NumPy arrays. Future extensions may include volume exclusion, angular alignment, or density biasing.

## Lennard-Jones Parameters

Each pairwise interaction is modeled using the standard Lennard-Jones potential:

\[
V(r) = 4\varepsilon \left[ \left(\frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^6 \right]
\]

Where:

- **ε (epsilon)**: The depth of the potential well. This controls the strength of the attractive force between particles. Higher ε results in stronger attraction. Units: energy (or reduced units).
- **σ (sigma)**: The distance at which the potential crosses zero. It roughly sets the preferred interparticle separation. Units: length (or reduced units).
- The potential minimum occurs at \( r = 2^{1/6} \sigma \approx 1.122 \sigma \).

A separate **cutoff distance** (typically \( \sim 2.5\sigma \) or higher) is used to truncate the interaction for computational efficiency. In this implementation, the cutoff is currently fixed to 3.0 (in reduced units), independent of σ.





