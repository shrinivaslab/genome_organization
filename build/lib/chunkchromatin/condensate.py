# condensate.py

import numpy as np
import openmm as mm


class Condensate:
    """
    Adds smooth repulsive and attractive interactions between multiple types
    of condensate particles and chromatin particles, using a smooth square-well
    (SSW-style) potential with type-specific epsilon values.

    Parameters
    ----------
    N : int
        Total number of particles in the system.
    chains : list of list of int
        Each sublist contains indices of one chromosome chain.
    simulation : openmm.app.Simulation
        The OpenMM simulation object to which forces will be added.
    condensate_types : np.ndarray of int
        Array of shape (n_condensates,) giving type index for each condensate.
    chromatin_types : np.ndarray of int
        Array of shape (N,) giving type index for each particle (chromatin or not).
    epsilon_cc : np.ndarray of shape (n_c_types, n_c_types)
        Epsilon matrix for condensate–condensate interactions.
    epsilon_cchr : np.ndarray of shape (n_c_types, n_chr_types)
        Epsilon matrix for condensate–chromatin interactions.
    cutoff : float
        Cutoff distance (rc) beyond which potential is zero.
    alpha : float
        Exponent controlling sharpness of repulsion core. Higher = harder.
    """

    def __init__(self, N, chains, simulation,
                 condensate_types, chromatin_types,
                 epsilon_cc, epsilon_cchr,
                 cutoff=3.0, alpha=6):
        self.N = N
        self.chains = chains
        self.simulation = simulation
        self.cutoff = cutoff
        self.alpha = alpha
        self.sigma = 1.0  # fixed for scaling

        self.chromosome_indices = sorted(set(i for chain in chains for i in range(chain[0], chain[1])))
        self.condensate_indices = [i for i in range(N) if i not in self.chromosome_indices]

        assert len(condensate_types) == len(self.condensate_indices), \
            "Length of condensate_types must match number of inferred condensate particles"
        assert len(chromatin_types) == N, \
            "chromatin_types must be defined for all particles (N)"

        self.condensate_types = condensate_types
        self.chromatin_types = chromatin_types
        self.epsilon_cc = epsilon_cc
        self.epsilon_cchr = epsilon_cchr

        self._add_condensate_condensate_force()
        self._add_condensate_chromatin_force()

    def _create_smooth_well_force(self, epsilon_matrix, type1_list, type2_list, type1_size, type2_size):
        """
        Returns a CustomNonbondedForce with smooth square-well potential:
        U(r) = epsilon_matrix(type1, type2) * (1 - (r/rc)^alpha)^2  for r < rc
        """

        potential = (
            "epsilon_matrix(type1, type2) * step(rc - r) * (1 - (r/rc)^alpha)^2;"
            "rc=cutoff; alpha=alpha"
        )

        force = mm.CustomNonbondedForce(potential)
        force.addPerParticleParameter("type")

        epsilon_flat = epsilon_matrix.flatten()
        force.addTabulatedFunction(
            "epsilon_matrix",
            mm.Discrete2DFunction(type1_size, type2_size, epsilon_flat.tolist())
        )

        force.addGlobalParameter("cutoff", self.cutoff)
        force.addGlobalParameter("alpha", float(self.alpha))

        force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
        force.setCutoffDistance(self.cutoff)

        return force

    def _add_condensate_condensate_force(self):
        n_types = self.epsilon_cc.shape[0]
        force = self._create_smooth_well_force(
            self.epsilon_cc,
            self.condensate_types,
            self.condensate_types,
            n_types,
            n_types
        )

        # Add exactly N particles with valid type values
        type_lookup = {i: float(self.condensate_types[self.condensate_indices.index(i)])
                       for i in self.condensate_indices}

        for i in range(self.N):
            type_val = type_lookup.get(i, -1.0)
            force.addParticle([type_val])

        force.addInteractionGroup(self.condensate_indices, self.condensate_indices)
        self.simulation.system.addForce(force)

    def _add_condensate_chromatin_force(self):
        n_ctypes, n_chtypes = self.epsilon_cchr.shape
        force = self._create_smooth_well_force(
            self.epsilon_cchr,
            self.condensate_types,
            self.chromatin_types,
            n_ctypes,
            n_chtypes
        )

        for i in range(self.N):
            if i in self.condensate_indices:
                idx = self.condensate_indices.index(i)
                force.addParticle([float(self.condensate_types[idx])])
            elif i in self.chromosome_indices:
                force.addParticle([float(self.chromatin_types[i])])
            else:
                force.addParticle([float(-1)])

        force.addInteractionGroup(self.condensate_indices, self.chromosome_indices)
        self.simulation.system.addForce(force)
