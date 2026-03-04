"""
chromosome_michrom.py

Exact OpenMiChroM force field implementations for 1-to-1 comparison.
All forces match OpenMiChroM's ChromDynamics.py exactly.

Based on: /home/pkv4601/genome_architecture/chunkchromatin/Megaenhancers/sim_fitting_benchmarking/michrom_code/OpenMiChroM/OpenMiChroM/ChromDynamics.py
"""

import logging
import time
from openmm import unit
import openmm as mm
import numpy as np
import pandas as pd

class ChromosomeMichroM(object):
    """
    Chromosome class with exact OpenMiChroM force implementations.
    
    This class implements the exact same forces as OpenMiChroM:
    - FENE bonds
    - Angles (cosine-based, not harmonic)
    - Repulsive soft-core
    - Type-to-type interactions (with Discrete2DFunction)
    - Ideal chromosome (gamma-form, no step(r-lim))
    - Loops
    - Flat bottom harmonic confinement
    """
    
    def __init__(self, N, chains, sim_object, extra_bonds=None, extra_triplets=None):
        """
        Initialize a ChromosomeMichroM object and generate bond/angle lists.

        Parameters
        ----------
        N : int
            Total number of particles.
        chains : list of tuples
            List of (start, end, isRing) tuples defining chain segments.
        sim_object : Simulation
            Simulation object, must have attribute `N`.
        extra_bonds : list of tuples, optional
            Additional (i, j) bonds.
        extra_triplets : list of tuples, optional
            Additional (i, j, k) angle triplets.
        """
        self.N = N
        self.chains = chains
        self.sim_object = sim_object

        self.bond_list = self._generate_bonds(sim_object, chains, extra_bonds)
        self.triplet_list = self._generate_triplets(sim_object, chains, extra_triplets)
    
    def _get_exceptions_for_nonbonded(self, sim_object):
        """
        Generate exceptions list for all CustomNonbondedForce objects.
        
        OpenMM requires all CustomNonbondedForce objects to have identical exclusions.
        OpenMiChroM only excludes bonded pairs (d=1) via bondsForException.
        We match this exactly - only exclude d=1, not d=2.
        The step(d-dinit) with dinit=3 should prevent d<3 from being evaluated.
        
        Returns
        -------
        set of tuples
            Set of (i, j) pairs to exclude, with i < j.
        """
        exceptions = set()
        
        # OpenMiChroM only excludes bonded pairs (d=1) from bondsForException
        # Match this exactly - only exclude d=1, not d=2
        for i, j in self.bond_list:
            exceptions.add((min(i, j), max(i, j)))
        
        return exceptions
        
    def add_fene_bonds(self, kFb=30.0, bonds=None, force_group=0):
        """
        Add FENE bonds exactly as in OpenMiChroM.
        
        Energy: -0.5 * kFb * fr0^2 * log(1 - (r / fr0)^2) + LJ * step(cutoff - r)
        LJ = 4 * epsilon * ((sigma / r)^12 - (sigma / r)^6) + epsilon
        
        Parameters
        ----------
        kFb : float
            Bond coefficient (default 30.0).
        bonds : list of tuples, optional
            Specific bonds to add. If None, uses self.bond_list.
        force_group : int
            OpenMM force group.
        
        Returns
        -------
        mm.CustomBondForce
        """
        # OpenMiChroM parameters
        fr0 = 1.5
        epsilon = 1.0  # OpenMiChroM uses epsilon=1.0 (in reduced units)
        sigma = 1.0    # OpenMiChroM uses sigma=1.0 (in reduced units)
        cutoff = 2.0 ** (1.0 / 6.0)  # 2^(1/6) * sigma
        
        # Energy expression exactly as in OpenMiChroM
        feneEnergy = (
            "-0.5 * kFb * fr0^2 * log(1 - (r / fr0)^2) + "
            "(4 * epsilon * ((sigma / r)^12 - (sigma / r)^6) + epsilon) * step(cutoff - r)"
        )
        
        feneBondForce = mm.CustomBondForce(feneEnergy)
        feneBondForce.setForceGroup(force_group)
        feneBondForce.addGlobalParameter("kFb", kFb)
        feneBondForce.addGlobalParameter("fr0", fr0)
        feneBondForce.addGlobalParameter("epsilon", epsilon)
        feneBondForce.addGlobalParameter("sigma", sigma)
        feneBondForce.addGlobalParameter("cutoff", cutoff)
        
        # Add bonds
        bonds_to_add = bonds if bonds is not None else self.bond_list
        for i, j in bonds_to_add:
            feneBondForce.addBond(int(i), int(j), [])
        
        return feneBondForce

    def add_angles(self, kA=2.0, force_group=1):
        """
        Add angle force exactly as in OpenMiChroM.
        
        Energy: kA * (1 - cos(theta - pi))
        
        Parameters
        ----------
        kA : float or array
            Angle coefficient(s) (default 2.0). If scalar, same for all angles.
        force_group : int
            OpenMM force group.
        
        Returns
        -------
        mm.CustomAngleForce
        """
        # Ensure kA is array
        if np.isscalar(kA):
            kA_array = np.full(len(self.triplet_list), kA, dtype=float)
        else:
            kA_array = np.asarray(kA, dtype=float)
            if len(kA_array) != len(self.triplet_list):
                raise ValueError(
                    f"The length of kA ({len(kA_array)}) must match the number of angles ({len(self.triplet_list)})."
                )
        
        # Energy expression exactly as in OpenMiChroM
        angleForceExpression = "kA * (1 - cos(theta - pi))"
        
        angleForce = mm.CustomAngleForce(angleForceExpression)
        angleForce.setForceGroup(force_group)
        angleForce.addPerAngleParameter("kA")
        angleForce.addGlobalParameter("pi", np.pi)
        
        for (i, j, k), k_val in zip(self.triplet_list, kA_array):
            angleForce.addAngle(int(i), int(j), int(k), [float(k_val)])
        
        return angleForce

    def add_repulsive_softcore(self, sim_object, eCut=4.0, cutoffDistance=3.0, force_group=3, name="repulsive_softcore"):
        """
        Add repulsive soft-core force exactly as in OpenMiChroM.
        
        Parameters
        ----------
        sim_object : Simulation
            Must have attributes: N (int).
        eCut : float
            Energy cost for chain crossing (default 4.0, in kT units).
        cutoffDistance : float
            Cutoff distance (default 3.0).
        force_group : int
            OpenMM force group.
        name : str
            Name for the force.
        
        Returns
        -------
        mm.CustomNonbondedForce
        """
        # OpenMiChroM uses sigma=1.0, epsilon=1.0 in reduced units
        sigma = 1.0
        epsilon = 1.0
        
        nbCutoffDist = sigma * 2.0 ** (1.0 / 6.0)
        
        # Scale eCut by epsilon (OpenMiChroM does this)
        eCut_scaled = eCut * epsilon
        
        # Calculate r0 exactly as in OpenMiChroM
        r0 = sigma * (((0.5 * eCut_scaled) / (4.0 * epsilon) - 0.25 + (0.5) ** 2.0) ** 0.5 + 0.5) ** (-1.0 / 6.0)
        
        # Energy expression exactly as in OpenMiChroM
        repulEnergy = (
            "LJ * step(r - r0) * step(cutoff - r)"
            " + step(r0 - r) * 0.5 * eCut * (1.0 + tanh((2.0 * LJ / eCut) - 1.0));"
            "LJ = 4.0 * epsilon * ((sigma / r)^12 - (sigma / r)^6) + epsilon"
        )
        
        repulForce = mm.CustomNonbondedForce(repulEnergy)
        repulForce.setForceGroup(force_group)
        repulForce.name = name
        repulForce.addGlobalParameter('epsilon', epsilon)
        repulForce.addGlobalParameter('sigma', sigma)
        repulForce.addGlobalParameter('eCut', eCut_scaled)
        repulForce.addGlobalParameter('r0', r0)
        repulForce.addGlobalParameter('cutoff', nbCutoffDist)
        repulForce.setCutoffDistance(cutoffDistance)
        
        # Add particles
        for _ in range(sim_object.N):
            repulForce.addParticle(())
        
        # Set nonbonded method (OpenMiChroM does this for all nonbonded forces)
        if hasattr(repulForce, "CutoffNonPeriodic") and hasattr(repulForce, "CutoffPeriodic"):
            repulForce.setNonbondedMethod(repulForce.CutoffNonPeriodic)
        
        # Add exclusions to match OpenMiChroM behavior
        # OpenMM requires all CustomNonbondedForce objects to have identical exclusions
        # OpenMiChroM only excludes bonded pairs (d=1), not d=2
        exceptions = self._get_exceptions_for_nonbonded(sim_object)
        for i, j in exceptions:
            repulForce.addExclusion(int(i), int(j))
        
        return repulForce

    def add_type_to_type_michrom(
        self,
        sim_object,
        interaction_matrix,
        monomer_types,
        mu=3.22,
        rc=1.78,
        rCutoff=3.0,
        force_group=2,
        name="type_to_type_michrom"
    ):
        """
        Add type-to-type interactions exactly as in OpenMiChroM.
        
        Energy: mapType(t1,t2) * 0.5 * (1. + tanh(mu*(rc - r))) * step(r-lim)
        
        Uses Discrete2DFunction for type mapping (not Kronecker deltas).
        NOTE: interaction_matrix values should be in energy units (NOT kT),
        as OpenMiChroM does not convert them.
        
        Parameters
        ----------
        sim_object : Simulation
            Must have attributes: N (int).
        interaction_matrix : ndarray
            KxK symmetric matrix of interaction energies between types.
        monomer_types : ndarray
            Array of length N assigning type index to each monomer.
        mu : float
            Tanh kernel parameter (default 3.22, OpenMiChroM default).
        rc : float
            Tanh kernel parameter (default 1.78, OpenMiChroM default).
        rCutoff : float
            Cutoff distance (default 3.0).
        force_group : int
            OpenMM force group.
        name : str
            Name for the force.
        
        Returns
        -------
        mm.CustomNonbondedForce
        """
        Ntypes = int(np.max(monomer_types)) + 1
        if interaction_matrix.shape[0] < Ntypes or interaction_matrix.shape[1] < Ntypes:
            raise ValueError(f"Interaction matrix must cover all {Ntypes} types.")
        if not np.allclose(interaction_matrix.T, interaction_matrix):
            raise ValueError("Interaction matrix must be symmetric.")
        
        # Energy expression exactly as in OpenMiChroM
        energy = "mapType(t1,t2)*0.5*(1. + tanh(mu*(rc - r)))*step(r-lim)"
        
        crossLP = mm.CustomNonbondedForce(energy)
        crossLP.setForceGroup(force_group)
        crossLP.name = name
        
        crossLP.addGlobalParameter('mu', mu)
        crossLP.addGlobalParameter('rc', rc)
        crossLP.addGlobalParameter('lim', 1.0)
        crossLP.setCutoffDistance(rCutoff)
        
        # Create Discrete2DFunction exactly as OpenMiChroM does
        # OpenMiChroM uses: lambdas = np.triu(tab.values) + np.triu(tab.values, k=1).T
        # Then: lambdas = list(np.ravel(lambdas))
        lambdas = np.triu(interaction_matrix) + np.triu(interaction_matrix, k=1).T
        lambdas_flat = list(np.ravel(lambdas))
        
        fTypes = mm.Discrete2DFunction(Ntypes, Ntypes, lambdas_flat)
        crossLP.addTabulatedFunction('mapType', fTypes)
        
        # Per-particle type parameter
        crossLP.addPerParticleParameter("t")
        for t in monomer_types:
            crossLP.addParticle([float(t)])
        
        # Set nonbonded method (OpenMiChroM does this for all nonbonded forces)
        if hasattr(crossLP, "CutoffNonPeriodic") and hasattr(crossLP, "CutoffPeriodic"):
            crossLP.setNonbondedMethod(crossLP.CutoffNonPeriodic)
        
        # Add exclusions to match OpenMiChroM behavior
        # OpenMM requires all CustomNonbondedForce objects to have identical exclusions
        # OpenMiChroM only excludes bonded pairs (d=1), not d=2
        exceptions = self._get_exceptions_for_nonbonded(sim_object)
        for i, j in exceptions:
            crossLP.addExclusion(int(i), int(j))
        
        return crossLP

    def add_ideal_chromosome_michrom(
        self,
        sim_object,
        gamma1=-0.030,
        gamma2=-0.351,
        gamma3=-3.727,
        d_init=3,
        d_end=500,
        mu=3.22,
        rc=1.78,
        rCutoff=3.0,
        force_group=4,
        name="ideal_chromosome_michrom"
    ):
        """
        Add ideal chromosome force exactly as in OpenMiChroM.
        
        Energy: step(d-dinit)*(gamma1/log(d) + gamma2/d + gamma3/d^2)*step(dend-d)*f
        where f=0.5*(1. + tanh(mu*(rc - r))) and d=abs(idx1-idx2)
        
        NOTE: gamma parameters are in energy units (NOT kT), as OpenMiChroM does not convert them.
        NOTE: No step(r-lim) - OpenMiChroM doesn't use this.
        NOTE: No max(d, dinit) - OpenMiChroM uses log(d) directly.
        
        Parameters
        ----------
        sim_object : Simulation
            Must have attributes: N (int).
        gamma1, gamma2, gamma3 : float
            Ideal chromosome parameters (OpenMiChroM defaults).
        d_init : int
            Minimum genomic distance (default 3).
        d_end : int
            Maximum genomic distance (default 500).
        mu : float
            Tanh kernel parameter (default 3.22, OpenMiChroM default).
        rc : float
            Tanh kernel parameter (default 1.78, OpenMiChroM default).
        rCutoff : float
            Cutoff distance (default 3.0).
        force_group : int
            OpenMM force group.
        name : str
            Name for the force.
        
        Returns
        -------
        mm.CustomNonbondedForce
        """
        # Energy expression exactly as in OpenMiChroM
        # NOTE: step(d-dinit) should prevent d<3, but we add a safeguard: max(d, dinit+0.5) to prevent log issues
        # However, OpenMiChroM uses log(d) directly, so we match exactly
        # The exclusions for d=1 should prevent log(1)=0 division issues
        energyIC = ("step(d-dinit)*(gamma1/log(d) + gamma2/d + gamma3/d^2)*step(dend -d)*f;"
                   "f=0.5*(1. + tanh(mu*(rc - r)));"
                   "d=abs(idx1-idx2)")
        
        IC = mm.CustomNonbondedForce(energyIC)
        IC.setForceGroup(force_group)
        IC.name = name
        
        IC.addGlobalParameter('gamma1', gamma1)
        IC.addGlobalParameter('gamma2', gamma2)
        IC.addGlobalParameter('gamma3', gamma3)
        IC.addGlobalParameter('dinit', float(d_init))
        IC.addGlobalParameter('dend', float(d_end))
        IC.addGlobalParameter('mu', mu)
        IC.addGlobalParameter('rc', rc)
        
        IC.setCutoffDistance(rCutoff)
        
        # Set nonbonded method (OpenMiChroM does this for all nonbonded forces)
        if hasattr(IC, "CutoffNonPeriodic") and hasattr(IC, "CutoffPeriodic"):
            IC.setNonbondedMethod(IC.CutoffNonPeriodic)
        
        # Per-particle index parameter
        IC.addPerParticleParameter("idx")
        for i in range(sim_object.N):
            IC.addParticle([i])
        
        # Add exclusions to match OpenMiChroM behavior
        # OpenMM requires all CustomNonbondedForce objects to have identical exclusions
        # OpenMiChroM only excludes bonded pairs (d=1) via bondsForException in createSimulation
        # The step(d-dinit) with dinit=3 should prevent d<3 from being evaluated
        exceptions = self._get_exceptions_for_nonbonded(sim_object)
        for i, j in exceptions:
            IC.addExclusion(int(i), int(j))
        
        return IC

    def add_ideal_chromosome_force(
        self,
        sim_object,
        lambda_IC,
        d_init=3,
        d_end=300,
        mu=4.22,
        rc=1.82,
        rCutoff=3.0,
        force_group=4,
        name="ideal_chromosome_force"
    ):
        """
        Ideal chromosome potential using per-distance lambdas (full phi).

        Energy: step(d-dinit)*IClist(d)*step(dend-d)*f*step(r-lim)
        where f = 0.5 * (1 + tanh(mu * (rc - r))) and d = abs(idx1-idx2).

        NOTE: lambda_IC values are in reduced energy units (OpenMiChroM style),
        i.e., not scaled by kT.

        Parameters
        ----------
        sim_object : Simulation
            Must have attributes: N (int).
        lambda_IC : ndarray
            1D array of Lagrange multipliers for each genomic distance.
            Shape: (d_end - d_init,).
        d_init : int
            Minimum genomic distance to consider (default 3).
        d_end : int
            Maximum genomic distance to consider (default 300).
        mu : float
            Tanh kernel parameter (default 4.22).
        rc : float
            Tanh kernel parameter (default 1.82).
        rCutoff : float
            Cutoff distance in reduced units (default 3.0).
        force_group : int
            OpenMM force group.
        name : str
            Name for the force.

        Returns
        -------
        mm.CustomNonbondedForce
        """
        dmax = d_end - d_init
        if len(lambda_IC) != dmax:
            raise ValueError(f"lambda_IC length ({len(lambda_IC)}) must equal dmax ({dmax})")

        energy_expr = (
            "step(d-dinit)*IClist(d)*step(dend-d)*f*step(r-lim);"
            "f=0.5*(1. + tanh(mu*(rc - r)));"
            "d=abs(idx2-idx1)"
        )

        force = mm.CustomNonbondedForce(energy_expr)
        force.setForceGroup(force_group)
        force.name = name
        force.setCutoffDistance(rCutoff)

        # Prepare IClist array with zero padding below d_init
        IClist_array = np.zeros(d_end, dtype=float)
        for d_idx, d in enumerate(range(d_init, d_end)):
            IClist_array[d] = float(lambda_IC[d_idx])

        tabIClist = mm.Discrete1DFunction(IClist_array.tolist())
        force.addTabulatedFunction('IClist', tabIClist)

        force.addGlobalParameter('dinit', float(d_init))
        force.addGlobalParameter('dend', float(d_end))
        force.addGlobalParameter('mu', float(mu))
        force.addGlobalParameter('rc', float(rc))
        force.addGlobalParameter('lim', 1.0)

        # Set nonbonded method (OpenMiChroM does this for all nonbonded forces)
        if hasattr(force, "CutoffNonPeriodic") and hasattr(force, "CutoffPeriodic"):
            force.setNonbondedMethod(force.CutoffNonPeriodic)

        # Per-particle index parameter
        force.addPerParticleParameter("idx")
        for i in range(sim_object.N):
            force.addParticle([float(i)])

        # Restrict IC interactions to within each chain only
        for start, end, _ in self.chains:
            end = sim_object.N if end is None else end
            if end <= start:
                continue
            group = list(range(start, end))
            force.addInteractionGroup(group, group)

        # Add exclusions to match OpenMiChroM behavior
        exceptions = self._get_exceptions_for_nonbonded(sim_object)
        for i, j in exceptions:
            force.addExclusion(int(i), int(j))

        return force

    def add_loops_michrom(
        self,
        sim_object,
        looplists,
        mu=3.22,
        rc=1.78,
        X=-1.612990,
        force_group=5,
        name="loops_michrom"
    ):
        """
        Add loop interactions exactly as in OpenMiChroM.
        
        Energy: qsi * 0.5 * (1. + tanh(mu*(rc - r)))
        
        NOTE: X (qsi) parameter is in energy units (NOT kT), as OpenMiChroM does not convert it.
        
        Parameters
        ----------
        sim_object : Simulation
            Must have attributes: N (int).
        looplists : list[str] or str
            List of loop file paths, one per chain.
        mu : float
            Tanh kernel parameter (default 3.22, OpenMiChroM default).
        rc : float
            Tanh kernel parameter (default 1.78, OpenMiChroM default).
        X : float
            Loop interaction parameter (default -1.612990, OpenMiChroM default).
        force_group : int
            OpenMM force group.
        name : str
            Name for the force.
        
        Returns
        -------
        mm.CustomBondForce
        """
        if isinstance(looplists, str):
            looplists = [looplists]
        
        if len(looplists) != len(self.chains):
            raise ValueError(
                f"Number of loop files ({len(looplists)}) must match "
                f"number of chains ({len(self.chains)})"
            )
        
        # Read loop positions exactly as OpenMiChroM's getLoops does
        # OpenMiChroM processes ALL lines (doesn't skip empty lines) and adds chain offset
        loop_positions = []
        for loop_file, chain in zip(looplists, self.chains):
            start_idx = int(chain[0])  # m = chain[0] in OpenMiChroM
            try:
                with open(loop_file, 'r') as f:
                    lines = f.read().splitlines()
                # OpenMiChroM processes all lines, even if empty (they'll fail on split/convert)
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 2:
                        # OpenMiChroM: pos[t][0] = int(pos[t][0]) + m
                        i = int(parts[0]) + start_idx
                        j = int(parts[1]) + start_idx
                        loop_positions.append([i, j])
            except FileNotFoundError:
                raise FileNotFoundError(f"Loop file not found: {loop_file}")
        
        # Energy expression exactly as in OpenMiChroM
        ELoop = "qsi*0.5*(1. + tanh(mu*(rc - r)))"
        
        Loop = mm.CustomBondForce(ELoop)
        Loop.setForceGroup(force_group)
        Loop.name = name
        
        Loop.addGlobalParameter('mu', mu)
        Loop.addGlobalParameter('rc', rc)
        Loop.addGlobalParameter('qsi', X)  # No unit conversion - OpenMiChroM uses it directly
        
        # Add bonds (OpenMiChroM converts from 1-indexed to 0-indexed: p[0]-1, p[1]-1)
        for pair in loop_positions:
            i = int(pair[0]) - 1  # Convert to 0-indexed
            j = int(pair[1]) - 1
            if i < 0 or j < 0 or i >= sim_object.N or j >= sim_object.N:
                raise ValueError(
                    f"Loop pair [{pair[0]}, {pair[1]}] results in out-of-bounds "
                    f"indices [{i}, {j}] for system with {sim_object.N} particles"
                )
            Loop.addBond(i, j)
        
        return Loop

    def add_flat_bottom_harmonic(
        self,
        sim_object,
        kR=5e-3,
        nRad=10.0,
        force_group=6,
        name="flat_bottom_harmonic"
    ):
        """
        Add flat-bottom harmonic confinement exactly as in OpenMiChroM.
        
        Energy: step(r - rRes) * 0.5 * kR * (r - rRes)^2
        where r = sqrt(x^2 + y^2 + z^2)
        
        Parameters
        ----------
        sim_object : Simulation
            Must have attributes: N (int).
        kR : float
            Spring constant (default 5e-3, OpenMiChroM default).
        nRad : float
            Nucleus radius (default 10.0, OpenMiChroM default).
        force_group : int
            OpenMM force group.
        name : str
            Name for the force.
        
        Returns
        -------
        mm.CustomExternalForce
        """
        # Energy expression exactly as in OpenMiChroM
        energyExpression = (
            "step(r - rRes) * 0.5 * kR * (r - rRes)^2;"
            "r = sqrt(x^2 + y^2 + z^2)"
        )
        
        restraintForce = mm.CustomExternalForce(energyExpression)
        restraintForce.setForceGroup(force_group)
        restraintForce.name = name
        restraintForce.addGlobalParameter('rRes', nRad)
        restraintForce.addGlobalParameter('kR', kR)
        
        # Add all particles
        for i in range(sim_object.N):
            restraintForce.addParticle(int(i), [])
        
        return restraintForce

    @staticmethod
    def _generate_bonds(sim_object, chains, extra_bonds=None):
        """
        Generate bond list from chains.
        
        NOTE: OpenMiChroM uses (start, end) for ring closure, where 'end' is the exclusive end.
        This matches their FENE bond code: if isRing: i1, i2 = start, end
        """
        bonds_list = [] if extra_bonds is None else [tuple(b) for b in extra_bonds]
        for start, end, is_ring in chains:
            end = sim_object.N if end is None else end
            # Linear bonds: (j, j+1) for j in [start, end-1)
            bonds_list.extend([(j, j + 1) for j in range(start, end - 1)])
            # Ring closure: OpenMiChroM uses (start, end) directly
            if is_ring:
                bonds_list.append((start, end))
        return np.array(bonds_list, dtype=int)

    @staticmethod
    def _generate_triplets(sim_object, chains, extra_triplets=None):
        """
        Generate triplet list from chains.
        
        NOTE: For rings, OpenMiChroM uses (end-1, end, start) and (end, start, start+1),
        where 'end' is the exclusive end index. This matches their angle generation logic.
        """
        triplets_list = [] if extra_triplets is None else [tuple(t) for t in extra_triplets]
        for start, end, is_ring in chains:
            end = sim_object.N if end is None else end
            # Linear chain angles: (j-1, j, j+1) for j in [start+1, end-1)
            for j in range(start + 1, end - 1):
                triplets_list.append((j - 1, j, j + 1))
            # Ring angles: OpenMiChroM uses (end-1, end, start) and (end, start, start+1)
            # Note: 'end' is exclusive, so this uses the actual last index (end-1) and wraps
            if is_ring:
                triplets_list.append((end - 1, end, start))  # Last three: (end-1, end, start)
                triplets_list.append((end, start, start + 1))  # Wrap: (end, start, start+1)
        return np.array(triplets_list, dtype=int)
