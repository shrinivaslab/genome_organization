import logging
import time
from openmm import unit
import openmm as mm
import numpy as np

class Chromosome(object):
    def __init__(self, N, chains, sim_object, extra_bonds=None, extra_triplets=None):
        """
        Initialize a Chromosome object and generate bond/angle lists.

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
        
    def add_harmonic_bond(self, force_group=0, bondWiggleDistance=0.05, bondLength=1.0):
        """
        Add harmonic bonds based on a physical 'wiggle' distance where energy = 1 kT.

        Parameters
        ----------
        force_group : int
            OpenMM force group.
        bondWiggleDistance : float or iterable
            Distance at which bond energy equals 1 kT. Smaller values = stiffer bonds.
        bondLength : float or iterable
            Equilibrium bond distance.
        
        Returns
        -------
        mm.HarmonicBondForce
        """
        from numpy import array, float64
        import numpy as np

        bond_force = mm.HarmonicBondForce()
        bond_force.setForceGroup(force_group)

        num_bonds = len(self.bond_list)
        ls = bondLength
        kT = self.sim_object.kT.value_in_unit(unit.kilojoule_per_mole)


        # Handle scalar or array input
        bondLength = np.array([bondLength]*num_bonds if np.isscalar(bondLength) else bondLength, dtype=float64) * ls
        bondWiggleDistance = np.array([bondWiggleDistance]*num_bonds if np.isscalar(bondWiggleDistance) else bondWiggleDistance, dtype=float64) * ls

        # Compute k = kT / wiggle^2, in OpenMM units
        kbond = kT / (bondWiggleDistance ** 2)
        kbond[bondWiggleDistance == 0] = 0.0

        for (i, j), r0, k in zip(self.bond_list, bondLength, kbond):
            bond_force.addBond(int(i), int(j), float(r0), float(k))

        return bond_force

    def add_harmonic_bond_old(self, force_group=0, k=30.0, r0=1.0):
            """
            Create a HarmonicBondForce from self.bond_list.

            Parameters
            ----------
            force_group : int
                Force group ID.
            k : float
                Spring constant in kT/nm².
            r0 : float
                Equilibrium bond distance in nm.

            Returns
            -------
            mm.HarmonicBondForce
            """
            bond_force = mm.HarmonicBondForce()
            bond_force.setForceGroup(force_group)
            # Convert k from kT/nm² to kJ/mol/nm²
            k_openmm = k * self.sim_object.kT._value
            for idx1, idx2 in self.bond_list:
                bond_force.addBond(int(idx1), int(idx2), r0, k_openmm)
            return bond_force

    def add_angle_force(self, k=1.5, theta_0=np.pi, force_group=1, override_checks=False):
        """
        Add harmonic angle force: U(θ) = 0.5 * k * (θ - θ₀)² for each triplet.

        Parameters
        ----------
        k : float or list
            Stiffness (unitless, in kT). Scalar or per-triplet.
        theta_0 : float or list
            Equilibrium angle(s), in radians. Scalar or per-triplet.
        force_group : int
            OpenMM force group ID.
        override_checks : bool
            Skip duplicate triplet checks.

        Returns
        -------
        mm.CustomAngleForce
        """
        if not override_checks:
            self._check_angle_bonds(self.triplet_list)

        k_array = self._to_array_1d(k, len(self.triplet_list))
        theta_array = self._to_array_1d(theta_0, len(self.triplet_list))

        # Convert k from (kT/rad²) to (kJ/mol/rad²)
        k_openmm = k_array * self.sim_object.kT.value_in_unit(unit.kilojoule_per_mole)

        # No extra kT here — 'angK' carries kJ/mol already
        energy = "0.5 * angK * (theta - angT0)^2"
        angle_force = mm.CustomAngleForce(energy)
        angle_force.setForceGroup(force_group)
        # per-angle parameters
        angle_force.addPerAngleParameter("angK")
        angle_force.addPerAngleParameter("angT0")

        for i, (a, b, c) in enumerate(self.triplet_list):
            angle_force.addAngle(int(a), int(b), int(c), (float(k_openmm[i]), float(theta_array[i])))

        return angle_force
    
    def add_polynomial_repulsive(self, sim_object, trunc=3.0, radiusMult=1.0, name="polynomial_repulsive"):
        """
        Adds a soft repulsive polynomial potential between all particles.

        The potential:
        - Is flat until r ≈ 0.7 (relative to REPsigma)
        - Decays smoothly to 0 at r = REPsigma
        - Has finite energy at r = 0 (equal to `trunc` × kT)

        Based on: https://gist.github.com/mimakaev/0327bf6ffe7057ee0e0625092ec8e318

        Parameters
        ----------
        sim_object : Simulation
            Must have attributes: N (int), kT (float), conlen (float).
        trunc : float
            Repulsion strength at r = 0 (in kT units).
        radiusMult : float
            Multiplier on `sim_object.conlen` to define the cutoff radius.
        name : str
            Descriptive name for the force.

        Returns
        -------
        CustomNonbondedForce
        """

        # Define cutoff radius in reduced units
        radius = sim_object.conlen * radiusMult
        energy_expr = (
            "rsc12 * (rsc2 - 1.0) * REPe / emin12 + REPe;"
            "rsc12 = rsc4 * rsc4 * rsc4;"
            "rsc4 = rsc2 * rsc2;"
            "rsc2 = rsc * rsc;"
            "rsc = r / REPsigma * rmin12;"
        )

        force = mm.CustomNonbondedForce(energy_expr)
        force.setCutoffDistance(radius)
        force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
        force.setForceGroup(3)  # Optional force group for repulsion
        force.name = name

        # Global parameters
        self._add_global_parameter(force, "REPe", trunc * sim_object.kT.value_in_unit(unit.kilojoule_per_mole))
        self._add_global_parameter(force, "REPsigma", radius)
        self._add_global_parameter(force, "emin12", 46656.0 / 823543.0)        # For x^12*(x²−1)
        self._add_global_parameter(force, "rmin12", np.sqrt(6.0 / 7.0))         # Scales distance into domain

        # Add particles
        for _ in range(sim_object.N):
            force.addParticle(())

        return force

    
    def add_spherical_confinement(
        self,
        sim_object,
        r="density",           # radius in reduced units or "density"
        k=5.0,                 # stiffness in kT / unit_length
        density=0.3,           # density for automatic radius estimation
        center=[0.0, 0.0, 0.0],# center of the sphere in reduced coordinates
        invert=False,          # exclude from sphere instead of confining
        particles=None,        # list of particle indices, or None for all
        name="spherical_confinement"
    ):
        """
        Constrain particles to be within (or outside) a sphere.

        Parameters
        ----------
        sim_object : Simulation
            Must have `N` (int), `kT` (float), `conlen` (float), and optionally `verbose`.
        r : float or "density"
            Confinement radius. If "density", computed from density and particle count.
        k : float
            Stiffness of the wall (in kT / conlen).
        density : float
            Density to use for automatic radius computation (in particles per unit volume).
        center : list of float
            Center of the sphere (3 values, in reduced coordinates).
        invert : bool
            If True, exclude particles from the sphere.
        particles : list of int, optional
            Which particles to apply the confinement to. Defaults to all.
        name : str
            Optional name for the force.

        Returns
        -------
        CustomExternalForce
            The spherical confinement potential.
        """
        # Calculate radius from density if requested
        if r == "density":
            r = (3 * sim_object.N / (4 * np.pi * density)) ** (1.0 / 3.0)

        if getattr(sim_object, "verbose", False):
            print(f"[spherical_confinement] radius = {r:.3f} (reduced units)")

        # Energy expression in reduced units
        energy_expr = (
            "step(invert_sign*(r-aa)) * kb * (sqrt((r-aa)^2 + t^2) - t); "
            "r = sqrt((x-x0)^2 + (y-y0)^2 + (z-z0)^2 + tt^2)"
        )

        force = mm.CustomExternalForce(energy_expr)
        force.name = name

        # Add particles
        particles = range(sim_object.N) if particles is None else particles
        for i in particles:
            force.addParticle(int(i), [])

        # Parameters (no units)
        self._add_global_parameter(force, "kb", k * sim_object.kT.value_in_unit(unit.kilojoule_per_mole))
        self._add_global_parameter(force, "aa", r - 1.0 / k)
        self._add_global_parameter(force, "t", (1.0 / k) / 10.0)
        self._add_global_parameter(force, "tt", 0.01)
        self._add_global_parameter(force, "invert_sign", -1.0 if invert else 1.0)

        # Center of confinement sphere
        self._add_global_parameter(force, "x0", center[0])
        self._add_global_parameter(force, "y0", center[1])
        self._add_global_parameter(force, "z0", center[2])

        sim_object.sphericalConfinementRadius = r  # for bookkeeping

        return force

    
    def add_nonbonded_pair_potential(
        self,
        sim_object,
        interactionMatrix,
        monomerTypes,
        rCutoff=1.8,
        name="custom_sticky_force"
    ):
        """
        Implements a sticky potential between monomer types.

        U_rep(r) = 5 * (1 + rRep^12 * (rRep^2 - 1) * c1)      for r < 1
        U_att(r) = -ε * (1 + rAtt^12 * (rAtt^2 - 1) * c1)     for 1 <= r < rCutoff
        ε is set by interactionMatrix[type1, type2]

        Parameters
        ----------
        sim_object : Simulation
            Must have attributes: N (int), conlen (float), kT (float).
        interactionMatrix : ndarray
            Symmetric matrix of ε values (float) between monomer types.
        monomerTypes : ndarray
            Array of length N assigning type index to each monomer.
        rCutoff : float
            Cutoff distance in reduced units (default 1.8).
        name : str
            Name for the force.

        Returns
        -------
        CustomNonbondedForce
        """

        Ntypes = np.max(monomerTypes) + 1
        if interactionMatrix.shape[0] < Ntypes or interactionMatrix.shape[1] < Ntypes:
            raise ValueError(f"Interaction matrix must cover all {Ntypes} types.")
        if not np.allclose(interactionMatrix.T, interactionMatrix):
            raise ValueError("Interaction matrix must be symmetric.")

        # Identify all interacting type pairs
        indexpairs = [(i, j) for i in range(Ntypes) for j in range(Ntypes) if interactionMatrix[i, j] != 0]

        # Constants
        c1 = (7.0 / 6.0) ** 6 * 7.0
        c2 = np.sqrt(6.0 / 7.0)

        # Construct energy expression
        energy = (
            "step(1.0 - r) * lambda_sticky * eRep + step(r - 1.0) * step(rCutoff - r) * lambda_sticky * eAttr;"
            "eRep = 5 * (1 + rRep12 * (rRep2 - 1) * c1);"
            "rRep12 = rRep4 * rRep4 * rRep4;"
            "rRep4 = rRep2 * rRep2;"
            "rRep2 = rRep * rRep;"
            "rRep = r * c2;"
            "eAttr = "
        )

        if indexpairs:
            terms = [f"delta(type1-{i})*delta(type2-{j})*INT_{i}_{j}" for i, j in indexpairs]
            energy += f"-1 * ({'+'.join(terms)}) * (1 + rAtt12 * (rAtt2 - 1) * c1);"
        else:
            energy += "0;"  # No attractions

        energy += (
            "rAtt12 = rAtt4 * rAtt4 * rAtt4;"
            "rAtt4 = rAtt2 * rAtt2;"
            "rAtt2 = rAtt * rAtt;"
            "rAtt = ((r - 1.4)/0.4) * c2;"
        )

        # Create force
        force = mm.CustomNonbondedForce(energy)
        force.setCutoffDistance(rCutoff * sim_object.conlen)
        force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
        force.setForceGroup(2)  # Optional: assign to group 2 for analysis
        force.name = name

        # Global parameters
        self._add_global_parameter(force, "rCutoff", rCutoff)
        self._add_global_parameter(force, "c1", c1)
        self._add_global_parameter(force, "c2", c2)
        self._add_global_parameter(force, "lambda_sticky", 1.0)

        for i, j in indexpairs:
            param_name = f"INT_{i}_{j}"
            self._add_global_parameter(force, param_name, interactionMatrix[i, j])

        # Per-particle parameter
        force.addPerParticleParameter("type")
        for t in monomerTypes:
            force.addParticle([float(t)])

        return force
    

    def add_tanh_type_force(
            self, 
            sim_object, 
            interaction_matrix, 
            monomer_types,
            mu=4.22,
            rc=1.82,
            rCutoff=3.0,
            name="tanh_type_force"
        ):
        """
        OpenMiChroM-style type–type interaction using the tanh distance kernel.
        units are kT

        Pair energy:
            U_ij = lambda_tanh * f(r; mu, rc) * alpha_{type_i, type_j}

        with f(r; mu, rc) = 0.5 * (1 + tanh(mu * (rc - r))).

        Notes
        -----
        - Negative alpha values are attractive by convention.
        """

        # Validate inputs
        Ntypes = int(np.max(monomer_types)) + 1
        if interaction_matrix.shape[0] < Ntypes or interaction_matrix.shape[1] < Ntypes:
            raise ValueError(f"Interaction matrix must cover all {Ntypes} types.")
        if not np.allclose(interaction_matrix.T, interaction_matrix):
            raise ValueError("Interaction matrix must be symmetric.")

        # Collect nonzero type pairs
        indexpairs = [(i, j) for i in range(Ntypes) for j in range(Ntypes)
                    if float(interaction_matrix[i, j]) != 0.0]

        # Build mixing term via Kronecker deltas on per-particle 'type'
        if indexpairs:
            mix_terms = [f"delta(type1-{i})*delta(type2-{j})*ALPHA_{i}_{j}" for (i, j) in indexpairs]
            mixing = "(" + "+".join(mix_terms) + ")"
        else:
            mixing = "0"

        # Energy expression: no step() gating
        energy = (
            "lambda_tanh * f * MIX;"
            "f = 0.5 * (1 + tanh(mu * (rc - r)));"
            f"MIX = {mixing};"
        )

        # Define force
        force = mm.CustomNonbondedForce(energy)
        force.name = name
        force.setCutoffDistance(rCutoff * sim_object.conlen)
        force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
        force.setForceGroup(2)  # match your other custom nonbonded force group

        # Global parameters
        self._add_global_parameter(force, "lambda_tanh", 1.0)
        self._add_global_parameter(force, "mu", float(mu))
        self._add_global_parameter(force, "rc", float(rc))
        self._add_global_parameter(force, "rcutoff", rCutoff)

        kT_kJmol = sim_object.kT.value_in_unit(unit.kilojoule_per_mole)
        # Alpha parameters per interacting type pair
        for (i, j) in indexpairs:
            self._add_global_parameter(force, f"ALPHA_{i}_{j}", float(interaction_matrix[i, j]) * kT_kJmol)

        # Per-particle 'type' parameter
        force.addPerParticleParameter("type")
        for t in monomer_types:
            force.addParticle([float(t)])

        return force

    def add_ideal_chromosome_force(
        self,
        sim_object,
        lambda_IC,
        d_init=3,
        d_end=300,
        mu=4.22,
        rc=1.82,
        rCutoff=3.0,
        name="ideal_chromosome_force"
    ):
        """
        Ideal chromosome potential: enforces contact probability vs genomic distance.
        
        Energy: E = sum_{d=d_init}^{d_end-1} [lambda_IC[d] * sum_i f(r_{i,i+d})]
        where f(r) = 0.5 * (1 + tanh(mu * (rc - r)))
        
        This potential applies the tanh distance kernel only to pairs (i, j) where
        d = |j - i| is the genomic distance (sequence separation) between monomers.
        
        Based on the reference implementation using CustomNonbondedForce with
        Discrete1DFunction for tabulated lambda_IC values.
        
        Parameters
        ----------
        sim_object : Simulation
            Must have attributes: N (int), conlen (float), kT (float).
        lambda_IC : ndarray
            1D array of Lagrange multipliers for each genomic distance.
            Shape: (dmax,) where dmax = d_end - d_init.
            lambda_IC[i] corresponds to genomic distance (i + d_init).
        d_init : int
            Minimum genomic distance to consider (in bins, default 3).
        d_end : int
            Maximum genomic distance to consider (in bins, default 300).
        mu : float
            Tanh kernel parameter (default 4.22).
        rc : float
            Tanh kernel parameter (default 1.82).
        rCutoff : float
            Cutoff distance in reduced units (default 3.0).
        name : str
            Name for the force.
        
        Returns
        -------
        CustomNonbondedForce
            The ideal chromosome potential as a CustomNonbondedForce.
        """
        dmax = d_end - d_init
        if len(lambda_IC) != dmax:
            raise ValueError(f"lambda_IC length ({len(lambda_IC)}) must equal dmax ({dmax})")
        
        # Build energy expression matching reference implementation
        # Energy: step(d-dinit)*IClist(d)*step(dend-d)*f*step(r-lim)
        # where d = abs(idx2-idx1), f = 0.5*(1 + tanh(mu*(rc - r)))
        energy_expr = (
            "step(d-dinit)*IClist(d)*step(dend-d)*f*step(r-lim);"
            "f=0.5*(1. + tanh(mu*(rc - r)));"
            "d=abs(idx2-idx1)"
        )
        
        force = mm.CustomNonbondedForce(energy_expr)
        force.setForceGroup(4)  # Use force group 4 for IC
        force.name = name
        force.setCutoffDistance(rCutoff * sim_object.conlen)
        force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
        
        # Prepare IClist array: pad with zeros for d < d_init, then lambda_IC values
        # Reference: IClist = np.append(np.zeros(dinit), IClist_listfromfile)[:-dinit]
        # But we want IClist[d] to return lambda_IC[d - d_init] for d in [d_init, d_end)
        # So we create an array of length d_end where:
        #   IClist[d] = 0 for d < d_init
        #   IClist[d] = lambda_IC[d - d_init] for d in [d_init, d_end)
        #   IClist[d] = 0 for d >= d_end (but we gate with step(dend-d) anyway)
        IClist_array = np.zeros(d_end, dtype=float)
        for d_idx, d in enumerate(range(d_init, d_end)):
            IClist_array[d] = lambda_IC[d_idx]
        
        # Convert to kJ/mol units
        kT_kJmol = sim_object.kT.value_in_unit(unit.kilojoule_per_mole)
        IClist_array = IClist_array * kT_kJmol
        
        # Create tabulated function
        tabIClist = mm.Discrete1DFunction(IClist_array.tolist())
        force.addTabulatedFunction('IClist', tabIClist)
        
        # Global parameters
        force.addGlobalParameter('dinit', float(d_init))
        force.addGlobalParameter('dend', float(d_end))
        force.addGlobalParameter('mu', float(mu))
        force.addGlobalParameter('rc', float(rc))
        force.addGlobalParameter('lim', 1.0)  # Minimum distance cutoff
        
        # Per-particle parameter for index
        force.addPerParticleParameter("idx")
        
        # Add all particles with their index
        for i in range(sim_object.N):
            force.addParticle([float(i)])

        # Restrict IC interactions to within each chain only
        for start, end, _ in self.chains:
            end = sim_object.N if end is None else end
            if end <= start:
                continue
            group = list(range(start, end))
            force.addInteractionGroup(group, group)
        
        return force

    def add_ideal_chromosome_gamma_force(
        self,
        sim_object,
        gamma1=-0.030,
        gamma2=-0.351,
        gamma3=-3.727,
        d_init=3,
        d_end=500,
        mu=4.22,
        rc=1.82,
        rCutoff=3.0,
        name="ideal_chromosome_gamma_force"
    ):
        """
        Ideal chromosome (gamma-form) potential using:
            gamma(d) = gamma1/log(d) + gamma2/d + gamma3/d^2
        and f(r) = 0.5 * (1 + tanh(mu * (rc - r))).

        Parameters
        ----------
        sim_object : Simulation
            Must have attributes: N (int), conlen (float), kT (float).
        gamma1, gamma2, gamma3 : float
            MiChroM gamma parameters (in kT units).
        d_init : int
            Minimum genomic distance to consider (in bins, default 3).
        d_end : int
            Maximum genomic distance to consider (in bins, default 500).
        mu : float
            Tanh kernel parameter (default 4.22).
        rc : float
            Tanh kernel parameter (default 1.82).
        rCutoff : float
            Cutoff distance in reduced units (default 3.0).
        name : str
            Name for the force.
        """
        # Energy expression mirrors OpenMiChroM addIdealChromosome (gamma-form)
        # Note: step(d-dinit) ensures d >= dinit (so d >= 3, avoiding log(1)=0 issue)
        # step(r-lim) ensures r >= lim (minimum distance cutoff, prevents infinite forces at r=0)
        energy_expr = (
            "step(d-dinit)*(gamma1/log(d) + gamma2/d + gamma3/d^2)*step(dend-d)*f*step(r-lim);"
            "f=0.5*(1. + tanh(mu*(rc - r)));"
            "d=abs(idx2-idx1)"
        )

        force = mm.CustomNonbondedForce(energy_expr)
        force.setForceGroup(4)  # Align with IC force group usage
        force.name = name
        force.setCutoffDistance(rCutoff * sim_object.conlen)
        force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)

        # Convert gamma parameters from kT to kJ/mol
        kT_kJmol = sim_object.kT.value_in_unit(unit.kilojoule_per_mole)
        force.addGlobalParameter('gamma1', float(gamma1) * kT_kJmol)
        force.addGlobalParameter('gamma2', float(gamma2) * kT_kJmol)
        force.addGlobalParameter('gamma3', float(gamma3) * kT_kJmol)
        force.addGlobalParameter('dinit', float(d_init))
        force.addGlobalParameter('dend', float(d_end))
        force.addGlobalParameter('mu', float(mu))
        force.addGlobalParameter('rc', float(rc))
        force.addGlobalParameter('lim', 1.0)  # Minimum distance cutoff (prevents infinite forces at r=0)

        # Per-particle parameter for index
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

        return force

    def add_loops(
        self,
        sim_object,
        looplists,
        mu=3.22,
        rc=1.78,
        X=-1.612990,
        name="loops"
    ):
        """
        Adds loop interactions using the tanh distance kernel.
        
        Loop potential: U = qsi * 0.5 * (1 + tanh(mu * (rc - r)))
        where qsi (X) is the loop interaction parameter.
        
        Parameters
        ----------
        sim_object : Simulation
            Must have attributes: N (int), conlen (float), kT (float).
        looplists : list[str] or str
            List of file paths containing loop information. Each file should be
            a two-column text file with indices i and j of loop anchor pairs.
            For multi-chain simulations, the order should match self.chains.
            If a single string is provided, it will be converted to a list.
        mu : float
            Parameter in the tanh distance kernel (default 3.22).
        rc : float
            Parameter in the tanh distance kernel (default 1.78).
        X : float
            Loop interaction parameter (default -1.612990, in kT units).
        name : str
            Name for the force.
        
        Returns
        -------
        CustomBondForce
            The loop force object.
        """
        # Handle single string input
        if isinstance(looplists, str):
            looplists = [looplists]
        
        # Validate looplists length matches chains
        if len(looplists) != len(self.chains):
            raise ValueError(
                f"Number of loop files ({len(looplists)}) must match "
                f"number of chains ({len(self.chains)})"
            )
        
        # Read loop positions and offset by chain start
        loop_positions = []
        for loop_file, chain in zip(looplists, self.chains):
            start_idx = int(chain[0])
            try:
                with open(loop_file, 'r') as f:
                    lines = f.read().splitlines()
                for line in lines:
                    if not line.strip():  # Skip empty lines
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        i = int(parts[0]) + start_idx
                        j = int(parts[1]) + start_idx
                        loop_positions.append([i, j])
            except FileNotFoundError:
                raise FileNotFoundError(f"Loop file not found: {loop_file}")
        
        # Energy expression matching benchmark
        energy_expr = "qsi*0.5*(1. + tanh(mu*(rc - r)))"
        
        # Create CustomBondForce
        force = mm.CustomBondForce(energy_expr)
        force.name = name
        
        # Convert X from kT to kJ/mol
        kT_kJmol = sim_object.kT.value_in_unit(unit.kilojoule_per_mole)
        
        # Add global parameters (CustomBondForce uses addGlobalParameter directly)
        force.addGlobalParameter('mu', float(mu))
        force.addGlobalParameter('rc', float(rc))
        force.addGlobalParameter('qsi', float(X) * kT_kJmol)
        
        # Add bonds (convert from 1-indexed to 0-indexed: p[0]-1, p[1]-1)
        for pair in loop_positions:
            i = int(pair[0]) - 1
            j = int(pair[1]) - 1
            # Validate indices
            if i < 0 or j < 0 or i >= sim_object.N or j >= sim_object.N:
                raise ValueError(
                    f"Loop pair [{pair[0]}, {pair[1]}] results in out-of-bounds "
                    f"indices [{i}, {j}] for system with {sim_object.N} particles"
                )
            force.addBond(i, j)
        
        return force

    @staticmethod
    def _generate_bonds(sim_object, chains, extra_bonds=None):
        """
        Generate list of bonds from chain definitions.
        
        Parameters
        ----------
        sim_object : Simulation
            Simulation object.
        chains : list of tuples
            List of (start, end, isRing) tuples.
        extra_bonds : list of tuples, optional
            Additional bonds to include.
            
        Returns
        -------
        numpy.ndarray
            Array of bond pairs.
        """
        bonds_list = [] if extra_bonds is None else [tuple(b) for b in extra_bonds]
        for start, end, is_ring in chains:
            end = sim_object.N if end is None else end
            bonds_list.extend([(j, j + 1) for j in range(start, end - 1)])
            if is_ring:
                bonds_list.append((start, end - 1))
        return np.array(bonds_list, dtype=int)

    @staticmethod
    def _generate_triplets(sim_object, chains, extra_triplets=None):
        """
        Generate list of angle triplets from chain definitions.
        
        Parameters
        ----------
        sim_object : Simulation
            Simulation object.
        chains : list of tuples
            List of (start, end, isRing) tuples.
        extra_triplets : list of tuples, optional
            Additional triplets to include.
            
        Returns
        -------
        numpy.ndarray
            Array of angle triplets.
        """
        triplets_list = [] if extra_triplets is None else [tuple(t) for t in extra_triplets]
        for start, end, is_ring in chains:
            end = sim_object.N if end is None else end
            triplets_list.extend([(j - 1, j, j + 1) for j in range(start + 1, end - 1)])
            if is_ring:
                triplets_list.append((end - 2, end - 1, start))
                triplets_list.append((end - 1, start, start + 1))
        return np.array(triplets_list, dtype=int)

    @staticmethod
    def _to_array_1d(val, length):
        return np.full(length, val) if np.isscalar(val) else np.asarray(val)

    @staticmethod
    def _check_angle_bonds(triplets):
        """
        Check for duplicate angle triplets.
        
        Parameters
        ----------
        triplets : array-like
            List of angle triplets to check.
            
        Raises
        ------
        ValueError
            If duplicate triplets are found.
        """
        seen = set()
        for t in triplets:
            # Convert numpy array to tuple for hashing
            t_tuple = tuple(t)
            if t_tuple in seen:
                raise ValueError(f"Duplicate angle triplet found: {t}")
            seen.add(t_tuple)

    
    def _add_global_parameter(self, force, name, value):
        """
        Add a unique global parameter to the force, rewriting the energy expression
        to include the unique (prefixed) name if necessary.

        Parameters
        ----------
        force : mm.Force
            The force to add the parameter to.
        name : str
            Desired base name of the parameter.
        value : float or unit.Quantity
            Parameter value.

        Returns
        -------
        str
            The actual parameter name used (with prefix).
        """
        # Always use prefixed name
        force_name = getattr(force, 'name', 'force')
        unique_name = f"{force_name}_{name}"

        # Replace in energy expression if needed
        if hasattr(force, 'getEnergyFunction'):
            energy = force.getEnergyFunction()
            import re
            # Replace only whole word matches
            energy = re.sub(rf'\b{name}\b', unique_name, energy)
            force.setEnergyFunction(energy)

        # Add the global parameter with the new unique name
        force.addGlobalParameter(unique_name, value)
        return unique_name
