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

        # Convert k from kT/rad² to kJ/mol/rad²
        k_openmm = k_array * self.sim_object.kT.value_in_unit(unit.kilojoule_per_mole)

        energy = "kT * angK * 0.5 * (theta - angT0)^2"
        angle_force = mm.CustomAngleForce(energy)
        angle_force.setForceGroup(force_group)
        self._add_global_parameter(angle_force, "kT", self.sim_object.kT)
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
        Add a global parameter to a force. If the parameter name is used literally
        in the energy function, it will be added as-is. Otherwise, it will be prefixed
        with the force name to avoid collisions.

        Parameters
        ----------
        force : mm.Force
            The force to add the parameter to.
        name : str
            Name of the parameter.
        value : float or unit.Quantity
            Value of the parameter.

        Returns
        -------
        str
            The actual parameter name used.
        """
        # Check if energy expression exists and parameter is used literally
        try:
            energy = force.getEnergyFunction()
            # Match full word occurrences only (avoid substrings)
            import re
            literal_usage = re.search(rf'\b{name}\b', energy) is not None
        except AttributeError:
            # Force has no energy function
            literal_usage = False

        if literal_usage:
            param_name = name
        else:
            force_name = getattr(force, 'name', 'force')
            param_name = f"{force_name}_{name}"

        force.addGlobalParameter(param_name, value)
        return param_name
