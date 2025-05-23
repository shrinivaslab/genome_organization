import logging
import time
from openmm import unit
import openmm as mm
import numpy as np

class Lamina(object):
    def __init__(self, N, chains, sim_object):
        """
        Initialize a lamina object.
        """
        self.N = N
        self.chains = chains
        self.sim_object = sim_object

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

        # Add particles
        particles = range(sim_object.N) if particles is None else particles
        for i in particles:
            force.addParticle(int(i), [])

        # Parameters (no units)
        force.addGlobalParameter("kb", k * sim_object.kT.value_in_unit(unit.kilojoule_per_mole))
        force.addGlobalParameter("aa", r - 1.0 / k)
        force.addGlobalParameter("t", (1.0 / k) / 10.0)
        force.addGlobalParameter("tt", 0.01)
        force.addGlobalParameter("invert_sign", -1.0 if invert else 1.0)

        # Center of confinement sphere
        force.addGlobalParameter("x0", center[0])
        force.addGlobalParameter("y0", center[1])
        force.addGlobalParameter("z0", center[2])

        sim_object.sphericalConfinementRadius = r  # for bookkeeping

        return force
    
    def add_C_monomer_pinning(self, sim_object, C_monomers, density=0.3, k=1.0):
        """Pins C monomers to random points on the nuclear lamina using spherical confinement potential.
        
        Parameters
        ----------
        sim_object : Simulation object
            The simulation object
        C_monomers : list
            List of C monomer indices
        density : float
            Density of monomers (0.33 as specified)
        k : float
            Force constant in units of kT
        """
        name = 'C_monomer_pinning'
        # Calculate local sphere radius based on density
        local_volume = len(C_monomers)/density
        local_radius = (local_volume)**(1/3.)


        # Generate random center points on the nuclear lamina for each group of C monomers
        theta = np.random.random() * 2 * np.pi
        phi = np.arccos(2 * np.random.random() - 1)
        
        # Convert to Cartesian coordinates on the lamina
        R = sim_object.conlen  # radius of main confining sphere
        center_x = R * np.sin(phi) * np.cos(theta)
        center_y = R * np.sin(phi) * np.sin(theta)
        center_z = R * np.cos(phi)
        
        # Create force using same form as spherical confinement
        force = mm.CustomExternalForce(
            "step(r-aa) * kb * (sqrt((r-aa)*(r-aa) + t*t) - t);"
            "r = sqrt((x-x0)^2 + (y-y0)^2 + (z-z0)^2 + tt^2)"
        )
        force.name = name
        # Add global parameters
        force.addGlobalParameter("kb", k * sim_object.kT.value_in_unit(unit.kilojoule_per_mole) / sim_object.conlen)
        force.addGlobalParameter("aa", (local_radius - 1.0/k) * sim_object.conlen)
        force.addGlobalParameter("t", (1.0/k) * sim_object.conlen / 10.0)
        force.addGlobalParameter("tt", 0.01 * sim_object.conlen)
        force.addGlobalParameter("x0", center_x * sim_object.conlen)
        force.addGlobalParameter("y0", center_y * sim_object.conlen)
        force.addGlobalParameter("z0", center_z * sim_object.conlen)
        
        # Add only C monomers
        for particle in C_monomers:
            force.addParticle(particle, [])

        #example usage
        # Identify C monomers (assuming they're type 2)
        # C_monomers = [i for i in range(N) if monomer_types[i] == 2]

        # # Create pinning points for each group of consecutive C monomers
        # # You might need to group consecutive C monomers if they should be pinned together
        # center_point = add_C_monomer_pinning(sim, C_monomers, density=0.33)
        
        return force

    def add_B_monomer_lamina_attraction(self,sim_object, B_monomers, BLam=1.0):
        """Implements the exact lamina attraction for B monomers as described in methods"""
        name = 'B_monomer_lamina_attraction'
        force = mm.CustomExternalForce(
            "BLam * (L - R + 1) * (R - L + 1) * step(1 - (R - L)) * step(R - L + 1);"
            "L = sqrt(r2 + tt2);"
            "r2 = x*x + y*y + z*z"
        )
        
        # Add parameters
        force.addGlobalParameter("BLam", BLam * sim_object.kT.value_in_unit(unit.kilojoule_per_mole))
        force.addGlobalParameter("R", sim_object.conlen)
        force.addGlobalParameter("tt2", 0.01 * 0.01)  # .01^2 as specified in methods
        force.name = name

        # Add only B monomers
        for particle in B_monomers:
            force.addParticle(particle, [])
        
        return force