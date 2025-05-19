import logging
import time
import openmm as mm
import simtk
import numpy as np

logging.basicConfig(level=logging.INFO)


class Simulation(object):
    def __init__(self, **kwargs):
        """
        Initialize a simulation object.

        Parameters
        ----------
        integrator_type : str, optional
            Type of integrator to use. Default is 'Langevin'.
        temperature : float, optional
            Temperature in reduced units. Default is 1.0.
        gamma : float, required
            Friction/damping constant in units of reciprocal time (1/τ). Default is 0.1.
        timestep : float, required
            Simulation time step in units of τ. Default is 0.005.
        platform : str, required
            Platform to use. Default is 'CPU'.
            Options are 'CUDA', 'OpenCL', or 'Reference'.
        reporter : HDF5Reporter, optional
            Reporter object for saving simulation data. If None, no data will be saved.
        """


        #integrator parameters
        self.integrator_type = kwargs.get("integrator_type", "Langevin")
        self.temperature = kwargs.get("temperature", 300.0) * simtk.unit.kelvin
        self.gamma = kwargs.get("gamma", 1.0) / simtk.unit.picosecond
        self.timestep = kwargs.get("timestep", 0.002) * simtk.unit.picosecond

        #platform parameters
        self.platform = kwargs.get("platform", "CPU")
        
        #system parameters
        self.system = mm.System()

        #global parameters
        self.N = kwargs.get("N")
        self.kB = simtk.unit.BOLTZMANN_CONSTANT_kB * simtk.unit.AVOGADRO_CONSTANT_NA
        self.kT = self.temperature * self.kB
        self.conlen = kwargs.get("conlen", 1.0)

        #Internal state
        self.positions = None
        self.velocities = None
        self.applied_forces = None
        self.step = 0
        self.block = 0
        self.time = 0

        #setup openmm system
        self.create_system_object()
        self.create_integrator_object()
        self.create_platform_object()

        #initialize reporter
        self.reporter = kwargs.get("reporter", None)
        
        assert self.temperature is not None, "Temperature must be specified."
        assert self.N is not None, "N must be specified."
    #need to add topology
    
    def set_positions(self, positions, center=np.zeros(3), random_offset=1e-5):
        """
        Set the positions of particles in the simulation, with centering and a small random offset.

        Parameters
        ----------
        positions : numpy.ndarray
            An Nx3 array of particle positions (in reduced units).
        center : array-like, optional
            The coordinate to center the positions around (default: [0, 0, 0]).
        random_offset : float, optional
            Magnitude of random offset to add to each coordinate (default: 1e-5).

        Returns
        -------
        None
        """
        # Validate input
        if positions.shape[0] != self.N:
            raise ValueError(f"Expected {self.N} particles, got {positions.shape[0]}")
        if positions.shape[1] != 3:
            raise ValueError("Positions must be Nx3 array")

        # Center positions
        centroid = np.mean(positions, axis=0)
        pos_centered = positions - centroid + np.asarray(center)

        # Add small random offset
        pos_final = pos_centered + np.random.uniform(-random_offset, random_offset, pos_centered.shape)

        self.positions = pos_final * simtk.unit.nanometers

    def set_velocities(self):
        """
        Set initial velocities according to Maxwell-Boltzmann distribution at the specified temperature.
        """
        if not hasattr(self, 'context'):
            raise RuntimeError("Context must be created before setting velocities")
        
        # Get the mass of the first particle (all particles have same mass in our case)
        mass = self.system.getParticleMass(0)
        
        # Calculate standard deviation based on temperature and mass
        sigma = np.sqrt((self.kB * self.temperature / mass)._value)
        
        # Generate random velocities from normal distribution
        velocities = np.random.normal(0, sigma, size=(self.N, 3))
        
        # Convert velocities to OpenMM units
        velocities = velocities * simtk.unit.nanometers / simtk.unit.picosecond
        
        # Set velocities in the context
        self.context.setVelocities(velocities)
    
    def add_force(self, force):
        """
        Add a force to the system.

        Parameters
        ----------
        force : mm.Force
            The force to add.

        Returns
        -------
        int
            Index of the added force.
        """
        return self.system.addForce(force)
    
    def run_simulation_block(
        self,
        steps=None,
        check_functions=[],
        get_velocities=False,
        save=True,
        save_extras={},
    ):
        """
        Perform one block of simulation steps.

        Parameters
        ----------
        steps : int or None
            Number of timesteps to perform. If None, uses default steps.
        check_functions : list of functions, optional
            List of functions to call every block. Coordinates are passed to each function.
            If any function returns False, simulation is stopped.
        get_velocities : bool, default=False
            If True, will return velocities in the result.
        save : bool, default=True
            If True, save results of this block.
        save_extras : dict
            Additional information to save with the results.

        Returns
        -------
        dict
            Dictionary containing simulation results including positions, energies, and time.
        """
        # Set default steps if not provided
        if steps is None:
            steps = 1000  # Default steps per block

        # Perform integration steps
        start_time = time.time()
        self.integrator.step(steps)
        end_time = time.time()

        # Get state information
        self.state = self.context.getState(
            getPositions=True,
            getVelocities=get_velocities,
            getEnergy=True
        )

        # Extract coordinates and convert to numpy array
        coords = self.state.getPositions(asNumpy=True)
        coords = np.array(coords, dtype=np.float32)

        # Calculate energies per particle (in units of kT)
        eK = self.state.getKineticEnergy()._value / self.N / self.kT._value
        eP = self.state.getPotentialEnergy()._value / self.N / self.kT._value
        
        # Get time in reduced units (τ)
        curtime = self.state.getTime()._value  # This gives time in reduced units

        # Calculate steps per second
        steps_per_second = steps / (end_time - start_time)

        # Log simulation progress
        msg = f"block {self.block:4d} "
        msg += f"pos[1]=[{coords[0][0]:.1f} {coords[0][1]:.1f} {coords[0][2]:.1f}] "
        msg += f"t={curtime:.1f}ps "
        msg += f"kin={eK:.2f} pot={eP:.2f} "
        msg += f"SPS={steps_per_second:.0f}"

        logging.info(msg)

        # Run check functions if provided
        check_fail = False
        for check_function in check_functions:
            if not check_function(coords):
                check_fail = True
                break

        # Basic error checks
        if np.isnan(coords).any():
            raise RuntimeError("Coordinates contain NaN values")
        if np.isnan(eK) or np.isnan(eP):
            raise RuntimeError("Energy values contain NaN")
        if check_fail:
            raise RuntimeError("Custom checks failed")

        # Prepare result dictionary
        result = {
            "pos": coords,
            "potentialEnergy": eP,
            "kineticEnergy": eK,
            "time": curtime,
            "block": self.block,
        }

        # Add velocities if requested
        if get_velocities:
            result["vel"] = self.state.getVelocities(asNumpy=True)

        # Add any extra information
        result.update(save_extras)

        # Save results using reporter if available and save is True
        if self.reporter is not None and save:
            self.reporter.report("data", result)

        # Update simulation state
        self.block += 1
        self.step += steps
        self.time = curtime

        return result
    
    def save_initial_state(self):
        """
        Save initial simulation state using the reporter if available.
        """
        if self.reporter is not None:
            # Save initial arguments
            init_args = {
                "integrator_type": self.integrator_type,
                "temperature": self.temperature,
                "gamma": self.gamma,
                "timestep": self.timestep,
                "platform": self.platform,
                "N": self.N,
                "conlen": self.conlen
            }
            self.reporter.report("initArgs", init_args)
            
            # Save starting conformation
            if self.positions is not None:
                self.reporter.report("starting_conformation", {"pos": self.positions})

    def print_stats(self):
        """
        Print current simulation statistics.
        """
        if not hasattr(self, 'state'):
            print("No simulation state available")
            return

        # Get current state
        self.state = self.context.getState(getEnergy=True)
        
        # Calculate energies per particle (in units of kT)
        eK = self.state.getKineticEnergy()._value / self.N / self.kT._value
        eP = self.state.getPotentialEnergy()._value / self.N / self.kT._value
        total_energy = eK + eP
        
        # Get time in reduced units
        curtime = self.state.getTime()._value

        # Print statistics
        print("\nSimulation Statistics:")
        print(f"Current block: {self.block}")
        print(f"Total steps: {self.step}")
        print(f"Simulation time: {curtime:.2f}τ")
        print(f"Kinetic energy: {eK:.2f} kT/particle")
        print(f"Potential energy: {eP:.2f} kT/particle")
        print(f"Total energy: {total_energy:.2f} kT/particle")
    

    def create_system_object(self):
        self.system = mm.System()
        #add particles to system
        for _ in range(self.N):
            self.system.addParticle(1.0)  # Add particles with mass 1.0

    def create_integrator_object(self):
        if self.integrator_type == 'Langevin':
            self.integrator = mm.LangevinIntegrator(
                self.temperature,  # In OpenMM units
                self.gamma,    
                self.timestep  
            )
        elif self.integrator_type == 'variableLangevin':
            self.integrator = mm.VariableLangevinIntegrator(
                self.temperature, #in openmm units
                self.gamma,
                self.timestep
            )

    def create_platform_object(self):
        if self.platform == 'CUDA':
            self.platform_object = mm.Platform.getPlatformByName('CUDA')
            self.platform_properties = {'CudaPrecision': 'double'}
        elif self.platform == 'OpenCL':
            self.platform_object = mm.Platform.getPlatformByName('OpenCL')
            self.platform_properties = {'OpenCLPrecision': 'double'}
        elif self.platform == 'Reference':
            self.platform_object = mm.Platform.getPlatformByName('Reference')
        elif self.platform == 'CPU':
            self.platform_object = mm.Platform.getPlatformByName('CPU')
        else:
            print("platform_type can be either CUDA, OpenCL, or CPU")

    def create_context(self):
        self.context = mm.Context(self.system, self.integrator, self.platform_object)
        self.context.setPositions(self.positions)
    