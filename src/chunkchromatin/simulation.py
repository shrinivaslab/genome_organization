# -----------------------------------------------------------------------------
# simulation.pys
# -----------------------------------------------------------------------------
# Units & conventions (read me):
# -----------------------------------------------------------------------------
# OpenMM is run in PHYSICAL units using its Quantity system.
# - Positions/velocities/timestep/temperature are carried with units
#   (nanometer, picosecond, kelvin, etc.) inside the integrator/context.
#
# What we store / report:
# - Trajectory frames written via the reporter are plain NumPy float32
#   positions in **nanometers** (we convert from Quantity -> nm -> np.float32).
# - Energies printed during the run (`potentialEnergy`, `kineticEnergy`) are
#   normalized to **kT per particle** for convenience in reduced-unit thinking.
# - The detailed breakdown from `get_energy_breakdown()` is left in **kJ/mol**
#   (OpenMM’s native energy unit) to preserve fidelity and avoid double scaling.
#
# Reduced-unit interpretation:
# - The model is conceptually “reduced” (σ ≈ 1, ε ≈ kT), but the engine runs
#   in physical units. If you need pure reduced coordinates, divide stored
#   positions by your σ (in nm). Likewise, divide energies by kT for kT units.
#
# Bottom line:
# - Inside OpenMM: physical units (Quantity).
# - On disk: positions = nm (float32); optional per-force energies = kJ/mol JSON.
# - On logs: eK/eP reported as kT/particle.
# ----------------------------------------------------------------------------- 

import logging
import time
from openmm import unit
import openmm as mm
import numpy as np

logging.basicConfig(level=logging.INFO)

class EKExceedsError(Exception):
    pass

# self.reporter must implement a `.report(...)` method.
# Supported signatures:
# - reporter.report(data)
# - reporter.report(name, data)

class Simulation(object):
    def __init__(self, **kwargs):
        self.integrator_type = kwargs.get("integrator_type", "Langevin")
        self.temperature = kwargs.get("temperature", 300.0) * unit.kelvin
        self.gamma = kwargs.get("gamma", 0.05) / unit.picosecond
        self.timestep = kwargs.get("timestep", 70) * unit.femtoseconds
        self.platform = kwargs.get("platform", "CPU")

        self.system = mm.System()
        self.N = kwargs.get("N")
        self.kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
        self.kT = self.temperature * self.kB
        self.conlen = kwargs.get("conlen", 1.0) * unit.nanometer
        self.eK_critical = kwargs.get("eK_critical", 10.0)

        self.positions = None
        self.velocities = None
        self.applied_forces = None
        self.step = 0
        self.block = 0
        self.time = 0

        #add force groups
        self._force_names = []      # parallel to groups
        self._force_groups = []     # integers 0..31

        self.create_system_object()
        self.create_integrator_object()
        self.create_platform_object()

        self.reporter = kwargs.get("reporter", None)

        assert self.temperature is not None, "Temperature must be specified."
        assert self.N is not None, "N must be specified."

    def set_positions(self, positions, center=np.zeros(3), random_offset=1e-5):
        if positions.shape[0] != self.N:
            raise ValueError(f"Expected {self.N} particles, got {positions.shape[0]}")
        if positions.shape[1] != 3:
            raise ValueError("Positions must be Nx3 array")

        centroid = np.mean(positions, axis=0)
        pos_centered = positions - centroid + np.asarray(center)
        pos_final = pos_centered + np.random.uniform(-random_offset, random_offset, pos_centered.shape)
        self.positions = pos_final * unit.nanometers

    def set_velocities(self):
        if not hasattr(self, 'context'):
            raise RuntimeError("Context must be created before setting velocities")

        mass = self.system.getParticleMass(0)
        kT = self.kB * self.temperature
        sigma = (kT / mass).sqrt()

        velocities = np.random.normal(0.0, 1.0, size=(self.N, 3)) * sigma.value_in_unit(unit.nanometer / unit.picosecond)
        velocities_quantity = unit.Quantity(velocities, unit.nanometer / unit.picosecond)
        self.context.setVelocities(velocities_quantity)

    def add_force(self, force, name=None):
        """
        Add a Force to the System, assign it a unique force group, and remember its name.
        Returns the index from System.addForce (unchanged behavior).
        """
        # Next group index (0..31). OpenMM supports up to 32 groups.
        next_group = len(self._force_groups)
        if next_group >= 32:
            raise ValueError("OpenMM supports at most 32 force groups; already have 32.")

        # Assign the group and add to system
        force.setForceGroup(next_group)
        idx = self.system.addForce(force)

        # Record metadata
        force_name = name if name is not None else f"force_{next_group}"
        self._force_names.append(force_name)
        self._force_groups.append(next_group)
        return idx

    def get_energy_breakdown(self):
        """
        Returns a dict with:
        - 'potential_total_kJmol'
        - 'kinetic_total_kJmol'
        - 'potential_by_force_kJmol' : {force_name: value, ...}
        Notes:
        * Kinetic energy is system-wide (not split by group).
        * Potential components are obtained by querying each force group.
        """
        if not hasattr(self, 'context'):
            raise RuntimeError("Context has not been created.")

        # Total energies
        state_all = self.context.getState(getEnergy=True)
        pot_total = state_all.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        kin_total = state_all.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)

        # Per-group potentials
        per_force = {}
        for name, grp in zip(self._force_names, self._force_groups):
            mask = (1 << grp)
            state_grp = self.context.getState(getEnergy=True, groups=mask)
            e_grp = state_grp.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            per_force[name] = e_grp

        return {
            "potential_total_kJmol": pot_total,
            "kinetic_total_kJmol": kin_total,
            "potential_by_force_kJmol": per_force,
        }


    def run_simulation_block(
        self,
        steps=None,
        check_functions=[],
        get_velocities=False,
        get_energies=False,       # <-- new argument
        save=True,
        save_extras={},
    ):
        if steps is None:
            steps = 1000

        start_time = time.time()
        self.integrator.step(steps)
        end_time = time.time()
        steps_per_second = steps / (end_time - start_time)

        self.state = self.context.getState(
            getPositions=True,
            getVelocities=get_velocities,
            getEnergy=True
        )

        curtime_ns = self.state.getTime().value_in_unit(unit.nanosecond)
        coords = self.state.getPositions(asNumpy=True)
        coords_nm = coords.value_in_unit(unit.nanometer)
        coords_nm = np.array(coords_nm, dtype=np.float32)

        kinetic_energy = self.state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        potential_energy = self.state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        kT_value = self.kT.value_in_unit(unit.kilojoule_per_mole)

        eK = kinetic_energy / (kT_value * self.N)
        eP = potential_energy / (kT_value * self.N)

        msg = f"block {self.block:4d} "
        msg += f"pos[1]=[{coords_nm[0][0]:.1f} {coords_nm[0][1]:.1f} {coords_nm[0][2]:.1f}] "
        msg += f"t={curtime_ns:.1f}ns "
        msg += f"kin={eK:.2f} pot={eP:.2f} "
        msg += f"SPS={steps_per_second:.0f}"

        logging.info(msg)

        check_fail = False
        for check_function in check_functions:
            if not check_function(coords):
                check_fail = True
                break

        if eK > self.eK_critical and self.integrator_type.lower() != "brownian":
            raise EKExceedsError("Ek={1} exceeds {0}".format(self.eK_critical, eK))
        if np.isnan(coords).any():
            raise RuntimeError("Coordinates contain NaN values")
        if np.isnan(eK) or np.isnan(eP):
            raise RuntimeError("Energy values contain NaN")
        if check_fail:
            raise RuntimeError("Custom checks failed")

        # Base result dict
        result = {
            "pos": coords_nm,
            "potentialEnergy": eP,
            "kineticEnergy": eK,
            "time": curtime_ns,
            "block": self.block,
        }

        # Optionally add velocities
        if get_velocities:
            velocities = self.state.getVelocities(asNumpy=True)
            result["vel"] = velocities.value_in_unit(unit.nanometer / unit.picosecond)

        # Optionally add detailed energy breakdown
        if get_energies:
            result["energy_breakdown"] = self.get_energy_breakdown()

        # Merge in any additional user-provided data
        result.update(save_extras)

        # Save via reporter if enabled
        if self.reporter is not None and save:
            self._dispatch_report(result)

        # Update counters
        self.block += 1
        self.step += steps
        self.time = curtime_ns

        return result


    def get_positions(self, as_numpy=True, in_units=unit.nanometer):
        if not hasattr(self, "context"):
            raise RuntimeError("Context has not been created.")
        state = self.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=as_numpy)
        return pos.value_in_unit(in_units) if as_numpy else pos

    def save_initial_state(self):
        if self.reporter is not None:
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
            if self.positions is not None:
                self.reporter.report("starting_conformation", {"pos": self.positions.value_in_unit(unit.nanometer)})

    def print_stats(self):
        if not hasattr(self, 'state'):
            print("No simulation state available")
            return

        self.state = self.context.getState(getEnergy=True)
        eK = self.state.getKineticEnergy()._value / self.N / self.kT._value
        eP = self.state.getPotentialEnergy()._value / self.N / self.kT._value
        total_energy = eK + eP
        curtime = self.state.getTime()._value

        print("\nSimulation Statistics:")
        print(f"Current block: {self.block}")
        print(f"Total steps: {self.step}")
        print(f"Simulation time: {curtime:.2f}τ")
        print(f"Kinetic energy: {eK:.2f} kT/particle")
        print(f"Potential energy: {eP:.2f} kT/particle")
        print(f"Total energy: {total_energy:.2f} kT/particle")

    def create_system_object(self):
        self.system = mm.System()
        for _ in range(self.N):
            self.system.addParticle(1.0)

    def create_integrator_object(self):
        if self.integrator_type == 'Langevin':
            self.integrator = mm.LangevinIntegrator(self.temperature, self.gamma, self.timestep)
        elif self.integrator_type == 'variableLangevin':
            self.integrator = mm.VariableLangevinIntegrator(self.temperature, self.gamma, self.timestep)

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

    def _dispatch_report(self, result):
        if self.reporter is None:
            return

        report_fn = getattr(self.reporter, "report", None)
        if not callable(report_fn):
            raise AttributeError("Reporter must have a callable 'report' method")

        try:
            # Always pass dicts through so energies (if present) are preserved
            report_fn(result)
        except TypeError:
            try:
                report_fn("data", result)
            except Exception as e:
                raise RuntimeError(f"Reporter could not handle report call: {e}")
