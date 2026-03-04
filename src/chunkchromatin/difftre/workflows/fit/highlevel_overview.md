difftre fit workflow high level overview

Steps;
driver, simulation, observable, reweighting

driver
driver.py is the entry point
it determines the state of the fit run
if it is a new run, it creates a run manifest and submits simulations
otherwise, it reads the manifest to determine state: current step and progress

simulation
simulation_worker.py
    read config for parameters and run simulation
submit_simulations.py
    create job array to run simulation worker for n replicates
    update manifest with progress
    submit observable step

observable / energy calc
- first figure out chunk size - how many trajectories to read into memory
- then stream through chunks, calculating observables and energies
- save per-frame observables.