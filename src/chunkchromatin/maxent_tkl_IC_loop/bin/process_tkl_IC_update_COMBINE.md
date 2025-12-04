This file should combine:
1. All functions from process_tkl_update.py (maxent_loop/bin/)
2. All functions from process_IC_update.py (maxent_IC_loop/bin/)
3. A combined worker that processes both TKL and IC from same trajectory
4. A combined reduce that updates both epsilon and lambda_IC

Key functions to combine:
- Shared: f_switch, load_all_positions, constants
- TKL: _load_exp_Tkl, UpperTriOnlineCov, _covariance_pass_upper, process_one_replicate
- IC: _load_exp_phi_IC, PhiICOnlineCov, _compute_phi_IC_from_positions, process_one_replicate_IC
- Combined worker: processes both observables, saves combined .npz
- Combined reduce: updates both epsilon and lambda_IC independently
