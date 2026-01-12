#!/usr/bin/env python3
"""
Standalone iteration driver for MaxEnt IC loop.

This version uses the standalone update step which only creates state.json
without updating parameters or submitting the next iteration.
"""

import argparse, os, json, shutil
from pathlib import Path
from chunkchromatin.maxent_IC_loop.bin.utils import ensure_dir, load_config, write_json, sbatch_submit, format_iter, make_executable

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--iter", required=True, type=int)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--proj-root", required=True)
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    proj_root = Path(args.proj_root).resolve()
    tpl_dir   = proj_root / "templates"
    bin_dir   = proj_root / "bin"
    cfg = load_config(Path(args.config))
    iterd = run_root / format_iter(args.iter)
    logd = ensure_dir(run_root / "logs")
    ensure_dir(iterd / "sims"); ensure_dir(iterd / "obs"); ensure_dir(iterd / "update")

    # Paths
    lambda_IC_path = (iterd / "params" / "lambda_IC.npy").resolve()
    # Epsilon is fixed and read directly from config path (not copied per iteration)
    # Check for both 'epsilon' (maxent_IC_loop) and 'epsilon_init' (maxent_tkl_IC_loop)
    epsilon_path_str = cfg["processing_inputs"].get("epsilon") or cfg["processing_inputs"].get("epsilon_init")
    if epsilon_path_str is None:
        raise ValueError("config.processing_inputs.epsilon or config.processing_inputs.epsilon_init must be set")
    epsilon_path = Path(epsilon_path_str).resolve()
    if not epsilon_path.exists():
        raise FileNotFoundError(f"Epsilon file not found at config path: {epsilon_path}")
    seeds_json = (run_root / "seeds.json").resolve()
    phi_exp_IC_path = (run_root / "exp_targets" / "phi_exp_IC.npy").resolve()

    # IC parameters
    d_init = cfg["ideal_chromosome"]["d_init"]
    d_end = cfg["ideal_chromosome"]["d_end"]

    # ------------------------------------
    # SIMULATION ARRAY (stage-specific res)
    # ------------------------------------
    sim_res = cfg["resources"]["simulation"]
    per_task = int(sim_res.get("per_task_replicates", 1))
    array_len = int(sim_res["array_len"])
    sim_script_tpl = (tpl_dir / "sbatch_sim_array_IC.sh")
    force_kwargs_json = json.dumps(cfg.get("force_kwargs", {}))
    sim_script_text = sim_script_tpl.read_text().format(
        job_name=f"{args.name}_sim_{args.iter:03d}",
        account=cfg["slurm"]["account"],
        partition=sim_res.get("partition", cfg["slurm"]["partition"]),
        time_limit=sim_res["time_limit"],
        cpus_per_task=sim_res["cpus_per_task"],
        mem=sim_res["mem"],
        gres=sim_res.get("gres", ""),
        array_max=array_len - 1,
        constraint_line=(f"#SBATCH --constraint={cfg['slurm']['constraint']}\n" if cfg["slurm"].get("constraint") else ""),
        qos_line=(f"#SBATCH --qos={cfg['slurm']['qos']}\n" if cfg["slurm"].get("qos") else ""),
        log_dir=str(logd),
        iter_dir=str(iterd),
        lambda_IC_path=str(lambda_IC_path),
        epsilon_path=str(epsilon_path),
        seeds_json=str(seeds_json),
        frames=cfg["simulation"]["frames"],
        burnin=cfg["simulation"]["burnin_frames"],
        save_frames=("1" if cfg["simulation"]["save_frames"] else "0"),
        n_reps=cfg["simulation"]["n_replicates"],
        obs_dir=str(iterd / "obs"),
        N=cfg["simulation"]["N"],
        density=cfg["simulation"]["density"],
        initialization_method=cfg["simulation"].get("initialization_method", "random_walk"),
        chains=json.dumps(cfg["simulation"]["chains"]),
        monomer_types=str(Path(cfg["processing_inputs"]["monomer_types"]).resolve()),
        run_replicates_array=str((bin_dir / "run_replicates_array_IC.py").resolve()),
        series_runner=str((bin_dir / "series_runner.py").resolve()),
        per_task_reps=per_task,
        d_init=d_init,
        d_end=d_end,
        force_kwargs=force_kwargs_json,
    )
    sim_sbatch = iterd / "sims" / "submit_sim_array.sh"
    sim_sbatch.write_text(sim_script_text); make_executable(sim_sbatch)
    sim_jobid = sbatch_submit(sim_sbatch)
    write_json(iterd / "sims" / "submit.json", {"jobid": sim_jobid})


    # ------------------------------------
    # PROCESSING WORKERS (array 0..N-1) using process_IC_update.py worker
    # ------------------------------------
    procw = cfg["resources"]["processing"]["workers"]
    inputs = cfg["processing_inputs"]
    kf = (inputs.get("kernel_flags") or {})
    kernel_cli = ""

    for key in ("mu", "rc", "rcut", "beta"):
        if key in kf and kf[key] is not None:
            kernel_cli += f" --{key} {kf[key]}"

    procw_tpl = (tpl_dir / "sbatch_process_worker_IC.sh")
    procw_text = procw_tpl.read_text().format(
        job_name=f"{args.name}_pwrk_{args.iter:03d}",
        account=cfg["slurm"]["account"],
        partition=cfg["slurm"]["partition"],
        time_limit=procw["time_limit"],
        cpus_per_task=procw["cpus_per_task"],
        workers=procw["cpus_per_task"],
        mem=procw["mem"],
        array_max=int(procw["array_len"]) - 1,
        constraint_line=(f"#SBATCH --constraint={cfg['slurm']['constraint']}\n" if cfg["slurm"].get("constraint") else ""),
        qos_line=(f"#SBATCH --qos={cfg['slurm']['qos']}\n" if cfg["slurm"].get("qos") else ""),
        log_dir=str(logd),
        iter_dir=str(iterd),
        obs_dir=str(iterd / "obs"),
        replicate_root=str(iterd / "sims"),
        n_reps=cfg["simulation"]["n_replicates"],
        io_k=int(procw.get("io_k", 2)),
        exp_phi_IC=str(Path(inputs["exp_phi_IC"]).resolve()),
        process_IC_update=str((bin_dir / "process_IC_update.py").resolve()),
        kernel_cli=kernel_cli.strip(),
        d_init=d_init,
        d_end=d_end,
        chains=json.dumps(cfg["simulation"]["chains"]),
    )
    procw_sbatch = iterd / "obs" / "submit_process_worker.sh"
    procw_sbatch.write_text(procw_text); make_executable(procw_sbatch)
    procw_jobid = sbatch_submit(procw_sbatch, extra_args=[f"--dependency=afterok:{sim_jobid}"])
    write_json(iterd / "obs" / "submit_workers.json", {"jobid": procw_jobid, "depends_on": sim_jobid})

    # ------------------------------------
    # PROCESSING REDUCE (single job) using process_IC_update.py reduce
    # ------------------------------------
    procr = cfg["resources"]["processing"]["reduce"]
    lambda_dir = procr.get("lambda_dir") or str(iterd / "update")
    procr_tpl = (tpl_dir / "sbatch_process_reduce_IC.sh")
    procr_text = procr_tpl.read_text().format(
        job_name=f"{args.name}_pred_{args.iter:03d}",
        account=cfg["slurm"]["account"],
        partition=cfg["slurm"]["partition"],
        time_limit=procr["time_limit"],
        cpus_per_task=procr["cpus_per_task"],
        mem=procr["mem"],
        constraint_line=(f"#SBATCH --constraint={cfg['slurm']['constraint']}\n" if cfg["slurm"].get("constraint") else ""),
        qos_line=(f"#SBATCH --qos={cfg['slurm']['qos']}\n" if cfg["slurm"].get("qos") else ""),
        log_dir=str(logd),
        iter_dir=str(iterd),
        obs_dir=str(iterd / "obs"),
        lambda_dir=lambda_dir,
        process_IC_update=str((bin_dir / "process_IC_update.py").resolve()),
        iteration=args.iter,
    )
    procr_sbatch = iterd / "obs" / "submit_process_reduce.sh"
    procr_sbatch.write_text(procr_text); make_executable(procr_sbatch)
    procr_jobid = sbatch_submit(procr_sbatch, extra_args=[f"--dependency=afterok:{procw_jobid}"])
    write_json(iterd / "obs" / "submit_reduce.json", {"jobid": procr_jobid, "depends_on": procw_jobid})

    # ------------------------------------
    # UPDATE (single job) using update_step_IC_standalone.py
    # ------------------------------------
    upd_script_tpl = (tpl_dir / "sbatch_update_IC.sh")
    upd_script_text = upd_script_tpl.read_text().format(
        job_name=f"{args.name}_update_{args.iter:03d}",
        account=cfg["slurm"]["account"],
        partition=cfg["slurm"]["partition"],
        time_limit="00:20:00",
        cpus_per_task=2,
        mem="4G",
        constraint_line=(f"#SBATCH --constraint={cfg['slurm']['constraint']}\n" if cfg["slurm"].get("constraint") else ""),
        qos_line=(f"#SBATCH --qos={cfg['slurm']['qos']}\n" if cfg["slurm"].get("qos") else ""),
        log_dir=str(logd),
        iter_dir=str(iterd),
        update_step=str((bin_dir / "update_step_IC_standalone.py").resolve()),
        run_root=str(run_root),
        iter_idx=args.iter,
        config_yaml=str(Path(args.config).resolve()),
    )
    upd_sbatch = iterd / "update" / "submit_update.sh"
    upd_sbatch.write_text(upd_script_text); make_executable(upd_sbatch)
    upd_jobid = sbatch_submit(upd_sbatch, extra_args=[f"--dependency=afterok:{procr_jobid}"])
    write_json(iterd / "update" / "submit.json", {"jobid": upd_jobid, "depends_on": procr_jobid})

if __name__ == "__main__":
    main()
