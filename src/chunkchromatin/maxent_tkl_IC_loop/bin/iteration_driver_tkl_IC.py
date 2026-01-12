#!/usr/bin/env python3
"""
Iteration driver for combined TKL and IC optimization loop.
Orchestrates simulation, processing, and update steps for both observables.
"""
import argparse, os, json, shutil
from pathlib import Path
from chunkchromatin.maxent_tkl_IC_loop.bin.utils import ensure_dir, load_config, write_json, sbatch_submit, format_iter, make_executable

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
    epsilon_path = (iterd / "params" / "epsilon.npy").resolve()
    lambda_IC_path = (iterd / "params" / "lambda_IC.npy").resolve()
    seeds_json = (run_root / "seeds.json").resolve()
    
    # IC parameters
    d_init = cfg["ideal_chromosome"]["d_init"]
    d_end = cfg["ideal_chromosome"]["d_end"]

    # Get max_lambda_step_size from config (default 0.5)
    max_lambda_step_size = cfg.get("update", {}).get("max_lambda_step_size", 0.5)
    max_lambda_step_size_flag = f" --max-lambda-step-size {max_lambda_step_size}" if max_lambda_step_size is not None else ""
    
    # Get gradient_normalization from config (default None for no normalization)
    gradient_normalization = cfg.get("update", {}).get("gradient_normalization", None)
    
    # Get optimization method from config (default "newton")
    method = cfg.get("update", {}).get("method", "newton")
    
    # Get Adam hyperparameters if method is Adam
    adam_config = cfg.get("update", {}).get("adam", {})
    adam_lr = adam_config.get("learning_rate", None)
    adam_lr_ic = adam_config.get("learning_rate_ic", None)  # Separate LR for IC
    adam_beta1 = adam_config.get("beta1", None)
    adam_beta2 = adam_config.get("beta2", None)
    adam_epsilon = adam_config.get("epsilon", None)
    
    # Get adaptive step size config
    adaptive_step_size_config = cfg.get("update", {}).get("adaptive_step_size", None)

    # Relative step cap config
    relstep_cfg = cfg.get("update", {}).get("relative_step", {})
    relstep_target = relstep_cfg.get("target_rms_frac", None)
    relstep_max = relstep_cfg.get("max_frac", None)
    relstep_target_flag = f" --relstep-target-frac {relstep_target}" if relstep_target is not None else ""
    relstep_max_flag = f" --relstep-max-frac {relstep_max}" if relstep_max is not None else ""
    
    # Write adaptive config to JSON file for reduce step to read
    if adaptive_step_size_config is not None:
        adaptive_config_path = iterd / "obs" / "adaptive_step_size_config.json"
        write_json(adaptive_config_path, adaptive_step_size_config)

    # ------------------------------------
    # SIMULATION ARRAY (stage-specific res)
    # ------------------------------------
    sim_res = cfg["resources"]["simulation"]
    per_task = int(sim_res.get("per_task_replicates", 1))
    array_len = int(sim_res["array_len"])
    sim_script_tpl = (tpl_dir / "sbatch_sim_array_tkl_IC.sh")
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
        run_replicates_array=str((bin_dir / "run_replicates_array_tkl_IC.py").resolve()),
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
    # PROCESSING WORKERS (array 0..N-1) using process_tkl_IC_update.py worker
    # ------------------------------------
    procw = cfg["resources"]["processing"]["workers"]
    inputs = cfg["processing_inputs"]
    kf = (inputs.get("kernel_flags") or {})
    kernel_cli = ""

    for key in ("mu", "rc", "rcut", "beta"):
        if key in kf and kf[key] is not None:
            kernel_cli += f" --{key} {kf[key]}"

    procw_tpl = (tpl_dir / "sbatch_process_worker_tkl_IC.sh")
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
        monomer_types=str(Path(inputs["monomer_types"]).resolve()),
        exp_tkl=str(Path(inputs["exp_tkl"]).resolve()),
        exp_phi_IC=str(Path(inputs["exp_phi_IC"]).resolve()),
        process_tkl_IC_update=str((bin_dir / "process_tkl_IC_update.py").resolve()),
        kernel_cli=kernel_cli.strip(),
        d_init=d_init,
        d_end=d_end,
        chains=json.dumps(cfg["simulation"]["chains"]),
        resolution=inputs.get("resolution", ""),
    )
    procw_sbatch = iterd / "obs" / "submit_process_worker.sh"
    procw_sbatch.write_text(procw_text); make_executable(procw_sbatch)
    procw_jobid = sbatch_submit(procw_sbatch, extra_args=[f"--dependency=afterok:{sim_jobid}"])
    write_json(iterd / "obs" / "submit_workers.json", {"jobid": procw_jobid, "depends_on": sim_jobid})

    # ------------------------------------
    # PROCESSING REDUCE (single job) using process_tkl_IC_update.py reduce
    # ------------------------------------
    procr = cfg["resources"]["processing"]["reduce"]
    epsilon_dir = procr.get("epsilon_dir") or str(iterd / "update")
    lambda_dir = procr.get("lambda_dir") or str(iterd / "update")
    procr_tpl = (tpl_dir / "sbatch_process_reduce_tkl_IC.sh")
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
        epsilon_dir=epsilon_dir,
        lambda_dir=lambda_dir,
        process_tkl_IC_update=str((bin_dir / "process_tkl_IC_update.py").resolve()),
        iteration=args.iter,
        max_lambda_step_size_flag=max_lambda_step_size_flag,
        gradient_normalization=gradient_normalization if gradient_normalization else "",
        gradient_normalization_flag=f" --gradient-normalization {gradient_normalization}" if gradient_normalization else "",
        method=method,
        method_flag=f" --method {method}" if method else "",
        adam_lr=adam_lr if adam_lr is not None else "",
        adam_lr_flag=f" --adam-lr {adam_lr}" if adam_lr is not None else "",
        adam_lr_ic=adam_lr_ic if adam_lr_ic is not None else "",
        adam_lr_ic_flag=f" --adam-lr-ic {adam_lr_ic}" if adam_lr_ic is not None else "",
        adam_beta1=adam_beta1 if adam_beta1 is not None else "",
        adam_beta1_flag=f" --adam-beta1 {adam_beta1}" if adam_beta1 is not None else "",
        adam_beta2=adam_beta2 if adam_beta2 is not None else "",
        adam_beta2_flag=f" --adam-beta2 {adam_beta2}" if adam_beta2 is not None else "",
        adam_epsilon=adam_epsilon if adam_epsilon is not None else "",
        adam_epsilon_flag=f" --adam-epsilon {adam_epsilon}" if adam_epsilon is not None else "",
        relstep_target_flag=relstep_target_flag,
        relstep_max_flag=relstep_max_flag,
    )
    procr_sbatch = iterd / "obs" / "submit_process_reduce.sh"
    procr_sbatch.write_text(procr_text); make_executable(procr_sbatch)
    procr_jobid = sbatch_submit(procr_sbatch, extra_args=[f"--dependency=afterok:{procw_jobid}"])
    write_json(iterd / "obs" / "submit_reduce.json", {"jobid": procr_jobid, "depends_on": procw_jobid})

    # ------------------------------------
    # UPDATE (single job) using update_step_tkl_IC.py
    # ------------------------------------
    upd_script_tpl = (tpl_dir / "sbatch_update_tkl_IC.sh")
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
        update_step=str((bin_dir / "update_step_tkl_IC.py").resolve()),
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
