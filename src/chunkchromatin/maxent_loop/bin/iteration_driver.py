#!/usr/bin/env python3
import argparse, os, json, shutil
from pathlib import Path
from chunkchromatin.maxent_loop.bin.utils import ensure_dir, load_config, write_json, sbatch_submit, format_iter, make_executable

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
    eps_path = (iterd / "params" / "epsilon.npy").resolve()
    seeds_json = (run_root / "seeds.json").resolve()
    kernel_json = (run_root / "exp_targets" / "kernel.json").resolve()
    targets_npy = (run_root / "exp_targets" / "T_type_kl.npy").resolve()

    # ------------------------------------
    # SIMULATION ARRAY (stage-specific res)
    # ------------------------------------
    sim_res = cfg["resources"]["simulation"]
    per_task = int(sim_res.get("per_task_replicates", 1))
    array_len = int(sim_res["array_len"])
    # Safety: array_len * per_task should cover n_replicates; extra slots are clipped in the template
    sim_script_tpl = (tpl_dir / "sbatch_sim_array.sh")
    sim_script_text = sim_script_tpl.read_text().format(
        job_name=f"{args.name}_sim_{args.iter:03d}",
        account=cfg["slurm"]["account"],
        partition=sim_res.get("partition", cfg["slurm"]["partition"]),  # use sim-specific or default
        time_limit=sim_res["time_limit"],
        cpus_per_task=sim_res["cpus_per_task"],
        mem=sim_res["mem"],
        gres=sim_res.get("gres", ""),  # GPU allocation for simulations
        array_max=array_len - 1,
        constraint_line=(f"#SBATCH --constraint={cfg['slurm']['constraint']}\n" if cfg["slurm"].get("constraint") else ""),
        qos_line=(f"#SBATCH --qos={cfg['slurm']['qos']}\n" if cfg["slurm"].get("qos") else ""),
        log_dir=str(logd),
        iter_dir=str(iterd),
        eps_path=str(eps_path),
        seeds_json=str(seeds_json),
        frames=cfg["simulation"]["frames"],
        burnin=cfg["simulation"]["burnin_frames"],
        save_frames=("1" if cfg["simulation"]["save_frames"] else "0"),
        n_reps=cfg["simulation"]["n_replicates"],
        kernel_json=str(kernel_json),
        targets_npy=str(targets_npy),
        obs_dir=str(iterd / "obs"),
        n_types=cfg["simulation"]["n_types"],
        monomer_types=str(Path(cfg["processing_inputs"]["monomer_types"]).resolve()),
        interaction_matrix=str(eps_path),
        run_replicates_array=str((bin_dir / "run_replicates_array.py").resolve()),
        series_runner=str((bin_dir / "series_runner.py").resolve()),
        per_task_reps=per_task,
    )
    sim_sbatch = iterd / "sims" / "submit_sim_array.sh"
    sim_sbatch.write_text(sim_script_text); make_executable(sim_sbatch)
    sim_jobid = sbatch_submit(sim_sbatch)
    write_json(iterd / "sims" / "submit.json", {"jobid": sim_jobid})


    # ------------------------------------
    # PROCESSING WORKERS (array 0..N-1) using process_tkl_update.py worker
    # ------------------------------------
    procw = cfg["resources"]["processing"]["workers"]
    inputs = cfg["processing_inputs"]
    kf = (inputs.get("kernel_flags") or {})
    kernel_cli = ""

    for key in ("mu", "rc", "rcut", "beta"):
        if key in kf and kf[key] is not None:
            kernel_cli += f" --{key} {kf[key]}"

    procw_tpl = (tpl_dir / "sbatch_process_worker.sh")
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
        process_tkl_update=str((bin_dir / "process_tkl_update.py").resolve()),
        kernel_cli=kernel_cli.strip(),
    )
    procw_sbatch = iterd / "obs" / "submit_process_worker.sh"
    procw_sbatch.write_text(procw_text); make_executable(procw_sbatch)
    procw_jobid = sbatch_submit(procw_sbatch, extra_args=[f"--dependency=afterok:{sim_jobid}"])
    write_json(iterd / "obs" / "submit_workers.json", {"jobid": procw_jobid, "depends_on": sim_jobid})

    # ------------------------------------
    # PROCESSING REDUCE (single job) using process_tkl_update.py reduce
    # ------------------------------------
    procr = cfg["resources"]["processing"]["reduce"]
    epsilon_dir = procr.get("epsilon_dir") or str(iterd / "update")
    procr_tpl = (tpl_dir / "sbatch_process_reduce.sh")
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
        process_tkl_update=str((bin_dir / "process_tkl_update.py").resolve()),
        iteration=args.iter,
    )
    procr_sbatch = iterd / "obs" / "submit_process_reduce.sh"
    procr_sbatch.write_text(procr_text); make_executable(procr_sbatch)
    procr_jobid = sbatch_submit(procr_sbatch, extra_args=[f"--dependency=afterok:{procw_jobid}"])
    write_json(iterd / "obs" / "submit_reduce.json", {"jobid": procr_jobid, "depends_on": procw_jobid})

    # ------------------------------------
    # UPDATE (single job) using update_step.py
    # ------------------------------------
    upd_script_tpl = (tpl_dir / "sbatch_update.sh")
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
        update_step=str((bin_dir / "update_step.py").resolve()),
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
