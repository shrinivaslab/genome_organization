
import argparse, os, json, shutil
from pathlib import Path
from utils import ensure_dir, load_config, write_json, sbatch_submit, format_iter, make_executable

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--iter", required=True, type=int)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    run_root = Path(args.run_root).resolve()
    cfg = load_config(Path(args.config))
    iterd = run_root / format_iter(args.iter)
    logd = ensure_dir(run_root / "logs")
    ensure_dir(iterd / "sims"); ensure_dir(iterd / "obs"); ensure_dir(iterd / "update")

    # Paths
    eps_path = (iterd / "params" / "epsilon.npy").resolve()
    seeds_json = (run_root / "seeds.json").resolve()
    kernel_json = (run_root / "exp_targets" / "kernel.json").resolve()
    targets_npy = (run_root / "exp_targets" / "T_type_kl.npy").resolve()

    # ---------- SIMULATION ARRAY ----------
    sim_script = (Path(__file__).resolve().parent.parent / "templates" / "sbatch_sim_array.sh")
    sim_script_text = sim_script.read_text().format(
        job_name=f"{args.name}_sim_{args.iter:03d}",
        account=cfg["slurm"]["account"],
        partition=cfg["slurm"]["partition"],
        time_limit=cfg["slurm"]["time_limit"],
        cpus=cfg["slurm"]["cpus_per_task"],
        mem=cfg["slurm"]["mem"],
        array_max=cfg["simulation"]["n_replicates"]-1,
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
        run_replicates_array=cfg["paths"]["run_replicates_array"],
    )
    sim_sbatch = iterd / "sims" / "submit_sim_array.sh"
    sim_sbatch.write_text(sim_script_text); make_executable(sim_sbatch)
    sim_jobid = sbatch_submit(sim_sbatch)
    write_json(iterd / "sims" / "submit.json", {"jobid": sim_jobid})

    # ---------- PROCESSING (depends on sim) ----------
    proc_script = (Path(__file__).resolve().parent.parent / "templates" / "sbatch_process.sh")
    proc_script_text = proc_script.read_text().format(
        job_name=f"{args.name}_proc_{args.iter:03d}",
        account=cfg["slurm"]["account"],
        partition=cfg["slurm"]["partition"],
        time_limit=cfg["slurm"]["time_limit"],
        cpus=max(2, cfg["slurm"]["cpus_per_task"]//2),
        mem=cfg["slurm"]["mem"],
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
        submit_process_obs=cfg["paths"]["submit_process_obs"],
    )
    proc_sbatch = iterd / "obs" / "submit_process.sh"
    proc_sbatch.write_text(proc_script_text); make_executable(proc_sbatch)
    proc_jobid = sbatch_submit(proc_sbatch, extra_args=[f"--dependency=afterok:{sim_jobid}"])
    write_json(iterd / "obs" / "submit.json", {"jobid": proc_jobid, "depends_on": sim_jobid})

    # ---------- UPDATE + CONTINUE (depends on process) ----------
    upd_script = (Path(__file__).resolve().parent.parent / "templates" / "sbatch_update.sh")
    upd_script_text = upd_script.read_text().format(
        job_name=f"{args.name}_update_{args.iter:03d}",
        account=cfg["slurm"]["account"],
        partition=cfg["slurm"]["partition"],
        time_limit="00:20:00",
        cpus=2,
        mem="4G",
        constraint_line=(f"#SBATCH --constraint={cfg['slurm']['constraint']}\n" if cfg["slurm"].get("constraint") else ""),
        qos_line=(f"#SBATCH --qos={cfg['slurm']['qos']}\n" if cfg["slurm"].get("qos") else ""),
        log_dir=str(logd),
        iter_dir=str(iterd),
        update_step=str((Path(__file__).resolve().parent / "update_step.py").resolve()),
        run_root=str(run_root),
        iter_idx=args.iter,
        config_yaml=str(Path(args.config).resolve()),
    )
    upd_sbatch = iterd / "update" / "submit_update.sh"
    upd_sbatch.write_text(upd_script_text); make_executable(upd_sbatch)
    upd_jobid = sbatch_submit(upd_sbatch, extra_args=[f"--dependency=afterok:{proc_jobid}"])
    write_json(iterd / "update" / "submit.json", {"jobid": upd_jobid, "depends_on": proc_jobid})

if __name__ == "__main__":
    main()
