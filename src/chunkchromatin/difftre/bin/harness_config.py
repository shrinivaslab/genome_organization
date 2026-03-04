from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def load_harness_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Harness config must be a JSON object.")
    return normalize_harness_config(cfg)


def normalize_harness_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Support legacy combined workflow config by filling harness defaults."""
    out = _deepcopy(cfg)
    out.setdefault("workflow", {})
    out["workflow"].setdefault("mode", "both")
    out.setdefault("run", {})
    if "output_root" not in out["run"]:
        if "output_dir" in out["run"]:
            out["run"]["output_root"] = out["run"]["output_dir"]
        else:
            raise ValueError("Config missing run.output_root")
    if "name" not in out["run"]:
        out["run"]["name"] = "difftre_harness"
    if "slurm" not in out:
        # Prefer fit-level slurm block, then reference-level slurm block.
        fit_slurm = out.get("fit", {}).get("slurm")
        ref_slurm = out.get("reference", {}).get("slurm")
        if fit_slurm:
            out["slurm"] = _deepcopy(fit_slurm)
        elif ref_slurm:
            out["slurm"] = _deepcopy(ref_slurm)
    return out


def validate_harness_config(cfg: dict[str, Any]) -> None:
    run = cfg.get("run", {})
    if "name" not in run:
        raise ValueError("Config missing run.name")

    workflow = cfg.get("workflow", {})
    mode = workflow.get("mode", "both")
    if mode not in {"reference", "fit", "both"}:
        raise ValueError(f"Unsupported workflow.mode: {mode}")

    if mode in {"reference", "both"} and "reference" not in cfg:
        raise ValueError("Config missing top-level reference block for workflow mode.")
    if mode in {"fit", "both"} and "fit" not in cfg:
        raise ValueError("Config missing top-level fit block for workflow mode.")


def _deepcopy(obj: Any) -> Any:
    return copy.deepcopy(obj)


def materialize_reference_config(cfg: dict[str, Any], run_root: Path) -> dict[str, Any]:
    ref = _deepcopy(cfg["reference"])
    ref.setdefault("run", {})
    ref["run"]["name"] = f"{cfg['run']['name']}_reference"
    ref["run"]["output_dir"] = str((run_root / "reference").resolve())
    ref.setdefault("slurm", _deepcopy(cfg.get("slurm", {})))
    return ref


def materialize_fit_config(cfg: dict[str, Any], run_root: Path) -> dict[str, Any]:
    fit = _deepcopy(cfg["fit"])
    fit.setdefault("run", {})
    fit["run"]["name"] = f"{cfg['run']['name']}_fit"
    fit["run"]["output_dir"] = str((run_root / "fit").resolve())
    fit.setdefault("slurm", _deepcopy(cfg.get("slurm", {})))
    return fit


def wire_fit_to_reference(fit_cfg: dict[str, Any], ref_root: Path) -> dict[str, Any]:
    fit_cfg = _deepcopy(fit_cfg)
    fit_cfg.setdefault("reference", {})
    fit_cfg["reference"]["targets_dir"] = str((ref_root / "exp_targets").resolve())
    fit_cfg["reference"]["reference_params_dir"] = str((ref_root / "params").resolve())
    fit_cfg.setdefault("monomer_types", {})
    fit_cfg["monomer_types"]["types_path"] = str((ref_root / "monomer_types.npy").resolve())
    fit_cfg["loops"] = {"looplist_path": str((ref_root / "looplist.txt").resolve())}
    return fit_cfg
