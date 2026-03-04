"""Core DiffTre engine module exports.

Use lazy attribute resolution to avoid importing heavy dependencies (notably JAX)
when only lightweight helpers under `diffTre.bin` are needed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["Chromosome", "ChromosomeMichroM", "DiffTREPipeline"]


def __getattr__(name: str) -> Any:
    if name == "Chromosome":
        return import_module("diffTre.bin.chromosome").Chromosome
    if name == "ChromosomeMichroM":
        return import_module("diffTre.bin.chromosome_michrom").ChromosomeMichroM
    if name == "DiffTREPipeline":
        return import_module("diffTre.bin.difftre_pipeline").DiffTREPipeline
    raise AttributeError(f"module 'diffTre.bin' has no attribute {name!r}")
