# Copyright (c) 2026
"""Shared paths and vLLM plugin env for benchmark scripts (benchall layout)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime

WORKSPACE = Path(__file__).resolve().parents[2]
_VLLM_FL_ROOT = WORKSPACE / "vllm-plugin-FL"

# Unified layout: workspace/benchall/<YYYYMMDD_HHMMSS>/...
DEFAULT_BENCH_ROOT = WORKSPACE / "benchall"
DEFAULT_LOG_BASENAME = "benchall.log"


def make_run_dir(base: Path, *, no_timestamp: bool) -> Path:
    if no_timestamp:
        return base.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (base / stamp).resolve()


def apply_vllm_plugins_env(cli_value: str | None) -> None:
    """See benchmark_overall docstring — only one vLLM platform plugin may be active."""
    if cli_value is not None:
        os.environ["VLLM_PLUGINS"] = cli_value
        return
    raw = os.environ.get("VLLM_PLUGINS", "").strip()
    if not raw:
        if os.environ.get("VLLM_TARGET_DEVICE") == "npu":
            os.environ["VLLM_PLUGINS"] = "ascend"
            print(
                "benchmark: VLLM_TARGET_DEVICE=npu and VLLM_PLUGINS unset; "
                "using VLLM_PLUGINS=ascend.",
                file=sys.stderr,
            )
        return
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if "ascend" in parts and "fl" in parts:
        os.environ["VLLM_PLUGINS"] = "ascend"
        print(
            "benchmark: VLLM_PLUGINS listed both 'ascend' and 'fl'; "
            "using VLLM_PLUGINS=ascend (FL hooks via vllm.general_plugins).",
            file=sys.stderr,
        )


def prepend_pythonpath_vllm_fl(env: dict[str, str]) -> dict[str, str]:
    if _VLLM_FL_ROOT.is_dir():
        extra = str(_VLLM_FL_ROOT)
        old = env.get("PYTHONPATH", "")
        if old:
            env["PYTHONPATH"] = f"{extra}{os.pathsep}{old}"
        else:
            env["PYTHONPATH"] = extra
    return env


def env_snapshot_for_inspect() -> dict[str, str]:
    """Relevant env for reproducing runs and spotting overhead-related flags."""
    out: dict[str, str] = {}
    for k, v in sorted(os.environ.items()):
        if (
            k.startswith("VLLM_")
            or k.startswith("VLLM_FL_")
            or k.startswith("TORCH_")
            or k.startswith("CUDA_")
            or k.startswith("NPU_")
            or k.startswith("ASCEND_")
            or k.startswith("HCCL_")
            or k.startswith("MASTER_")
            or k.startswith("WORLD_")
            or k in ("PATH", "PYTHONPATH", "USE_FLAGGEMS")
        ):
            out[k] = v
    return out
