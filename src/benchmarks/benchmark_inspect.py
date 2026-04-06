#!/usr/bin/env python3
"""
Deep inspect: env snapshot + JSON aggregation + compare deltas for overhead analysis.

Writes under ``<run_dir>/inspect/``:
  - ``env_snapshot.json`` — filtered environment
  - ``results_summary.json`` — parsed throughput/latency metrics per subdir
  - ``compare_overhead.json`` — if ``prefer_flagos`` and ``prefer_reference`` exist, pct deltas
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from benchmark_common import env_snapshot_for_inspect


def _load_json(p: Path) -> Any | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _summarize_dir(d: Path, label: str) -> dict[str, Any]:
    out: dict[str, Any] = {"label": label, "path": str(d), "throughput": {}, "latency": {}}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("throughput_*.json")):
        data = _load_json(p)
        if isinstance(data, dict):
            out["throughput"][p.stem] = data
    for p in sorted(d.glob("latency_*.json")):
        data = _load_json(p)
        if isinstance(data, dict):
            out["latency"][p.stem] = data
    return out


def _pct_delta(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return (a - b) / b * 100.0


def _compare_prefers(run_dir: Path) -> dict[str, Any] | None:
    f = run_dir / "prefer_flagos"
    r = run_dir / "prefer_reference"
    if not f.is_dir() or not r.is_dir():
        return None
    sf = _summarize_dir(f, "prefer_flagos")
    sr = _summarize_dir(r, "prefer_reference")
    report: dict[str, Any] = {
        "prefer_flagos": sf,
        "prefer_reference": sr,
        "deltas": {"throughput": {}, "latency": {}},
    }
    for name in set(sf["throughput"].keys()) & set(sr["throughput"].keys()):
        tf = sf["throughput"].get(name, {})
        tr = sr["throughput"].get(name, {})
        tps_f = tf.get("tokens_per_second")
        tps_r = tr.get("tokens_per_second")
        if isinstance(tps_f, (int, float)) and isinstance(tps_r, (int, float)):
            report["deltas"]["throughput"][name] = {
                "tokens_per_second_pct_reference": _pct_delta(tps_f, tps_r),
                "flagos_tokens_per_second": tps_f,
                "reference_tokens_per_second": tps_r,
            }
    for name in set(sf["latency"].keys()) & set(sr["latency"].keys()):
        lf = sf["latency"].get(name, {})
        lr = sr["latency"].get(name, {})
        av_f = lf.get("avg_latency")
        av_r = lr.get("avg_latency")
        if isinstance(av_f, (int, float)) and isinstance(av_r, (int, float)):
            # Lower latency is better; positive pct means flagos slower if flagos-reference
            report["deltas"]["latency"][name] = {
                "avg_latency_seconds_pct_vs_reference": _pct_delta(av_f, av_r),
                "flagos_avg_latency_s": av_f,
                "reference_avg_latency_s": av_r,
            }
    return report


def write_inspect_artifacts(run_dir: Path, *, compare: bool) -> Path:
    """Write inspect/ under run_dir. Returns inspect directory path."""
    inspect_dir = run_dir / "inspect"
    inspect_dir.mkdir(parents=True, exist_ok=True)

    env_path = inspect_dir / "env_snapshot.json"
    env_path.write_text(
        json.dumps(env_snapshot_for_inspect(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "compare_mode": compare,
    }
    if compare:
        summary["prefer_flagos"] = _summarize_dir(run_dir / "prefer_flagos", "prefer_flagos")
        summary["prefer_reference"] = _summarize_dir(
            run_dir / "prefer_reference", "prefer_reference"
        )
        cmp = _compare_prefers(run_dir)
        if cmp is not None:
            (inspect_dir / "compare_overhead.json").write_text(
                json.dumps(cmp, indent=2) + "\n", encoding="utf-8"
            )
            summary["compare_note"] = (
                "deltas: throughput tokens_per_second_pct_reference is (flagos-ref)/ref*100; "
                "latency avg_latency_seconds_pct_vs_reference is (flagos-ref)/ref*100"
            )
    else:
        summary["single_run"] = _summarize_dir(run_dir, "single")

    (inspect_dir / "results_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    readme = inspect_dir / "README.txt"
    readme.write_text(
        "inspect/\n"
        "  env_snapshot.json   — environment variables relevant to vLLM / FL / device\n"
        "  results_summary.json — parsed bench JSON metrics\n"
        "  compare_overhead.json — (only if --compare) flagos vs reference deltas\n"
        "\n"
        "For kernel/device profiling, use vendor tools (e.g. Ascend profiling) or pass\n"
        "extra flags via benchmark_overall.py --extra-bench-args '...' to each vllm bench call.\n",
        encoding="utf-8",
    )

    print(f"benchmark_inspect: wrote {inspect_dir}", file=sys.stderr)
    return inspect_dir
