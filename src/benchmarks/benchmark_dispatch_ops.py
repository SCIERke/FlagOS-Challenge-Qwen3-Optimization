#!/usr/bin/env python3
"""
Micro-benchmarks for vllm-plugin-FL dispatch operators (see knowledge/vllm-plugin-fl-fixed.md).

Measures latency of call_op() for silu_and_mul, rms_norm, rotary_embedding (and optional gelu_and_mul).
For full-model throughput/latency (vllm bench), use benchmark_overall.py instead.

Usage (from workspace root, where vllm-plugin-FL/ lives):

  # Default: auto device, prefer flagos
  PYTHONPATH=vllm-plugin-FL python src/benchmarks/benchmark_dispatch_ops.py

  # PyTorch reference only (CPU-friendly)
  VLLM_FL_PREFER=reference PYTHONPATH=vllm-plugin-FL python src/benchmarks/benchmark_dispatch_ops.py --device cpu

  # Compare flagos vs reference in isolated subprocesses
  PYTHONPATH=vllm-plugin-FL python src/benchmarks/benchmark_dispatch_ops.py --compare

Default output layout matches benchmark_overall.py: ``workspace/benchall/<YYYYMMDD_HHMMSS>/benchall.log``.

On Ascend/NPU, vLLM allows only one ``vllm.platform_plugins`` entry; this script calls the same
``VLLM_PLUGINS`` handling as benchmark_overall.py (e.g. ``ascend`` only, not ``ascend,fl``).

Environment knobs (documented in vllm-plugin-fl-fixed.md): VLLM_FL_PREFER, VLLM_FL_STRICT,
VLLM_FL_PLATFORM, VLLM_FL_CONFIG, VLLM_FL_PER_OP, VLLM_FL_LOG_LEVEL, etc.
Set VLLM_FL_LOG_LEVEL=INFO to print which impl_id is selected per operator.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Callable, Iterable, Optional

from benchmark_common import (
    DEFAULT_BENCH_ROOT,
    DEFAULT_LOG_BASENAME,
    apply_vllm_plugins_env,
    make_run_dir,
)

# Ensure vllm_fl is importable when run as a script
_WORKSPACE = Path(__file__).resolve().parents[2]
_VLLM_FL_ROOT = _WORKSPACE / "vllm-plugin-FL"
if _VLLM_FL_ROOT.is_dir() and str(_VLLM_FL_ROOT) not in sys.path:
    sys.path.insert(0, str(_VLLM_FL_ROOT))


def _pick_device(name: str) -> str:
    if name != "auto":
        return name
    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _device_sync(device: str) -> None:
    import torch

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def _tensor(device: str, *shape: int, dtype=None):
    import torch

    dt = dtype or torch.float16
    dev = torch.device(device)
    return torch.randn(shape, device=dev, dtype=dt)


@dataclass
class BenchResult:
    name: str
    ms_per_iter: float
    impl_id: str


def _run_loop(fn: Callable[[], None], warmup: int, iters: int, device: str) -> float:
    for _ in range(warmup):
        fn()
    _device_sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _device_sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000.0


def benchmark_silu_and_mul(device: str, tokens: int, hidden2: int, warmup: int, iters: int) -> BenchResult:
    from vllm_fl.dispatch import call_op, get_default_manager

    mgr = get_default_manager()
    mgr.ensure_initialized()
    # last dim is 2 * d for SiLU gate
    x = _tensor(device, tokens, hidden2)

    def fn():
        call_op("silu_and_mul", None, x)

    ms = _run_loop(fn, warmup, iters, device)
    impl = mgr.get_selected_impl_id("silu_and_mul")
    return BenchResult(f"silu_and_mul (tokens={tokens}, dim={hidden2})", ms, impl)


def benchmark_gelu_and_mul(device: str, tokens: int, hidden2: int, warmup: int, iters: int) -> BenchResult:
    from vllm_fl.dispatch import call_op, get_default_manager

    class _Stub:
        approximate = "none"

    mgr = get_default_manager()
    mgr.ensure_initialized()
    obj = _Stub()
    x = _tensor(device, tokens, hidden2)

    def fn():
        call_op("gelu_and_mul", obj, x)

    ms = _run_loop(fn, warmup, iters, device)
    impl = mgr.get_selected_impl_id("gelu_and_mul")
    return BenchResult(f"gelu_and_mul (tokens={tokens}, dim={hidden2})", ms, impl)


def benchmark_rms_norm(
    device: str, tokens: int, hidden: int, warmup: int, iters: int, with_residual: bool
) -> BenchResult:
    from vllm_fl.dispatch import call_op, get_default_manager

    class _RMSStub:
        def __init__(self):
            import torch

            self.weight = torch.ones(hidden, device=device, dtype=torch.float16)
            self.variance_epsilon = 1e-6

    mgr = get_default_manager()
    mgr.ensure_initialized()
    stub = _RMSStub()
    x = _tensor(device, tokens, hidden)
    residual = _tensor(device, tokens, hidden) if with_residual else None

    def fn():
        call_op("rms_norm", stub, x, residual)

    ms = _run_loop(fn, warmup, iters, device)
    impl = mgr.get_selected_impl_id("rms_norm")
    tag = f"rms_norm (tokens={tokens}, hidden={hidden}, residual={with_residual})"
    return BenchResult(tag, ms, impl)


def benchmark_rotary_embedding(
    device: str,
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    max_positions: int,
    warmup: int,
    iters: int,
) -> BenchResult:
    from vllm_fl.dispatch import call_op, get_default_manager

    import torch

    mgr = get_default_manager()
    mgr.ensure_initialized()
    half = head_dim // 2
    cos = _tensor(device, max_positions, half)
    sin = _tensor(device, max_positions, half)
    q_src = _tensor(device, num_tokens, num_heads, head_dim)
    k_src = _tensor(device, num_tokens, num_heads, head_dim)
    q = torch.empty_like(q_src)
    k = torch.empty_like(k_src)
    position_ids = torch.randint(0, max_positions, (num_tokens,), device=device, dtype=torch.int32)

    def fn():
        q.copy_(q_src)
        k.copy_(k_src)
        call_op(
            "rotary_embedding",
            None,
            q,
            k,
            cos,
            sin,
            position_ids,
            False,
            True,
        )

    ms = _run_loop(fn, warmup, iters, device)
    impl = mgr.get_selected_impl_id("rotary_embedding")
    return BenchResult(
        f"rotary_embedding (tokens={num_tokens}, heads={num_heads}, head_dim={head_dim})",
        ms,
        impl,
    )


def _print_results(device: str, prefer: str, results: Iterable[BenchResult]) -> None:
    print()
    print(f"device={device}  VLLM_FL_PREFER={prefer}")
    print("-" * 72)
    for r in results:
        print(f"{r.ms_per_iter:8.4f} ms/iter  impl={r.impl_id}")
        print(f"           {r.name}")
    print("-" * 72)


class _TeeStdout:
    """Write to multiple text streams (e.g. console + benchall.log)."""

    def __init__(self, *files: IO[str]) -> None:
        self._files = files

    def write(self, s: str) -> int:
        for f in self._files:
            f.write(s)
        return len(s)

    def flush(self) -> None:
        for f in self._files:
            f.flush()


def _run_subprocess_tee(
    cmd: list[str],
    env: dict[str, str],
    *,
    log_fp: Optional[IO[str]] = None,
) -> int:
    line = " ".join(cmd)
    print(line, file=sys.stderr)
    if log_fp is not None:
        log_fp.write(line + "\n")
        log_fp.flush()
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for out_line in proc.stdout:
        sys.stdout.write(out_line)
        sys.stdout.flush()
        if log_fp is not None:
            log_fp.write(out_line)
            log_fp.flush()
    return proc.wait()


def _parse_ops(s: str) -> list[str]:
    allowed = {"silu", "rms", "rotary", "gelu"}
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    for p in parts:
        if p not in allowed:
            raise argparse.ArgumentTypeError(f"unknown op token {p!r}; allowed: {sorted(allowed)}")
    return parts


def run_suite(args: argparse.Namespace) -> list[BenchResult]:
    device = _pick_device(args.device)
    prefer = os.environ.get("VLLM_FL_PREFER", "flagos")

    # Import after env is fully set (CLI cannot override already-imported policy easily)
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    if device == "npu":
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise SystemExit("NPU requested but torch.npu is not available.")

    results: list[BenchResult] = []
    ops = args.ops

    if "silu" in ops:
        results.append(
            benchmark_silu_and_mul(device, args.tokens, args.hidden * 2, args.warmup, args.iters)
        )
    if "gelu" in ops:
        results.append(
            benchmark_gelu_and_mul(device, args.tokens, args.hidden * 2, args.warmup, args.iters)
        )
    if "rms" in ops:
        results.append(
            benchmark_rms_norm(device, args.tokens, args.hidden, args.warmup, args.iters, False)
        )
        if args.rms_residual:
            results.append(
                benchmark_rms_norm(device, args.tokens, args.hidden, args.warmup, args.iters, True)
            )
    if "rotary" in ops:
        results.append(
            benchmark_rotary_embedding(
                device,
                args.tokens,
                args.num_heads,
                args.head_dim,
                args.max_positions,
                args.warmup,
                args.iters,
            )
        )

    _print_results(device, prefer, results)
    return results


def _compare_subprocess(
    base_args: list[str],
    prefer: str,
    *,
    run_dir: Path,
    log_fp: Optional[IO[str]] = None,
) -> int:
    """Child runs without --compare; uses same run_dir so parent owns the log and one timestamp folder."""
    env = os.environ.copy()
    env["VLLM_FL_PREFER"] = prefer
    child_args = [
        *base_args,
        "--no-timestamp",
        "--output-dir",
        str(run_dir),
        "--no-log",
    ]
    cmd = [sys.executable, str(Path(__file__).resolve()), *child_args]
    banner = f"\n=== subprocess: VLLM_FL_PREFER={prefer} ===\n"
    sys.stdout.write(banner)
    sys.stdout.flush()
    if log_fp is not None:
        log_fp.write(banner)
        log_fp.flush()
    return _run_subprocess_tee(cmd, env, log_fp=log_fp)


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark vllm-plugin-FL dispatch operators.")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "npu"))
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--tokens", type=int, default=4096, help="Token rows for silu/rms; sequence tokens for rotary")
    p.add_argument("--hidden", type=int, default=4096, help="Hidden size (silu input last dim = 2*hidden)")
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--max-positions", type=int, default=8192, help="Cos/sin cache length for rotary")
    p.add_argument("--rms-residual", action="store_true", help="Also benchmark rms_norm with residual add")
    p.add_argument(
        "--ops",
        type=_parse_ops,
        default=["silu", "rms", "rotary"],
        help="Comma-separated: silu,rms,rotary,gelu",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Run this script twice in subprocesses with VLLM_FL_PREFER=flagos and reference",
    )
    p.add_argument(
        "--vllm-plugins",
        metavar="LIST",
        default=None,
        help="Set VLLM_PLUGINS before vLLM loads (e.g. 'ascend' on NPU). "
        "If unset, conflicts between ascend+fl are resolved like benchmark_overall.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BENCH_ROOT,
        help=f"Base directory (default: {DEFAULT_BENCH_ROOT}); each run uses YYYYMMDD_HHMMSS unless --no-timestamp",
    )
    p.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Write logs/results under --output-dir directly (no time-stamped subfolder)",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Log path. Default: <run-dir>/{DEFAULT_LOG_BASENAME}. Use --no-log to disable",
    )
    p.add_argument(
        "--no-log",
        action="store_true",
        help="Do not write a log file (stdout/stderr only)",
    )
    args = p.parse_args()

    # Before any subprocess or vllm/vllm_fl import path that touches platform plugins.
    apply_vllm_plugins_env(args.vllm_plugins)

    run_dir = make_run_dir(args.output_dir, no_timestamp=args.no_timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"benchmark_dispatch_ops: run directory -> {run_dir}", file=sys.stderr)

    log_path: Optional[Path] = None
    if not args.no_log:
        log_path = run_dir / DEFAULT_LOG_BASENAME if args.log_file is None else args.log_file

    log_fh: Optional[IO[str]] = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"benchmark_dispatch_ops: log file -> {log_path}", file=sys.stderr)
        log_fh = open(log_path, "w", encoding="utf-8")
        log_fh.write(f"# benchmark_dispatch_ops {datetime.now().isoformat()}\n")
        log_fh.write(f"# run_dir={run_dir}\n")
        log_fh.write(f"# compare={args.compare}\n\n")
        log_fh.flush()

    base_flags = [
        "--device",
        args.device,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--tokens",
        str(args.tokens),
        "--hidden",
        str(args.hidden),
        "--num-heads",
        str(args.num_heads),
        "--head-dim",
        str(args.head_dim),
        "--max-positions",
        str(args.max_positions),
        "--ops",
        ",".join(args.ops),
    ]
    if args.rms_residual:
        base_flags.append("--rms-residual")

    exit_code = 0
    old_stdout = sys.stdout
    try:
        if args.compare:
            rc_flagos = _compare_subprocess(base_flags, "flagos", run_dir=run_dir, log_fp=log_fh)
            rc_ref = _compare_subprocess(base_flags, "reference", run_dir=run_dir, log_fp=log_fh)
            exit_code = rc_flagos if rc_flagos != 0 else rc_ref
        else:
            if log_fh is not None:
                sys.stdout = _TeeStdout(old_stdout, log_fh)
            try:
                run_suite(args)
            finally:
                sys.stdout = old_stdout
    finally:
        if log_fh is not None:
            log_fh.write(f"\n# benchmark_dispatch_ops finished {datetime.now().isoformat()}\n")
            log_fh.close()

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
