#!/usr/bin/env python3
"""
Overall (end-to-end) benchmarks for vLLM with vllm-plugin-FL.

This complements benchmark_dispatch_ops.py (operator-level call_op latency). Here we measure
full-model throughput and latency via the official vLLM bench CLI, so dispatch + attention +
memory + scheduler costs are all included.

Typical setup (install vllm-plugin-fl so entry points register; or extend PYTHONPATH):

  export PYTHONPATH="/path/to/workspace/vllm-plugin-FL:${PYTHONPATH}"
  # Platform-specific, e.g. Ascend NPU — use ONLY ``ascend`` as platform plugin:
  #   export VLLM_PLUGINS="ascend"
  # vLLM allows a single active ``vllm.platform_plugins`` entry; ``fl`` is also registered
  # there, so ``ascend,fl`` raises RuntimeError. FL still registers models via
  # ``vllm.general_plugins`` when the package is installed.
  # export VLLM_TARGET_DEVICE="npu"
  # ...

  python src/benchmarks/benchmark_overall.py --model /path/to/Qwen3-4B

Compare two dispatch preferences in isolated runs (separate output dirs):

Each run creates a timestamped subfolder under ``--output-dir`` (default: ``workspace/benchall/YYYYMMDD_HHMMSS/``)
with JSON results, default log ``benchall.log``, and optional ``inspect/`` (see ``--inspect``). Delete the folder to discard a run.

  # Custom log path (otherwise: ``<run-dir>/benchall.log``)
  python src/benchmarks/benchmark_overall.py --model /path/to/Qwen3-4B --compare --log-file /tmp/other.txt

  # Write JSON directly under --output-dir (no timestamp subfolder; old layout)
  python src/benchmarks/benchmark_overall.py --model /path/to/Qwen3-4B --no-timestamp

Other knobs: VLLM_FL_PREFER, VLLM_FL_STRICT, VLLM_FL_CONFIG, etc. (see knowledge/vllm-plugin-fl-fixed.md).

CLI note: some environments ship a ``vllm`` wrapper that runs ``python -m vllm``, but vLLM has no
``vllm.__main__``. This script calls the real entrypoint:
``python -m vllm.entrypoints.cli.main bench ...`` (see ``project.scripts`` in vLLM's pyproject.toml).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import IO, Iterable, Optional

from benchmark_common import (
    DEFAULT_BENCH_ROOT,
    DEFAULT_LOG_BASENAME,
    apply_vllm_plugins_env,
    make_run_dir,
    prepend_pythonpath_vllm_fl,
)
from benchmark_inspect import write_inspect_artifacts


# Mirrors vllm-plugin-FL/benchmarks/flagos_eval/run_benchmark.sh defaults
THROUGHPUT_SCENARIOS: dict[str, tuple[int, int, int]] = {
    # name -> (input_len, output_len, num_prompts)
    "chat_1k": (1024, 1024, 300),
    "chat_4k": (4096, 1024, 300),
    "chat_6k": (6144, 1024, 300),
}

LATENCY_SCENARIOS: dict[str, tuple[int, int, int, int]] = {
    # name -> (input_len, output_len, batch_size, num_iters)
    "batch_8": (4096, 1024, 8, 10),
}


def _vllm_cli_prefix(mode: str) -> list[str]:
    """
    argv prefix to run ``vllm bench ...`` equivalent.

    ``module`` (default): ``sys.executable -m vllm.entrypoints.cli.main`` — matches setuptools
    console_scripts and avoids broken ``python -m vllm`` wrappers.

    ``executable``: first ``vllm`` on PATH (use if your install has no cli.main module).
    """
    if mode == "executable":
        exe = shutil.which("vllm")
        if not exe:
            raise SystemExit(
                "vllm CLI not found on PATH. Install vLLM or use --vllm-cli module "
                "(default) with a Python where vllm is importable."
            )
        return [exe]
    if mode != "module":
        raise SystemExit(f"Unknown --vllm-cli mode {mode!r}; use module or executable.")
    if importlib.util.find_spec("vllm.entrypoints.cli.main") is None:
        raise SystemExit(
            "Cannot import vllm.entrypoints.cli.main. Use the same Python as vLLM "
            "(where `import vllm` works), or pass --vllm-cli executable if `vllm` on PATH is valid."
        )
    return [sys.executable, "-m", "vllm.entrypoints.cli.main"]


def _emit(msg: str, log_fp: Optional[IO[str]]) -> None:
    print(msg, flush=True)
    if log_fp is not None:
        log_fp.write(msg + "\n")
        log_fp.flush()


def _run(
    cmd: list[str],
    env: dict[str, str],
    *,
    dry_run: bool,
    log_fp: Optional[IO[str]] = None,
) -> int:
    line = " ".join(cmd)
    _emit(line, log_fp)
    if dry_run:
        return 0
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


def bench_throughput(
    model: str,
    out_dir: Path,
    scenarios: Iterable[str],
    env: dict[str, str],
    vllm_cli: list[str],
    *,
    trust_remote_code: bool,
    dtype: str,
    enforce_eager: bool,
    dry_run: bool,
    log_fp: Optional[IO[str]] = None,
    extra_bench_args: Optional[list[str]] = None,
) -> int:
    extra_bench_args = extra_bench_args or []
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in scenarios:
        if name not in THROUGHPUT_SCENARIOS:
            msg = f"Unknown throughput scenario {name!r}, skip."
            print(msg, file=sys.stderr)
            if log_fp is not None:
                log_fp.write(msg + "\n")
                log_fp.flush()
            continue
        inp, out, n = THROUGHPUT_SCENARIOS[name]
        json_path = out_dir / f"throughput_{name}.json"
        cmd = [
            *vllm_cli,
            "bench",
            "throughput",
            "--model",
            model,
            "--input-len",
            str(inp),
            "--output-len",
            str(out),
            "--num-prompts",
            str(n),
            "--dtype",
            dtype,
            "--output-json",
            str(json_path),
        ]
        if trust_remote_code:
            cmd.append("--trust-remote-code")
        if enforce_eager:
            cmd.append("--enforce-eager")
        cmd.extend(extra_bench_args)
        rc = _run(cmd, env, dry_run=dry_run, log_fp=log_fp)
        if rc != 0:
            return rc
    return 0


def bench_latency(
    model: str,
    out_dir: Path,
    scenarios: Iterable[str],
    env: dict[str, str],
    vllm_cli: list[str],
    *,
    trust_remote_code: bool,
    dtype: str,
    enforce_eager: bool,
    dry_run: bool,
    log_fp: Optional[IO[str]] = None,
    extra_bench_args: Optional[list[str]] = None,
) -> int:
    extra_bench_args = extra_bench_args or []
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in scenarios:
        if name not in LATENCY_SCENARIOS:
            msg = f"Unknown latency scenario {name!r}, skip."
            print(msg, file=sys.stderr)
            if log_fp is not None:
                log_fp.write(msg + "\n")
                log_fp.flush()
            continue
        inp, out, bs, iters = LATENCY_SCENARIOS[name]
        json_path = out_dir / f"latency_{name}.json"
        cmd = [
            *vllm_cli,
            "bench",
            "latency",
            "--model",
            model,
            "--input-len",
            str(inp),
            "--output-len",
            str(out),
            "--batch-size",
            str(bs),
            "--num-iters",
            str(iters),
            "--dtype",
            dtype,
            "--output-json",
            str(json_path),
        ]
        if trust_remote_code:
            cmd.append("--trust-remote-code")
        if enforce_eager:
            cmd.append("--enforce-eager")
        cmd.extend(extra_bench_args)
        rc = _run(cmd, env, dry_run=dry_run, log_fp=log_fp)
        if rc != 0:
            return rc
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end vLLM bench (throughput + latency).")
    p.add_argument("--model", required=True, help="Local path or HF id for the model")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BENCH_ROOT,
        help=f"Base directory (default: {DEFAULT_BENCH_ROOT}); each run uses YYYYMMDD_HHMMSS unless --no-timestamp",
    )
    p.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Write results directly under --output-dir (no time-stamped subfolder)",
    )
    p.add_argument(
        "--throughput-scenarios",
        default=",".join(THROUGHPUT_SCENARIOS.keys()),
        help=f"Comma-separated subset of: {','.join(THROUGHPUT_SCENARIOS.keys())}",
    )
    p.add_argument(
        "--latency-scenarios",
        default=",".join(LATENCY_SCENARIOS.keys()),
        help=f"Comma-separated subset of: {','.join(LATENCY_SCENARIOS.keys())}",
    )
    p.add_argument("--skip-throughput", action="store_true")
    p.add_argument("--skip-latency", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--enforce-eager", action="store_true", default=True)
    p.add_argument("--no-enforce-eager", action="store_false", dest="enforce_eager")
    p.add_argument("--dry-run", action="store_true", help="Print vllm bench commands only")
    p.add_argument(
        "--compare",
        action="store_true",
        help="Run twice: VLLM_FL_PREFER=flagos vs reference (separate subdirs)",
    )
    p.add_argument(
        "--vllm-cli",
        choices=("module", "executable"),
        default="module",
        help="How to spawn vLLM: -m vllm.entrypoints.cli.main (default) or vllm on PATH",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Log .txt path. Default: <run-dir>/{DEFAULT_LOG_BASENAME}. Use --no-log to disable",
    )
    p.add_argument(
        "--no-log",
        action="store_true",
        help="Do not write a .txt log file",
    )
    p.add_argument(
        "--vllm-plugins",
        metavar="LIST",
        default=None,
        help="Set VLLM_PLUGINS before importing vLLM (e.g. 'ascend' on NPU). "
        "Do not use 'ascend,fl' — see module docstring.",
    )
    p.add_argument(
        "--inspect",
        action="store_true",
        help="After the run, write inspect/ (env snapshot, JSON summary, compare deltas if --compare)",
    )
    p.add_argument(
        "--extra-bench-args",
        default="",
        metavar="STRING",
        help="Extra arguments appended to every `vllm bench` invocation (quoted; use shlex rules).",
    )
    args = p.parse_args()

    extra_bench_args: list[str] = []
    if args.extra_bench_args.strip():
        try:
            extra_bench_args = shlex.split(args.extra_bench_args)
        except ValueError as e:
            raise SystemExit(f"--extra-bench-args: invalid shell string: {e}") from e

    # Must run before any import of vllm (including find_spec in _vllm_cli_prefix).
    apply_vllm_plugins_env(args.vllm_plugins)

    tp_names = [s.strip() for s in args.throughput_scenarios.split(",") if s.strip()]
    lat_names = [s.strip() for s in args.latency_scenarios.split(",") if s.strip()]

    run_dir = make_run_dir(args.output_dir, no_timestamp=args.no_timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"benchmark_overall: run directory -> {run_dir}", file=sys.stderr)

    log_path: Optional[Path] = None
    if not args.no_log:
        log_path = (
            run_dir / DEFAULT_LOG_BASENAME if args.log_file is None else args.log_file
        )

    log_fh: Optional[IO[str]] = None
    log_fp: Optional[IO[str]] = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"benchmark_overall: log file -> {log_path}", file=sys.stderr)
        log_fh = open(log_path, "w", encoding="utf-8")
        log_fp = log_fh
        log_fp.write(f"# benchmark_overall {datetime.now().isoformat()}\n")
        log_fp.write(f"# run_dir={run_dir}\n")
        log_fp.write(f"# model={args.model!r} compare={args.compare}\n\n")
        log_fp.flush()

    vllm_cli = _vllm_cli_prefix(args.vllm_cli)

    def one_run(prefer: str | None, suffix: str) -> int:
        env = prepend_pythonpath_vllm_fl(dict(os.environ))
        if prefer is not None:
            env["VLLM_FL_PREFER"] = prefer
        out = run_dir if suffix == "" else run_dir / suffix
        banner = (
            f"\n=== output -> {out}  VLLM_FL_PREFER={env.get('VLLM_FL_PREFER', '(unset)')} ===\n"
        )
        sys.stdout.write(banner)
        sys.stdout.flush()
        if log_fp is not None:
            log_fp.write(banner)
            log_fp.flush()
        rc = 0
        if not args.skip_throughput:
            rc = bench_throughput(
                args.model,
                out,
                tp_names,
                env,
                vllm_cli,
                trust_remote_code=args.trust_remote_code,
                dtype=args.dtype,
                enforce_eager=args.enforce_eager,
                dry_run=args.dry_run,
                log_fp=log_fp,
                extra_bench_args=extra_bench_args,
            )
        if rc != 0:
            return rc
        if not args.skip_latency:
            rc = bench_latency(
                args.model,
                out,
                lat_names,
                env,
                vllm_cli,
                trust_remote_code=args.trust_remote_code,
                dtype=args.dtype,
                enforce_eager=args.enforce_eager,
                dry_run=args.dry_run,
                log_fp=log_fp,
                extra_bench_args=extra_bench_args,
            )
        return rc

    exit_code = 0
    try:
        if args.compare:
            exit_code = one_run("flagos", "prefer_flagos")
            if exit_code == 0:
                exit_code = one_run("reference", "prefer_reference")
        else:
            exit_code = one_run(None, "")

        if exit_code == 0 and args.inspect:
            write_inspect_artifacts(run_dir, compare=args.compare)
    finally:
        if log_fh is not None:
            log_fh.write(f"\n# benchmark_overall finished {datetime.now().isoformat()}\n")
            log_fh.close()

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
