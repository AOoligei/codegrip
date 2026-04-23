#!/usr/bin/env python3
"""Shared helpers for experiment monitors and queue scripts."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class GpuStatus:
    index: int
    used_mb: int
    total_mb: int

    @property
    def free_mb(self) -> int:
        return max(0, self.total_mb - self.used_mb)


@dataclass(frozen=True)
class GpuLockClaim:
    gpu: GpuStatus
    lock_path: str


def resolve_candidate_path(
    bm25_candidates: Optional[str] = None,
    graph_candidates: Optional[str] = None,
    hybrid_candidates: Optional[str] = None,
) -> Tuple[str, str]:
    provided = [
        ("bm25", bm25_candidates),
        ("graph", graph_candidates),
        ("hybrid", hybrid_candidates),
    ]
    chosen = [(pool, path) for pool, path in provided if path]
    if len(chosen) != 1:
        raise ValueError(
            "Exactly one of --bm25_candidates, --graph_candidates, or "
            "--hybrid_candidates must be provided"
        )
    pool_name, path = chosen[0]
    return path, pool_name


def _load_summary(path: os.PathLike | str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _coerce_metric(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def read_summary_metric(path: os.PathLike | str, metric_name: str) -> Optional[float]:
    payload = _load_summary(path)
    if not isinstance(payload, dict):
        return None
    overall = payload.get("overall")
    if not isinstance(overall, dict):
        return None
    return _coerce_metric(overall.get(metric_name))


def summary_is_valid(
    path: os.PathLike | str,
    required_metrics: Sequence[str] = ("hit@1",),
) -> bool:
    for metric_name in required_metrics:
        if read_summary_metric(path, metric_name) is None:
            return False
    return True


def resolve_adapter_dir(exp_dir: os.PathLike | str) -> str:
    base = Path(exp_dir)
    candidates = [
        base / "final",
        base / "best",
    ]
    for candidate in candidates:
        if (candidate / "adapter_config.json").is_file() or (
            candidate / "adapter_model.safetensors"
        ).is_file():
            return str(candidate)
    raise FileNotFoundError(f"No adapter directory found under {base}")


def select_eligible_gpu(
    gpu_statuses: Iterable[GpuStatus],
    preferred_gpu_ids: Sequence[int],
    min_free_mb: int,
    max_used_mb: Optional[int] = None,
    safety_buffer_mb: int = 0,
) -> Optional[GpuStatus]:
    statuses = {status.index: status for status in gpu_statuses}
    for gpu_id in preferred_gpu_ids:
        status = statuses.get(gpu_id)
        if status is None:
            continue
        effective_free = status.free_mb - max(0, safety_buffer_mb)
        if effective_free < min_free_mb:
            continue
        if max_used_mb is not None and status.used_mb > max_used_mb:
            continue
        return status
    return None


def query_gpu_statuses(gpu_ids: Optional[Sequence[int]] = None) -> list[GpuStatus]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    if gpu_ids:
        cmd.extend(["-i", ",".join(str(gpu_id) for gpu_id in gpu_ids)])
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return []

    statuses: list[GpuStatus] = []
    for raw_line in completed.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 3:
            continue
        try:
            statuses.append(
                GpuStatus(
                    index=int(parts[0]),
                    used_mb=int(parts[1]),
                    total_mb=int(parts[2]),
                )
            )
        except ValueError:
            continue
    return statuses


def _lock_file_path(lock_dir: os.PathLike | str, gpu_id: int) -> Path:
    return Path(lock_dir) / f"gpu{gpu_id}.lock"


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _lock_is_stale(lock_path: Path) -> bool:
    payload = _load_summary(lock_path)
    if not isinstance(payload, dict):
        return True
    pid = payload.get("pid")
    if not isinstance(pid, int):
        return True
    return not _pid_is_alive(pid)


def claim_gpu_lock(
    gpu_statuses: Iterable[GpuStatus],
    preferred_gpu_ids: Sequence[int],
    min_free_mb: int,
    max_used_mb: Optional[int],
    safety_buffer_mb: int,
    lock_dir: os.PathLike | str,
    owner_pid: int,
    label: str,
) -> Optional[GpuLockClaim]:
    Path(lock_dir).mkdir(parents=True, exist_ok=True)
    statuses = {status.index: status for status in gpu_statuses}

    for gpu_id in preferred_gpu_ids:
        status = select_eligible_gpu(
            gpu_statuses=[statuses[gpu_id]] if gpu_id in statuses else [],
            preferred_gpu_ids=[gpu_id],
            min_free_mb=min_free_mb,
            max_used_mb=max_used_mb,
            safety_buffer_mb=safety_buffer_mb,
        )
        if status is None:
            continue

        lock_path = _lock_file_path(lock_dir, gpu_id)
        if lock_path.exists() and _lock_is_stale(lock_path):
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass

        payload = {
            "gpu_id": gpu_id,
            "pid": owner_pid,
            "label": label,
            "created_at": time.time(),
        }
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return GpuLockClaim(gpu=status, lock_path=str(lock_path))

    return None


def release_gpu_lock(lock_path: os.PathLike | str) -> None:
    try:
        Path(lock_path).unlink()
    except FileNotFoundError:
        pass


def _cmd_claim_gpu(args: argparse.Namespace) -> int:
    claim = claim_gpu_lock(
        gpu_statuses=query_gpu_statuses(args.gpus),
        preferred_gpu_ids=args.gpus,
        min_free_mb=args.min_free_mb,
        max_used_mb=args.max_used_mb,
        safety_buffer_mb=args.safety_buffer_mb,
        lock_dir=args.lock_dir,
        owner_pid=args.owner_pid,
        label=args.label,
    )
    if claim is None:
        return 1
    sys.stdout.write(f"{claim.gpu.index}\t{claim.lock_path}\n")
    return 0


def _cmd_release_lock(args: argparse.Namespace) -> int:
    release_gpu_lock(args.lock_path)
    return 0


def _cmd_summary_ok(args: argparse.Namespace) -> int:
    return 0 if summary_is_valid(args.path, tuple(args.metric)) else 1


def _cmd_metric(args: argparse.Namespace) -> int:
    value = read_summary_metric(args.path, args.metric)
    if value is None:
        return 1
    sys.stdout.write(f"{value:.2f}\n")
    return 0


def _cmd_resolve_adapter(args: argparse.Namespace) -> int:
    try:
        resolved = resolve_adapter_dir(args.exp_dir)
    except FileNotFoundError:
        return 1
    sys.stdout.write(f"{resolved}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiment automation helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    claim_gpu = subparsers.add_parser("claim-gpu")
    claim_gpu.add_argument("--gpus", nargs="+", type=int, required=True)
    claim_gpu.add_argument("--min-free-mb", type=int, required=True)
    claim_gpu.add_argument("--max-used-mb", type=int, default=None)
    claim_gpu.add_argument("--safety-buffer-mb", type=int, default=0)
    claim_gpu.add_argument("--lock-dir", required=True)
    claim_gpu.add_argument("--owner-pid", type=int, default=os.getpid())
    claim_gpu.add_argument("--label", default="job")
    claim_gpu.set_defaults(func=_cmd_claim_gpu)

    release_lock = subparsers.add_parser("release-lock")
    release_lock.add_argument("--lock-path", required=True)
    release_lock.set_defaults(func=_cmd_release_lock)

    summary_ok = subparsers.add_parser("summary-ok")
    summary_ok.add_argument("--path", required=True)
    summary_ok.add_argument("--metric", action="append", default=["hit@1"])
    summary_ok.set_defaults(func=_cmd_summary_ok)

    metric = subparsers.add_parser("metric")
    metric.add_argument("--path", required=True)
    metric.add_argument("--metric", required=True)
    metric.set_defaults(func=_cmd_metric)

    resolve_adapter = subparsers.add_parser("resolve-adapter")
    resolve_adapter.add_argument("--exp-dir", required=True)
    resolve_adapter.set_defaults(func=_cmd_resolve_adapter)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
