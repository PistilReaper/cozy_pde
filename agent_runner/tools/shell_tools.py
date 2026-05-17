from __future__ import annotations

import os
import re
import subprocess
import time
from datetime import datetime, timezone

from ..config import RunnerConfig
from ..safety import WorkspaceSafety
from . import failure, success

REHEARSAL_LIMIT_PATTERNS = (
    "--smoke",
    "--dry-run",
    "--subset",
    "--fast-dev-run",
    "--max-batches",
    "--limit-train-batches",
    "--limit_train_batches",
    "--max-steps",
)
TRAIN_COMMAND_RE = re.compile(r"(^|\s)(python\s+)?[^ ]*train(?:_[^ ]+)?\.py(\s|$)|(^|\s)train(\s|$)")
EPOCH_ONE_RE = re.compile(r"--epochs(?:=|\s+)1(\D|$)")


def run_shell(
    *,
    command: str,
    safety: WorkspaceSafety,
    config: RunnerConfig,
    cwd: str | None = None,
    timeout_seconds: int | None = None,
    profile: str = "default",
) -> dict:
    max_timeout = config.budget.max_single_shell_seconds
    if profile == "rehearsal":
        max_timeout = min(max_timeout, 900)
    timeout_seconds = timeout_seconds or max_timeout
    timeout_seconds = min(timeout_seconds, max_timeout)

    command_check = safety.validate_shell_command(command)
    if not command_check.ok:
        return failure("run_shell", command_check.error or "shell command rejected", command=command)

    normalized = " ".join(command.lower().split())
    if profile == "rehearsal":
        is_train_command = bool(TRAIN_COMMAND_RE.search(normalized))
        has_smoke_limit = any(flag in normalized for flag in REHEARSAL_LIMIT_PATTERNS) or bool(EPOCH_ONE_RE.search(normalized))
        if is_train_command and not has_smoke_limit:
            return failure(
                "run_shell",
                "Rehearsal profile requires smoke-limited training flags such as --smoke, --max-batches 1, --epochs 1, --dry-run, or --subset.",
                command=command,
                profile=profile,
            )

    cwd_check = safety.validate_cwd(cwd)
    if not cwd_check.ok:
        return failure("run_shell", cwd_check.error or "cwd rejected", cwd=cwd)
    assert cwd_check.resolved_path is not None

    config.shell_log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = config.shell_log_dir / f"shell_{stamp}.log"

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            ["/bin/bash", "-lc", command],
            cwd=str(cwd_check.resolved_path),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        elapsed = time.perf_counter() - started
        combined = (
            f"$ {command}\n"
            f"[stdout]\n{completed.stdout}\n"
            f"[stderr]\n{completed.stderr}\n"
            f"[returncode] {completed.returncode}\n"
        )
        log_path.write_text(combined, encoding="utf-8")
        return success(
            "run_shell",
            f"Command finished with return code {completed.returncode}",
            command=command,
            cwd=str(cwd_check.resolved_path),
            profile=profile,
            returncode=completed.returncode,
            stdout_tail=completed.stdout[-4000:],
            stderr_tail=completed.stderr[-4000:],
            elapsed_seconds=elapsed,
            timed_out=False,
            log_path=str(log_path),
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        log_path.write_text(
            f"$ {command}\n[stdout]\n{stdout}\n[stderr]\n{stderr}\n[timed_out]\n",
            encoding="utf-8",
        )
        return failure(
            "run_shell",
            f"Command timed out after {timeout_seconds} seconds",
            command=command,
            cwd=str(cwd_check.resolved_path),
            profile=profile,
            stdout_tail=stdout[-4000:],
            stderr_tail=stderr[-4000:],
            elapsed_seconds=elapsed,
            timed_out=True,
            log_path=str(log_path),
        )
