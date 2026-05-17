from __future__ import annotations

import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

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
SECRET_NAME_RE = re.compile(r"(api[_-]?key|token|secret|password|auth|credential|cookie|bearer)", re.IGNORECASE)
SECRET_VALUE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{10,}", re.IGNORECASE),
    re.compile(r"(bearer\s+)[A-Za-z0-9._\-]{10,}", re.IGNORECASE),
)


def _snapshot_submission_code(code_dir: Path) -> dict[str, bytes]:
    if not code_dir.exists():
        return {}
    snapshot: dict[str, bytes] = {}
    for file_path in sorted(candidate for candidate in code_dir.rglob("*") if candidate.is_file()):
        snapshot[file_path.relative_to(code_dir).as_posix()] = file_path.read_bytes()
    return snapshot


def _diff_submission_code(before: dict[str, bytes], after: dict[str, bytes]) -> dict[str, list[str]]:
    before_paths = set(before)
    after_paths = set(after)
    added = sorted(after_paths - before_paths)
    removed = sorted(before_paths - after_paths)
    modified = sorted(path for path in before_paths & after_paths if before[path] != after[path])
    return {"added": added, "removed": removed, "modified": modified}


def _restore_submission_code(code_dir: Path, snapshot: dict[str, bytes]) -> None:
    code_dir.mkdir(parents=True, exist_ok=True)
    for file_path in sorted(candidate for candidate in code_dir.rglob("*") if candidate.is_file()):
        relative = file_path.relative_to(code_dir).as_posix()
        if relative not in snapshot:
            file_path.unlink()
    for relative, payload in snapshot.items():
        target = code_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    for directory in sorted((candidate for candidate in code_dir.rglob("*") if candidate.is_dir()), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()


def _sanitized_env() -> tuple[dict[str, str], list[str]]:
    sanitized: dict[str, str] = {}
    secret_values: list[str] = []
    for key, value in os.environ.items():
        if SECRET_NAME_RE.search(key):
            if value:
                secret_values.append(value)
            continue
        sanitized[key] = value
    sanitized["PYTHONDONTWRITEBYTECODE"] = "1"
    return sanitized, secret_values


def _redact_text(text: str, secret_values: list[str]) -> str:
    redacted = text
    for secret_value in sorted({value for value in secret_values if len(value) >= 6}, key=len, reverse=True):
        redacted = redacted.replace(secret_value, "[REDACTED]")
    for pattern in SECRET_VALUE_PATTERNS:
        redacted = pattern.sub(lambda match: f"{match.group(1)}[REDACTED]" if match.lastindex else "[REDACTED]", redacted)
    return redacted


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
    code_dir = config.workspace_root / "submission" / "code"
    before_snapshot = _snapshot_submission_code(code_dir)
    env, secret_values = _sanitized_env()

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            ["/bin/bash", "-lc", command],
            cwd=str(cwd_check.resolved_path),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env=env,
        )
        elapsed = time.perf_counter() - started
        stdout = _redact_text(completed.stdout, secret_values)
        stderr = _redact_text(completed.stderr, secret_values)
        after_snapshot = _snapshot_submission_code(code_dir)
        diff = _diff_submission_code(before_snapshot, after_snapshot)
        if any(diff.values()):
            _restore_submission_code(code_dir, before_snapshot)
            log_path.write_text(
                (
                    f"$ {command}\n"
                    f"[stdout]\n{stdout}\n"
                    f"[stderr]\n{stderr}\n"
                    f"[returncode] {completed.returncode}\n"
                    f"[submission_code_diff] {diff}\n"
                ),
                encoding="utf-8",
            )
            return failure(
                "run_shell",
                "run_shell modified submission/code; only write_file may mutate submission/code",
                command=command,
                cwd=str(cwd_check.resolved_path),
                profile=profile,
                returncode=completed.returncode,
                stdout_tail=stdout[-4000:],
                stderr_tail=stderr[-4000:],
                elapsed_seconds=elapsed,
                timed_out=False,
                log_path=str(log_path),
                submission_code_diff=diff,
            )
        combined = (
            f"$ {command}\n"
            f"[stdout]\n{stdout}\n"
            f"[stderr]\n{stderr}\n"
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
            stdout_tail=stdout[-4000:],
            stderr_tail=stderr[-4000:],
            elapsed_seconds=elapsed,
            timed_out=False,
            log_path=str(log_path),
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        stdout = _redact_text(exc.stdout or "", secret_values)
        stderr = _redact_text(exc.stderr or "", secret_values)
        after_snapshot = _snapshot_submission_code(code_dir)
        diff = _diff_submission_code(before_snapshot, after_snapshot)
        if any(diff.values()):
            _restore_submission_code(code_dir, before_snapshot)
            return failure(
                "run_shell",
                "Command timed out and modified submission/code; only write_file may mutate submission/code",
                command=command,
                cwd=str(cwd_check.resolved_path),
                profile=profile,
                stdout_tail=stdout[-4000:],
                stderr_tail=stderr[-4000:],
                elapsed_seconds=elapsed,
                timed_out=True,
                log_path=str(log_path),
                submission_code_diff=diff,
            )
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
