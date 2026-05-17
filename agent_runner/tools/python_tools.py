from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone

from ..config import RunnerConfig
from ..safety import WorkspaceSafety
from . import failure, success


def run_python(
    *,
    code: str,
    safety: WorkspaceSafety,
    config: RunnerConfig,
    timeout_seconds: int = 120,
) -> dict:
    timeout_seconds = min(timeout_seconds, 120)
    scratch_dir = config.workspace_root / "runs" / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    script_path = scratch_dir / f"scratch_{stamp}.py"
    write_check = safety.validate_write_path(script_path)
    if not write_check.ok:
        return failure("run_python", write_check.error or "scratch path rejected", path=str(script_path))

    script_path.write_text(code, encoding="utf-8")
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(config.workspace_root),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed = time.perf_counter() - started
        return success(
            "run_python",
            f"Scratch Python finished with return code {completed.returncode}",
            script_path=str(script_path),
            returncode=completed.returncode,
            stdout_tail=completed.stdout[-4000:],
            stderr_tail=completed.stderr[-4000:],
            elapsed_seconds=elapsed,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        return failure(
            "run_python",
            f"Scratch Python timed out after {timeout_seconds} seconds",
            script_path=str(script_path),
            stdout_tail=(exc.stdout or "")[-4000:],
            stderr_tail=(exc.stderr or "")[-4000:],
            elapsed_seconds=elapsed,
            timed_out=True,
        )

