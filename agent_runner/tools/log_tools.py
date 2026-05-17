from __future__ import annotations

import re
from pathlib import Path

from ..safety import WorkspaceSafety
from . import failure, success

LOSS_RE = re.compile(r"(?:loss|val_loss|train_loss)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
ELAPSED_RE = re.compile(r"(?:elapsed(?:_seconds)?|time)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
SHAPE_MISMATCH_RE = re.compile(r"shape mismatch|mismatch shape|expected shape|invalid shape", re.IGNORECASE)
IMPORT_ERROR_RE = re.compile(r"importerror|module not found|no module named", re.IGNORECASE)
OOM_RE = re.compile(r"out of memory|cuda oom|cublas.*alloc|cudnn.*oom", re.IGNORECASE)
NAN_INF_RE = re.compile(r"\bnan\b|\binf\b", re.IGNORECASE)
SMOKE_SUCCESS_RE = re.compile(r"smoke\s+(success|succeeded|passed|complete)|smoke\s+train\s+(success|succeeded|passed)", re.IGNORECASE)


def analyze_log(*, path: str, safety: WorkspaceSafety) -> dict:
    check = safety.validate_read_path(path)
    if not check.ok:
        return failure("analyze_log", check.error or "read check failed", path=path)
    assert check.resolved_path is not None
    if not check.resolved_path.exists():
        return failure("analyze_log", "Log file does not exist", path=str(check.resolved_path))

    text = check.resolved_path.read_text(encoding="utf-8", errors="replace")
    losses = [float(match.group(1)) for match in LOSS_RE.finditer(text)]
    elapsed_values = [float(match.group(1)) for match in ELAPSED_RE.finditer(text)]
    lower = text.lower()
    nan_detected = bool(NAN_INF_RE.search(text)) and "nan" in lower
    inf_detected = bool(NAN_INF_RE.search(text)) and "inf" in lower
    oom_detected = bool(OOM_RE.search(text))
    shape_mismatch_detected = bool(SHAPE_MISMATCH_RE.search(text))
    import_error_detected = bool(IMPORT_ERROR_RE.search(text))
    smoke_success = bool(SMOKE_SUCCESS_RE.search(text))

    if nan_detected or inf_detected:
        status = "failed"
        recommendation = "rollback"
    elif oom_detected:
        status = "failed"
        recommendation = "reduce_model"
    elif shape_mismatch_detected or import_error_detected:
        status = "failed"
        recommendation = "fix_code"
    elif smoke_success:
        status = "smoke_passed"
        recommendation = "finalize_rehearsal"
    elif losses:
        status = "improving"
        recommendation = "continue"
    else:
        status = "unknown"
        recommendation = "inspect"

    return success(
        "analyze_log",
        f"Analyzed log with status {status}",
        status=status,
        best_metric=min(losses) if losses else None,
        best_epoch=None,
        nan_detected=nan_detected,
        inf_detected=inf_detected,
        oom_detected=oom_detected,
        shape_mismatch_detected=shape_mismatch_detected,
        import_error_detected=import_error_detected,
        smoke_success=smoke_success,
        recommendation=recommendation,
        loss_count=len(losses),
        elapsed_seconds=max(elapsed_values) if elapsed_values else None,
    )
