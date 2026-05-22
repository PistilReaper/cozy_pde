from __future__ import annotations


def compatibility_checks_for_patch(
    *,
    supported_tasks: list[str],
    current_task: str,
    cli_ok: bool,
    smoke_ok: bool,
    infer_shape_ok: bool,
) -> dict[str, bool]:
    compatibility: dict[str, bool] = {}
    supported_ok = cli_ok and smoke_ok and infer_shape_ok

    for task_name in supported_tasks:
        compatibility[f"{task_name}_compat_ok"] = supported_ok

    compatibility.setdefault(f"{current_task}_compat_ok", True)
    return compatibility


def patch_compatibility_gate(
    *,
    supported_tasks: list[str],
    current_task: str,
    validation: dict[str, bool] | None = None,
) -> dict[str, object]:
    prior_supported_tasks = [task for task in supported_tasks if task and task != current_task]
    validation_payload = validation or {}
    needs_explicit_validation = bool(prior_supported_tasks)
    default_ok = not needs_explicit_validation
    compatibility = compatibility_checks_for_patch(
        supported_tasks=prior_supported_tasks,
        current_task=current_task,
        cli_ok=bool(validation_payload.get("cli_ok", default_ok)),
        smoke_ok=bool(validation_payload.get("smoke_ok", default_ok)),
        infer_shape_ok=bool(validation_payload.get("infer_shape_ok", default_ok)),
    )
    accepted = all(compatibility.get(f"{task}_compat_ok", False) for task in prior_supported_tasks)
    if not prior_supported_tasks:
        accepted = True
    reason = "compatibility evidence accepted"
    if not accepted:
        reason = "compatibility evidence required before modifying shared code for a later task"
    return {
        "accepted": accepted,
        "reason": reason,
        "compatibility": compatibility,
        "validated_against_tasks": prior_supported_tasks,
    }
