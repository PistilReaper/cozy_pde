from __future__ import annotations

import hashlib
from pathlib import Path

from cozy_pde_v3.code_evolution import (
    directory_api_contract_hash,
    read_shared_code_directory_content,
    snapshot_shared_code_directory,
    stable_shared_code_hash,
    CodePatchRecord,
)
from cozy_pde_v3.memory_store import MemoryStore
from cozy_pde_v3.validation.provenance import build_shared_code_union


def _write_shared_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _derive_manifest_entries(
    *,
    code_dir: Path,
    latest_snapshot: dict[str, object],
    baseline_task: str,
    patch_record: dict[str, object],
) -> list[dict[str, object]]:
    changed_paths = {str(path) for path in patch_record["changed_files"]}
    entries: list[dict[str, object]] = []
    for file_path in sorted(path for path in code_dir.rglob("*") if path.is_file()):
        relative_path = f"submission/code/{file_path.relative_to(code_dir).as_posix()}"
        payload = file_path.read_bytes()
        changed_by_patch = relative_path in changed_paths
        entries.append(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
                "code_version": latest_snapshot["code_version"],
                "originating_task": patch_record["task_context"] if changed_by_patch else baseline_task,
                "patch_id": patch_record["patch_id"] if changed_by_patch else f"snapshot:{latest_snapshot['code_version']}",
                "step_id": patch_record["patch_id"] if changed_by_patch else f"snapshot:{latest_snapshot['code_version']}",
                "task_id": patch_record["task_context"] if changed_by_patch else baseline_task,
                "timestamp": latest_snapshot["created_at"],
                "llm_call_ids": list(patch_record["llm_call_ids"]) if changed_by_patch else [],
            }
        )
    return entries


def test_shared_code_evolution_tracks_real_hashes_lineage_and_manifest_interplay(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    code_dir = workspace / "submission" / "code"
    store = MemoryStore(workspace / "internal_logs" / "memory.db")
    store.initialize()

    _write_shared_file(
        code_dir / "train.py",
        "def train(task, config, data_dir, output_dir):\n    return (task, config, data_dir, output_dir)\n",
    )
    _write_shared_file(
        code_dir / "infer.py",
        "def infer(task, config, data_dir, output):\n    return (task, config, data_dir, output)\n",
    )
    _write_shared_file(code_dir / "model.py", "class SharedOperator:\n    width = 32\n")

    api_contract_payload = {
        "required_files": ["submission/code/train.py", "submission/code/infer.py"],
        "train_flags": ["--task", "--config", "--data_dir", "--output_dir"],
        "infer_flags": ["--task", "--config", "--data_dir", "--output"],
        "layout": "shared",
    }
    initial_content = read_shared_code_directory_content(code_dir)
    initial_snapshot = snapshot_shared_code_directory(
        code_dir=code_dir,
        api_contract_payload=api_contract_payload,
        parent_version=None,
        supported_tasks=["task1"],
        task_support_matrix={"task1": {"status": "pass", "cli_smoke": "pass"}},
        created_by_run_id="run-task1",
        created_at="2026-05-22T09:00:00Z",
    )

    assert initial_snapshot.content_hash == stable_shared_code_hash(initial_content)
    assert initial_snapshot.api_contract_hash == directory_api_contract_hash(api_contract_payload)
    store.record_code_snapshot(initial_snapshot)

    _write_shared_file(
        code_dir / "model.py",
        "class SharedOperator:\n    width = 48\n    task2_adapter = True\n",
    )
    updated_content = read_shared_code_directory_content(code_dir)
    updated_snapshot = snapshot_shared_code_directory(
        code_dir=code_dir,
        api_contract_payload=api_contract_payload,
        parent_version=initial_snapshot.code_version,
        supported_tasks=["task1", "task2"],
        task_support_matrix={
            "task1": {"status": "pass", "compatibility_preserved": True},
            "task2": {"status": "accepted", "cli_smoke": "pass"},
        },
        created_by_run_id="run-task2",
        created_at="2026-05-22T10:00:00Z",
    )

    assert updated_snapshot.parent_version == initial_snapshot.code_version
    assert updated_snapshot.api_contract_hash == initial_snapshot.api_contract_hash
    assert updated_snapshot.content_hash == stable_shared_code_hash(updated_content)
    assert updated_snapshot.content_hash != initial_snapshot.content_hash
    assert updated_snapshot.code_version != initial_snapshot.code_version
    store.record_code_snapshot(updated_snapshot)

    patch_record = CodePatchRecord(
        patch_id="patch-task2-001",
        base_code_version=initial_snapshot.code_version,
        new_code_version=updated_snapshot.code_version,
        task_context="task2",
        changed_files=["submission/code/model.py"],
        change_intent="Extend the shared operator for task2 while keeping task1 runnable",
        backward_compatibility_claim="task1 compatibility preserved after task2 patch",
        affected_interfaces=["submission/code/model.py"],
        llm_call_ids=["call-task2-1"],
        validation_results={
            "task_compatibility": {"task1": True, "task2": True},
            "patch_acceptance": {"task1": "preserved", "task2": "accepted"},
            "api_contract_preserved": True,
        },
    )
    store.record_patch(patch_record)

    snapshots = store.list_code_snapshots()
    patch_rows = store.list_patch_records()

    assert [row["code_version"] for row in snapshots] == [
        initial_snapshot.code_version,
        updated_snapshot.code_version,
    ]
    assert snapshots[1]["parent_version"] == initial_snapshot.code_version
    assert snapshots[1]["content_hash"] == updated_snapshot.content_hash
    assert snapshots[1]["api_contract_hash"] == updated_snapshot.api_contract_hash
    assert snapshots[1]["supported_tasks"] == ["task1", "task2"]
    assert snapshots[1]["task_support_matrix"]["task1"]["compatibility_preserved"] is True

    assert patch_rows == [
        {
            "id": 1,
            "patch_id": "patch-task2-001",
            "base_code_version": initial_snapshot.code_version,
            "new_code_version": updated_snapshot.code_version,
            "task_context": "task2",
            "changed_files": ["submission/code/model.py"],
            "change_intent": "Extend the shared operator for task2 while keeping task1 runnable",
            "backward_compatibility_claim": "task1 compatibility preserved after task2 patch",
            "affected_interfaces": ["submission/code/model.py"],
            "llm_call_ids": ["call-task2-1"],
            "validation_results": {
                "task_compatibility": {"task1": True, "task2": True},
                "patch_acceptance": {"task1": "preserved", "task2": "accepted"},
                "api_contract_preserved": True,
            },
        }
    ]

    assert patch_rows[0]["validation_results"]["task_compatibility"]["task1"] is True
    assert patch_rows[0]["validation_results"]["patch_acceptance"]["task1"] == "preserved"

    manifest_entries = _derive_manifest_entries(
        code_dir=code_dir,
        latest_snapshot=snapshots[-1],
        baseline_task="task1",
        patch_record=patch_rows[0],
    )

    assert manifest_entries == [
        {
            "path": "submission/code/infer.py",
            "sha256": hashlib.sha256((code_dir / "infer.py").read_bytes()).hexdigest(),
            "size": (code_dir / "infer.py").stat().st_size,
            "code_version": updated_snapshot.code_version,
            "originating_task": "task1",
            "patch_id": f"snapshot:{updated_snapshot.code_version}",
            "step_id": f"snapshot:{updated_snapshot.code_version}",
            "task_id": "task1",
            "timestamp": "2026-05-22T10:00:00Z",
            "llm_call_ids": [],
        },
        {
            "path": "submission/code/model.py",
            "sha256": hashlib.sha256((code_dir / "model.py").read_bytes()).hexdigest(),
            "size": (code_dir / "model.py").stat().st_size,
            "code_version": updated_snapshot.code_version,
            "originating_task": "task2",
            "patch_id": "patch-task2-001",
            "step_id": "patch-task2-001",
            "task_id": "task2",
            "timestamp": "2026-05-22T10:00:00Z",
            "llm_call_ids": ["call-task2-1"],
        },
        {
            "path": "submission/code/train.py",
            "sha256": hashlib.sha256((code_dir / "train.py").read_bytes()).hexdigest(),
            "size": (code_dir / "train.py").stat().st_size,
            "code_version": updated_snapshot.code_version,
            "originating_task": "task1",
            "patch_id": f"snapshot:{updated_snapshot.code_version}",
            "step_id": f"snapshot:{updated_snapshot.code_version}",
            "task_id": "task1",
            "timestamp": "2026-05-22T10:00:00Z",
            "llm_call_ids": [],
        },
    ]

    shared_union = build_shared_code_union(
        [
            {
                "version": snapshots[0]["code_version"],
                "files": sorted(initial_content),
                "validated_tasks": snapshots[0]["supported_tasks"],
            },
            {
                "version": snapshots[1]["code_version"],
                "created_during": patch_rows[0]["task_context"],
                "parent": snapshots[1]["parent_version"],
                "changed_files": patch_rows[0]["changed_files"],
                "validated_tasks": snapshots[1]["supported_tasks"],
                "llm_call_ids": patch_rows[0]["llm_call_ids"],
            },
        ]
    )

    assert len(shared_union["shared_code_versions"]) == 2
    union_by_version = {
        entry["version"]: entry
        for entry in shared_union["shared_code_versions"]
    }
    assert union_by_version[initial_snapshot.code_version] == {
        "version": initial_snapshot.code_version,
        "files": [
            "submission/code/infer.py",
            "submission/code/model.py",
            "submission/code/train.py",
        ],
        "validated_tasks": ["task1"],
        "llm_call_ids": [],
    }
    assert union_by_version[updated_snapshot.code_version] == {
        "version": updated_snapshot.code_version,
        "created_during": "task2",
        "parent": initial_snapshot.code_version,
        "changed_files": ["submission/code/model.py"],
        "validated_tasks": ["task1", "task2"],
        "llm_call_ids": ["call-task2-1"],
    }
