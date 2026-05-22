from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from cozy_pde_v3.code_evolution import CodePatchRecord, CodeSnapshot
from cozy_pde_v3.memory_store import MemoryStore
from cozy_pde_v3.package import build_code_manifest_entries, build_shared_code_union_for_workspace


def _write_shared_code(workspace: Path) -> dict[str, str]:
    files = {
        "model.py": "print('model')\n",
        "infer.py": "print('infer')\n",
    }
    for name, content in files.items():
        path = workspace / "submission" / "code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return files


def _seed_memory_store(workspace: Path, files: dict[str, str]) -> None:
    store = MemoryStore(workspace / "memory.db")
    store.initialize()
    store.record_code_snapshot(
        CodeSnapshot(
            code_version="sha256:v1",
            parent_version=None,
            content_hash="sha256:content-v1",
            api_contract_hash="sha256:api-v1",
            supported_tasks=["task1"],
            task_support_matrix={"task1": {"status": "pass"}},
            created_by_run_id="run-001",
            created_at="2026-05-22T00:00:00Z",
        )
    )
    store.record_code_snapshot(
        CodeSnapshot(
            code_version="sha256:v2",
            parent_version="sha256:v1",
            content_hash="sha256:content-v2",
            api_contract_hash="sha256:api-v2",
            supported_tasks=["task1", "task2"],
            task_support_matrix={"task1": {"status": "pass"}, "task2": {"status": "pass"}},
            created_by_run_id="run-002",
            created_at="2026-05-22T00:10:00Z",
        )
    )
    store.record_patch(
        CodePatchRecord(
            patch_id="patch-002",
            base_code_version="sha256:v1",
            new_code_version="sha256:v2",
            task_context="task2",
            changed_files=[f"submission/code/{name}" for name in files],
            change_intent="Unify shared inference path",
            backward_compatibility_claim="task1 remains supported",
            affected_interfaces=["infer()", "model()"],
            llm_call_ids=["call-2", "call-1"],
            validation_results={"validated_tasks": ["task1", "task2"]},
        )
    )


def test_build_code_manifest_entries_records_required_v3_fields(workspace: Path) -> None:
    files = _write_shared_code(workspace)
    _seed_memory_store(workspace, files)

    entries = build_code_manifest_entries(workspace_root=workspace, tasks=["task1", "task2"])

    assert [entry["path"] for entry in entries] == [
        "submission/code/infer.py",
        "submission/code/model.py",
    ]
    assert entries[0]["sha256"] == sha256(files["infer.py"].encode("utf-8")).hexdigest()
    assert entries[1]["sha256"] == sha256(files["model.py"].encode("utf-8")).hexdigest()
    assert all(entry["code_version"] == "sha256:v2" for entry in entries)
    assert all(entry["originating_task"] == "task2" for entry in entries)
    assert all(entry["llm_call_ids"] == ["call-1", "call-2"] for entry in entries)
    assert all(entry["patch_id"] == "patch-002" for entry in entries)
    assert all("size" in entry for entry in entries)
    assert all("step_id" in entry for entry in entries)
    assert all("task_id" in entry for entry in entries)
    assert all("timestamp" in entry for entry in entries)


def test_build_shared_code_union_for_workspace_preserves_version_chain(workspace: Path) -> None:
    files = _write_shared_code(workspace)
    _seed_memory_store(workspace, files)
    code_manifest_entries = build_code_manifest_entries(workspace_root=workspace, tasks=["task1", "task2"])

    shared_code_union = build_shared_code_union_for_workspace(
        workspace_root=workspace,
        tasks=["task1", "task2"],
        code_manifest_entries=code_manifest_entries,
    )

    assert shared_code_union == {
        "shared_code_versions": [
            {
                "version": "sha256:v1",
                "validated_tasks": ["task1"],
                "llm_call_ids": [],
            },
            {
                "version": "sha256:v2",
                "created_during": "task2",
                "parent": "sha256:v1",
                "changed_files": ["submission/code/infer.py", "submission/code/model.py"],
                "validated_tasks": ["task1", "task2"],
                "llm_call_ids": ["call-1", "call-2"],
            },
        ]
    }
