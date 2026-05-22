from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from cozy_pde_v3.code_evolution import (
    CodePatchRecord,
    CodeSnapshot,
    directory_api_contract_hash,
    snapshot_shared_code_directory,
    stable_shared_code_hash,
)
from cozy_pde_v3.memory_store import MemoryStore
from cozy_pde_v3.state import AgentState


def test_agent_state_tracks_v3_fields() -> None:
    state = AgentState(task="task-4")

    assert state.task == "task-4"
    assert state.mode == "formal"
    assert state.run_id == ""
    assert state.current_phase == "capability_check"
    assert state.current_objective == ""
    assert state.latest_error_type is None
    assert state.latest_error_summary is None
    assert state.latest_tool_name is None
    assert state.latest_tool_result_ok is None
    assert state.last_tool_call_id is None
    assert state.last_llm_call_id is None
    assert state.best_artifact_version is None
    assert state.best_artifact_path is None
    assert state.shared_code_version is None
    assert state.latest_checkpoint_path is None
    assert state.submission_snapshot_id is None
    assert state.supported_tasks == []
    assert state.finalize_gate_status == {}
    assert state.preflight_complete is False
    assert state.data_inspection_summary == {}


def test_agent_state_accepts_structured_finalize_gate_status() -> None:
    state = AgentState(
        task="task-2",
        finalize_gate_status={"status": "blocked", "reasons": ["missing-review"]},
    )

    assert state.finalize_gate_status == {
        "status": "blocked",
        "reasons": ["missing-review"],
    }


def test_memory_store_initialize_creates_persistence_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"

    store = MemoryStore(db_path)
    store.initialize()

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

    assert [row[0] for row in rows] == [
        "code_patch_records",
        "code_snapshots",
        "decision_records",
    ]

    with sqlite3.connect(db_path) as connection:
        decision_columns = connection.execute(
            "SELECT name FROM pragma_table_info('decision_records') ORDER BY cid"
        ).fetchall()

    assert [row[0] for row in decision_columns] == [
        "id",
        "state_hash",
        "reason_code",
        "route",
        "selected_profile",
        "selected_phase",
        "selected_tools",
        "outcome",
        "created_at",
    ]


def test_record_decision_and_list_round_trip_structured_fields(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.initialize()

    store.record_decision(
        state_hash="sha256:state-1",
        reason_code="baseline_missing",
        route="establish_shared_code_baseline",
        selected_profile="strong_planner",
        selected_phase="baseline_guard",
        selected_tools=["read_file"],
        outcome="accepted",
        created_at="2026-05-21T10:05:00Z",
    )

    assert store.list_decision_records() == [
        {
            "id": 1,
            "state_hash": "sha256:state-1",
            "reason_code": "baseline_missing",
            "route": "establish_shared_code_baseline",
            "selected_profile": "strong_planner",
            "selected_phase": "baseline_guard",
            "selected_tools": ["read_file"],
            "outcome": "accepted",
            "created_at": "2026-05-21T10:05:00Z",
        }
    ]


def test_record_code_snapshot_and_list_round_trip_structured_fields(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.initialize()

    snapshot = CodeSnapshot(
        code_version="sha256:version-a",
        parent_version="sha256:version-root",
        content_hash="sha256:content-a",
        api_contract_hash="sha256:api-a",
        supported_tasks=["task1", "task2"],
        task_support_matrix={
            "task1": {"status": "pass", "artifacts": ["model.py"]},
            "task2": {"status": "warn", "notes": ["needs-regression-check"]},
        },
        created_by_run_id="run-123",
        created_at="2026-05-21T10:00:00Z",
    )

    store.record_code_snapshot(snapshot)
    rows = store.list_code_snapshots()

    assert rows == [
        {
            "id": 1,
            "code_version": "sha256:version-a",
            "parent_version": "sha256:version-root",
            "content_hash": "sha256:content-a",
            "api_contract_hash": "sha256:api-a",
            "supported_tasks": ["task1", "task2"],
            "task_support_matrix": {
                "task1": {"status": "pass", "artifacts": ["model.py"]},
                "task2": {"status": "warn", "notes": ["needs-regression-check"]},
            },
            "created_by_run_id": "run-123",
            "created_at": "2026-05-21T10:00:00Z",
        }
    ]


def test_record_patch_and_list_round_trip_structured_fields(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.initialize()

    record = CodePatchRecord(
        patch_id="patch-1",
        base_code_version="sha256:version-a",
        new_code_version="sha256:version-b",
        task_context="task-3 repair flow after validation regression",
        changed_files=["submission/code/model.py"],
        change_intent="Fix task-3 regression without changing task-1 behavior",
        backward_compatibility_claim="Task1 behavior preserved; task2 not yet revalidated",
        affected_interfaces=["Model.forward()"],
        llm_call_ids=["call-7"],
        validation_results={"smoke": {"status": "pass"}},
    )

    store.record_patch(record)
    rows = store.list_patch_records()

    assert rows == [
        {
            "id": 1,
            "patch_id": "patch-1",
            "base_code_version": "sha256:version-a",
            "new_code_version": "sha256:version-b",
            "task_context": "task-3 repair flow after validation regression",
            "changed_files": ["submission/code/model.py"],
            "change_intent": "Fix task-3 regression without changing task-1 behavior",
            "backward_compatibility_claim": "Task1 behavior preserved; task2 not yet revalidated",
            "affected_interfaces": ["Model.forward()"],
            "llm_call_ids": ["call-7"],
            "validation_results": {"smoke": {"status": "pass"}},
        }
    ]


def test_stable_shared_code_hash_is_independent_of_dict_order() -> None:
    hash_a = stable_shared_code_hash(
        {
            "submission/code/b.py": "print('b')\n",
            "submission/code/a.py": "print('a')\n",
        }
    )
    hash_b = stable_shared_code_hash(
        {
            "submission/code/a.py": "print('a')\n",
            "submission/code/b.py": "print('b')\n",
        }
    )

    assert hash_a == hash_b


def test_stable_shared_code_hash_changes_when_content_changes() -> None:
    original_hash = stable_shared_code_hash({"submission/code/a.py": "print('a')\n"})
    updated_hash = stable_shared_code_hash({"submission/code/a.py": "print('changed')\n"})

    assert original_hash != updated_hash


def test_directory_helpers_use_real_submission_code_content_and_contract(tmp_path: Path) -> None:
    code_dir = tmp_path / "submission" / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "model.py").write_text("print('model')\n", encoding="utf-8")
    (code_dir / "train.py").write_text("print('train')\n", encoding="utf-8")

    contract_payload = {
        "task_id": "task1",
        "task_spec": {"equation": "Burgers", "input_steps": 10},
        "tool_schemas": [{"name": "write_file"}, {"name": "validate_submission"}],
    }

    snapshot = snapshot_shared_code_directory(
        code_dir=code_dir,
        api_contract_payload=contract_payload,
        parent_version=None,
        supported_tasks=["task1"],
        task_support_matrix={"task1": {"status": "baseline"}},
        created_by_run_id="run-123",
        created_at="2026-05-21T11:00:00Z",
    )

    assert snapshot.content_hash == stable_shared_code_hash(
        {
            "submission/code/model.py": "print('model')\n",
            "submission/code/train.py": "print('train')\n",
        }
    )
    assert snapshot.api_contract_hash == directory_api_contract_hash(contract_payload)
    assert snapshot.code_version.startswith("sha256:")
    assert json.loads(json.dumps(snapshot.task_support_matrix)) == {"task1": {"status": "baseline"}}
