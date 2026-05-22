from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cozy_pde_v3.code_evolution import CodePatchRecord, CodeSnapshot


def _serialize_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _deserialize_json(value: str) -> Any:
    return json.loads(value)


class MemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS code_snapshots (
                    id INTEGER PRIMARY KEY,
                    code_version TEXT NOT NULL,
                    parent_version TEXT,
                    content_hash TEXT NOT NULL,
                    api_contract_hash TEXT NOT NULL,
                    supported_tasks TEXT NOT NULL,
                    task_support_matrix TEXT NOT NULL,
                    created_by_run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS code_patch_records (
                    id INTEGER PRIMARY KEY,
                    patch_id TEXT NOT NULL,
                    base_code_version TEXT NOT NULL,
                    new_code_version TEXT NOT NULL,
                    task_context TEXT NOT NULL,
                    changed_files TEXT NOT NULL,
                    change_intent TEXT NOT NULL,
                    backward_compatibility_claim TEXT NOT NULL,
                    affected_interfaces TEXT NOT NULL,
                    llm_call_ids TEXT NOT NULL,
                    validation_results TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_records (
                    id INTEGER PRIMARY KEY,
                    state_hash TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    route TEXT NOT NULL,
                    selected_profile TEXT NOT NULL,
                    selected_phase TEXT NOT NULL,
                    selected_tools TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def record_code_snapshot(self, snapshot: CodeSnapshot) -> None:
        payload = asdict(snapshot)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO code_snapshots (
                    code_version,
                    parent_version,
                    content_hash,
                    api_contract_hash,
                    supported_tasks,
                    task_support_matrix,
                    created_by_run_id,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["code_version"],
                    payload["parent_version"],
                    payload["content_hash"],
                    payload["api_contract_hash"],
                    _serialize_json(payload["supported_tasks"]),
                    _serialize_json(payload["task_support_matrix"]),
                    payload["created_by_run_id"],
                    payload["created_at"],
                ),
            )

    def list_code_snapshots(self) -> list[dict[str, object]]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    id,
                    code_version,
                    parent_version,
                    content_hash,
                    api_contract_hash,
                    supported_tasks,
                    task_support_matrix,
                    created_by_run_id,
                    created_at
                FROM code_snapshots
                ORDER BY id ASC
                """
            ).fetchall()
        return [
            {
                "id": row["id"],
                "code_version": row["code_version"],
                "parent_version": row["parent_version"],
                "content_hash": row["content_hash"],
                "api_contract_hash": row["api_contract_hash"],
                "supported_tasks": _deserialize_json(row["supported_tasks"]),
                "task_support_matrix": _deserialize_json(row["task_support_matrix"]),
                "created_by_run_id": row["created_by_run_id"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def record_patch(self, record: CodePatchRecord) -> None:
        payload = asdict(record)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO code_patch_records (
                    patch_id,
                    base_code_version,
                    new_code_version,
                    task_context,
                    changed_files,
                    change_intent,
                    backward_compatibility_claim,
                    affected_interfaces,
                    llm_call_ids,
                    validation_results
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["patch_id"],
                    payload["base_code_version"],
                    payload["new_code_version"],
                    payload["task_context"],
                    _serialize_json(payload["changed_files"]),
                    payload["change_intent"],
                    payload["backward_compatibility_claim"],
                    _serialize_json(payload["affected_interfaces"]),
                    _serialize_json(payload["llm_call_ids"]),
                    _serialize_json(payload["validation_results"]),
                ),
            )

    def list_patch_records(self) -> list[dict[str, object]]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    id,
                    patch_id,
                    base_code_version,
                    new_code_version,
                    task_context,
                    changed_files,
                    change_intent,
                    backward_compatibility_claim,
                    affected_interfaces,
                    llm_call_ids,
                    validation_results
                FROM code_patch_records
                ORDER BY id ASC
                """
            ).fetchall()
        return [
            {
                "id": row["id"],
                "patch_id": row["patch_id"],
                "base_code_version": row["base_code_version"],
                "new_code_version": row["new_code_version"],
                "task_context": row["task_context"],
                "changed_files": _deserialize_json(row["changed_files"]),
                "change_intent": row["change_intent"],
                "backward_compatibility_claim": row["backward_compatibility_claim"],
                "affected_interfaces": _deserialize_json(row["affected_interfaces"]),
                "llm_call_ids": _deserialize_json(row["llm_call_ids"]),
                "validation_results": _deserialize_json(row["validation_results"]),
            }
            for row in rows
        ]

    def record_decision(
        self,
        *,
        state_hash: str,
        reason_code: str,
        route: str,
        selected_profile: str,
        selected_phase: str,
        selected_tools: list[str],
        outcome: str,
        created_at: str,
    ) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO decision_records (
                    state_hash,
                    reason_code,
                    route,
                    selected_profile,
                    selected_phase,
                    selected_tools,
                    outcome,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state_hash,
                    reason_code,
                    route,
                    selected_profile,
                    selected_phase,
                    _serialize_json(selected_tools),
                    outcome,
                    created_at,
                ),
            )

    def list_decision_records(self) -> list[dict[str, object]]:
        with sqlite3.connect(self.db_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT
                    id,
                    state_hash,
                    reason_code,
                    route,
                    selected_profile,
                    selected_phase,
                    selected_tools,
                    outcome,
                    created_at
                FROM decision_records
                ORDER BY id ASC
                """
            ).fetchall()
        return [
            {
                "id": row["id"],
                "state_hash": row["state_hash"],
                "reason_code": row["reason_code"],
                "route": row["route"],
                "selected_profile": row["selected_profile"],
                "selected_phase": row["selected_phase"],
                "selected_tools": _deserialize_json(row["selected_tools"]),
                "outcome": row["outcome"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def latest_code_snapshot(self) -> dict[str, object] | None:
        snapshots = self.list_code_snapshots()
        if not snapshots:
            return None
        return snapshots[-1]
