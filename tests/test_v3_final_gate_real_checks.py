from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from textwrap import dedent

import h5py
import numpy as np

from cozy_pde_v3.code_evolution import CodePatchRecord, CodeSnapshot
from cozy_pde_v3.memory_store import MemoryStore
from cozy_pde_v3.validation.submission import validate_submission_bundle_v3


STRUCTURED_METHODOLOGY_SOURCES = [
    "agent_state_snapshots",
    "decision_records",
    "experiment_cards",
    "validation_reports",
    "artifact_metadata",
    "final_package_snapshot",
    "code_snapshots",
    "code_patch_records",
]


def _write_task_test(path: Path, *, samples: int, total_steps: int) -> None:
    array = np.arange(samples * total_steps * 256, dtype=np.float32).reshape(samples, total_steps, 256)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=array)


def _write_submission_outputs(workspace: Path, task: str, *, samples: int, total_steps: int, input_steps: int) -> None:
    test_path = workspace / "data" / f"{task}_test.hdf5"
    _write_task_test(test_path, samples=samples, total_steps=total_steps)
    with h5py.File(test_path, "r") as source:
        tensor = source["tensor"][:]
    pred = tensor.copy()
    pred[:, input_steps:, :] = pred[:, input_steps:, :] + np.float32(0.01)
    with h5py.File(workspace / "submission" / f"{task}_pred.hdf5", "w") as handle:
        handle.create_dataset("pred", data=pred)
    (workspace / "submission" / f"{task}_time.csv").write_text(
        "train_time,inference_time\n1.0,0.2\n",
        encoding="utf-8",
    )
    (workspace / "submission" / f"{task}_logs.log").write_text(
        json.dumps(
            {
                "timestamp": "2026-05-22T00:00:00+00:00",
                "elapsed_seconds": 0.1,
                "response": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_shared_code(workspace: Path) -> list[dict[str, object]]:
    files = {
        "train.py": dedent(
            """
            from __future__ import annotations

            import argparse
            import json
            from pathlib import Path


            def main() -> int:
                parser = argparse.ArgumentParser()
                parser.add_argument("--task", required=True)
                parser.add_argument("--config", required=True)
                parser.add_argument("--data_dir", required=True)
                parser.add_argument("--output_dir", required=True)
                args = parser.parse_args()
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "checkpoint.pt").write_text(json.dumps({"task": args.task}), encoding="utf-8")
                return 0


            if __name__ == "__main__":
                raise SystemExit(main())
            """
        ).strip()
        + "\n",
        "infer.py": dedent(
            """
            from __future__ import annotations

            import argparse
            import json
            from pathlib import Path

            import h5py
            import numpy as np


            _INPUT_STEPS = {"task1": 10, "task2": 10, "task3": 20}


            def _first_dataset(handle: h5py.File):
                datasets = []

                def collect(_, obj):
                    if isinstance(obj, h5py.Dataset):
                        datasets.append(obj)

                handle.visititems(collect)
                return datasets[0]


            def main() -> int:
                parser = argparse.ArgumentParser()
                parser.add_argument("--task", required=True)
                parser.add_argument("--config", required=True)
                parser.add_argument("--data_dir", required=True)
                parser.add_argument("--output_dir", required=True)
                parser.add_argument("--output", required=True)
                args = parser.parse_args()
                checkpoint_path = Path(args.output_dir) / "checkpoint.pt"
                json.loads(checkpoint_path.read_text(encoding="utf-8"))
                with h5py.File(Path(args.data_dir) / f"{args.task}_test.hdf5", "r") as source:
                    tensor = np.asarray(_first_dataset(source)[...])
                pred = tensor.copy()
                pred[:, _INPUT_STEPS[args.task] :, :] = pred[:, _INPUT_STEPS[args.task] :, :] + np.float32(0.01)
                with h5py.File(args.output, "w") as handle:
                    handle.create_dataset("pred", data=pred)
                return 0


            if __name__ == "__main__":
                raise SystemExit(main())
            """
        ).strip()
        + "\n",
        "validate.py": "print('validate')\n",
    }
    entries: list[dict[str, object]] = []
    code_dir = workspace / "submission" / "code"
    for index, (name, content) in enumerate(files.items(), start=1):
        path = code_dir / name
        path.write_text(content, encoding="utf-8")
        payload = content.encode("utf-8")
        entries.append(
            {
                "path": f"submission/code/{name}",
                "sha256": sha256(payload).hexdigest(),
                "size": len(payload),
                "code_version": "v2",
                "originating_task": "task1",
                "patch_id": "patch-001",
                "step_id": f"step-{index:03d}",
                "task_id": "task1",
                "timestamp": "2026-05-22T00:00:00+00:00",
                "llm_call_ids": [f"call-{index}"],
            }
        )
    return entries


def _write_memory_records(
    workspace: Path,
    *,
    supported_tasks: list[str],
    baseline_version: str = "v1",
    final_version: str = "v2",
) -> None:
    store = MemoryStore(workspace / "internal_logs" / "memory.db")
    store.initialize()
    store.record_code_snapshot(
        CodeSnapshot(
            code_version=baseline_version,
            parent_version=None,
            content_hash="sha256:root",
            api_contract_hash="sha256:shared-cli-v1",
            supported_tasks=[],
            task_support_matrix={},
            created_by_run_id="run-001",
            created_at="2026-05-22T00:00:00Z",
        )
    )
    store.record_code_snapshot(
        CodeSnapshot(
            code_version=final_version,
            parent_version=baseline_version,
            content_hash="sha256:final",
            api_contract_hash="sha256:shared-cli-v1",
            supported_tasks=supported_tasks,
            task_support_matrix={task: {"compat": True} for task in supported_tasks},
            created_by_run_id="run-001",
            created_at="2026-05-22T00:10:00Z",
        )
    )
    store.record_patch(
        CodePatchRecord(
            patch_id="patch-001",
            base_code_version=baseline_version,
            new_code_version=final_version,
            task_context="shared submission finalize",
            changed_files=[
                "submission/code/train.py",
                "submission/code/infer.py",
                "submission/code/validate.py",
            ],
            change_intent="Finalize shared CLI",
            backward_compatibility_claim="Earlier task API contract preserved",
            affected_interfaces=["train.py", "infer.py"],
            llm_call_ids=["call-1"],
            validation_results={"task_compatibility": {task: True for task in supported_tasks}},
        )
    )


def test_validate_submission_bundle_v3_reports_real_gate_success(workspace: Path) -> None:
    _write_submission_outputs(workspace, "task1", samples=2, total_steps=200, input_steps=10)
    _write_submission_outputs(workspace, "task3", samples=1000, total_steps=400, input_steps=20)
    code_manifest_entries = _write_shared_code(workspace)
    _write_memory_records(workspace, supported_tasks=["task1", "task3"])
    (workspace / "submission" / "methodology.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (workspace / "submission" / "submission.json").write_text("{}", encoding="utf-8")

    result = validate_submission_bundle_v3(
        workspace_root=workspace,
        tasks=["task1", "task3"],
        strict=True,
        code_manifest_entries=code_manifest_entries,
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
    )

    assert result["ok"] is True
    gate = result["data"]["finalize_gate"]
    assert gate["task_rule_ok"] is True
    assert gate["api_contract_ok"] is True
    assert gate["incremental_patch_ok"] is True
    assert gate["task1_cli_parse_ok"] is True
    assert gate["task1_train_smoke_ok"] is True
    assert gate["task1_infer_smoke_ok"] is True
    assert gate["task1_checkpoint_load_ok"] is True
    assert gate["task3_checkpoint_load_ok"] is True
    assert gate["secret_scan_ok"] is True
    assert gate["overall_ok"] is True


def test_validate_submission_bundle_v3_requires_methodology_pdf_artifact(workspace: Path) -> None:
    _write_submission_outputs(workspace, "task1", samples=2, total_steps=200, input_steps=10)
    code_manifest_entries = _write_shared_code(workspace)
    _write_memory_records(workspace, supported_tasks=["task1"])
    (workspace / "submission" / "submission.json").write_text("{}", encoding="utf-8")

    result = validate_submission_bundle_v3(
        workspace_root=workspace,
        tasks=["task1"],
        strict=True,
        code_manifest_entries=code_manifest_entries,
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
    )

    assert result["ok"] is False
    gate = result["data"]["finalize_gate"]
    assert gate["methodology_ok"] is False
    assert "methodology.pdf missing" in "\n".join(gate["failures"])


def test_validate_submission_bundle_v3_rejects_explicit_manifest_entries_missing_required_fields(workspace: Path) -> None:
    _write_submission_outputs(workspace, "task1", samples=2, total_steps=200, input_steps=10)
    code_manifest_entries = _write_shared_code(workspace)
    _write_memory_records(workspace, supported_tasks=["task1"])
    (workspace / "submission" / "methodology.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (workspace / "submission" / "submission.json").write_text("{}", encoding="utf-8")
    invalid_entries = []
    for entry in code_manifest_entries:
        broken = dict(entry)
        broken.pop("code_version")
        invalid_entries.append(broken)

    result = validate_submission_bundle_v3(
        workspace_root=workspace,
        tasks=["task1"],
        strict=True,
        code_manifest_entries=invalid_entries,
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
    )

    assert result["ok"] is False
    gate = result["data"]["finalize_gate"]
    assert gate["code_manifest_ok"] is False
    assert any(
        failure.startswith("code manifest entry missing code_version:")
        for failure in gate["failures"]
    )


def test_validate_submission_bundle_v3_keeps_prediction_gate_true_when_only_logs_fail(workspace: Path) -> None:
    _write_submission_outputs(workspace, "task1", samples=2, total_steps=200, input_steps=10)
    code_manifest_entries = _write_shared_code(workspace)
    _write_memory_records(workspace, supported_tasks=["task1"])
    (workspace / "submission" / "methodology.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (workspace / "submission" / "submission.json").write_text("{}", encoding="utf-8")
    (workspace / "submission" / "task1_logs.log").write_text("{not-json}\n", encoding="utf-8")

    result = validate_submission_bundle_v3(
        workspace_root=workspace,
        tasks=["task1"],
        strict=True,
        code_manifest_entries=code_manifest_entries,
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
    )

    assert result["ok"] is False
    gate = result["data"]["finalize_gate"]
    assert gate["logs_ok"] is False
    assert gate["prediction_ok"] is True


def test_validate_submission_bundle_v3_accepts_hash_version_lineage_in_recorded_order(workspace: Path) -> None:
    _write_submission_outputs(workspace, "task1", samples=2, total_steps=200, input_steps=10)
    code_manifest_entries = _write_shared_code(workspace)
    _write_memory_records(
        workspace,
        supported_tasks=["task1"],
        baseline_version="sha256:z-base",
        final_version="sha256:a-next",
    )
    (workspace / "submission" / "methodology.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (workspace / "submission" / "submission.json").write_text("{}", encoding="utf-8")

    result = validate_submission_bundle_v3(
        workspace_root=workspace,
        tasks=["task1"],
        strict=True,
        code_manifest_entries=code_manifest_entries,
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
    )

    assert result["ok"] is True
    assert result["data"]["finalize_gate"]["incremental_patch_ok"] is True


def test_validate_submission_bundle_v3_scans_internal_and_proxy_logs_for_secrets(workspace: Path) -> None:
    _write_submission_outputs(workspace, "task1", samples=2, total_steps=200, input_steps=10)
    code_manifest_entries = _write_shared_code(workspace)
    _write_memory_records(workspace, supported_tasks=["task1"])
    (workspace / "submission" / "methodology.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (workspace / "submission" / "submission.json").write_text("{}", encoding="utf-8")
    (workspace / "internal_logs" / "tool_calls.jsonl").write_text('{"token":"sk-abcdefghijklmnopqrstuvwxyz123456"}\n', encoding="utf-8")
    (workspace / "proxy_logs").mkdir(parents=True, exist_ok=True)
    (workspace / "proxy_logs" / "llm.jsonl").write_text('{"authorization":"Bearer abc"}\n', encoding="utf-8")

    result = validate_submission_bundle_v3(
        workspace_root=workspace,
        tasks=["task1"],
        strict=True,
        code_manifest_entries=code_manifest_entries,
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
    )

    assert result["ok"] is False
    gate = result["data"]["finalize_gate"]
    assert gate["secret_scan_ok"] is False
    joined = "\n".join(gate["failures"])
    assert "internal_logs/tool_calls.jsonl" in joined
    assert "proxy_logs/llm.jsonl" in joined
