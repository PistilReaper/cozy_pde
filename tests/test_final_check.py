from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent

import h5py
import numpy as np

from cozy_pde_v3.code_evolution import CodePatchRecord, CodeSnapshot
from cozy_pde_v3.memory_store import MemoryStore
from cozy_pde_v3.package import package_submission_v3
from cozy_pde_v3.validation.submission import build_finalize_gate_status, validate_submission_bundle_v3


def _write_valid_task_bundle(workspace: Path, task: str) -> None:
    tensor = np.linspace(0.0, 1.0, num=2 * 200 * 256, dtype=np.float32).reshape(2, 200, 256)
    with h5py.File(workspace / "data" / f"{task}_test.hdf5", "w") as handle:
        handle.create_dataset("tensor", data=tensor)
    with h5py.File(workspace / "submission" / f"{task}_pred.hdf5", "w") as handle:
        pred = tensor.copy()
        pred[:, 10:, :] = pred[:, 10:, :] + 0.02
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


def _write_shared_code(workspace: Path, *, task_fork: bool = False) -> list[dict[str, object]]:
    code_dir = workspace / "submission" / "code"
    target_dir = code_dir / "task1" if task_fork else code_dir
    target_dir.mkdir(parents=True, exist_ok=True)
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
                json.loads((Path(args.output_dir) / "checkpoint.pt").read_text(encoding="utf-8"))
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
    }
    manifest_entries: list[dict[str, object]] = []
    for index, (name, content) in enumerate(files.items(), start=1):
        relative_path = Path("submission/code") / ("task1" if task_fork else "") / name
        (target_dir / name).write_text(content, encoding="utf-8")
        payload = content.encode("utf-8")
        manifest_entries.append(
            {
                "path": relative_path.as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
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
    return manifest_entries


def _submission_task_code_path(task: str, filename: str) -> str:
    return "/".join(("submission", "code", task, filename))


def _write_incremental_records(workspace: Path, *, supported_tasks: list[str]) -> None:
    store = MemoryStore(workspace / "internal_logs" / "memory.db")
    store.initialize()
    store.record_code_snapshot(
        CodeSnapshot(
            code_version="v1",
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
            code_version="v2",
            parent_version="v1",
            content_hash="sha256:final",
            api_contract_hash="sha256:shared-cli-v1",
            supported_tasks=supported_tasks,
            task_support_matrix={task: {"compat": True} for task in supported_tasks},
            created_by_run_id="run-001",
            created_at="2026-05-22T00:05:00Z",
        )
    )
    store.record_patch(
        CodePatchRecord(
            patch_id="patch-001",
            base_code_version="v1",
            new_code_version="v2",
            task_context="shared finalize",
            changed_files=[
                "submission/code/train.py",
                "submission/code/infer.py",
            ],
            change_intent="Finalize shared CLI",
            backward_compatibility_claim="Earlier task API contract preserved",
            affected_interfaces=["train.py", "infer.py"],
            llm_call_ids=["call-1"],
            validation_results={"task_compatibility": {task: True for task in supported_tasks}},
        )
    )


def test_finalize_gate_requires_structured_methodology_sources() -> None:
    status = build_finalize_gate_status(
        prediction_ok=True,
        logs_ok=True,
        provenance_log_ok=True,
        package_ok=True,
        methodology_records_only_ok=False,
        final_code_paths=["submission/code/train.py"],
        provenance_links={"submission/code/train.py": ["call-1"]},
        cli_smoke_status={"task1": True},
        cli_smoke_details={"task1": {"cli_parse_ok": True, "train_smoke_ok": True, "infer_smoke_ok": True, "checkpoint_load_ok": True}},
        api_contract_ok=True,
        incremental_patch_ok=True,
        shared_code_ok=True,
        code_manifest_entries=[
            {
                "path": "submission/code/train.py",
                "llm_call_ids": ["call-1"],
            }
        ],
        methodology_sources=None,
        supported_tasks=["task1"],
    )

    assert status["methodology_records_only_ok"] is False
    assert status["overall_ok"] is False
    assert any("methodology structured sources missing" in failure for failure in status["failures"])


def test_v3_final_check_marks_missing_code_provenance_as_failure(workspace: Path) -> None:
    _write_valid_task_bundle(workspace, "task1")
    manifest_entries = _write_shared_code(workspace)
    _write_incremental_records(workspace, supported_tasks=["task1"])
    manifest_entries[0]["llm_call_ids"] = []

    result = validate_submission_bundle_v3(
        workspace_root=workspace,
        tasks=["task1"],
        strict=True,
        code_manifest_entries=manifest_entries,
        methodology_sources=["code_patch_records"],
    )

    assert result["ok"] is False
    gate = result["data"]["finalize_gate"]
    assert gate["code_provenance_ok"] is False
    assert "submission/code/train.py" in gate["missing_code_provenance_paths"]


def test_v3_final_check_rejects_task_specific_code_forks(workspace: Path) -> None:
    _write_valid_task_bundle(workspace, "task1")
    manifest_entries = _write_shared_code(workspace, task_fork=True)
    _write_incremental_records(workspace, supported_tasks=["task1"])

    result = validate_submission_bundle_v3(
        workspace_root=workspace,
        tasks=["task1"],
        strict=True,
        code_manifest_entries=manifest_entries,
        methodology_sources=["code_patch_records"],
    )

    assert result["ok"] is False
    gate = result["data"]["finalize_gate"]
    assert gate["no_task_specific_code_fork_ok"] is False
    assert gate["task_specific_code_fork_violations"] == [
        _submission_task_code_path("task1", "infer.py"),
        _submission_task_code_path("task1", "train.py"),
    ]


def test_v3_package_submission_writes_manifest_for_valid_shared_submission(workspace: Path) -> None:
    _write_valid_task_bundle(workspace, "task1")
    manifest_entries = _write_shared_code(workspace)
    _write_incremental_records(workspace, supported_tasks=["task1"])
    (workspace / "submission" / "methodology.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (workspace / "submission" / "submission.json").write_text("{}", encoding="utf-8")

    result = package_submission_v3(
        submission_dir=workspace / "submission",
        tasks=["task1"],
        test_data_roots=[workspace / "data"],
        strict=True,
        code_manifest_entries=manifest_entries,
        methodology_sources=["code_patch_records"],
    )

    assert result["ok"] is True
    assert (workspace / "submission" / "manifest.json").exists()
    assert (workspace / "submission" / "submission.zip").exists()
