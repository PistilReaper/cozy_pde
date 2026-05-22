from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from textwrap import dedent

from dataclasses import asdict

import pytest

from cozy_pde_v3.code_evolution import CodePatchRecord, CodeSnapshot
from cozy_pde_v3.memory_store import MemoryStore
from cozy_pde_v3.research.cache import ResearchCache, canonicalize_url
from cozy_pde_v3.research.providers import ResearchProviderFlags
from cozy_pde_v3.research.tools import ResearchToolFlags
from cozy_pde_v3.validation.provenance import build_shared_code_union
from cozy_pde_v3.validation.submission import build_finalize_gate_status, validate_submission_bundle_v3
from cozy_pde_v3.package import package_submission_v3

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


def _task_fork_path(*parts: str) -> str:
    return "/".join(("code", *parts))


def test_finalize_gate_contains_required_keys() -> None:
    status = build_finalize_gate_status(
        prediction_ok=True,
        time_csv_ok=True,
        logs_ok=True,
        provenance_log_ok=True,
        provenance_ok=True,
        inference_time_ok=True,
        package_ok=True,
        code_manifest_ok=True,
        methodology_records_only_ok=True,
        methodology_ok=True,
        secret_scan_ok=True,
        task_rule_ok=True,
        final_code_paths=["submission/code/model.py"],
        provenance_links={"submission/code/model.py": ["call-1"]},
        code_manifest_entries=[
            {
                "path": "submission/code/model.py",
                "llm_call_ids": ["call-1"],
            }
        ],
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
        cli_smoke_status={"task1": True, "task2": True, "task3": False},
        cli_smoke_details={
            "task1": {"cli_parse_ok": True, "train_smoke_ok": True, "infer_smoke_ok": True, "checkpoint_load_ok": True},
            "task2": {"cli_parse_ok": True, "train_smoke_ok": True, "infer_smoke_ok": True, "checkpoint_load_ok": True},
        },
        api_contract_ok=True,
        incremental_patch_ok=True,
        shared_code_ok=True,
        supported_tasks=["task1", "task2"],
    )

    assert status["time_csv_ok"] is True
    assert status["provenance_ok"] is True
    assert status["inference_time_ok"] is True
    assert status["code_manifest_ok"] is True
    assert status["methodology_ok"] is True
    assert status["secret_scan_ok"] is True
    assert status["task_rule_ok"] is True
    assert status["shared_code_ok"] is True
    assert status["code_provenance_ok"] is True
    assert status["api_contract_ok"] is True
    assert status["task1_compat_ok"] is True
    assert status["task2_compat_ok"] is True
    assert status["task3_compat_ok"] is False
    assert status["task1_cli_parse_ok"] is True
    assert status["task1_train_smoke_ok"] is True
    assert status["task1_infer_smoke_ok"] is True
    assert status["task1_checkpoint_load_ok"] is True
    assert status["incremental_patch_ok"] is True
    assert status["no_task_specific_code_fork_ok"] is True
    assert status["failures"] == []
    assert status["warnings"] == []
    assert status["overall_ok"] is True


def test_finalize_gate_rejects_task_specific_code_fork_paths() -> None:
    fork_path = _task_fork_path("task" + "1", "train.py")
    status = build_finalize_gate_status(
        prediction_ok=True,
        time_csv_ok=True,
        logs_ok=True,
        provenance_log_ok=True,
        provenance_ok=True,
        inference_time_ok=True,
        package_ok=True,
        code_manifest_ok=True,
        methodology_records_only_ok=True,
        methodology_ok=True,
        secret_scan_ok=True,
        task_rule_ok=True,
        final_code_paths=[fork_path],
        provenance_links={fork_path: ["call-1"]},
        code_manifest_entries=[
            {
                "path": fork_path,
                "llm_call_ids": ["call-1"],
            }
        ],
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
        cli_smoke_status={"task1": True, "task2": True, "task3": True},
        cli_smoke_details={
            "task1": {"cli_parse_ok": True, "train_smoke_ok": True, "infer_smoke_ok": True, "checkpoint_load_ok": True},
            "task2": {"cli_parse_ok": True, "train_smoke_ok": True, "infer_smoke_ok": True, "checkpoint_load_ok": True},
            "task3": {"cli_parse_ok": True, "train_smoke_ok": True, "infer_smoke_ok": True, "checkpoint_load_ok": True},
        },
        api_contract_ok=True,
        incremental_patch_ok=True,
        shared_code_ok=True,
    )

    assert status["no_task_specific_code_fork_ok"] is False
    assert status["task_specific_code_fork_violations"] == [fork_path]
    assert f"task-specific code fork detected: {fork_path}" in status["failures"]


def test_finalize_gate_rejects_missing_provenance_linkage() -> None:
    status = build_finalize_gate_status(
        prediction_ok=True,
        time_csv_ok=True,
        logs_ok=True,
        provenance_log_ok=True,
        provenance_ok=True,
        inference_time_ok=True,
        package_ok=True,
        code_manifest_ok=True,
        methodology_records_only_ok=True,
        methodology_ok=True,
        secret_scan_ok=True,
        task_rule_ok=True,
        final_code_paths=["submission/code/model.py", "submission/code/train.py"],
        provenance_links={"submission/code/model.py": ["call-1"]},
        code_manifest_entries=[
            {
                "path": "submission/code/model.py",
                "llm_call_ids": ["call-1"],
            },
            {
                "path": "submission/code/train.py",
                "llm_call_ids": [],
            },
        ],
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
        cli_smoke_status={"task1": True, "task2": True, "task3": True},
        cli_smoke_details={
            "task1": {"cli_parse_ok": True, "train_smoke_ok": True, "infer_smoke_ok": True, "checkpoint_load_ok": True},
            "task2": {"cli_parse_ok": True, "train_smoke_ok": True, "infer_smoke_ok": True, "checkpoint_load_ok": True},
            "task3": {"cli_parse_ok": True, "train_smoke_ok": True, "infer_smoke_ok": True, "checkpoint_load_ok": True},
        },
        api_contract_ok=True,
        incremental_patch_ok=True,
        shared_code_ok=True,
    )

    assert status["code_provenance_ok"] is False
    assert status["missing_code_provenance_paths"] == ["submission/code/train.py"]
    assert "missing code provenance linkage: submission/code/train.py" in status["failures"]


def test_finalize_gate_propagates_cli_smoke_inputs() -> None:
    status = build_finalize_gate_status(
        prediction_ok=True,
        time_csv_ok=True,
        logs_ok=True,
        provenance_log_ok=True,
        provenance_ok=True,
        inference_time_ok=True,
        package_ok=True,
        code_manifest_ok=True,
        methodology_records_only_ok=True,
        methodology_ok=True,
        secret_scan_ok=True,
        task_rule_ok=True,
        final_code_paths=["submission/code/model.py"],
        provenance_links={"submission/code/model.py": ["call-1"]},
        code_manifest_entries=[
            {
                "path": "submission/code/model.py",
                "llm_call_ids": ["call-1"],
            }
        ],
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
        cli_smoke_status={"task1": True, "task2": False, "task3": True},
        cli_smoke_details={
            "task1": {"cli_parse_ok": True, "train_smoke_ok": True, "infer_smoke_ok": True, "checkpoint_load_ok": True},
            "task2": {"cli_parse_ok": False, "train_smoke_ok": True, "infer_smoke_ok": True, "checkpoint_load_ok": True},
            "task3": {"cli_parse_ok": True, "train_smoke_ok": True, "infer_smoke_ok": True, "checkpoint_load_ok": True},
        },
        api_contract_ok=True,
        incremental_patch_ok=True,
        shared_code_ok=True,
    )

    assert status["task1_compat_ok"] is True
    assert status["task2_compat_ok"] is False
    assert status["task3_compat_ok"] is True


def test_finalize_gate_only_requires_supported_task_compatibility() -> None:
    status = build_finalize_gate_status(
        prediction_ok=True,
        time_csv_ok=True,
        logs_ok=True,
        provenance_log_ok=True,
        provenance_ok=True,
        inference_time_ok=True,
        package_ok=True,
        code_manifest_ok=True,
        methodology_records_only_ok=True,
        methodology_ok=True,
        secret_scan_ok=True,
        task_rule_ok=True,
        final_code_paths=["submission/code/model.py"],
        provenance_links={"submission/code/model.py": ["call-1"]},
        code_manifest_entries=[
            {
                "path": "submission/code/model.py",
                "llm_call_ids": ["call-1"],
            }
        ],
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
        cli_smoke_status={"task1": True, "task2": True, "task3": False},
        cli_smoke_details={
            "task1": {"cli_parse_ok": True, "train_smoke_ok": True, "infer_smoke_ok": True, "checkpoint_load_ok": True},
            "task2": {"cli_parse_ok": True, "train_smoke_ok": True, "infer_smoke_ok": True, "checkpoint_load_ok": True},
        },
        api_contract_ok=True,
        incremental_patch_ok=True,
        shared_code_ok=True,
        supported_tasks=["task1", "task2"],
    )

    assert status["task3_compat_ok"] is False
    assert status["overall_ok"] is True


def test_finalize_gate_derives_manifest_and_methodology_checks_from_structured_inputs() -> None:
    status = build_finalize_gate_status(
        prediction_ok=True,
        time_csv_ok=True,
        logs_ok=True,
        provenance_log_ok=True,
        provenance_ok=True,
        inference_time_ok=True,
        package_ok=True,
        code_manifest_ok=True,
        methodology_records_only_ok=True,
        methodology_ok=True,
        secret_scan_ok=True,
        task_rule_ok=True,
        final_code_paths=["submission/code/model.py"],
        provenance_links={},
        code_manifest_entries=[
            {
                "path": "submission/code/model.py",
                "sha256": "abc",
                "size": 12,
                "code_version": "v2",
                "originating_task": "task1",
                "patch_id": "patch-001",
                "step_id": "patch-001",
                "task_id": "task1",
                "timestamp": "2026-05-22T00:00:00Z",
                "llm_call_ids": ["call-1"],
            }
        ],
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
        cli_smoke_status={"task1": True},
        cli_smoke_details={
            "task1": {"cli_parse_ok": True, "train_smoke_ok": True, "infer_smoke_ok": True, "checkpoint_load_ok": True},
        },
        api_contract_ok=True,
        incremental_patch_ok=True,
        shared_code_ok=True,
        supported_tasks=["task1"],
    )

    assert status["code_manifest_ok"] is True
    assert status["methodology_records_only_ok"] is True
    assert status["methodology_ok"] is True
    assert status["code_provenance_ok"] is True
    assert status["missing_code_manifest_paths"] == []
    assert status["invalid_methodology_sources"] == []
    assert status["overall_ok"] is True


def test_build_shared_code_union_preserves_version_lineage_fields() -> None:
    shared = build_shared_code_union(
        [
            {
                "version": "v2",
                "created_during": "task2",
                "parent": "v1",
                "changed_files": ["submission/code/model.py"],
                "validated_tasks": ["task1"],
                "llm_call_ids": ["call-1"],
            },
            {
                "version": "v2",
                "created_during": "task2",
                "parent": "v1",
                "changed_files": ["submission/code/task_specs.py", "submission/code/model.py"],
                "validated_tasks": ["task2", "task3"],
                "llm_call_ids": ["call-2"],
            },
        ]
    )

    assert shared == {
        "shared_code_versions": [
            {
                "version": "v2",
                "created_during": "task2",
                "parent": "v1",
                "changed_files": ["submission/code/model.py", "submission/code/task_specs.py"],
                "validated_tasks": ["task1", "task2", "task3"],
                "llm_call_ids": ["call-1", "call-2"],
            }
        ]
    }


def test_research_wrappers_import_cleanly() -> None:
    assert callable(canonicalize_url)
    assert ResearchCache is not None
    assert asdict(ResearchProviderFlags()) == {
        "arxiv_enabled": True,
        "github_enabled": True,
        "allow_unauthenticated_github": True,
    }
    assert ResearchToolFlags().__dict__ == {
        "fetch_pdf": True,
        "fetch_url": True,
        "parse_html": True,
        "parse_pdf": True,
        "search_arxiv": True,
        "search_github": True,
    }


def _write_task_test_stub(path: Path, *, samples: int, total_steps: int, spatial_points: int = 256) -> None:
    import h5py
    import numpy as np

    array = np.arange(samples * total_steps * spatial_points, dtype=np.float32).reshape(samples, total_steps, spatial_points)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=array)


def _write_valid_task_bundle(workspace: Path, task: str) -> None:
    import h5py
    import numpy as np

    if task == "task3":
        samples = 1000
        total_steps = 400
        input_steps = 20
    else:
        samples = 2
        total_steps = 200
        input_steps = 10

    test_path = workspace / "data" / f"{task}_test.hdf5"
    _write_task_test_stub(test_path, samples=samples, total_steps=total_steps)

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
                "response": f"{task} complete",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_shared_submission_code(workspace: Path) -> list[dict[str, object]]:
    code_dir = workspace / "submission" / "code"
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
                (output_dir / "checkpoint.pt").write_text(
                    json.dumps({"task": args.task, "config": args.config}),
                    encoding="utf-8",
                )
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
                if not datasets:
                    raise ValueError("missing dataset")
                return datasets[0]


            def main() -> int:
                parser = argparse.ArgumentParser()
                parser.add_argument("--task", required=True)
                parser.add_argument("--config", required=True)
                parser.add_argument("--data_dir", required=True)
                parser.add_argument("--output_dir")
                parser.add_argument("--output")
                args = parser.parse_args()

                artifact_dir = Path(args.output_dir or Path(args.config).parent)
                checkpoint_path = artifact_dir / "checkpoint.pt"
                if not checkpoint_path.exists():
                    raise FileNotFoundError("checkpoint.pt missing")
                json.loads(checkpoint_path.read_text(encoding="utf-8"))

                data_dir = Path(args.data_dir)
                output_path = Path(args.output) if args.output else artifact_dir / f"{args.task}_pred.hdf5"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(data_dir / f"{args.task}_test.hdf5", "r") as source:
                    tensor = np.asarray(_first_dataset(source)[...])
                pred = tensor.copy()
                pred[:, _INPUT_STEPS[args.task] :, :] = pred[:, _INPUT_STEPS[args.task] :, :] + np.float32(0.01)
                with h5py.File(output_path, "w") as target:
                    target.create_dataset("pred", data=pred)
                return 0


            if __name__ == "__main__":
                raise SystemExit(main())
            """
        ).strip()
        + "\n",
        "validate.py": "print('validate')\n",
    }
    entries: list[dict[str, object]] = []
    for index, (name, content) in enumerate(files.items(), start=1):
        path = code_dir / name
        path.write_text(content, encoding="utf-8")
        entries.append(
            {
                "path": f"submission/code/{name}",
                "sha256": sha256(content.encode("utf-8")).hexdigest(),
                "size": len(content.encode("utf-8")),
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


def _write_structured_submission_metadata(workspace: Path) -> None:
    (workspace / "submission" / "methodology.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (workspace / "submission" / "submission.json").write_text("{}", encoding="utf-8")


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
            task_context="shared submission finalize",
            changed_files=[
                "submission/code/train.py",
                "submission/code/infer.py",
                "submission/code/validate.py",
            ],
            change_intent="Finalize shared submission CLI",
            backward_compatibility_claim="Shared task CLI remains compatible",
            affected_interfaces=["train.py", "infer.py"],
            llm_call_ids=["call-1"],
            validation_results={
                "task_compatibility": {task: True for task in supported_tasks},
            },
        )
    )


def test_validate_submission_bundle_v3_accepts_shared_code_layout_and_task_bundle(workspace: Path) -> None:
    _write_valid_task_bundle(workspace, "task1")
    code_manifest_entries = _write_shared_submission_code(workspace)
    _write_structured_submission_metadata(workspace)
    _write_incremental_records(workspace, supported_tasks=["task1"])

    result = validate_submission_bundle_v3(
        workspace_root=workspace,
        tasks=["task1"],
        strict=True,
        code_manifest_entries=code_manifest_entries,
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
    )

    assert result["ok"] is True
    assert result["data"]["finalize_gate"]["overall_ok"] is True
    assert result["data"]["finalize_gate"]["no_task_specific_code_fork_ok"] is True
    assert result["data"]["finalize_gate"]["supported_tasks"] == ["task1"]


def test_validate_submission_bundle_v3_rejects_task_specific_code_forks(workspace: Path) -> None:
    _write_valid_task_bundle(workspace, "task1")
    task_segment = "task" + "1"
    task_dir = workspace / "submission" / "code" / task_segment
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / "train.py"
    content = "print('task1 train')\n"
    manifest_path = "/".join(("submission", "code", task_segment, "train.py"))
    path.write_text(content, encoding="utf-8")
    (workspace / "submission" / "methodology.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (workspace / "submission" / "submission.json").write_text("{}", encoding="utf-8")

    result = validate_submission_bundle_v3(
        workspace_root=workspace,
        tasks=["task1"],
        strict=True,
        code_manifest_entries=[
            {
                "path": manifest_path,
                "sha256": sha256(content.encode("utf-8")).hexdigest(),
                "size": len(content.encode("utf-8")),
                "code_version": "v1",
                "originating_task": "task1",
                "patch_id": "patch-001",
                "step_id": "step-001",
                "task_id": "task1",
                "timestamp": "2026-05-22T00:00:00+00:00",
                "llm_call_ids": ["call-1"],
            }
        ],
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
    )

    assert result["ok"] is False
    assert "task-specific code fork detected" in result["error"]


def test_package_submission_v3_refuses_packaging_when_finalize_gate_fails(workspace: Path) -> None:
    package_result = package_submission_v3(
        submission_dir=workspace / "submission",
        tasks=["task1"],
        test_data_roots=[workspace / "data"],
        strict=True,
    )

    assert package_result["ok"] is False
    assert not (workspace / "submission" / "submission.zip").exists()


def test_package_submission_v3_writes_manifest_and_zip_for_valid_shared_submission(workspace: Path) -> None:
    _write_valid_task_bundle(workspace, "task1")
    code_manifest_entries = _write_shared_submission_code(workspace)
    _write_structured_submission_metadata(workspace)
    _write_incremental_records(workspace, supported_tasks=["task1"])

    result = package_submission_v3(
        submission_dir=workspace / "submission",
        tasks=["task1"],
        test_data_roots=[workspace / "data"],
        strict=True,
        code_manifest_entries=code_manifest_entries,
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
    )

    assert result["ok"] is True
    assert (workspace / "submission" / "manifest.json").exists()
    assert (workspace / "submission" / "submission.zip").exists()
