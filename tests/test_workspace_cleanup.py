from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from textwrap import dedent

import h5py
import numpy as np

from cozy_pde_v3.code_evolution import CodePatchRecord, CodeSnapshot
from cozy_pde_v3.memory_store import MemoryStore
from cozy_pde_v3.package import package_submission_v3
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


def _removed_run_dir_name() -> str:
    return "_".join(["autonomous", "dry", "run"])


def _write_valid_task_bundle(workspace: Path, task: str) -> None:
    samples = 2
    total_steps = 200
    input_steps = 10

    test_path = workspace / "data" / f"{task}_test.hdf5"
    tensor = np.arange(samples * total_steps * 256, dtype=np.float32).reshape(samples, total_steps, 256)
    with h5py.File(test_path, "w") as handle:
        handle.create_dataset("tensor", data=tensor)

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
            validation_results={"task_compatibility": {task: True for task in supported_tasks}},
        )
    )


def _write_valid_workspace(workspace: Path) -> list[dict[str, object]]:
    _write_valid_task_bundle(workspace, "task1")
    code_manifest_entries = _write_shared_submission_code(workspace)
    _write_structured_submission_metadata(workspace)
    _write_incremental_records(workspace, supported_tasks=["task1"])
    return code_manifest_entries


def test_v3_validation_does_not_create_removed_run_dirs(workspace: Path) -> None:
    manifest_entries = _write_valid_workspace(workspace)
    existing_log = workspace / "runs" / "task1-session" / "logs" / "shell.log"
    existing_log.parent.mkdir(parents=True, exist_ok=True)
    existing_log.write_text("preserve-me\n", encoding="utf-8")

    result = validate_submission_bundle_v3(
        workspace_root=workspace,
        tasks=["task1"],
        strict=True,
        code_manifest_entries=manifest_entries,
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
    )

    assert result["ok"] is True
    assert existing_log.read_text(encoding="utf-8") == "preserve-me\n"
    assert not (workspace / "runs" / "rehearsal").exists()
    assert not (workspace / "runs" / _removed_run_dir_name()).exists()
    assert not (workspace / "runs" / "archive").exists()


def test_v3_packaging_keeps_existing_run_artifacts_untouched(workspace: Path) -> None:
    manifest_entries = _write_valid_workspace(workspace)
    existing_log = workspace / "runs" / "task1-session" / "logs" / "shell.log"
    existing_log.parent.mkdir(parents=True, exist_ok=True)
    existing_log.write_text("preserve-me\n", encoding="utf-8")

    result = package_submission_v3(
        submission_dir=workspace / "submission",
        tasks=["task1"],
        test_data_roots=[workspace / "data"],
        strict=True,
        code_manifest_entries=manifest_entries,
        methodology_sources=STRUCTURED_METHODOLOGY_SOURCES,
    )

    assert result["ok"] is True
    assert existing_log.read_text(encoding="utf-8") == "preserve-me\n"
    assert not (workspace / "runs" / "rehearsal").exists()
    assert not (workspace / "runs" / _removed_run_dir_name()).exists()
    assert not (workspace / "runs" / "archive").exists()
