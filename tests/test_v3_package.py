from __future__ import annotations

import json
import zipfile
from pathlib import Path

import h5py
import numpy as np

from cozy_pde_v3.code_evolution import CodePatchRecord, CodeSnapshot
from cozy_pde_v3.memory_store import MemoryStore
from cozy_pde_v3.package import package_submission_v3


def _write_task_test(path: Path, *, samples: int, total_steps: int) -> None:
    array = np.arange(samples * total_steps * 256, dtype=np.float32).reshape(samples, total_steps, 256)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=array)


def _write_valid_task_bundle(workspace: Path, task: str) -> None:
    samples = 1000 if task == "task3" else 2
    total_steps = 400 if task == "task3" else 200
    input_steps = 20 if task == "task3" else 10
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
                "response": f"{task} complete",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _seed_shared_code_state(workspace: Path) -> None:
    internal_logs = workspace / "internal_logs"
    internal_logs.mkdir(parents=True, exist_ok=True)
    code_dir = workspace / "submission" / "code"
    files = {
        "model.py": "print('model')\n",
        "train.py": "\n".join(
            [
                "import argparse",
                "from pathlib import Path",
                "",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--task', required=True)",
                "parser.add_argument('--config', required=True)",
                "parser.add_argument('--data_dir', required=True)",
                "parser.add_argument('--output_dir', required=True)",
                "args = parser.parse_args()",
                "output_dir = Path(args.output_dir)",
                "output_dir.mkdir(parents=True, exist_ok=True)",
                "(output_dir / 'checkpoint.txt').write_text(args.task, encoding='utf-8')",
            ]
        )
        + "\n",
        "infer.py": "\n".join(
            [
                "import argparse",
                "from pathlib import Path",
                "",
                "import h5py",
                "import numpy as np",
                "",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--task', required=True)",
                "parser.add_argument('--config', required=True)",
                "parser.add_argument('--data_dir', required=True)",
                "parser.add_argument('--output_dir', required=True)",
                "parser.add_argument('--output', required=True)",
                "args = parser.parse_args()",
                "shape = (1000, 400, 256) if args.task == 'task3' else (2, 200, 256)",
                "output_path = Path(args.output)",
                "output_path.parent.mkdir(parents=True, exist_ok=True)",
                "with h5py.File(output_path, 'w') as handle:",
                "    handle.create_dataset('pred', data=np.zeros(shape, dtype=np.float32))",
            ]
        )
        + "\n",
    }
    for name, content in files.items():
        (code_dir / name).write_text(content, encoding="utf-8")

    store = MemoryStore(internal_logs / "memory.db")
    store.initialize()
    store.record_code_snapshot(
        CodeSnapshot(
            code_version="sha256:v1",
            parent_version="sha256:root",
            content_hash="sha256:content-v1",
            api_contract_hash="sha256:api-stable",
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
            api_contract_hash="sha256:api-stable",
            supported_tasks=["task1"],
            task_support_matrix={"task1": {"status": "pass"}},
            created_by_run_id="run-002",
            created_at="2026-05-22T00:05:00Z",
        )
    )
    store.record_patch(
        CodePatchRecord(
            patch_id="patch-002",
            base_code_version="sha256:v1",
            new_code_version="sha256:v2",
            task_context="task1",
            changed_files=["submission/code/infer.py", "submission/code/model.py", "submission/code/train.py"],
            change_intent="Keep shared code aligned",
            backward_compatibility_claim="task1 preserved",
            affected_interfaces=["infer()", "model()"],
            llm_call_ids=["call-1"],
            validation_results={"validated_tasks": ["task1"], "task_compatibility": {"task1": True}},
        )
    )


def test_package_submission_v3_never_bypasses_validation_before_zip(workspace: Path, monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_validate_submission_bundle_v3(**kwargs):
        calls.append(kwargs)
        return {"ok": False, "error": "validation blocked", "data": {"finalize_gate": {"overall_ok": False}}}

    monkeypatch.setattr("cozy_pde_v3.package.validate_submission_bundle_v3", fake_validate_submission_bundle_v3)

    result = package_submission_v3(
        submission_dir=workspace / "submission",
        tasks=["task1"],
        test_data_roots=[workspace / "data"],
        strict=True,
    )

    assert result["ok"] is False
    assert calls
    assert not (workspace / "submission" / "submission.zip").exists()


def test_package_submission_v3_generates_manifest_code_manifest_and_methodology(workspace: Path) -> None:
    _write_valid_task_bundle(workspace, "task1")
    _seed_shared_code_state(workspace)

    result = package_submission_v3(
        submission_dir=workspace / "submission",
        tasks=["task1"],
        test_data_roots=[workspace / "data"],
        strict=True,
    )

    assert result["ok"] is True
    manifest = json.loads((workspace / "submission" / "manifest.json").read_text(encoding="utf-8"))
    code_manifest = json.loads((workspace / "submission" / "code_manifest.json").read_text(encoding="utf-8"))
    submission_json = json.loads((workspace / "submission" / "submission.json").read_text(encoding="utf-8"))
    shared_code_union = json.loads((workspace / "submission" / "shared_code_union.json").read_text(encoding="utf-8"))
    assert submission_json["tasks"] == ["task1"]
    assert submission_json["shared_code_union"]["shared_code_versions"][-1]["version"] == "sha256:v2"
    assert shared_code_union["shared_code_versions"][-1]["version"] == "sha256:v2"
    assert [entry["archive_path"] for entry in manifest if entry["path"].startswith("code/")] == [
        "shared_code/infer.py",
        "shared_code/model.py",
        "shared_code/train.py",
    ]
    assert [entry["path"] for entry in code_manifest] == [
        "submission/code/infer.py",
        "submission/code/model.py",
        "submission/code/train.py",
    ]
    assert all("patch_id" in entry for entry in code_manifest)
    assert (workspace / "submission" / "submission.json").exists()
    assert (workspace / "submission" / "methodology.pdf").exists()
    assert (workspace / "submission" / "submission.zip").exists()

    with zipfile.ZipFile(workspace / "submission" / "submission.zip", "r") as archive:
        names = sorted(archive.namelist())

    assert "shared_code/infer.py" in names
    assert "shared_code/model.py" in names
    assert "shared_code/train.py" in names
    assert "task1_pred.hdf5" in names
    assert "task1_time.csv" in names
    assert "task1_logs.log" in names
    assert "submission.json" in names
    assert "methodology.pdf" in names


def test_package_submission_v3_rejects_invalid_existing_code_manifest(workspace: Path) -> None:
    _write_valid_task_bundle(workspace, "task1")
    _seed_shared_code_state(workspace)
    (workspace / "submission" / "methodology.pdf").write_bytes(b"%PDF-1.4\n%%EOF\n")
    (workspace / "submission" / "submission.json").write_text("{}", encoding="utf-8")
    (workspace / "submission" / "code_manifest.json").write_text(
        json.dumps([{"path": "submission/code/model.py", "sha256": "abc"}]),
        encoding="utf-8",
    )

    result = package_submission_v3(
        submission_dir=workspace / "submission",
        tasks=["task1"],
        test_data_roots=[workspace / "data"],
        strict=True,
    )

    assert result["ok"] is False
    assert "code_manifest.json" in result["error"]
