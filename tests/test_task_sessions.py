from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent

from cozy_pde_v3.agent_loop import run_formal_task_session
from cozy_pde_v3.code_evolution import CodePatchRecord, CodeSnapshot
from cozy_pde_v3.log_export import export_task_logs
from cozy_pde_v3.memory_store import MemoryStore
from cozy_pde_v3.task_specs import DEFAULT_TASK_SPECS
from cozy_pde_v3.validation.submission import validate_submission_bundle_v3


def _write_task_log(path: Path, task: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-22T00:00:00+00:00",
                "elapsed_seconds": 0.1,
                "response": f"{task} ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_valid_task_bundle(workspace: Path, task: str) -> None:
    import h5py
    import numpy as np

    data_path = workspace / "data" / f"{task}_test.hdf5"
    tensor = np.linspace(0.0, 1.0, num=2 * 200 * 256, dtype=np.float32).reshape(2, 200, 256)
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("tensor", data=tensor)
    with h5py.File(workspace / "submission" / f"{task}_pred.hdf5", "w") as handle:
        pred = tensor.copy()
        pred[:, 10:, :] = pred[:, 10:, :] + 0.05
        handle.create_dataset("pred", data=pred)
    (workspace / "submission" / f"{task}_time.csv").write_text(
        "train_time,inference_time\n1.0,0.2\n",
        encoding="utf-8",
    )
    _write_task_log(workspace / "submission" / f"{task}_logs.log", task)


def _write_shared_submission_code(workspace: Path) -> list[dict[str, object]]:
    code_dir = workspace / "submission" / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
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

                output_path = Path(args.output) if args.output else artifact_dir / f"{args.task}_pred.hdf5"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(Path(args.data_dir) / f"{args.task}_test.hdf5", "r") as source:
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
    manifest_entries: list[dict[str, object]] = []
    for index, (name, content) in enumerate(files.items(), start=1):
        path = code_dir / name
        path.write_text(content, encoding="utf-8")
        payload = content.encode("utf-8")
        manifest_entries.append(
            {
                "path": f"submission/code/{name}",
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
            task_context="task1",
            changed_files=[
                "submission/code/train.py",
                "submission/code/infer.py",
                "submission/code/validate.py",
            ],
            change_intent="Finalize shared session CLI",
            backward_compatibility_claim="Shared CLI remains compatible",
            affected_interfaces=["train.py", "infer.py"],
            llm_call_ids=["call-1"],
            validation_results={"task_compatibility": {task: True for task in supported_tasks}},
        )
    )


class FakeResponsesClient:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        if not self.responses:
            raise AssertionError("no queued fake response")
        return dict(self.responses.pop(0))


def _provider_report_payload() -> dict[str, object]:
    return {
        "formal_ready": True,
        "primary": {
            "provider": "primary",
            "model_id": "gpt-5.4",
            "formal_ready": True,
        },
        "forced_failover": {"required": False},
    }


def _turn(response_id: str, output_items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "provider": "primary",
        "model": "gpt-5.4",
        "raw_response": {"id": response_id, "model": "gpt-5.4", "output": output_items},
        "standard_output_items": output_items,
        "provider_output_items": output_items,
        "usage": {"total_tokens": 10},
    }


def _config_stub(workspace: Path) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        workspace_root=workspace,
        task_specs=DEFAULT_TASK_SPECS,
    )


def test_single_task_log_export_writes_only_requested_task_log(workspace: Path) -> None:
    _write_task_log(workspace / "llm_logs" / "task1_all_llm_calls.jsonl", "task1")
    _write_task_log(workspace / "llm_logs" / "task2_all_llm_calls.jsonl", "task2")

    result = export_task_logs(workspace, ["task2"])

    assert result["ok"] is True
    assert not (workspace / "submission" / "task1_logs.log").exists()
    assert (workspace / "submission" / "task2_logs.log").read_text(encoding="utf-8") == (
        workspace / "llm_logs" / "task2_all_llm_calls.jsonl"
    ).read_text(encoding="utf-8")


def test_shared_multi_task_log_export_requires_explicit_override(workspace: Path) -> None:
    _write_task_log(workspace / "llm_logs" / "task1_task2_all_llm_calls.jsonl", "shared")

    result = export_task_logs(workspace, ["task1", "task2"], allow_multi_task_session=False)

    assert result["ok"] is False
    assert "independent task sessions" in result["error"]


def test_shared_multi_task_log_export_can_copy_shared_session_log(workspace: Path) -> None:
    _write_task_log(workspace / "llm_logs" / "task1_task2_all_llm_calls.jsonl", "shared")

    result = export_task_logs(workspace, ["task1", "task2"], allow_multi_task_session=True)

    assert result["ok"] is True
    exported = (workspace / "submission" / "task1_logs.log").read_text(encoding="utf-8")
    assert exported == (workspace / "submission" / "task2_logs.log").read_text(encoding="utf-8")


def test_single_task_validation_uses_shared_submission_code_layout(workspace: Path) -> None:
    _write_valid_task_bundle(workspace, "task1")
    code_manifest_entries = _write_shared_submission_code(workspace)
    _write_structured_submission_metadata(workspace)
    _write_incremental_records(workspace, supported_tasks=["task1"])

    result = validate_submission_bundle_v3(
        workspace_root=workspace,
        tasks=["task1"],
        strict=True,
        code_manifest_entries=code_manifest_entries,
        methodology_sources=["code_patch_records"],
    )

    assert result["ok"] is True
    finalize_gate = result["data"]["finalize_gate"]
    assert finalize_gate["shared_code_ok"] is True
    assert finalize_gate["no_task_specific_code_fork_ok"] is True
    assert finalize_gate["supported_tasks"] == ["task1"]


def test_single_task_formal_session_writes_shared_submission_code_and_single_log(workspace: Path) -> None:
    provider_report_path = workspace / "provider_report.json"
    provider_report_path.write_text(json.dumps(_provider_report_payload()), encoding="utf-8")
    client = FakeResponsesClient(
        [
            _turn(
                "resp_write",
                [
                    {
                        "type": "function_call",
                        "name": "write_file",
                        "call_id": "call_1",
                        "arguments": json.dumps(
                            {
                                "path": "submission/code/train.py",
                                "content": "print('train shared')\n",
                            }
                        ),
                    }
                ],
            ),
            _turn(
                "resp_done",
                [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "FORMAL_DONE"}],
                    }
                ],
            ),
        ]
    )

    result = run_formal_task_session(
        config=_config_stub(workspace),
        task="task1",
        provider_report_path=provider_report_path,
        responses_client=client,
    )

    assert result["ok"] is True
    assert result["state"]["task"] == "task1"
    assert (workspace / "submission" / "code" / "train.py").exists()
    assert not (workspace / "submission" / "code" / "task1").exists()
    assert (workspace / "llm_logs" / "all_llm_calls.jsonl").exists()
