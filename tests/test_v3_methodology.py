from __future__ import annotations

import json

from cozy_pde_v3.methodology import (
    STRUCTURED_METHODOLOGY_SOURCES,
    build_methodology_record_bundle,
    write_deterministic_methodology_pdf,
)


def test_build_methodology_record_bundle_uses_only_structured_sources() -> None:
    bundle = build_methodology_record_bundle(
        tasks=["task1", "task3"],
        finalize_gate={"overall_ok": False, "failures": ["missing code provenance linkage"]},
        code_manifest_entries=[
            {
                "path": "submission/code/model.py",
                "sha256": "abc",
                "code_version": "sha256:v2",
                "originating_task": "task3",
                "llm_call_ids": ["call-1"],
                "patch_id": "patch-1",
            }
        ],
        shared_code_union={
            "shared_code_versions": [
                {
                    "version": "sha256:v2",
                    "parent": "sha256:v1",
                    "changed_files": ["submission/code/model.py"],
                    "validated_tasks": ["task1", "task3"],
                    "llm_call_ids": ["call-1"],
                }
            ]
        },
    )

    assert sorted(bundle) == sorted(STRUCTURED_METHODOLOGY_SOURCES)
    assert bundle["artifact_metadata"]["tasks"] == ["task1", "task3"]
    assert bundle["validation_reports"]["finalize_gate"]["overall_ok"] is False
    assert bundle["code_patch_records"][0]["patch_id"] == "patch-1"


def test_write_deterministic_methodology_pdf_creates_mechanical_pdf(workspace) -> None:
    bundle = build_methodology_record_bundle(
        tasks=["task2"],
        finalize_gate={"overall_ok": True, "failures": [], "supported_tasks": ["task2"]},
        code_manifest_entries=[],
        shared_code_union={"shared_code_versions": []},
    )

    result = write_deterministic_methodology_pdf(workspace_root=workspace, tasks=["task2"], record_bundle=bundle)

    pdf_path = workspace / "submission" / "methodology.pdf"
    records_path = workspace / "submission" / "methodology_records.json"
    assert result["ok"] is True
    assert pdf_path.exists()
    assert records_path.exists()
    assert pdf_path.read_bytes().startswith(b"%PDF-1.4")
    assert b"Task 2" in pdf_path.read_bytes()
    assert json.loads(records_path.read_text(encoding="utf-8"))["artifact_metadata"]["tasks"] == ["task2"]
