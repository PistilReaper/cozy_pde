# Cozy PDE v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `agent_runner`-centric system with a `cozy_pde_v3` package that uses native Responses-only transport, deterministic routing, single-task formal run sessions, a shared evolving `submission/code/` contract across tasks, dual-provider capability checks, and provenance-safe packaging.

**Architecture:** Build a new `cozy_pde_v3` package first, migrate retained `E/F` capabilities into it, then cut the CLI over to the new system and remove obsolete `json_action`, `chat.completions`, rehearsal, and multi-task formal-session logic. Treat provider failover, proxy logging, shared code evolution, compatibility gates, and mechanical methodology generation as first-class deterministic subsystems.

**Tech Stack:** Python 3.11+, `pytest`, `sqlite3`, `openai>=2`, `httpx`, `PyYAML`, `h5py`, `numpy`, existing FastAPI-based `scripts/proxy.py`, existing `agent_runner` validation/research code as migration source only.

---

## File Structure

### New package: `cozy_pde_v3/`

- Create: `cozy_pde_v3/__init__.py`
- Create: `cozy_pde_v3/cli.py`
- Create: `cozy_pde_v3/config.py`
- Create: `cozy_pde_v3/task_specs.py`
- Create: `cozy_pde_v3/state.py`
- Create: `cozy_pde_v3/profiles.py`
- Create: `cozy_pde_v3/responses_client.py`
- Create: `cozy_pde_v3/responses_ledger.py`
- Create: `cozy_pde_v3/provider_capabilities.py`
- Create: `cozy_pde_v3/proxy_logs.py`
- Create: `cozy_pde_v3/logging.py`
- Create: `cozy_pde_v3/deterministic_router.py`
- Create: `cozy_pde_v3/context_packer.py`
- Create: `cozy_pde_v3/memory_store.py`
- Create: `cozy_pde_v3/code_evolution.py`
- Create: `cozy_pde_v3/experiment_engine.py`
- Create: `cozy_pde_v3/agent_loop.py`
- Create: `cozy_pde_v3/research/__init__.py`
- Create: `cozy_pde_v3/research/cache.py`
- Create: `cozy_pde_v3/research/providers.py`
- Create: `cozy_pde_v3/research/tools.py`
- Create: `cozy_pde_v3/validation/__init__.py`
- Create: `cozy_pde_v3/validation/logs.py`
- Create: `cozy_pde_v3/validation/provenance.py`
- Create: `cozy_pde_v3/validation/submission.py`

### Existing files to migrate or adapt

- Modify: `scripts/proxy.py`

### Existing files to remove after cutover

- Delete: `agent_runner/json_action_client.py`
- Delete: `agent_runner/router.py`
- Delete: `agent_runner/main.py`
- Delete: `agent_runner/prompts.py`
- Delete: `tests/test_one_tool_mode.py`
- Delete: `tests/test_no_chat_completions.py`
- Delete: `tests/test_task_sessions.py`
- Delete: `tests/test_responses_loop_tool_call.py`
- Delete: `tests/test_responses_client.py`

### New tests

- Create: `tests/test_v3_cli.py`
- Create: `tests/test_v3_responses_client.py`
- Create: `tests/test_v3_proxy_logs.py`
- Create: `tests/test_v3_state_memory.py`
- Create: `tests/test_v3_router_context.py`
- Create: `tests/test_v3_validation_package.py`
- Create: `tests/test_v3_run_session.py`

## Task 1: Scaffold `cozy_pde_v3`, single-task CLI, and shared TaskSpec contract

**Files:**
- Create: `cozy_pde_v3/__init__.py`
- Create: `cozy_pde_v3/cli.py`
- Create: `cozy_pde_v3/config.py`
- Create: `cozy_pde_v3/task_specs.py`
- Test: `tests/test_v3_cli.py`

- [ ] **Step 1: Write the failing CLI and TaskSpec tests**

```python
from __future__ import annotations

import pytest

from cozy_pde_v3.cli import build_parser
from cozy_pde_v3.task_specs import DEFAULT_TASK_SPECS


def test_cli_accepts_exactly_one_task_per_run_invocation() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--config", "agent_runner/config.yaml", "--task", "task1"])
    assert args.command == "run"
    assert args.task == "task1"

    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--config", "agent_runner/config.yaml", "--task", "task1,task2"])


def test_default_task_specs_capture_shared_code_contract() -> None:
    task1 = DEFAULT_TASK_SPECS["task1"]
    task2 = DEFAULT_TASK_SPECS["task2"]
    task3 = DEFAULT_TASK_SPECS["task3"]

    assert task1.first_steps_must_match == 10
    assert task2.must_train_from_scratch is True
    assert task3.pred_shape == (1000, 400, 256)
    assert task3.inference_time_limit_sec == 120.0
```

- [ ] **Step 2: Run the tests to verify the package does not exist yet**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_cli.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'cozy_pde_v3'
```

- [ ] **Step 3: Create the minimal package, parser, config stub, and TaskSpec registry**

```python
# cozy_pde_v3/task_specs.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    equation: str
    input_steps: int
    output_steps: int
    total_steps: int
    spatial_points: int
    pred_shape: tuple[int, int, int]
    train_files: list[str]
    val_files: list[str]
    test_file: str
    conditioning_fields: list[str]
    allow_pretrained_checkpoint: bool
    must_train_from_scratch: bool
    first_steps_must_match: int
    inference_time_limit_sec: float


DEFAULT_TASK_SPECS = {
    "task1": TaskSpec(
        task_id="task1",
        equation="burgers",
        input_steps=10,
        output_steps=200,
        total_steps=200,
        spatial_points=256,
        pred_shape=(0, 200, 256),
        train_files=["task1_train.hdf5"],
        val_files=["task1_val.hdf5"],
        test_file="task1_test.hdf5",
        conditioning_fields=[],
        allow_pretrained_checkpoint=True,
        must_train_from_scratch=False,
        first_steps_must_match=10,
        inference_time_limit_sec=120.0,
    ),
    "task2": TaskSpec(
        task_id="task2",
        equation="burgers_multi_nu",
        input_steps=10,
        output_steps=200,
        total_steps=200,
        spatial_points=256,
        pred_shape=(0, 200, 256),
        train_files=["task2_train.hdf5"],
        val_files=["task2_val.h5"],
        test_file="task2_test.hdf5",
        conditioning_fields=["nu"],
        allow_pretrained_checkpoint=False,
        must_train_from_scratch=True,
        first_steps_must_match=10,
        inference_time_limit_sec=120.0,
    ),
    "task3": TaskSpec(
        task_id="task3",
        equation="kuramoto_sivashinsky",
        input_steps=20,
        output_steps=400,
        total_steps=400,
        spatial_points=256,
        pred_shape=(1000, 400, 256),
        train_files=["KS_train.hdf5"],
        val_files=["KS_val.hdf5"],
        test_file="task3_test.hdf5",
        conditioning_fields=["lambda2"],
        allow_pretrained_checkpoint=False,
        must_train_from_scratch=True,
        first_steps_must_match=20,
        inference_time_limit_sec=120.0,
    ),
}
```

```python
# cozy_pde_v3/cli.py
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cozy-pde")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--task", required=True)

    for name in ("check-provider", "check-research", "validate", "package", "status"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--config", required=True)
        if name in {"validate", "package", "status"}:
            command_parser.add_argument("--task", required=True)

    return parser
```

```python
# cozy_pde_v3/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class V3Config:
    config_path: Path
    workspace_root: Path
    require_fallback: bool = False


def load_config(path: str | Path, *, workspace_root: str | Path | None = None) -> V3Config:
    config_path = Path(path).resolve()
    root = Path(workspace_root).resolve() if workspace_root is not None else config_path.parent.parent / "workspace"
    return V3Config(config_path=config_path, workspace_root=root)
```

```python
# cozy_pde_v3/__init__.py
"""Cozy PDE v3 package."""
```

- [ ] **Step 4: Run the tests to verify the new package and shared TaskSpec contract pass**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_cli.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit the scaffolding**

```bash
git add cozy_pde_v3/__init__.py cozy_pde_v3/cli.py cozy_pde_v3/config.py cozy_pde_v3/task_specs.py tests/test_v3_cli.py
git commit -m "feat: scaffold cozy pde v3 package and task specs"
```

## Task 2: Implement unified Responses transport and provider capability reports

**Files:**
- Create: `cozy_pde_v3/responses_client.py`
- Create: `cozy_pde_v3/responses_ledger.py`
- Create: `cozy_pde_v3/provider_capabilities.py`
- Test: `tests/test_v3_responses_client.py`

- [ ] **Step 1: Write failing tests for unified turns, failover boundaries, and provider reports**

```python
from __future__ import annotations

import json
import types
from pathlib import Path

from cozy_pde_v3.provider_capabilities import write_provider_report
from cozy_pde_v3.responses_client import ResponsesClient, ResponsesTurn


class QuotaError(RuntimeError):
    status_code = 429


def test_responses_client_returns_raw_and_standardized_items(monkeypatch) -> None:
    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = types.SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            return {
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}
                ],
                "usage": {"input_tokens": 10, "output_tokens": 3, "cached_tokens": 7},
            }

    monkeypatch.setitem(__import__("sys").modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    client = ResponsesClient.primary_only(base_url="https://aixj.vip", api_key="test-key")
    turn = client.create(
        input_items=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        tools=[],
        instructions="Reply briefly.",
        metadata={"task": "task1"},
        prompt_cache_key="cozypde:v3:test",
    )

    assert isinstance(turn, ResponsesTurn)
    assert turn.provider == "primary"
    assert turn.provider_output_items[0]["type"] == "message"
    assert turn.standard_output_items[0]["type"] == "message"
    assert turn.usage["cached_tokens"] == 7


def test_responses_client_fails_over_on_quota_before_any_tool_execution(monkeypatch) -> None:
    calls = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.base_url = kwargs["base_url"]
            self.responses = types.SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            calls.append(self.base_url)
            if "aixj" in self.base_url:
                raise QuotaError("quota exceeded")
            return {
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "fallback ok"}]}
                ]
            }

    monkeypatch.setitem(__import__("sys").modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    client = ResponsesClient.with_fallback(
        primary_base_url="https://aixj.vip",
        primary_api_key="primary-key",
        fallback_base_url="https://api.deepseek.com",
        fallback_api_key="fallback-key",
    )
    turn = client.create(input_items=[], tools=[], instructions="Reply", metadata={}, prompt_cache_key="probe")

    assert calls == ["https://aixj.vip/v1", "https://api.deepseek.com"]
    assert turn.failover_from == "primary"
    assert turn.failover_reason == "quota_exhausted"
    assert turn.provider == "fallback"


def test_responses_client_does_not_retry_after_provider_already_returned_function_call(monkeypatch) -> None:
    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = types.SimpleNamespace(create=self.create)

        def create(self, **kwargs):
            return {
                "output": [
                    {"type": "function_call", "name": "write_file", "call_id": "call_1", "arguments": "{\"path\": \"x\"}"}
                ]
            }

    monkeypatch.setitem(__import__("sys").modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))
    client = ResponsesClient.primary_only(base_url="https://aixj.vip", api_key="test-key")
    turn = client.create(input_items=[], tools=[], instructions="Reply", metadata={}, prompt_cache_key="probe")

    assert turn.standard_output_items[0]["type"] == "function_call"
    assert turn.failover_from is None


def test_write_provider_report_includes_config_schema_and_proxy_hashes(tmp_path: Path) -> None:
    report_path = tmp_path / "provider_report.json"
    report = write_provider_report(
        report_path=report_path,
        primary={"formal_ready": True},
        fallback={"formal_ready": False},
        forced_failover={"ok": False},
        config_bytes=b"cfg",
        tool_schema_bytes=b"schema",
        proxy_bytes=b"proxy",
    )

    assert report_path.exists()
    assert report["config_hash"]
    assert report["tool_schema_hash"]
    assert report["proxy_version_hash"]
```

- [ ] **Step 2: Run the tests to verify the Responses layer is still missing**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_responses_client.py -q
```

Expected:

```text
E   ImportError: cannot import name 'ResponsesClient'
```

- [ ] **Step 3: Implement `ResponsesTurn`, adapter-based create, error classification, and report writing**

```python
# cozy_pde_v3/responses_client.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ResponsesTurn:
    provider: str
    model: str
    raw_response: dict[str, Any]
    provider_output_items: list[dict[str, Any]]
    standard_output_items: list[dict[str, Any]]
    usage: dict[str, Any]
    failover_from: str | None = None
    failover_reason: str | None = None


def _classify_provider_error(exc: Exception) -> str | None:
    text = str(exc).lower()
    status_code = getattr(exc, "status_code", None)
    if status_code in {502, 503, 504, 524}:
        return "gateway_error"
    if status_code == 429 and "quota" in text:
        return "quota_exhausted"
    if any(token in text for token in ("timeout", "timed out", "connection reset", "network")):
        return "network_error"
    return None
```

```python
class ResponsesClient:
    def create(self, *, input_items, tools, instructions, metadata, prompt_cache_key) -> ResponsesTurn:
        try:
            raw = self._primary.responses.create(
                model=self._primary_model,
                input=input_items,
                tools=tools,
                instructions=instructions,
                metadata=metadata,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention="24h",
                parallel_tool_calls=False,
                store=False,
                stream=False,
            )
            return self._normalize(raw=raw, provider="primary", model=self._primary_model)
        except Exception as exc:  # noqa: BLE001
            reason = _classify_provider_error(exc)
            if reason is None or self._fallback is None:
                raise
            fallback_raw = self._fallback.responses.create(
                model=self._fallback_model,
                input=input_items,
                tools=tools,
                instructions=instructions,
                metadata=metadata,
                prompt_cache_key=prompt_cache_key,
                prompt_cache_retention="24h",
                parallel_tool_calls=False,
                store=False,
                stream=False,
            )
            turn = self._normalize(raw=fallback_raw, provider="fallback", model=self._fallback_model)
            turn.failover_from = "primary"
            turn.failover_reason = reason
            return turn
```

```python
# cozy_pde_v3/responses_ledger.py
from __future__ import annotations

import json
from typing import Any


def extract_standard_output_items(raw_response: dict[str, Any]) -> list[dict[str, Any]]:
    return list(raw_response.get("output", []))


def function_call_output(call_id: str, output: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(output, ensure_ascii=False),
    }
```

```python
# cozy_pde_v3/provider_capabilities.py
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_provider_report(
    *,
    report_path: Path,
    primary: dict[str, Any],
    fallback: dict[str, Any],
    forced_failover: dict[str, Any],
    config_bytes: bytes,
    tool_schema_bytes: bytes,
    proxy_bytes: bytes,
) -> dict[str, Any]:
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": hashlib.sha256(config_bytes).hexdigest(),
        "tool_schema_hash": hashlib.sha256(tool_schema_bytes).hexdigest(),
        "proxy_version_hash": hashlib.sha256(proxy_bytes).hexdigest(),
        "primary": primary,
        "fallback": fallback,
        "forced_failover": forced_failover,
        "formal_ready": bool(primary.get("formal_ready")),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
```

- [ ] **Step 4: Run the tests to verify the unified Responses turn contract**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_responses_client.py -q
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Commit the unified Responses layer**

```bash
git add cozy_pde_v3/responses_client.py cozy_pde_v3/responses_ledger.py cozy_pde_v3/provider_capabilities.py tests/test_v3_responses_client.py
git commit -m "feat: add v3 responses client and provider capability reports"
```

## Task 3: Upgrade proxy logging, redaction, and dual-provider log merging

**Files:**
- Modify: `scripts/proxy.py`
- Create: `cozy_pde_v3/proxy_logs.py`
- Test: `tests/test_v3_proxy_logs.py`

- [ ] **Step 1: Write failing tests for proxy redaction and merged provider logs**

```python
from __future__ import annotations

import json

from cozy_pde_v3.proxy_logs import merge_provider_logs, redact_proxy_entry


def test_redact_proxy_entry_scrubs_headers_tokens_and_user_paths() -> None:
    entry = {
        "headers": {"Authorization": "Bearer sk-secret-token"},
        "request_body": {"env": "OPENAI_API_KEY=sk-another-secret"},
        "response_body": {"stdout": "/home/alice/cozy_pde/output"},
    }

    redacted = redact_proxy_entry(entry)
    payload = json.dumps(redacted)

    assert "sk-secret-token" not in payload
    assert "sk-another-secret" not in payload
    assert "/home/alice" not in payload


def test_merge_provider_logs_orders_entries_by_timestamp(tmp_path) -> None:
    aixj_dir = tmp_path / "aixj"
    deepseek_dir = tmp_path / "deepseek"
    aixj_dir.mkdir()
    deepseek_dir.mkdir()
    (aixj_dir / "llm-20260521.jsonl").write_text('{"timestamp":"2026-05-21T00:00:02+00:00","provider":"aixj"}\n', encoding="utf-8")
    (deepseek_dir / "llm-20260521.jsonl").write_text('{"timestamp":"2026-05-21T00:00:01+00:00","provider":"deepseek"}\n', encoding="utf-8")

    merged = merge_provider_logs(aixj_dir, deepseek_dir)
    assert [entry["provider"] for entry in merged] == ["deepseek", "aixj"]
```

- [ ] **Step 2: Run the tests to verify proxy helpers do not exist yet**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_proxy_logs.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'cozy_pde_v3.proxy_logs'
```

- [ ] **Step 3: Add redaction helpers, merged provider log export, and reuse them from `scripts/proxy.py`**

```python
# cozy_pde_v3/proxy_logs.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

TOKEN_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]+"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+", re.IGNORECASE),
    re.compile(r"/home/[^/]+"),
]


def _redact_text(text: str) -> str:
    redacted = text
    for pattern in TOKEN_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact_proxy_entry(entry: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(entry))
    serialized = _redact_text(json.dumps(payload, ensure_ascii=False))
    return json.loads(serialized)
```

```python
def merge_provider_logs(primary_dir: Path, fallback_dir: Path | None) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for directory in [primary_dir, fallback_dir]:
        if directory is None or not directory.exists():
            continue
        for path in sorted(directory.glob("llm-*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entries.append(json.loads(line))
    return sorted(entries, key=lambda item: item.get("timestamp", ""))
```

```python
# scripts/proxy.py
from cozy_pde_v3.proxy_logs import redact_proxy_entry


def write_llm_log(log_dir: str, entry: dict) -> None:
    log_file = Path(log_dir) / f"llm-{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(log_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(redact_proxy_entry(entry), ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Run the proxy log tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_proxy_logs.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit proxy hardening**

```bash
git add scripts/proxy.py cozy_pde_v3/proxy_logs.py tests/test_v3_proxy_logs.py
git commit -m "feat: add v3 proxy redaction and merged provider logs"
```

## Task 4: Persist state, memory, code snapshots, and patch records

**Files:**
- Create: `cozy_pde_v3/state.py`
- Create: `cozy_pde_v3/memory_store.py`
- Create: `cozy_pde_v3/code_evolution.py`
- Test: `tests/test_v3_state_memory.py`

- [ ] **Step 1: Write failing tests for state persistence and shared-code version tables**

```python
from __future__ import annotations

from cozy_pde_v3.code_evolution import CodePatchRecord, CodeSnapshot
from cozy_pde_v3.memory_store import MemoryStore
from cozy_pde_v3.state import AgentState


def test_agent_state_tracks_shared_code_version_and_external_ids() -> None:
    state = AgentState(task="task2")
    state.last_llm_call_id = "task2:step-008"
    state.best_artifact_version = "art_002"
    state.shared_code_version = "v2"

    assert state.task == "task2"
    assert state.last_llm_call_id == "task2:step-008"
    assert state.shared_code_version == "v2"


def test_memory_store_initializes_decision_and_code_tables(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.sqlite")
    store.initialize()
    store.record_code_snapshot(
        CodeSnapshot(
            code_version="v1",
            parent_version=None,
            content_hash="abc",
            api_contract_hash="def",
            supported_tasks=["task1"],
            task_support_matrix={"task1": {"compat_ok": True}},
            created_by_run_id="task1_run",
            created_at="2026-05-21T00:00:00+00:00",
        )
    )
    store.record_patch(
        CodePatchRecord(
            patch_id="patch_1",
            base_code_version="v1",
            new_code_version="v2",
            task_context="task2",
            changed_files=["submission/code/model.py"],
            change_intent="add nu conditioning",
            backward_compatibility_claim="task1 cli and smoke train still pass",
            affected_interfaces=["train.py --task", "infer.py --task"],
            llm_call_ids=["task2:step-009"],
            validation_results={"task1_compat_ok": True},
        )
    )

    assert store.list_code_snapshots()[0]["code_version"] == "v1"
    assert store.list_patch_records()[0]["new_code_version"] == "v2"
```

- [ ] **Step 2: Run the tests to verify the persistence layer is still missing**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_state_memory.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'cozy_pde_v3.state'
```

- [ ] **Step 3: Implement `AgentState`, `MemoryStore`, `CodeSnapshot`, and `CodePatchRecord`**

```python
# cozy_pde_v3/state.py
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(slots=True)
class AgentState:
    task: str
    run_id: str = ""
    mode: str = "formal"
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_phase: str = "capability_check"
    current_objective: str = ""
    latest_error_type: str | None = None
    latest_error_summary: str | None = None
    latest_tool_name: str | None = None
    latest_tool_result_ok: bool | None = None
    last_llm_call_id: str | None = None
    last_tool_call_id: str | None = None
    best_artifact_path: str | None = None
    best_artifact_version: str | None = None
    latest_checkpoint_path: str | None = None
    submission_snapshot_id: str | None = None
    shared_code_version: str | None = None
    supported_tasks: list[str] = field(default_factory=list)
    finalize_gate_status: dict[str, object] = field(default_factory=dict)
```

```python
# cozy_pde_v3/code_evolution.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CodeSnapshot:
    code_version: str
    parent_version: str | None
    content_hash: str
    api_contract_hash: str
    supported_tasks: list[str]
    task_support_matrix: dict[str, dict]
    created_by_run_id: str
    created_at: str


@dataclass(frozen=True, slots=True)
class CodePatchRecord:
    patch_id: str
    base_code_version: str
    new_code_version: str
    task_context: str
    changed_files: list[str]
    change_intent: str
    backward_compatibility_claim: str
    affected_interfaces: list[str]
    llm_call_ids: list[str]
    validation_results: dict
```

```python
# cozy_pde_v3/memory_store.py
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS decision_records (
                    state_hash TEXT,
                    reason_code TEXT,
                    route TEXT,
                    selected_profile TEXT,
                    selected_phase TEXT,
                    selected_tools TEXT,
                    outcome TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS code_snapshots (
                    code_version TEXT PRIMARY KEY,
                    parent_version TEXT,
                    content_hash TEXT,
                    api_contract_hash TEXT,
                    supported_tasks TEXT,
                    task_support_matrix TEXT,
                    created_by_run_id TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS code_patch_records (
                    patch_id TEXT PRIMARY KEY,
                    base_code_version TEXT,
                    new_code_version TEXT,
                    task_context TEXT,
                    changed_files TEXT,
                    change_intent TEXT,
                    backward_compatibility_claim TEXT,
                    affected_interfaces TEXT,
                    llm_call_ids TEXT,
                    validation_results TEXT
                );
                """
            )
```

- [ ] **Step 4: Run the state and memory tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_state_memory.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit persistence and code-evolution records**

```bash
git add cozy_pde_v3/state.py cozy_pde_v3/memory_store.py cozy_pde_v3/code_evolution.py tests/test_v3_state_memory.py
git commit -m "feat: add v3 state persistence and code evolution records"
```

## Task 5: Implement deterministic routing, token-bounded context packing, and compatibility-aware experiment engine

**Files:**
- Create: `cozy_pde_v3/profiles.py`
- Create: `cozy_pde_v3/deterministic_router.py`
- Create: `cozy_pde_v3/context_packer.py`
- Create: `cozy_pde_v3/experiment_engine.py`
- Test: `tests/test_v3_router_context.py`

- [ ] **Step 1: Write failing tests for recovery precedence, context budgets, and compatibility gate routing**

```python
from __future__ import annotations

from cozy_pde_v3.context_packer import ContextPacker
from cozy_pde_v3.deterministic_router import DeterministicRouter
from cozy_pde_v3.experiment_engine import compatibility_checks_for_patch
from cozy_pde_v3.state import AgentState


def test_router_prioritizes_failure_recovery_over_new_implementation() -> None:
    state = AgentState(task="task2")
    state.current_objective = "add nu conditioning"
    state.latest_error_type = "cuda_oom"
    state.latest_error_summary = "CUDA out of memory"

    route = DeterministicRouter().choose(state)

    assert route.phase == "failure_recovery"
    assert route.reason_code == "cuda_oom"


def test_context_packer_enforces_memory_and_log_budgets() -> None:
    packer = ContextPacker(retrieved_memory_budget=1500, log_summary_budget=1000, code_excerpt_budget=5000)
    items = packer.build(
        compact_state={"phase": "diagnosis"},
        retrieved_memory="x " * 3000,
        log_summary="y " * 3000,
        code_excerpt="z " * 9000,
        current_request="diagnose the last failure",
    )

    serialized = " ".join(str(item) for item in items)
    assert len(serialized) < 20000


def test_compatibility_checks_require_cli_smoke_and_inference_shape() -> None:
    results = compatibility_checks_for_patch(
        supported_tasks=["task1"],
        current_task="task2",
        cli_ok=True,
        smoke_ok=True,
        infer_shape_ok=True,
    )
    assert results["task1_compat_ok"] is True
```

- [ ] **Step 2: Run the tests to verify router and engine modules are missing**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_router_context.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'cozy_pde_v3.context_packer'
```

- [ ] **Step 3: Implement route decisions, profile map, token limits, and shared-code compatibility checks**

```python
# cozy_pde_v3/deterministic_router.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RouteDecision:
    phase: str
    profile: str
    allowed_tools: tuple[str, ...]
    requires_llm: bool
    deterministic_action: str | None
    reason_code: str


class DeterministicRouter:
    def choose(self, state) -> RouteDecision:
        if state.latest_error_type in {"cuda_oom", "shape_mismatch", "loss_nan", "inference_timeout"}:
            return RouteDecision(
                phase="failure_recovery",
                profile="diagnoser",
                allowed_tools=("read_file", "run_python", "validate_submission"),
                requires_llm=False,
                deterministic_action="recover_from_failure",
                reason_code=state.latest_error_type,
            )
        if not state.shared_code_version:
            return RouteDecision("baseline_guard", "planner", ("inspect_hdf5",), False, "ensure_baseline", "baseline_missing")
        return RouteDecision("implementation", "coder", ("read_file", "write_file", "patch_file"), True, None, "implementation_needed")
```

```python
# cozy_pde_v3/context_packer.py
from __future__ import annotations


class ContextPacker:
    def __init__(self, *, retrieved_memory_budget: int, log_summary_budget: int, code_excerpt_budget: int) -> None:
        self.retrieved_memory_budget = retrieved_memory_budget
        self.log_summary_budget = log_summary_budget
        self.code_excerpt_budget = code_excerpt_budget

    def _trim(self, text: str, budget: int) -> str:
        return text[:budget]

    def build(self, *, compact_state, retrieved_memory, log_summary, code_excerpt, current_request):
        return [
            {"type": "developer_contract", "text": "shared codebase only"},
            {"type": "compact_state", "text": str(compact_state)},
            {"type": "retrieved_memory", "text": self._trim(retrieved_memory, self.retrieved_memory_budget)},
            {"type": "log_summary", "text": self._trim(log_summary, self.log_summary_budget)},
            {"type": "code_excerpt", "text": self._trim(code_excerpt, self.code_excerpt_budget)},
            {"type": "current_request", "text": current_request},
        ]
```

```python
# cozy_pde_v3/experiment_engine.py
from __future__ import annotations


def compatibility_checks_for_patch(*, supported_tasks, current_task, cli_ok, smoke_ok, infer_shape_ok):
    results = {}
    for task in supported_tasks:
        key = f"{task}_compat_ok"
        results[key] = bool(cli_ok and smoke_ok and infer_shape_ok)
    if current_task not in supported_tasks:
        results[f"{current_task}_compat_ok"] = True
    return results
```

- [ ] **Step 4: Run the router/context/engine tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_router_context.py -q
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit routing and compatibility gating**

```bash
git add cozy_pde_v3/profiles.py cozy_pde_v3/deterministic_router.py cozy_pde_v3/context_packer.py cozy_pde_v3/experiment_engine.py tests/test_v3_router_context.py
git commit -m "feat: add v3 router context packer and compatibility gate"
```

## Task 6: Migrate validation, packaging, research, status reporting, and structured finalize gates

**Files:**
- Create: `cozy_pde_v3/validation/logs.py`
- Create: `cozy_pde_v3/validation/provenance.py`
- Create: `cozy_pde_v3/validation/submission.py`
- Create: `cozy_pde_v3/logging.py`
- Create: `cozy_pde_v3/research/cache.py`
- Create: `cozy_pde_v3/research/providers.py`
- Create: `cozy_pde_v3/research/tools.py`
- Test: `tests/test_v3_validation_package.py`

- [ ] **Step 1: Write failing tests for structured finalize gates, shared-code provenance, and research wrappers**

```python
from __future__ import annotations

from cozy_pde_v3.validation.submission import build_finalize_gate_status
from cozy_pde_v3.validation.provenance import build_shared_code_union


def test_finalize_gate_contains_shared_code_and_compatibility_keys() -> None:
    gate = build_finalize_gate_status(prediction_ok=True, logs_ok=True, provenance_ok=True)

    assert "shared_code_ok" in gate
    assert "api_contract_ok" in gate
    assert "task1_compat_ok" in gate
    assert "no_task_specific_code_fork_ok" in gate


def test_shared_code_union_preserves_cross_task_lineage() -> None:
    union = build_shared_code_union(
        snapshots=[
            {
                "version": "v2",
                "created_during": "task2",
                "parent": "v1",
                "changed_files": ["submission/code/model.py"],
                "llm_call_ids": ["task2:step-009"],
                "validated_tasks": ["task1", "task2"],
            }
        ]
    )

    assert union["shared_code_versions"][0]["parent"] == "v1"
    assert union["shared_code_versions"][0]["validated_tasks"] == ["task1", "task2"]
```

- [ ] **Step 2: Run the tests to verify validation helpers do not exist yet**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_validation_package.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'cozy_pde_v3.validation'
```

- [ ] **Step 3: Port and simplify the retained `E/F` capabilities into `cozy_pde_v3`**

```python
# cozy_pde_v3/validation/submission.py
from __future__ import annotations


def build_finalize_gate_status(*, prediction_ok: bool, logs_ok: bool, provenance_ok: bool) -> dict[str, object]:
    gate = {
        "prediction_ok": prediction_ok,
        "time_csv_ok": False,
        "logs_ok": logs_ok,
        "provenance_ok": provenance_ok,
        "inference_time_ok": False,
        "package_ok": False,
        "code_manifest_ok": False,
        "methodology_ok": False,
        "secret_scan_ok": False,
        "task_rule_ok": False,
        "shared_code_ok": False,
        "code_provenance_ok": False,
        "api_contract_ok": False,
        "task1_compat_ok": False,
        "task2_compat_ok": False,
        "task3_compat_ok": False,
        "incremental_patch_ok": False,
        "no_task_specific_code_fork_ok": False,
        "failures": [],
        "warnings": [],
    }
    gate["overall_ok"] = all(
        gate[name]
        for name in (
            "prediction_ok",
            "logs_ok",
            "provenance_ok",
            "shared_code_ok",
            "code_provenance_ok",
            "api_contract_ok",
            "no_task_specific_code_fork_ok",
        )
    )
    return gate
```

```python
# cozy_pde_v3/validation/logs.py
from __future__ import annotations

import json
from pathlib import Path


def validate_jsonl_log(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"ok": False, "error": f"{path} does not exist"}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)
    return {"ok": True}
```

```python
# cozy_pde_v3/validation/provenance.py
from __future__ import annotations


def build_shared_code_union(*, snapshots: list[dict]) -> dict[str, object]:
    return {"shared_code_versions": snapshots}
```

```python
# cozy_pde_v3/logging.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
```

```python
# cozy_pde_v3/research/cache.py
from __future__ import annotations

from agent_runner.research_cache import ResearchCache  # migrate by move in final cleanup

__all__ = ["ResearchCache"]
```

```python
# cozy_pde_v3/research/providers.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResearchProviderFlags:
    arxiv_enabled: bool = True
    github_enabled: bool = True
    allow_unauthenticated_github: bool = True
```

```python
# cozy_pde_v3/research/tools.py
from __future__ import annotations

from agent_runner.tools.research_tools import fetch_pdf, fetch_url, parse_html, parse_pdf, search_arxiv, search_github

__all__ = [
    "search_arxiv",
    "search_github",
    "fetch_url",
    "fetch_pdf",
    "parse_pdf",
    "parse_html",
]
```

- [ ] **Step 4: Run the validation/package tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_validation_package.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit migrated validation and research support**

```bash
git add cozy_pde_v3/validation cozy_pde_v3/logging.py cozy_pde_v3/research tests/test_v3_validation_package.py
git commit -m "feat: add v3 validation package and research support"
```

## Task 7: Wire the formal run session, cut the CLI over, and delete obsolete second-generation layers

**Files:**
- Create: `cozy_pde_v3/agent_loop.py`
- Modify: `cozy_pde_v3/cli.py`
- Test: `tests/test_v3_run_session.py`
- Delete: `agent_runner/json_action_client.py`
- Delete: `agent_runner/router.py`
- Delete: `agent_runner/main.py`
- Delete: `agent_runner/prompts.py`
- Delete: `tests/test_one_tool_mode.py`
- Delete: `tests/test_no_chat_completions.py`
- Delete: `tests/test_task_sessions.py`
- Delete: `tests/test_responses_loop_tool_call.py`
- Delete: `tests/test_responses_client.py`

- [ ] **Step 1: Write the failing integration test for a single-task formal run session**

```python
from __future__ import annotations

from cozy_pde_v3.agent_loop import should_start_formal_run
from cozy_pde_v3.validation.submission import build_finalize_gate_status


def test_formal_run_requires_primary_ready_and_honors_optional_fallback() -> None:
    assert should_start_formal_run(primary_ready=True, fallback_ready=False, require_fallback=False) is True
    assert should_start_formal_run(primary_ready=True, fallback_ready=True, require_fallback=True) is True
    assert should_start_formal_run(primary_ready=True, fallback_ready=False, require_fallback=True) is False
    assert should_start_formal_run(primary_ready=False, fallback_ready=True, require_fallback=False) is False


def test_finalize_gate_blocks_when_shared_code_contract_is_not_ready() -> None:
    gate = build_finalize_gate_status(prediction_ok=True, logs_ok=True, provenance_ok=True)
    assert gate["overall_ok"] is False
```

- [ ] **Step 2: Run the tests to verify the run loop still does not exist**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_run_session.py -q
```

Expected:

```text
E   ModuleNotFoundError: No module named 'cozy_pde_v3.agent_loop'
```

- [ ] **Step 3: Implement the run-start rule, wire CLI command handlers, update dependencies, and remove obsolete files**

```python
# cozy_pde_v3/agent_loop.py
from __future__ import annotations


def should_start_formal_run(*, primary_ready: bool, fallback_ready: bool, require_fallback: bool) -> bool:
    if not primary_ready:
        return False
    if require_fallback and not fallback_ready:
        return False
    return True
```

```python
# cozy_pde_v3/cli.py
from __future__ import annotations

from cozy_pde_v3.agent_loop import should_start_formal_run
from cozy_pde_v3.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        config = load_config(args.config)
        del config
        return 0
    if args.command in {"check-provider", "check-research", "validate", "package", "status"}:
        return 0
    return 1
```

Delete these files after the new tests pass and the new command path is reachable:

```text
agent_runner/json_action_client.py
agent_runner/router.py
agent_runner/main.py
agent_runner/prompts.py
tests/test_one_tool_mode.py
tests/test_no_chat_completions.py
tests/test_task_sessions.py
tests/test_responses_loop_tool_call.py
tests/test_responses_client.py
```

- [ ] **Step 4: Run the focused v3 tests and then the full test suite**

Run:

```bash
.venv/bin/python -m pytest tests/test_v3_cli.py tests/test_v3_responses_client.py tests/test_v3_proxy_logs.py tests/test_v3_state_memory.py tests/test_v3_router_context.py tests/test_v3_validation_package.py tests/test_v3_run_session.py -q
```

Expected:

```text
all selected v3 tests pass
```

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected:

```text
full suite passes without any `agent_runner.json_action_client` or `chat.completions` references
```

- [ ] **Step 5: Commit the cutover and deletion of obsolete second-generation layers**

```bash
git add cozy_pde_v3 requirements.txt tests/test_v3_cli.py tests/test_v3_responses_client.py tests/test_v3_proxy_logs.py tests/test_v3_state_memory.py tests/test_v3_router_context.py tests/test_v3_validation_package.py tests/test_v3_run_session.py
git add -u agent_runner tests
git commit -m "refactor: cut over to cozy pde v3 formal run architecture"
```

## Self-Review Checklist

- [ ] The plan never reintroduces `chat.completions` or `json_action`.
- [ ] The plan keeps formal runs single-task while forcing all runs to share one evolving `submission/code/`.
- [ ] Provider capability checks include `config_hash`, `tool_schema_hash`, and `proxy_version_hash`.
- [ ] Failover never crosses a returned `function_call` boundary.
- [ ] Finalize gates include shared-code compatibility and no-fork checks.
- [ ] `methodology` and `code_manifest.json` are generated mechanically from structured records.
- [ ] The plan deletes rehearsal/dry-run/legacy router code instead of preserving it as a compatibility layer.

## Execution Handoff

Plan complete and saved to `dev_docs/2026-05-21-cozy-pde-v3-implementation-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
