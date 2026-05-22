from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_module(module_name: str, relative_path: str):
    path = Path(__file__).resolve().parent.parent / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scan_v3_banned_patterns = _load_module("scan_v3_banned_patterns", "scripts/scan_v3_banned_patterns.py")
check_v3_deletion_readiness = _load_module(
    "check_v3_deletion_readiness",
    "scripts/check_v3_deletion_readiness.py",
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _token(*parts: str) -> str:
    return "".join(parts)


def test_scan_report_finds_banned_pattern_in_scoped_path(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    pattern = _token("json", "_action")
    line_text = f"WIRE_API = '{pattern}'"
    _write(repo_root / "cozy_pde_v3" / "module.py", line_text + "\n")

    report = scan_v3_banned_patterns.scan_paths(
        repo_root=repo_root,
        targets=[Path("cozy_pde_v3")],
    )

    assert report["ok"] is False
    assert report["scanned_file_count"] == 1
    assert report["matches"] == [
        {
            "path": "cozy_pde_v3/module.py",
            "line": 1,
            "pattern": pattern,
            "line_text": line_text,
        }
    ]


def test_retained_scope_ignores_legacy_paths_outside_default_targets(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write(repo_root / "cozy_pde_v3" / "clean.py", "VALUE = 'responses'\n")
    _write(repo_root / "tests" / "test_clean.py", "def test_ok():\n    assert True\n")
    _write(repo_root / "pytest.ini", "[pytest]\n")
    _write(repo_root / "requirements.txt", "pytest\n")
    pattern = _token("Json", "Action", "Client")
    _write(repo_root / "agent_runner" / "legacy.py", f"client = '{pattern}'\n")

    report = scan_v3_banned_patterns.scan_scope(
        repo_root=repo_root,
        scope="retained",
    )

    assert report["ok"] is True
    assert report["matches"] == []
    assert report["targets"] == [
        "cozy_pde_v3",
        "tests",
        "pytest.ini",
        "requirements.txt",
        "scripts/scan_v3_banned_patterns.py",
        "scripts/check_v3_deletion_readiness.py",
    ]
    assert report["scanned_file_count"] == 4


def test_retained_scope_catches_banned_patterns_in_retained_tests(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write(repo_root / "cozy_pde_v3" / "clean.py", "VALUE = 'responses'\n")
    _write(repo_root / "pytest.ini", "[pytest]\n")
    _write(repo_root / "requirements.txt", "pytest\n")
    pattern = _token("code/", "task2")
    line_text = f'assert value == "{pattern}/train.py"'
    _write(repo_root / "tests" / "test_gate.py", line_text + "\n")

    report = scan_v3_banned_patterns.scan_scope(
        repo_root=repo_root,
        scope="retained",
    )

    assert report["ok"] is False
    assert report["matches"] == [
        {
            "path": "tests/test_gate.py",
            "line": 1,
            "pattern": pattern,
            "line_text": line_text,
        }
    ]


def test_retained_scope_catches_banned_patterns_in_retained_top_level_config(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write(repo_root / "cozy_pde_v3" / "clean.py", "VALUE = 'responses'\n")
    _write(repo_root / "tests" / "test_clean.py", "def test_ok():\n    assert True\n")
    _write(repo_root / "requirements.txt", "pytest\n")
    pattern = _token("json", "_action")
    line_text = f"legacy_wire_api = {pattern}"
    _write(repo_root / "pytest.ini", line_text + "\n")

    report = scan_v3_banned_patterns.scan_scope(
        repo_root=repo_root,
        scope="retained",
    )

    assert report["ok"] is False
    assert report["matches"] == [
        {
            "path": "pytest.ini",
            "line": 1,
            "pattern": pattern,
            "line_text": line_text,
        }
    ]


def test_full_scope_catches_legacy_hits_anywhere_in_repo(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write(repo_root / "cozy_pde_v3" / "clean.py", "VALUE = 'responses'\n")
    pattern = _token("Json", "Action", "Client")
    line_text = f"client = '{pattern}'"
    _write(repo_root / "agent_runner" / "legacy.py", line_text + "\n")

    report = scan_v3_banned_patterns.scan_scope(
        repo_root=repo_root,
        scope="full",
    )

    assert report["ok"] is False
    assert report["matches"] == [
        {
            "path": "agent_runner/legacy.py",
            "line": 1,
            "pattern": pattern,
            "line_text": line_text,
        }
    ]


def test_deletion_readiness_runner_returns_failure_for_retained_scope_hit(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    pattern = _token("autonomous", "_dry_run")
    _write(repo_root / "cozy_pde_v3" / "bad.py", f"MODE = '{pattern}'\n")

    report = check_v3_deletion_readiness.run_readiness_check(
        repo_root=repo_root,
        scope="retained",
    )

    assert report["ok"] is False
    assert report["scope"] == "retained"
    assert report["match_count"] == 1


def test_cli_json_output_reports_matches_and_exit_code(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    pattern = _token("autonomous", "_rehearsal")
    _write(repo_root / "cozy_pde_v3" / "bad.py", f"MODE = '{pattern}'\n")

    script = Path(__file__).resolve().parent.parent / "scripts" / "scan_v3_banned_patterns.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(repo_root),
            "--path",
            "cozy_pde_v3",
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert f'"pattern": "{pattern}"' in result.stdout
