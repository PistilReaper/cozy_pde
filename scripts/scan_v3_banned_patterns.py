from __future__ import annotations

import argparse
import json
from pathlib import Path


def _build_banned_patterns() -> tuple[str, ...]:
    return (
        "chat" + ".completions",
        "json" + "_action",
        "Json" + "Action" + "Client",
        "autonomous" + "_dry_run",
        "autonomous" + "_rehearsal",
        "code/" + "task1",
        "code/" + "task2",
        "code/" + "task3",
    )


BANNED_PATTERNS = _build_banned_patterns()
DEFAULT_RETAINED_TARGETS = (
    Path("cozy_pde_v3"),
    Path("tests"),
    Path("pytest.ini"),
    Path("requirements.txt"),
    Path("scripts/scan_v3_banned_patterns.py"),
    Path("scripts/check_v3_deletion_readiness.py"),
)
IGNORED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "workspace",
}


def _is_binary(path: Path) -> bool:
    try:
        payload = path.read_bytes()
    except OSError:
        return True
    return b"\x00" in payload


def _iter_target_files(repo_root: Path, targets: list[Path]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        resolved = repo_root / target
        if not resolved.exists():
            continue
        if resolved.is_file():
            files.append(resolved)
            continue
        for path in sorted(resolved.rglob("*")):
            if not path.is_file():
                continue
            if any(part in IGNORED_DIR_NAMES for part in path.parts):
                continue
            files.append(path)
    return sorted({path for path in files})


def _iter_full_repo_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(repo_root).parts
        if any(part in IGNORED_DIR_NAMES for part in relative_parts):
            continue
        files.append(path)
    return files


def scan_paths(*, repo_root: Path, targets: list[Path]) -> dict[str, object]:
    repo_root = repo_root.resolve()
    matches: list[dict[str, object]] = []
    files = _iter_target_files(repo_root, targets)
    for path in files:
        if _is_binary(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="replace")
        relative_path = path.relative_to(repo_root).as_posix()
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern in BANNED_PATTERNS:
                if pattern in line:
                    matches.append(
                        {
                            "path": relative_path,
                            "line": line_number,
                            "pattern": pattern,
                            "line_text": line,
                        }
                    )
    return {
        "ok": not matches,
        "repo_root": repo_root.as_posix(),
        "targets": [target.as_posix() for target in targets],
        "scanned_file_count": len(files),
        "match_count": len(matches),
        "matches": matches,
    }


def scan_scope(*, repo_root: Path, scope: str) -> dict[str, object]:
    repo_root = repo_root.resolve()
    if scope == "retained":
        return scan_paths(repo_root=repo_root, targets=list(DEFAULT_RETAINED_TARGETS))
    if scope == "full":
        matches: list[dict[str, object]] = []
        files = _iter_full_repo_files(repo_root)
        for path in files:
            if _is_binary(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="utf-8", errors="replace")
            relative_path = path.relative_to(repo_root).as_posix()
            for line_number, line in enumerate(text.splitlines(), start=1):
                for pattern in BANNED_PATTERNS:
                    if pattern in line:
                        matches.append(
                            {
                                "path": relative_path,
                                "line": line_number,
                                "pattern": pattern,
                                "line_text": line,
                            }
                        )
        return {
            "ok": not matches,
            "repo_root": repo_root.as_posix(),
            "targets": ["."],
            "scanned_file_count": len(files),
            "match_count": len(matches),
            "matches": matches,
        }
    raise ValueError(f"Unsupported scope: {scope}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan repo files for v3 deletion-gate banned patterns.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to scan. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--scope",
        choices=("retained", "full"),
        default="retained",
        help="Use retained targets first, or full repo after deletion.",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Relative file or directory path to scan. Overrides scope defaults when provided.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a human summary.",
    )
    return parser


def _format_human_report(report: dict[str, object]) -> str:
    lines = [
        f"scan_ok={report['ok']}",
        f"scanned_file_count={report['scanned_file_count']}",
        f"match_count={report['match_count']}",
    ]
    matches = report["matches"]
    assert isinstance(matches, list)
    for match in matches:
        assert isinstance(match, dict)
        lines.append(
            f"{match['path']}:{match['line']}: banned pattern {match['pattern']!r}: {match['line_text']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    if args.path:
        report = scan_paths(repo_root=repo_root, targets=[Path(path) for path in args.path])
    else:
        report = scan_scope(repo_root=repo_root, scope=args.scope)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_human_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
