from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load_scan_module():
    path = Path(__file__).resolve().parent / "scan_v3_banned_patterns.py"
    spec = importlib.util.spec_from_file_location("scan_v3_banned_patterns", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load scan module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCAN_MODULE = _load_scan_module()


def run_readiness_check(
    *,
    repo_root: Path,
    scope: str = "retained",
    targets: list[Path] | None = None,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    if targets:
        scan_report = SCAN_MODULE.scan_paths(repo_root=repo_root, targets=targets)
    else:
        scan_report = SCAN_MODULE.scan_scope(repo_root=repo_root, scope=scope)
    return {
        "ok": bool(scan_report["ok"]),
        "scope": scope,
        "repo_root": repo_root.as_posix(),
        "targets": scan_report["targets"],
        "scanned_file_count": scan_report["scanned_file_count"],
        "match_count": scan_report["match_count"],
        "matches": scan_report["matches"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether the repo is ready for v3 deletion gating.")
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
        help="Retained scope is the pre-deletion gate. Full is the post-deletion gate.",
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
    status = "READY" if report["ok"] else "BLOCKED"
    lines = [
        f"deletion_readiness={status}",
        f"scope={report['scope']}",
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
    targets = [Path(path) for path in args.path]
    report = run_readiness_check(
        repo_root=args.repo_root.resolve(),
        scope=args.scope,
        targets=targets or None,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_human_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
