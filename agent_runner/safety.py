from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SafetyCheck:
    ok: bool
    resolved_path: Path | None = None
    error: str | None = None


class WorkspaceSafety:
    _SENSITIVE_NAMES = {
        ".env",
        ".env.local",
        "secrets.env",
    }
    _SENSITIVE_KEYWORDS = ("api_key", "apikey", "token", "secret", "credential")
    _FORBIDDEN_COMMAND_FRAGMENTS = (
        "rm -rf /",
        "mkfs",
        "shutdown",
        "reboot",
        "curl | bash",
        "curl|bash",
        "wget ",
        "curl ",
        "generate_data",
        "numerical_solver",
        "finite_difference_solver",
        "burgers_solver",
        "simulate_extra",
        "download_dataset",
    )

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        allowed_write_roots: tuple[str | Path, ...] | list[str | Path] | None = None,
        extra_read_roots: tuple[str | Path, ...] | list[str | Path] | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        default_write_roots = (
            self.workspace_root / "submission",
            self.workspace_root / "submission" / "code",
            self.workspace_root / "runs",
        )
        self.allowed_write_roots = tuple(Path(root).resolve() for root in (allowed_write_roots or default_write_roots))
        self.extra_read_roots = tuple(Path(root).resolve() for root in (extra_read_roots or ()))

    def resolve_path(self, path: str | Path) -> SafetyCheck:
        candidate = Path(path)
        allowed_roots = (self.workspace_root, *self.extra_read_roots)
        if candidate.is_absolute():
            resolved_candidates = [candidate.resolve()]
        else:
            resolved_candidates = [(self.workspace_root / candidate).resolve()]
            for root in self.extra_read_roots:
                resolved_candidates.append((root / candidate).resolve())
                if candidate.parts and candidate.parts[0] == root.name:
                    resolved_candidates.append((root.parent / candidate).resolve())

        allowed_candidates = [
            resolved for resolved in resolved_candidates if any(self._is_relative_to(resolved, root) for root in allowed_roots)
        ]
        if not allowed_candidates:
            return SafetyCheck(ok=False, error=f"Path is outside workspace: {path}")

        existing_candidates = [resolved for resolved in allowed_candidates if resolved.exists()]
        if existing_candidates:
            return SafetyCheck(ok=True, resolved_path=existing_candidates[0])

        return SafetyCheck(ok=True, resolved_path=allowed_candidates[0])

    def validate_read_path(self, path: str | Path) -> SafetyCheck:
        check = self.resolve_path(path)
        if not check.ok:
            return check
        assert check.resolved_path is not None
        name = check.resolved_path.name.lower()
        path_text = str(check.resolved_path).lower()
        if name in self._SENSITIVE_NAMES or any(keyword in path_text for keyword in self._SENSITIVE_KEYWORDS):
            return SafetyCheck(ok=False, error=f"Reading sensitive path is not allowed: {path}")
        return check

    def validate_write_path(self, path: str | Path) -> SafetyCheck:
        check = self.resolve_path(path)
        if not check.ok:
            return check
        assert check.resolved_path is not None
        for root in self.allowed_write_roots:
            if self._is_relative_to(check.resolved_path, root):
                return check
        return SafetyCheck(ok=False, error=f"Writing to path is not allowed: {path}")

    def validate_cwd(self, cwd: str | Path | None) -> SafetyCheck:
        path = self.workspace_root if cwd is None else cwd
        candidate = Path(path)
        resolved = candidate.resolve() if candidate.is_absolute() else (self.workspace_root / candidate).resolve()
        if not self._is_relative_to(resolved, self.workspace_root):
            return SafetyCheck(ok=False, error=f"cwd is outside workspace: {path}")
        return SafetyCheck(ok=True, resolved_path=resolved)

    def validate_shell_command(self, command: str) -> SafetyCheck:
        normalized = " ".join(command.lower().split())
        for fragment in self._FORBIDDEN_COMMAND_FRAGMENTS:
            if fragment in normalized:
                return SafetyCheck(ok=False, error=f"Command contains forbidden fragment: {fragment}")
        return SafetyCheck(ok=True)

    @staticmethod
    def _is_relative_to(path: Path, base: Path) -> bool:
        try:
            path.relative_to(base)
            return True
        except ValueError:
            return False
