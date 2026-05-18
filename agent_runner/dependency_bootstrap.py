from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class DependencySpec:
    module_name: str
    package_spec: str
    install_timeout_seconds: int = 120


CORE_BOOTSTRAP_DEPENDENCIES: tuple[DependencySpec, ...] = (
    DependencySpec(module_name="yaml", package_spec="PyYAML>=6.0"),
    DependencySpec(module_name="h5py", package_spec="h5py>=3.10"),
    DependencySpec(module_name="numpy", package_spec="numpy>=1.26"),
)

MODE_DEPENDENCIES: dict[str, tuple[DependencySpec, ...]] = {
    "preflight": (
        DependencySpec(module_name="openai", package_spec="openai>=2.0.0"),
        DependencySpec(module_name="torch", package_spec="torch", install_timeout_seconds=240),
    ),
    "test_tool_loop": (DependencySpec(module_name="openai", package_spec="openai>=2.0.0"),),
    "provider_health_check": (DependencySpec(module_name="openai", package_spec="openai>=2.0.0"),),
    "local_research_check": (DependencySpec(module_name="pypdf", package_spec="pypdf>=4.0"),),
    "autonomous": (
        DependencySpec(module_name="openai", package_spec="openai>=2.0.0"),
        DependencySpec(module_name="pypdf", package_spec="pypdf>=4.0"),
        DependencySpec(module_name="torch", package_spec="torch", install_timeout_seconds=240),
    ),
    "autonomous_dry_run": (
        DependencySpec(module_name="openai", package_spec="openai>=2.0.0"),
        DependencySpec(module_name="pypdf", package_spec="pypdf>=4.0"),
    ),
    "autonomous_rehearsal": (
        DependencySpec(module_name="openai", package_spec="openai>=2.0.0"),
        DependencySpec(module_name="pypdf", package_spec="pypdf>=4.0"),
        DependencySpec(module_name="torch", package_spec="torch", install_timeout_seconds=240),
    ),
    "export_task_logs": (),
    "package_final": (),
    "final_check": (),
    "readiness_check": (
        DependencySpec(module_name="openai", package_spec="openai>=2.0.0"),
        DependencySpec(module_name="pypdf", package_spec="pypdf>=4.0"),
        DependencySpec(module_name="torch", package_spec="torch", install_timeout_seconds=240),
    ),
}


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _pip_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    return env


def ensure_dependency(spec: DependencySpec) -> None:
    if _module_available(spec.module_name):
        return
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", spec.package_spec],
            capture_output=True,
            text=True,
            timeout=spec.install_timeout_seconds,
            check=False,
            env=_pip_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out after {spec.install_timeout_seconds} seconds while installing {spec.package_spec!r}"
        ) from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()[-2000:]
        stdout = completed.stdout.strip()[-2000:]
        detail = stderr or stdout or f"pip exited with code {completed.returncode}"
        raise RuntimeError(f"Failed to install {spec.package_spec!r}: {detail}")
    importlib.invalidate_caches()
    if not _module_available(spec.module_name):
        raise RuntimeError(
            f"Installed {spec.package_spec!r}, but module {spec.module_name!r} is still unavailable in {sys.executable}"
        )


def ensure_main_bootstrap_dependencies() -> None:
    for spec in CORE_BOOTSTRAP_DEPENDENCIES:
        ensure_dependency(spec)


def ensure_mode_dependencies(mode: str) -> None:
    for spec in MODE_DEPENDENCIES.get(mode, ()):
        ensure_dependency(spec)
