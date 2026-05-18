from __future__ import annotations

import sys
from subprocess import CompletedProcess

import pytest

from agent_runner import dependency_bootstrap


def test_ensure_dependency_installs_missing_package_with_current_python(monkeypatch):
    installed = {"value": False}
    commands: list[tuple[list[str], int]] = []

    def fake_module_available(module_name: str) -> bool:
        assert module_name == "pypdf"
        return installed["value"]

    def fake_run(command, *, capture_output, text, timeout, check, env):  # noqa: ANN001
        assert capture_output is True
        assert text is True
        assert check is False
        assert env["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"
        commands.append((list(command), timeout))
        installed["value"] = True
        return CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(dependency_bootstrap, "_module_available", fake_module_available)
    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    dependency_bootstrap.ensure_dependency(
        dependency_bootstrap.DependencySpec(
            module_name="pypdf",
            package_spec="pypdf>=4.0",
            install_timeout_seconds=90,
        )
    )

    assert commands == [([sys.executable, "-m", "pip", "install", "pypdf>=4.0"], 90)]


def test_ensure_dependency_skips_install_when_module_exists(monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "_module_available", lambda module_name: module_name == "openai")

    def fail_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("subprocess.run should not be called when dependency already exists")

    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fail_run)

    dependency_bootstrap.ensure_dependency(
        dependency_bootstrap.DependencySpec(
            module_name="openai",
            package_spec="openai>=2.0.0",
        )
    )


def test_ensure_mode_dependencies_installs_expected_modules(monkeypatch):
    installed_specs: list[str] = []

    def fake_ensure_dependency(spec):  # noqa: ANN001
        installed_specs.append(spec.module_name)

    monkeypatch.setattr(dependency_bootstrap, "ensure_dependency", fake_ensure_dependency)

    dependency_bootstrap.ensure_mode_dependencies("autonomous")

    assert installed_specs == ["openai", "pypdf", "torch"]


def test_package_final_mode_does_not_install_optional_dependencies(monkeypatch):
    installed_specs: list[str] = []

    def fake_ensure_dependency(spec):  # noqa: ANN001
        installed_specs.append(spec.module_name)

    monkeypatch.setattr(dependency_bootstrap, "ensure_dependency", fake_ensure_dependency)

    dependency_bootstrap.ensure_mode_dependencies("package_final")

    assert installed_specs == []
