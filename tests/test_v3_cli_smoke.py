from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import h5py
import numpy as np

from cozy_pde_v3.validation.submission import run_submission_cli_smoke


def _write_test_hdf5(path: Path, *, samples: int, total_steps: int) -> None:
    tensor = np.arange(samples * total_steps * 256, dtype=np.float32).reshape(samples, total_steps, 256)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("tensor", data=tensor)


def _write_cli_scripts(workspace: Path) -> None:
    train_path = workspace / "submission" / "code" / "train.py"
    train_path.write_text(
        dedent(
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
        encoding="utf-8",
    )
    infer_path = workspace / "submission" / "code" / "infer.py"
    infer_path.write_text(
        dedent(
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
                return datasets[0]


            def main() -> int:
                parser = argparse.ArgumentParser()
                parser.add_argument("--task", required=True)
                parser.add_argument("--config", required=True)
                parser.add_argument("--data_dir", required=True)
                parser.add_argument("--output_dir", required=True)
                parser.add_argument("--output", required=True)
                args = parser.parse_args()
                output_dir = Path(args.output_dir)
                checkpoint_path = output_dir / "checkpoint.pt"
                json.loads(checkpoint_path.read_text(encoding="utf-8"))
                with h5py.File(Path(args.data_dir) / f"{args.task}_test.hdf5", "r") as source:
                    tensor = np.asarray(_first_dataset(source)[...])
                pred = tensor.copy()
                pred[:, _INPUT_STEPS[args.task] :, :] = pred[:, _INPUT_STEPS[args.task] :, :] + np.float32(0.01)
                with h5py.File(args.output, "w") as target:
                    target.create_dataset("pred", data=pred)
                return 0


            if __name__ == "__main__":
                raise SystemExit(main())
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_run_submission_cli_smoke_executes_real_subprocess_checks(workspace: Path) -> None:
    _write_cli_scripts(workspace)
    _write_test_hdf5(workspace / "data" / "task1_test.hdf5", samples=2, total_steps=200)
    _write_test_hdf5(workspace / "data" / "task3_test.hdf5", samples=1000, total_steps=400)

    result = run_submission_cli_smoke(
        workspace_root=workspace,
        submission_dir=workspace / "submission",
        tasks=["task1", "task3"],
    )

    assert result["ok"] is True
    assert result["status"] == {"task1": True, "task3": True}
    assert result["details"]["task1"] == {
        "cli_parse_ok": True,
        "train_smoke_ok": True,
        "infer_smoke_ok": True,
        "checkpoint_load_ok": True,
    }
    assert result["details"]["task3"] == {
        "cli_parse_ok": True,
        "train_smoke_ok": True,
        "infer_smoke_ok": True,
        "checkpoint_load_ok": True,
    }
