from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data_utils import (
    DataNotAvailableError,
    build_persistence_prediction,
    load_array_from_hdf5,
    summarize_array,
    write_prediction_hdf5,
    write_time_csv,
)
from model import TinyTemporalConvNet, rollout_model


def load_model_checkpoint(path: str | None, device: torch.device) -> TinyTemporalConvNet | None:
    if not path:
        return None
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device)
    model = TinyTemporalConvNet(width=int(payload.get("width", 32))).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def run_infer(args: argparse.Namespace) -> int:
    start = time.perf_counter()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_path.parent / f"task{args.task}_infer.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    log(
        json.dumps(
            {
                "event": "hypothesis",
                "task": args.task,
                "text": "Persistence rollout is a robust fallback; optional tiny checkpoint can be used only for same-task smoke inference.",
            }
        )
    )

    try:
        test_input, key = load_array_from_hdf5(args.test_hdf5, args.dataset_key)
    except DataNotAvailableError as exc:
        log(json.dumps({"event": "data_missing", "error": str(exc)}))
        return 2

    log(json.dumps({"event": "data_loaded", "dataset_key": key, "summary": summarize_array(test_input)}))
    inference_start = time.perf_counter()
    if args.checkpoint:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        model = load_model_checkpoint(args.checkpoint, device)
        assert model is not None
        batches = []
        with torch.no_grad():
            for i in range(0, test_input.shape[0], args.batch_size):
                context = torch.from_numpy(test_input[i : i + args.batch_size].astype(np.float32)).to(device)
                pred = rollout_model(model, context, total_steps=args.total_steps, context_steps=args.context_steps)
                batches.append(pred.cpu().numpy())
        prediction = np.concatenate(batches, axis=0).astype(np.float32)
        prediction[:, : args.context_steps, :] = test_input[:, : args.context_steps, :]
        mode = "checkpoint_rollout"
    else:
        prediction = build_persistence_prediction(
            test_input,
            total_steps=args.total_steps,
            context_steps=args.context_steps,
        )
        mode = "persistence"
    inference_time = time.perf_counter() - inference_start

    write_prediction_hdf5(out_path, prediction, dataset_key=args.output_key)
    if args.time_csv:
        write_time_csv(args.time_csv, train_time=args.train_time, inference_time=inference_time)
    elapsed = time.perf_counter() - start
    log(
        json.dumps(
            {
                "event": "conclusion",
                "task": args.task,
                "mode": mode,
                "prediction_shape": list(prediction.shape),
                "inference_time": inference_time,
                "elapsed_seconds": elapsed,
                "output": str(out_path),
            }
        )
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal inference for PDE neural operator rehearsal.")
    parser.add_argument("--task", type=int, choices=[1, 2], required=True)
    parser.add_argument("--test-hdf5", type=str, required=True)
    parser.add_argument("--dataset-key", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--output-key", type=str, default="pred")
    parser.add_argument("--time-csv", type=str, default=None)
    parser.add_argument("--train-time", type=float, default=0.0)
    parser.add_argument("--total-steps", type=int, default=200)
    parser.add_argument("--context-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(run_infer(build_arg_parser().parse_args()))
