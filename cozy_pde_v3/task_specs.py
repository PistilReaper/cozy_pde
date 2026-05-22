from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    equation: str
    input_steps: int
    output_steps: int
    total_steps: int
    spatial_points: int
    pred_shape: tuple[int, int, int]
    first_steps_must_match: int
    inference_time_limit_sec: float
    must_train_from_scratch: bool
    allow_public_pretrained_weights: bool
    default_train_filenames: tuple[str, ...]
    default_validation_filenames: tuple[str, ...]
    default_test_filenames: tuple[str, ...]


DEFAULT_TASK_SPECS: dict[str, TaskSpec] = {
    "task1": TaskSpec(
        task_id="task1",
        equation="Burgers",
        input_steps=10,
        output_steps=200,
        total_steps=200,
        spatial_points=256,
        pred_shape=(0, 200, 256),
        first_steps_must_match=10,
        inference_time_limit_sec=120.0,
        must_train_from_scratch=False,
        allow_public_pretrained_weights=True,
        default_train_filenames=("train.hdf5", "train.h5"),
        default_validation_filenames=("validation.hdf5", "validation.h5", "val.hdf5", "val.h5"),
        default_test_filenames=("test.hdf5", "test.h5"),
    ),
    "task2": TaskSpec(
        task_id="task2",
        equation="Advection",
        input_steps=10,
        output_steps=200,
        total_steps=200,
        spatial_points=256,
        pred_shape=(0, 200, 256),
        first_steps_must_match=10,
        inference_time_limit_sec=120.0,
        must_train_from_scratch=True,
        allow_public_pretrained_weights=True,
        default_train_filenames=("train.hdf5", "train.h5"),
        default_validation_filenames=("validation.hdf5", "validation.h5", "val.hdf5", "val.h5"),
        default_test_filenames=("test.hdf5", "test.h5"),
    ),
    "task3": TaskSpec(
        task_id="task3",
        equation="Kuramoto-Sivashinsky",
        input_steps=20,
        output_steps=400,
        total_steps=400,
        spatial_points=256,
        pred_shape=(1000, 400, 256),
        first_steps_must_match=20,
        inference_time_limit_sec=120.0,
        must_train_from_scratch=True,
        allow_public_pretrained_weights=False,
        default_train_filenames=("train.hdf5", "train.h5"),
        default_validation_filenames=("validation.hdf5", "validation.h5", "val.hdf5", "val.h5"),
        default_test_filenames=("test.hdf5", "test.h5"),
    ),
}

TASK_IDS: tuple[str, ...] = tuple(DEFAULT_TASK_SPECS)
