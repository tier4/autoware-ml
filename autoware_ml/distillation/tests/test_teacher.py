"""Unit tests for the teacher: config discovery, build and the frozen forward."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.distillation.teacher import Teacher
from autoware_ml.distillation.tests.toy_models import (
    ToySegModel,
    build_gt_batch,
    toy_model_config,
    write_teacher_run,
)
from autoware_ml.utils.mlflow_helpers import find_run_config


def test_find_run_config_walks_up_from_the_checkpoint(tmp_path: Path) -> None:
    checkpoint = write_teacher_run(tmp_path / "run", ToySegModel(), toy_model_config(6))

    assert (
        find_run_config(checkpoint) == tmp_path / "run" / "artifacts" / "config" / "resolved.yaml"
    )
    assert find_run_config(tmp_path / "elsewhere" / "best.ckpt") is None


def test_teacher_is_rebuilt_from_the_run_config_and_frozen(tmp_path: Path) -> None:
    source = ToySegModel(hidden_channels=8)
    checkpoint = write_teacher_run(tmp_path / "run", source, toy_model_config(8))
    teacher = Teacher(weights=[checkpoint], autocast_dtype=None)

    with pytest.raises(RuntimeError, match="not been built"):
        teacher.model  # noqa: B018
    model = teacher.build(torch.device("cpu"))

    assert isinstance(model, ToySegModel)
    assert model.hidden.out_features == 8
    assert torch.equal(model.classifier.weight, source.classifier.weight)
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_teacher_drops_the_training_only_keys_of_its_config(tmp_path: Path) -> None:
    """A teacher that was itself distilled must not rebuild its own teacher."""
    config = toy_model_config(
        6,
        metrics=None,
        optimizer={"_target_": "torch.optim.AdamW", "_partial_": True, "lr": 1e-3},
        distillation={"_target_": "autoware_ml.no.such.Module"},
    )
    checkpoint = write_teacher_run(tmp_path / "run", ToySegModel(), config)

    model = Teacher(weights=[checkpoint]).build(torch.device("cpu"))

    assert model.distillation is None
    assert model.optimizer_partial is None


def test_teacher_needs_full_weight_coverage(tmp_path: Path) -> None:
    checkpoint = write_teacher_run(
        tmp_path / "run", ToySegModel(hidden_channels=6), toy_model_config(6)
    )
    state = torch.load(checkpoint, weights_only=True)
    del state["state_dict"]["classifier.bias"]
    torch.save(state, checkpoint)

    with pytest.raises(RuntimeError, match="not covered"):
        Teacher(weights=[checkpoint]).build(torch.device("cpu"))


def test_teacher_without_a_config_asks_for_one(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.ckpt"
    torch.save({"state_dict": {}}, checkpoint)

    with pytest.raises(FileNotFoundError, match="pass `config`"):
        Teacher(weights=[checkpoint])


def test_teacher_checks_its_checkpoints_up_front(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Teacher(weights=[tmp_path / "missing.ckpt"])


def test_a_teacher_without_weights_instantiates_but_cannot_build() -> None:
    """An experiment composes with teacher_weights unset; the run fails at fit setup."""
    teacher = Teacher(weights=None)

    with pytest.raises(ValueError, match="teacher_weights="):
        teacher.build(torch.device("cpu"))


def test_teacher_rejects_a_non_float_autocast_dtype(tmp_path: Path) -> None:
    checkpoint = write_teacher_run(tmp_path / "run", ToySegModel(), toy_model_config(6))

    with pytest.raises(ValueError, match="floating point"):
        Teacher(weights=[checkpoint], autocast_dtype="int32")


def test_teacher_run_returns_detached_outputs_on_the_student_inputs(tmp_path: Path) -> None:
    checkpoint = write_teacher_run(tmp_path / "run", ToySegModel(), toy_model_config(6))
    teacher = Teacher(weights=[checkpoint], autocast_dtype=None)
    teacher.build(torch.device("cpu"))
    batch = build_gt_batch()
    inputs = ModelBatchInputs.from_gt_batch(batch)

    outputs = teacher.run(batch, inputs)

    logits = outputs.segmentation3d().logits
    assert logits.shape == (24, 3)
    assert not logits.requires_grad
    assert torch.equal(logits, teacher.model(inputs).segmentation3d().logits)


def test_teacher_can_preprocess_the_batch_itself(tmp_path: Path) -> None:
    checkpoint = write_teacher_run(tmp_path / "run", ToySegModel(), toy_model_config(6))
    teacher = Teacher(weights=[checkpoint], autocast_dtype=None, share_student_inputs=False)
    teacher.build(torch.device("cpu"))
    batch = build_gt_batch()

    outputs = teacher.run(batch, ModelBatchInputs.from_gt_batch(batch))

    assert outputs.segmentation3d().logits.shape == (24, 3)
