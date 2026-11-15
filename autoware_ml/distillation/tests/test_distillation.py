"""Unit tests for the distillation attached to a model: targets, terms, loss and lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from autoware_ml.distillation.distillation import Distillation
from autoware_ml.distillation.taps import Tap
from autoware_ml.distillation.teacher import Teacher
from autoware_ml.distillation.term import DistillationTerm
from autoware_ml.distillation.tests.toy_models import (
    InMemoryTeacher,
    ToySegModel,
    ToySegModelWithGroups,
    build_gt_batch,
    toy_model_config,
    write_teacher_run,
)
from autoware_ml.losses.distillation.feature import FeatureMSELoss
from autoware_ml.losses.distillation.soft_target import SoftTargetKLDivLoss

LOGITS = "segmentation3d_head_outputs.logits"


def _logits_term(weight: float = 1.0) -> DistillationTerm:
    return DistillationTerm(
        name="logits", student=LOGITS, teacher=LOGITS, loss=SoftTargetKLDivLoss(), weight=weight
    )


def _hidden_term(adapter: nn.Module | None = None, weight: float = 1.0) -> DistillationTerm:
    return DistillationTerm(
        name="hidden",
        student="module:hidden",
        teacher="module:hidden",
        loss=FeatureMSELoss(),
        weight=weight,
        adapter=adapter,
    )


def _file_teacher(tmp_path: Path, model: ToySegModel, hidden_channels: int) -> Teacher:
    checkpoint = write_teacher_run(tmp_path / "run", model, toy_model_config(hidden_channels))
    return Teacher(weights=[checkpoint], autocast_dtype=None)


def _silence_logging(model: ToySegModel) -> dict[str, Any]:
    logged: dict[str, Any] = {}
    model.log_dict = lambda values, **kwargs: logged.update(values)
    return logged


def test_prepare_batch_attaches_the_teacher_outputs_and_features(tmp_path: Path) -> None:
    teacher_model = ToySegModel(hidden_channels=8)
    student = ToySegModel(
        distillation=Distillation(
            teacher=_file_teacher(tmp_path, teacher_model, 8),
            terms=[_logits_term(), _hidden_term(adapter=nn.Linear(6, 8))],
        )
    )
    student.setup("fit")
    batch = build_gt_batch()

    inputs = student.on_after_batch_transfer(batch, dataloader_idx=0)

    targets = inputs.distillation_targets
    assert targets is not None
    expected = teacher_model.eval()(inputs.replace(distillation_targets=None))
    assert torch.allclose(targets.outputs.segmentation3d().logits, expected.segmentation3d().logits)
    assert set(targets.features) == {"module:hidden"}
    assert targets.features["module:hidden"].shape == (24, 8)


def test_compute_adds_the_terms_and_redefines_the_loss(tmp_path: Path) -> None:
    student = ToySegModel(
        distillation=Distillation(
            teacher=_file_teacher(tmp_path, ToySegModel(hidden_channels=8), 8),
            terms=[_logits_term(weight=2.0), _hidden_term(adapter=nn.Linear(6, 8), weight=0.5)],
        )
    )
    student.setup("fit")
    _silence_logging(student)
    inputs = student.on_after_batch_transfer(build_gt_batch(), dataloader_idx=0)

    metrics, _ = student._core_step(inputs, step_prefix="train")

    assert set(metrics) >= {
        "loss",
        "gt_loss",
        "loss_ce",
        "distill/logits",
        "distill/hidden",
        "distill_loss",
    }
    assert torch.equal(metrics["gt_loss"], metrics["loss_ce"])
    assert torch.allclose(
        metrics["distill_loss"], 2.0 * metrics["distill/logits"] + 0.5 * metrics["distill/hidden"]
    )
    assert torch.allclose(metrics["loss"], metrics["gt_loss"] + metrics["distill_loss"])
    assert metrics["distill/logits"] > 0.0
    assert metrics["distill/hidden"] > 0.0


def test_the_loss_trains_the_student_and_the_adapter_but_not_the_teacher(tmp_path: Path) -> None:
    adapter = nn.Linear(6, 8)
    student = ToySegModel(
        distillation=Distillation(
            teacher=_file_teacher(tmp_path, ToySegModel(hidden_channels=8), 8),
            terms=[_logits_term(), _hidden_term(adapter=adapter)],
        )
    )
    student.setup("fit")
    _silence_logging(student)
    inputs = student.on_after_batch_transfer(build_gt_batch(), dataloader_idx=0)

    student.training_step(inputs, batch_idx=0).backward()

    assert student.hidden.weight.grad is not None
    assert adapter.weight.grad is not None
    teacher_model = student.distillation.teacher.model
    assert all(parameter.grad is None for parameter in teacher_model.parameters())


def test_a_teacher_equal_to_the_student_gives_zero_terms(tmp_path: Path) -> None:
    student = ToySegModel(hidden_channels=6)
    teacher_model = ToySegModel(hidden_channels=6)
    teacher_model.load_state_dict(student.state_dict())
    student.distillation = Distillation(
        teacher=_file_teacher(tmp_path, teacher_model, 6), terms=[_logits_term(), _hidden_term()]
    )
    student.setup("fit")
    _silence_logging(student)
    inputs = student.on_after_batch_transfer(build_gt_batch(), dataloader_idx=0)

    metrics, _ = student._core_step(inputs, step_prefix="train")

    assert torch.allclose(metrics["distill/logits"], torch.zeros(()), atol=1e-6)
    assert torch.allclose(metrics["distill/hidden"], torch.zeros(()), atol=1e-6)
    assert torch.allclose(metrics["loss"], metrics["gt_loss"])


def test_an_inactive_split_runs_no_teacher_and_adds_no_terms(tmp_path: Path) -> None:
    student = ToySegModel(
        distillation=Distillation(
            teacher=_file_teacher(tmp_path, ToySegModel(), 6), terms=[_logits_term()]
        )
    )
    student.setup("fit")
    _silence_logging(student)
    student.eval()

    inputs = student.on_after_batch_transfer(build_gt_batch(), dataloader_idx=0)
    metrics, _ = student._core_step(inputs, step_prefix="val")

    assert inputs.distillation_targets is None
    assert set(metrics) == {"loss", "loss_ce"}


def test_val_can_be_opted_in(tmp_path: Path) -> None:
    student = ToySegModel(
        distillation=Distillation(
            teacher=_file_teacher(tmp_path, ToySegModel(), 6),
            terms=[_logits_term()],
            stages=["train", "val"],
        )
    )
    student.setup("fit")
    student.eval()

    inputs = student.on_after_batch_transfer(build_gt_batch(), dataloader_idx=0)

    assert inputs.distillation_targets is not None


def test_the_teacher_stays_in_eval_mode_while_the_student_trains(tmp_path: Path) -> None:
    student = ToySegModel(
        distillation=Distillation(
            teacher=_file_teacher(tmp_path, ToySegModel(), 6), terms=[_logits_term()]
        )
    )
    student.setup("fit")

    student.train()

    assert student.training
    assert not student.distillation.teacher.model.training


def test_the_student_checkpoint_holds_the_adapters_but_no_teacher(tmp_path: Path) -> None:
    student = ToySegModel(
        distillation=Distillation(
            teacher=_file_teacher(tmp_path, ToySegModel(hidden_channels=8), 8),
            terms=[_hidden_term(adapter=nn.Linear(6, 8))],
        )
    )
    student.setup("fit")

    keys = set(student.state_dict())

    assert {"hidden.weight", "hidden.bias", "classifier.weight", "classifier.bias"} <= keys
    assert "distillation.terms.0.adapter.weight" in keys
    assert not any("teacher" in key for key in keys)


def test_adapters_join_the_default_group_of_a_model_without_explicit_groups(tmp_path: Path) -> None:
    adapter = nn.Linear(6, 8)
    student = ToySegModel(
        distillation=Distillation(
            teacher=_file_teacher(tmp_path, ToySegModel(hidden_channels=8), 8),
            terms=[_hidden_term(adapter=adapter)],
        ),
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
    )
    student.setup("fit")

    optimizer = student.configure_optimizers()

    assert len(optimizer.param_groups) == 1
    assert any(parameter is adapter.weight for parameter in optimizer.param_groups[0]["params"])


def test_adapters_form_their_own_group_next_to_explicit_model_groups(tmp_path: Path) -> None:
    adapter = nn.Linear(6, 8)
    student = ToySegModelWithGroups(
        distillation=Distillation(
            teacher=_file_teacher(tmp_path, ToySegModel(hidden_channels=8), 8),
            terms=[_hidden_term(adapter=adapter)],
        ),
        optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
        optimizer_group_overrides={"distillation": {"lr": 0.01}},
    )
    student.setup("fit")

    optimizer = student.configure_optimizers()

    assert len(optimizer.param_groups) == 3
    distillation_group = optimizer.param_groups[2]
    assert distillation_group["lr"] == 0.01
    assert [id(p) for p in distillation_group["params"]] == [id(adapter.weight), id(adapter.bias)]


def test_teardown_removes_the_hooks(tmp_path: Path) -> None:
    student = ToySegModel(
        distillation=Distillation(
            teacher=_file_teacher(tmp_path, ToySegModel(), 6), terms=[_hidden_term()]
        )
    )
    student.setup("fit")
    assert len(student.hidden._forward_hooks) == 1

    student.teardown("fit")

    assert len(student.hidden._forward_hooks) == 0


def test_the_in_memory_teacher_serves_models_without_a_config() -> None:
    teacher_model = ToySegModel(hidden_channels=6)
    student = ToySegModel(
        distillation=Distillation(teacher=InMemoryTeacher(teacher_model), terms=[_logits_term()])
    )
    student.setup("fit")

    inputs = student.on_after_batch_transfer(build_gt_batch(), dataloader_idx=0)

    assert inputs.distillation_targets is not None


class _PartsTerm(nn.Module):
    """A term made of two logged parts, bound to the student at setup."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "parts"
        self.weight = 2.0
        self.student_tap = Tap(LOGITS)
        self.teacher_tap = Tap(LOGITS)
        self.bound: nn.Module | None = None

    def bind(self, student: nn.Module) -> None:
        self.bound = student

    def forward(self, student_outputs, student_features, targets) -> dict[str, torch.Tensor]:
        student = self.student_tap.resolve(student_outputs, student_features)
        teacher = self.teacher_tap.resolve(targets.outputs, targets.features)
        return {"a": (student - teacher).abs().mean(), "b": torch.ones(())}


def test_a_term_may_return_parts_which_are_logged_and_summed(tmp_path: Path) -> None:
    term = _PartsTerm()
    student = ToySegModel(
        distillation=Distillation(teacher=_file_teacher(tmp_path, ToySegModel(), 6), terms=[term])
    )
    student.setup("fit")
    _silence_logging(student)
    inputs = student.on_after_batch_transfer(build_gt_batch(), dataloader_idx=0)

    metrics, _ = student._core_step(inputs, step_prefix="train")

    assert term.bound is student
    assert torch.allclose(
        metrics["distill/parts"], metrics["distill/parts/a"] + metrics["distill/parts/b"]
    )
    assert torch.allclose(metrics["distill_loss"], 2.0 * metrics["distill/parts"])


def test_terms_must_have_distinct_names() -> None:
    with pytest.raises(ValueError, match="share the name"):
        Distillation(teacher=InMemoryTeacher(ToySegModel()), terms=[_logits_term(), _logits_term()])


def test_stages_must_be_train_val_or_test() -> None:
    with pytest.raises(ValueError, match="stages must be among"):
        Distillation(
            teacher=InMemoryTeacher(ToySegModel()), terms=[_logits_term()], stages=["predict"]
        )


def test_a_term_names_a_shape_mismatch_and_the_remedy() -> None:
    student = ToySegModel(
        distillation=Distillation(
            teacher=InMemoryTeacher(ToySegModel(hidden_channels=8)), terms=[_hidden_term()]
        )
    )
    student.setup("fit")
    _silence_logging(student)
    inputs = student.on_after_batch_transfer(build_gt_batch(), dataloader_idx=0)

    with pytest.raises(ValueError, match="Add an adapter"):
        student._core_step(inputs, step_prefix="train")


def test_compute_without_targets_fails_loudly() -> None:
    distillation = Distillation(teacher=InMemoryTeacher(ToySegModel()), terms=[_logits_term()])
    inputs = ToySegModel().on_after_batch_transfer(build_gt_batch(), dataloader_idx=0)

    with pytest.raises(ValueError, match="no distillation targets"):
        distillation.compute(inputs, None, {"loss": torch.zeros(())})
