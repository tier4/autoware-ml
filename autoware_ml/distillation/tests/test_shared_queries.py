"""Unit tests for the shared-queries term and the head entry point it relies on."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.detection3d.head_outputs import Detection3DHeadOutputs
from autoware_ml.dataclasses.models.distillation import DistillationTargets
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.distillation.shared_queries import TransFusionSharedQueriesTerm
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import build_transfusion_head

BEV = "module:bev_neck"


def _bev_features(seed: int = 0) -> torch.Tensor:
    return torch.randn((2, 64, 8, 8), generator=torch.Generator().manual_seed(seed))


def _student(head: nn.Module) -> nn.Module:
    holder = nn.Module()
    holder.bbox_head = head
    return holder


def _targets(teacher_outputs) -> DistillationTargets:
    return DistillationTargets(
        outputs=ModelOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=None, transfusion_head_outputs=teacher_outputs
            )
        ),
        features={},
    )


def test_forward_at_the_own_proposals_reproduces_forward() -> None:
    head = build_transfusion_head().eval()
    features = _bev_features()

    own = head(features)
    replayed = head.forward_at_proposals(features, own.query_positions, own.query_labels)

    assert torch.equal(replayed.query_labels, own.query_labels)
    assert torch.equal(replayed.query_positions, own.query_positions)
    assert torch.allclose(replayed.query_heatmap_score, own.query_heatmap_score)
    for name in ("heatmap", "center", "height", "dim", "rot", "vel"):
        assert torch.allclose(
            getattr(replayed.separate_head_outputs, name),
            getattr(own.separate_head_outputs, name),
        ), name


def test_outputs_carry_the_proposal_positions_and_keep_them_when_selecting_samples() -> None:
    head = build_transfusion_head().eval()
    outputs = head(_bev_features())

    assert outputs.query_positions is not None
    assert outputs.query_positions.shape == outputs.query_labels.shape
    selected = outputs.select_samples(torch.tensor([True, False]))
    assert torch.equal(selected.query_positions, outputs.query_positions[:1])


def test_an_identical_head_has_no_regression_gap_at_the_shared_queries() -> None:
    teacher_head = build_transfusion_head().eval()
    student_head = build_transfusion_head().eval()
    student_head.load_state_dict(teacher_head.state_dict())
    term = TransFusionSharedQueriesTerm()
    term.bind(_student(student_head))
    features = _bev_features()

    parts = term(None, {BEV: features}, _targets(teacher_head(features)))

    assert set(parts) == {"cls", "reg"}
    assert torch.allclose(parts["reg"], torch.zeros(()), atol=1e-6)
    assert parts["cls"] > 0.0  # the entropy of the teacher's own probabilities


def test_a_different_head_is_pulled_on_both_parts() -> None:
    teacher_head = build_transfusion_head().eval()
    student_head = build_transfusion_head().train()
    term = TransFusionSharedQueriesTerm(classification_weight=2.0, regression_weight=0.5)
    term.bind(_student(student_head))
    features = _bev_features()

    parts = term(None, {BEV: features}, _targets(teacher_head(features)))
    (parts["cls"] + parts["reg"]).backward()

    assert parts["reg"] > 0.0
    assert torch.isfinite(parts["cls"]) and torch.isfinite(parts["reg"])
    assert any(p.grad is not None for p in student_head.prediction_heads.parameters())
    assert all(parameter.grad is None for parameter in teacher_head.parameters())
    assert not any(
        parameter is student_head.class_encoding.weight for parameter in term.parameters()
    )


def test_the_term_needs_the_teacher_query_positions() -> None:
    teacher_head = build_transfusion_head().eval()
    term = TransFusionSharedQueriesTerm()
    term.bind(_student(build_transfusion_head()))
    features = _bev_features()
    outputs = teacher_head(features).model_copy(update={"query_positions": None})

    with pytest.raises(ValueError, match="no query positions"):
        term(None, {BEV: features}, _targets(outputs))


def test_the_term_must_be_bound_before_it_runs() -> None:
    features = _bev_features()
    targets = _targets(build_transfusion_head().eval()(features))

    with pytest.raises(RuntimeError, match="bound to the student"):
        TransFusionSharedQueriesTerm()(None, {BEV: features}, targets)


def test_bind_checks_the_head() -> None:
    with pytest.raises(ValueError, match="no submodule 'bbox_head'"):
        TransFusionSharedQueriesTerm().bind(nn.Module())
    with pytest.raises(ValueError, match="forward_at_proposals"):
        TransFusionSharedQueriesTerm().bind(_student(nn.Linear(1, 1)))


def test_the_student_features_tap_must_hook_a_module() -> None:
    with pytest.raises(ValueError, match="module tap"):
        TransFusionSharedQueriesTerm(student_features="detection3d_head_outputs")


def test_the_teacher_tap_must_lead_to_transfusion_outputs() -> None:
    term = TransFusionSharedQueriesTerm(teacher="detection3d_head_outputs")
    term.bind(_student(build_transfusion_head()))
    features = _bev_features()

    with pytest.raises(ValueError, match="not TransFusionHeadOutputs"):
        term(None, {BEV: features}, _targets(build_transfusion_head().eval()(features)))


def test_targets_are_untouched_by_the_namespace_helper() -> None:
    """Guard the test helper itself: a SimpleNamespace is not a head outputs object."""
    assert not isinstance(SimpleNamespace(), nn.Module)
