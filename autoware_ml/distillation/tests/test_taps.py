"""Unit tests for the taps: output paths, module hooks and what they capture."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.distillation.taps import (
    CaptureStore,
    FeatureCapture,
    Tap,
    register_taps,
    select_tensor,
    walk_path,
)
from autoware_ml.utils.point_cloud.structures import Point


def _outputs() -> ModelOutputs:
    return ModelOutputs(
        segmentation3d_head_outputs=Segmentation3DHeadOutputs(logits=torch.arange(6.0).view(3, 2))
    )


def test_output_tap_reads_the_tensor_at_the_attribute_path() -> None:
    tap = Tap("segmentation3d_head_outputs.logits")

    assert not tap.is_module
    assert torch.equal(tap.resolve(_outputs(), {}), torch.arange(6.0).view(3, 2))


def test_output_tap_rejects_a_path_to_a_non_tensor() -> None:
    with pytest.raises(ValueError, match="not a tensor"):
        Tap("segmentation3d_head_outputs").resolve(_outputs(), {})


def test_output_tap_names_the_missing_part() -> None:
    with pytest.raises(ValueError, match="no 'probs'"):
        Tap("segmentation3d_head_outputs.probs").resolve(_outputs(), {})


def test_output_tap_rejects_a_path_through_an_unset_task() -> None:
    with pytest.raises(ValueError, match="crosses None"):
        Tap("detection3d_head_outputs.dense_heatmap").resolve(_outputs(), {})


def test_walk_path_indexes_mappings_and_sequences() -> None:
    value = {"stages": [torch.zeros(1), torch.ones(1)]}

    assert torch.equal(walk_path(value, "stages.1"), torch.ones(1))


def test_module_tap_parses_path_and_selector() -> None:
    tap = Tap("module:encoder.enc.enc3@feat")

    assert tap.is_module
    assert tap.module_path == "encoder.enc.enc3"
    assert tap.selector == "feat"


@pytest.mark.parametrize("spec", ["", "module:", "module:@feat"])
def test_tap_rejects_empty_specs(spec: str) -> None:
    with pytest.raises(ValueError):
        Tap(spec)


def test_module_tap_reads_the_captured_feature() -> None:
    tap = Tap("module:hidden")

    assert torch.equal(tap.resolve(None, {"module:hidden": torch.ones(2)}), torch.ones(2))


def test_module_tap_without_a_capture_fails_loudly() -> None:
    with pytest.raises(ValueError, match="captured no output"):
        Tap("module:hidden").resolve(None, {})


def test_select_tensor_takes_feat_of_a_point_by_default() -> None:
    point = Point(feat=torch.ones(2), coord=torch.zeros(2))

    assert torch.equal(select_tensor(point, None), torch.ones(2))
    assert torch.equal(select_tensor(point, "coord"), torch.zeros(2))


def test_select_tensor_needs_an_index_for_a_tuple() -> None:
    with pytest.raises(ValueError, match="index"):
        select_tensor((torch.ones(1), torch.zeros(1)), None)
    assert torch.equal(select_tensor((torch.ones(1), torch.zeros(1)), "1"), torch.zeros(1))


def test_select_tensor_rejects_a_selector_on_a_tensor() -> None:
    with pytest.raises(ValueError, match="returns a tensor"):
        select_tensor(torch.ones(1), "feat")


def test_capture_clones_so_a_later_in_place_write_does_not_reach_it() -> None:
    """PTv3 unpooling overwrites parent features in place after they were returned."""
    store = CaptureStore()
    store.enabled = True
    module = nn.Identity()
    module.register_forward_hook(FeatureCapture(store, "module:identity", None))
    output = module(torch.ones(3))

    output.mul_(0.0)

    assert torch.equal(store.tensors["module:identity"], torch.ones(3))


def test_capture_keeps_the_autograd_graph_of_the_student() -> None:
    store = CaptureStore()
    store.enabled = True
    module = nn.Linear(2, 2)
    module.register_forward_hook(FeatureCapture(store, "module:linear", None))

    module(torch.ones(1, 2))

    assert store.tensors["module:linear"].requires_grad


def test_disabled_store_captures_nothing() -> None:
    store = CaptureStore()
    module = nn.Identity()
    module.register_forward_hook(FeatureCapture(store, "module:identity", None))

    module(torch.ones(3))

    assert store.tensors == {}


def test_deep_copy_of_a_hooked_module_shares_the_hook() -> None:
    """Export deep-copies heads; the copy must not drag the capture store along."""
    store = CaptureStore()
    module = nn.Identity()
    hook = FeatureCapture(store, "module:identity", None)
    module.register_forward_hook(hook)

    copied = deepcopy(module)

    assert list(copied._forward_hooks.values()) == [hook]


def test_register_taps_hooks_each_module_once_and_skips_output_taps() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    store = CaptureStore()
    store.enabled = True
    taps = [Tap("module:0"), Tap("module:0"), Tap("segmentation3d_head_outputs.logits")]

    handles = register_taps(model, taps, store)
    model(torch.ones(1, 2))

    assert len(handles) == 1
    assert set(store.tensors) == {"module:0"}


def test_register_taps_names_a_missing_module() -> None:
    with pytest.raises(ValueError, match="no submodule 'missing'"):
        register_taps(nn.Linear(2, 2), [Tap("module:missing")], CaptureStore())
