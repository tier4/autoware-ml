"""The classification rows of a StreamPETR checkpoint follow the class order of its taxonomy.

Stage 1 of the three-stage flow trains the nuScenes taxonomy, stages 2 and 3 the dynamic T4
taxonomy: the same seven classes with the last two swapped. ``--weights`` loads by shape, so
the final classification rows of the decoder (``cls_branches``) and of the auxiliary 2D head
(``img_roi_head.cls``) are permuted into the target order before the next stage loads them.
``permute_class_rows`` is the snippet of docs/models/streampetr.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from autoware_ml.models.detection3d.tests.test_streampetr import (
    IMAGE_HW,
    NUM_CAMS,
    _build_batch_inputs,
    _build_focal_head,
    _build_model,
)

NUSCENES_CLASSES = ["car", "truck", "bus", "bicycle", "pedestrian", "traffic_cone", "barrier"]
T4_CLASSES = ["car", "truck", "bus", "bicycle", "pedestrian", "barrier", "traffic_cone"]


def permute_class_rows(
    state_dict: dict[str, torch.Tensor], source: Sequence[str], target: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Reorder the per-class rows of the classification layers from ``source`` to ``target``.

    The rows of the final Linear of every decoder classification branch (one row per class)
    and of the 1x1 classification conv of the auxiliary 2D head are gathered in the order
    ``target`` lists the classes of ``source``. Every other tensor is untouched.
    """
    rows = [source.index(name) for name in target]
    permuted = dict(state_dict)
    for key, value in state_dict.items():
        is_decoder_cls = key.startswith("bbox_head.cls_branches.") and value.shape[0] == len(rows)
        if is_decoder_cls or key.startswith("img_roi_head.cls."):
            permuted[key] = value[rows].clone()
    return permuted


def _class_logits(model, batch_inputs) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        outputs = model(batch_inputs)
        features = model.image_feature_extractor(batch_inputs.image_data.images)
        auxiliary = model.img_roi_head(features, image_height=IMAGE_HW[0], image_width=IMAGE_HW[1])
    return outputs.detection3d().streampetr_head().all_cls_scores, auxiliary.cls_scores


def test_permuted_rows_move_the_class_logits_with_their_class() -> None:
    """
    Input: a StreamPETR model with the auxiliary 2D head on the nuScenes classes, with random
    classification biases, and a model of the same shapes on the T4 classes, the same seven
    classes with the last two swapped.
    Expected: the raw checkpoint loads without an error and scores the traffic cones in the
    barrier column. Permuted by ``permute_class_rows``, the snippet of
    docs/models/streampetr.md, the logits of every class move with their class, and only the
    final classification layers of the decoder and of the 2D head change.
    Check: the logits of the raw load against the source ones, the decoder and 2D logits of the
    permuted load against the source ones reordered, the two swapped columns, and the set of
    checkpoint keys that changed.
    """
    torch.manual_seed(0)
    source = _build_model(
        img_roi_head=_build_focal_head(NUSCENES_CLASSES), class_names=NUSCENES_CLASSES
    )
    target = _build_model(img_roi_head=_build_focal_head(T4_CLASSES), class_names=T4_CLASSES)
    batch_inputs = _build_batch_inputs(torch.randn(2, NUM_CAMS, 3, *IMAGE_HW))
    rows = [NUSCENES_CLASSES.index(name) for name in T4_CLASSES]
    assert rows == [0, 1, 2, 3, 4, 6, 5]
    # The classification biases start at one constant per class; a trained checkpoint tells
    # the classes apart in the biases too.
    torch.nn.init.normal_(source.bbox_head.cls_branches[0][-1].bias)
    torch.nn.init.normal_(source.img_roi_head.cls.bias)

    # Both models have the same shapes, so the raw checkpoint loads silently: the target then
    # scores the logits of traffic_cone (source column 5) in its barrier column (column 5).
    checkpoint = {"state_dict": source.state_dict()}
    target.load_state_dict(checkpoint["state_dict"], strict=True)
    source_decoder, source_auxiliary = _class_logits(source, batch_inputs)
    naive_decoder, _ = _class_logits(target, batch_inputs)
    assert torch.equal(naive_decoder, source_decoder)

    permuted = permute_class_rows(checkpoint["state_dict"], NUSCENES_CLASSES, T4_CLASSES)
    target.load_state_dict(permuted, strict=True)
    decoder, auxiliary = _class_logits(target, batch_inputs)

    # The logits of every class moved with it: the two swapped columns and the rest alike.
    assert torch.equal(decoder, source_decoder[..., rows])
    assert torch.equal(auxiliary, source_auxiliary[..., rows])
    for name in ("barrier", "traffic_cone"):
        assert torch.equal(
            decoder[..., T4_CLASSES.index(name)], source_decoder[..., NUSCENES_CLASSES.index(name)]
        )
    # Only the final Linear of every decoder classification branch (weight and bias, one
    # branch per decoder layer) and the 1x1 classification conv of the 2D head changed.
    changed = {
        key for key in permuted if not torch.equal(permuted[key], checkpoint["state_dict"][key])
    }
    num_layers = len(target.bbox_head.cls_branches)
    assert changed == {
        f"bbox_head.cls_branches.{layer}.{len(target.bbox_head.cls_branches[layer]) - 1}.{kind}"
        for layer in range(num_layers)
        for kind in ("weight", "bias")
    } | {"img_roi_head.cls.weight", "img_roi_head.cls.bias"}
