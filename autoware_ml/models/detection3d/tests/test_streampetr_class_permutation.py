# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The class row permutation of docs/models/streampetr.md that carries stage 1 into the T4 order.

The nuScenes and the dynamic T4 taxonomy list the same seven classes with the last two swapped,
and ``--weights`` loads by shape, so the per-class rows of a stage-1 checkpoint are permuted
before stage 2 loads it.
"""

from __future__ import annotations

from collections.abc import Sequence
import unittest

from jaxtyping import Float, Shaped
import torch

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.detection3d.streampetr import StreamPETRDetection3DModel
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
    state_dict: dict[str, Shaped[torch.Tensor, "*shape"]],
    source: Sequence[str],
    target: Sequence[str],
) -> dict[str, Shaped[torch.Tensor, "*shape"]]:
    """Reorder the per-class rows of the classification layers from ``source`` to ``target``.

    Args:
        state_dict: The checkpoint state dict of a model trained on ``source``.
        source: The class names in the order the checkpoint was trained on.
        target: The same class names in the order of the next stage.

    Returns:
        A copy of ``state_dict`` whose final decoder classification Linear layers and 2D head
        classification and heatmap convs list their rows in the ``target`` order.
    """
    rows = [source.index(name) for name in target]
    permuted = dict(state_dict)
    for key, value in state_dict.items():
        is_decoder_cls = key.startswith("bbox_head.cls_branches.") and value.shape[0] == len(rows)
        if is_decoder_cls or key.startswith(
            ("auxiliary_2d_head.cls.", "auxiliary_2d_head.heatmap.")
        ):
            permuted[key] = value[rows].clone()
    return permuted


class TestStreamPETRClassPermutation(unittest.TestCase):
    """The permuted checkpoint scores every class in its new column."""

    def _class_logits(
        self, model: StreamPETRDetection3DModel, batch_inputs: ModelBatchInputs
    ) -> tuple[
        Float[torch.Tensor, "num_layers batch_size num_queries num_classes"],
        Float[torch.Tensor, "num_images num_tokens num_classes"],
        Float[torch.Tensor, "num_images num_tokens num_classes"],
    ]:
        """Run the model in evaluation and return its per-class logits.

        Args:
            model: The StreamPETR model with the auxiliary 2D head.
            batch_inputs: The model inputs of one batch.

        Returns:
            The decoder class logits, and the auxiliary 2D class and heatmap logits.
        """
        model.eval()
        with torch.no_grad():
            outputs = model(batch_inputs)
            features = model.image_feature_extractor(batch_inputs.image_data.images)
            auxiliary = model.auxiliary_2d_head(
                features, image_height=IMAGE_HW[0], image_width=IMAGE_HW[1]
            )
        return (
            outputs.detection3d().streampetr_head().all_cls_scores,
            auxiliary.cls_scores,
            auxiliary.heatmap,
        )

    def test_permuted_rows_move_the_class_logits_with_their_class(self) -> None:
        """Test that the permuted checkpoint moves the logits of every class with the class.

        Input: a StreamPETR model with the auxiliary 2D head on the nuScenes classes, with
        random classification biases, and a model of the same shapes on the T4 classes, the
        same seven classes with the last two swapped.
        Expected: the raw checkpoint loads without an error and scores the traffic cones in the
        barrier column. Permuted by ``permute_class_rows``, the snippet of
        docs/models/streampetr.md, the logits of every class move with their class, and only
        the final classification layers of the decoder and the classification and heatmap
        convs of the 2D head change.
        Check: the logits of the raw load against the source ones, the decoder, 2D class and
        2D heatmap logits of the permuted load against the source ones reordered, the two
        swapped columns, and the set of checkpoint keys that changed.
        """
        torch.manual_seed(0)
        source = _build_model(
            auxiliary_2d_head=_build_focal_head(NUSCENES_CLASSES), class_names=NUSCENES_CLASSES
        )
        target = _build_model(
            auxiliary_2d_head=_build_focal_head(T4_CLASSES), class_names=T4_CLASSES
        )
        batch_inputs = _build_batch_inputs(torch.randn(2, NUM_CAMS, 3, *IMAGE_HW))
        rows = [NUSCENES_CLASSES.index(name) for name in T4_CLASSES]
        self.assertEqual(rows, [0, 1, 2, 3, 4, 6, 5])
        # The biases start equal for every class, unlike those of a trained checkpoint.
        torch.nn.init.normal_(source.bbox_head.cls_branches[0][-1].bias)
        torch.nn.init.normal_(source.auxiliary_2d_head.cls.bias)
        torch.nn.init.normal_(source.auxiliary_2d_head.heatmap.bias)

        # Same shapes, so the raw checkpoint loads silently with the cones in the barrier column.
        checkpoint = {"state_dict": source.state_dict()}
        target.load_state_dict(checkpoint["state_dict"], strict=True)
        source_decoder, source_auxiliary, source_heatmap = self._class_logits(source, batch_inputs)
        naive_decoder, _, _ = self._class_logits(target, batch_inputs)
        self.assertTrue(torch.equal(naive_decoder, source_decoder))

        permuted = permute_class_rows(checkpoint["state_dict"], NUSCENES_CLASSES, T4_CLASSES)
        target.load_state_dict(permuted, strict=True)
        decoder, auxiliary, heatmap = self._class_logits(target, batch_inputs)

        self.assertTrue(torch.equal(decoder, source_decoder[..., rows]))
        self.assertTrue(torch.equal(auxiliary, source_auxiliary[..., rows]))
        self.assertTrue(torch.equal(heatmap, source_heatmap[..., rows]))
        for name in ("barrier", "traffic_cone"):
            self.assertTrue(
                torch.equal(
                    decoder[..., T4_CLASSES.index(name)],
                    source_decoder[..., NUSCENES_CLASSES.index(name)],
                ),
                name,
            )
        changed = {
            key for key in permuted if not torch.equal(permuted[key], checkpoint["state_dict"][key])
        }
        num_layers = len(target.bbox_head.cls_branches)
        expected = {
            f"bbox_head.cls_branches.{layer}.{len(target.bbox_head.cls_branches[layer]) - 1}.{kind}"
            for layer in range(num_layers)
            for kind in ("weight", "bias")
        } | {
            f"auxiliary_2d_head.{conv}.{kind}"
            for conv in ("cls", "heatmap")
            for kind in ("weight", "bias")
        }
        self.assertEqual(changed, expected)


if __name__ == "__main__":
    unittest.main()
