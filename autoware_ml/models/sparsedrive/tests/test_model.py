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

"""Tests for the SparseDrive model."""

from __future__ import annotations

import itertools
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import onnx
import torch
from jaxtyping import Shaped
from omegaconf import OmegaConf

from autoware_ml.dataclasses.batch.frame_meta import FrameMetaBatch
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.dataclasses.models.detection3d.head_outputs import Detection3DHeadOutputs
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.dataclasses.models.segmentation3d.head_outputs import Segmentation3DHeadOutputs
from autoware_ml.models.detection3d.ptv3 import rebuild_point_chain
from autoware_ml.models.sparsedrive.lidar_stages import PTv3Det3DFeaturePyramid
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    POINT_CLOUD_RANGE,
    build_inputs,
    build_ptv3_encoder,
    build_seg3d_head,
)
from autoware_ml.models.sparsedrive.sparse4d_head import (
    ANCHOR_DIM,
    TEMPORAL_STATE_INPUT_NAMES,
    TemporalStateInputs,
)
from autoware_ml.models.detection3d.task_modules.partial_annotations import PartialAnnotations
from autoware_ml.models.sparsedrive.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    gt_batch,
    make_anchor_recipe,
    random_stage,
)
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.sparsedrive.model import (
    SPARSE4D_EXPORT_OUTPUT_NAMES,
    SparseDriveModel,
    Sparse4DHeadExportModule,
)
from autoware_ml.models.segmentation3d.encoders.voxel import SweepSplitVoxelFeatureEncoder
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.deploy import export_to_onnx

requires_cuda_spconv = unittest.skipUnless(
    IS_SPCONV_AVAILABLE and torch.cuda.is_available(),
    "PTv3 sparse-convolution tests require CUDA spconv",
)


def _build_model(
    partial_annotations: PartialAnnotations | None = None, **head_overrides: object
) -> SparseDriveModel:
    """Build a small temporal model whose head samples the stage the deepest stage fuses into."""
    return SparseDriveModel(
        encoder=build_ptv3_encoder(),
        voxel_encoder=SweepSplitVoxelFeatureEncoder(),
        seg3d_head=build_seg3d_head(),
        det_fusion=PTv3Det3DFeaturePyramid(stage_channels=(8, 16), out_channels=32),
        bbox_head=build_sparse4d_head(
            make_anchor_recipe(), **{"num_temp_instances": 4, **head_overrides}
        ),
        log_dict_configs=LogDictConfigs(prog_bar=False),
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        grid_size=1.0,
        point_cloud_range=POINT_CLOUD_RANGE,
        partial_annotations=partial_annotations,
    )


def _head_graph_inputs(
    grid_dims: Sequence[tuple[int, int, int]], channels: Sequence[int]
) -> list[Shaped[torch.Tensor, "..."]]:
    """Det head graph inputs of one sample over random encoder stages.

    Args:
        grid_dims: Voxel grid of every stage but the deepest, finest first.
        channels: Feature width of every stage, finest first.

    Returns:
        The stage features from the finest to the deepest stage, then the pooling clusters and
        the voxel coordinates of every stage but the deepest.
    """
    generator = torch.Generator().manual_seed(0)
    counts = [24 // 2**stage for stage in range(len(channels))]
    feats = [
        torch.randn((count, width), generator=generator)
        for count, width in zip(counts, channels, strict=True)
    ]
    clusters = [
        torch.randint(0, coarse, (fine,), generator=generator)
        for fine, coarse in itertools.pairwise(counts)
    ]
    grid_coords = []
    for count, (size_x, size_y, size_z) in zip(counts[:-1], grid_dims, strict=True):
        cells = torch.randperm(size_x * size_y * size_z, generator=generator)[:count]
        grid_coords.append(
            torch.stack([cells % size_x, cells // size_x % size_y, cells // (size_x * size_y)], 1)
        )
    return [*feats, *clusters, *grid_coords]


def frame_inputs(device: torch.device, timestamp: float) -> ModelBatchInputs:
    """One frame of the test scene with its frame metadata."""
    batch = build_inputs(device=device)
    detections = batch.multi_task_gt_batch.detection3d_gt_batch
    gt_batch = batch.multi_task_gt_batch._replace(
        detection3d_gt_batch=detections._replace(
            gt_bboxes_num_points=torch.full_like(detections.gt_bboxes_num_points, 5)
        ),
        frame_meta_batch=FrameMetaBatch(
            ego2globals=torch.eye(4, device=device)[None],
            scene_tokens=["db/scene/0"],
            timestamps=torch.tensor([timestamp], dtype=torch.float64, device=device),
            key_frames=[True],
            lanelet_map_paths=[None],
        ),
    )
    return batch.replace(multi_task_gt_batch=gt_batch)


def _three_frame_step(
    model: SparseDriveModel, padding: list[bool], key_frames: list[bool]
) -> tuple[ModelBatchInputs, ModelOutputs]:
    """A batch of three frames with one box each and the outputs of the detection head on it."""
    car = torch.tensor([[4.0, 4.0, 0.0, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])
    gt = gt_batch([car] * 3, [torch.tensor([0])] * 3)
    frame_meta = FrameMetaBatch(
        ego2globals=torch.eye(4, dtype=torch.float64).repeat(3, 1, 1),
        scene_tokens=["db/scene/0", "db/scene/1", "db/scene/2"],
        timestamps=torch.full((3,), 100.0, dtype=torch.float64),
        key_frames=key_frames,
        lanelet_map_paths=[None] * 3,
    )
    batch = ModelBatchInputs(
        multi_task_gt_batch=ModelGTBatch(
            point_cloud_gt_batch=None,
            detection3d_gt_batch=gt,
            segmentation3d_gt_batch=None,
            image_gt_batch=None,
            padding_mask=torch.tensor(padding),
            frame_meta_batch=frame_meta,
        ),
        voxels_data=None,
        image_data=None,
        range_view_data=None,
    )
    stages = [
        random_stage(grid.grid_dims, 16, batch_size=3) for grid in model.bbox_head.level_grids
    ]
    det3d_outputs = model.bbox_head(
        stages,
        timestamps_seconds=frame_meta.timestamps,
        lidar2global=frame_meta.ego2globals,
        scene_tokens=frame_meta.scene_tokens,
        gt_detections=gt,
    )
    outputs = ModelOutputs(
        detection3d_head_outputs=Detection3DHeadOutputs(sparse4d_head_outputs=det3d_outputs),
        segmentation3d_head_outputs=Segmentation3DHeadOutputs(logits=torch.zeros((4, 3))),
    )
    return batch, outputs


class TestSparseDriveModel(unittest.TestCase):
    def test_the_losses_leave_out_padding_and_non_key_frames(self) -> None:
        """
        Input: a key frame, its padding copy and a non-key frame, each with one box, then three
          frames that are all padding.
        Expected: only the key frame reaches the assignment, a step without a supervised frame
          gives zero detection terms that still backpropagate.
        """
        model = _build_model()
        zero = torch.zeros((), requires_grad=True)
        seg3d_losses = {"loss": zero, "loss_ce": zero, "loss_lovasz": zero}
        cases = [
            ([False, True, False], [True, True, False], [0]),
            ([True, True, True], [True, True, True], []),
        ]
        for padding, key_frames, lanes in cases:
            with self.subTest(padding=padding, key_frames=key_frames):
                model.reset_streams()
                batch, outputs = _three_frame_step(model, padding, key_frames)
                with (
                    mock.patch(
                        f"{SparseDriveModel.__module__}.segmentation3d_point_loss",
                        return_value=seg3d_losses,
                    ),
                    mock.patch.object(
                        model.bbox_head, "assign", wraps=model.bbox_head.assign
                    ) as assign,
                ):
                    losses = model.detection_segmentation_metrics(batch, outputs)

                self.assertEqual(losses.det3d_mask.tolist(), [lane in lanes for lane in range(3)])
                self.assertEqual(assign.call_args.args[4], lanes)
                self.assertEqual(len(assign.call_args.args[1]), len(lanes))
                self.assertTrue(torch.isfinite(losses.metrics["loss"]))
                if not lanes:
                    self.assertTrue(
                        all(
                            float(value) == 0.0
                            for name, value in losses.metrics.items()
                            if name.startswith("det3d_") and "loss" in name
                        )
                    )
                losses.metrics["loss"].backward()

    def test_the_model_refuses_a_head_sampling_other_strides_than_the_pyramid(self) -> None:
        with self.assertRaisesRegex(ValueError, r"returns the strides \[1\] but the head samples"):
            _build_model(feature_strides=(2,))

    @requires_cuda_spconv
    def test_temporal_model_trains_on_a_stream_and_reaches_every_parameter(self) -> None:
        device = torch.device("cuda")
        model = _build_model(dn_query_budget=8).to(device)

        for frame_index in range(2):
            model.zero_grad()
            batch = frame_inputs(device, 100.0 + 0.1 * frame_index)
            outputs = model(batch)
            metrics = model.compute_metrics(batch, outputs)
            metrics["loss"].backward()

            self.assertTrue(torch.isfinite(metrics["loss"]))
            self.assertIsNotNone(outputs.detection3d().sparse4d_head().denoising)
            missing = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and parameter.grad is None
            ]
            self.assertEqual(missing, [])

        # The second frame of the scene reads the instances the first one cached
        self.assertEqual(model.bbox_head.instance_bank._slot_valid.tolist(), [True])

    def test_partial_annotations_hand_the_head_the_annotated_classes(self) -> None:
        """
        Input: a model ignoring cones in the frames flagged as not annotated, three frames flagged
          not annotated, not annotated and annotated, the second padding its lane, then the same
          frames without the flag.
        Expected: the head loss gets the class masks of the supervised frames, without cones for
          the first, and the frames without the flag raise.
        """
        model = _build_model(
            PartialAnnotations(class_names=("car", "cone"), partially_annotated_classes=["cone"]),
        )
        zero = torch.zeros((), requires_grad=True)
        seg3d_losses = {"loss": zero, "loss_ce": zero, "loss_lovasz": zero}
        batch, outputs = _three_frame_step(model, [False, True, False], [True, True, True])
        detections = batch.multi_task_gt_batch.detection3d_gt_batch
        flagged = batch.replace(
            multi_task_gt_batch=batch.multi_task_gt_batch._replace(
                detection3d_gt_batch=detections._replace(
                    gt_traffic_cone_barrier_bbox_status=torch.tensor([False, False, True])
                )
            )
        )
        head = model.bbox_head

        with (
            mock.patch(
                f"{SparseDriveModel.__module__}.segmentation3d_point_loss",
                return_value=seg3d_losses,
            ),
            mock.patch.object(head, "loss_from_targets", wraps=head.loss_from_targets) as spy,
        ):
            model.detection_segmentation_metrics(flagged, outputs)
            self.assertEqual(spy.call_args.args[3].tolist(), [[True, False], [True, True]])
            with self.assertRaisesRegex(ValueError, "gt_traffic_cone_barrier_bbox_status"):
                model.detection_segmentation_metrics(batch, outputs)

    @requires_cuda_spconv
    def test_model_eval_output_and_predictions(self) -> None:
        device = torch.device("cuda")
        model = _build_model().to(device).eval()
        batch = frame_inputs(device, 100.0)

        with torch.no_grad():
            outputs = model(batch)
            eval_output = model.build_eval_output(batch, outputs)
            predictions = model.decode_outputs(batch, outputs)

        assert eval_output.detection3d is not None
        frame_meta = eval_output.detection3d.frames[0].frame_meta
        assert frame_meta is not None
        self.assertEqual(frame_meta.scene_token, "db/scene/0")
        self.assertEqual(predictions.detection3d_predictions[0].bboxes_3d.shape[1], 9)
        self.assertIsNotNone(predictions.segmentation3d_predictions)

    @requires_cuda_spconv
    def test_temporal_model_eval_output_carries_the_tracks_of_one_stream(self) -> None:
        device = torch.device("cuda")
        model = _build_model().to(device).eval()

        eval_outputs = []
        with torch.no_grad():
            for frame_index in range(2):
                batch = frame_inputs(device, 100.0 + 0.1 * frame_index)
                outputs = model(batch)
                eval_outputs.append(model.build_eval_output(batch, outputs))

        first, second = (output.detection3d.frames[0] for output in eval_outputs)
        assert first.gt_instance_ids is not None and second.gt_instance_ids is not None
        self.assertEqual(first.frame_meta.scene_token, second.frame_meta.scene_token)
        self.assertIsNotNone(second.predictions.instance_ids)
        self.assertEqual(len(second.gt_instance_ids), len(second.gt_boxes))

    @requires_cuda_spconv
    def test_the_model_predicts_without_detection_ground_truth(self) -> None:
        """
        Input: one frame without detection ground truth, the model in eval mode.
        Expected: the model predicts boxes, it reads the ground truth only to train and evaluate.
        """
        device = torch.device("cuda")
        model = _build_model().to(device).eval()
        batch = frame_inputs(device, 100.0)
        batch = batch.replace(
            multi_task_gt_batch=batch.multi_task_gt_batch._replace(detection3d_gt_batch=None)
        )

        with torch.no_grad():
            predictions = model.decode_outputs(batch, model(batch))

        self.assertEqual(predictions.detection3d_predictions[0].bboxes_3d.shape[1], 9)

    def test_forward_needs_frame_metadata(self) -> None:
        model = _build_model()
        with self.assertRaisesRegex(ValueError, "frame metadata"):
            model(build_inputs())

    def test_split_export_decodes_like_the_head_and_carries_the_bank_state(self) -> None:
        """
        Input: the det head graph of three encoder stages whose head samples the two fused ones,
          run on the stage inputs of one sample for two frames, the second with the state the
          first returned.
        Expected: every graph output equals the export form of the head on the stages the pyramid
          fuses, and the second frame reads the state of the first.
        """
        det_fusion = PTv3Det3DFeaturePyramid(stage_channels=(8, 16, 32), out_channels=32).eval()
        head = build_sparse4d_head(
            make_anchor_recipe(), num_temp_instances=4, feature_strides=(1, 2)
        ).eval()
        module = Sparse4DHeadExportModule(
            det_fusion, head.prepare_for_export(), SPARSE4D_EXPORT_OUTPUT_NAMES, stage_count=3
        )
        inputs = _head_graph_inputs([grid.grid_dims for grid in head.level_grids], (8, 16, 32))
        stages = det_fusion(rebuild_point_chain(inputs[:3], inputs[3:5], inputs[5:]))
        state = head.temporal_state_export_inputs(torch.device("cpu"))

        frames = []
        with torch.no_grad():
            for _ in range(2):
                outputs = dict(zip(module.output_names, module(*inputs, *state), strict=True))
                reference = head.forward_temporal_export(stages, state)._asdict()
                for name in module.output_names:
                    with self.subTest(frame=len(frames), name=name):
                        torch.testing.assert_close(
                            outputs[name], reference[name], atol=1e-5, rtol=0
                        )
                frames.append(outputs)
                state = TemporalStateInputs(
                    temp_instance_feature=outputs["temp_instance_feature_out"],
                    temp_anchor=outputs["temp_anchor_out"],
                    temp_confidence=outputs["temp_confidence_out"],
                    temp_valid=torch.ones(1),
                    time_interval=torch.full((1,), 0.1),
                    prev2current=torch.eye(4)[None],
                )

        self.assertEqual(frames[0]["temp_anchor_out"].shape, (1, 4, ANCHOR_DIM))
        self.assertEqual(frames[0]["temp_query_index_out"].shape, (1, 4))
        self.assertFalse(torch.equal(frames[1]["classification"], frames[0]["classification"]))

    def test_the_head_graph_refuses_outputs_the_head_does_not_return(self) -> None:
        det_fusion = PTv3Det3DFeaturePyramid(stage_channels=(8, 16), out_channels=32)
        head = build_sparse4d_head(make_anchor_recipe(), num_temp_instances=4)
        with self.assertRaisesRegex(ValueError, r"has no outputs \['scores'\]"):
            Sparse4DHeadExportModule(det_fusion, head, ("classification", "scores"), stage_count=2)

    @requires_cuda_spconv
    def test_det_head_exports_to_onnx_with_the_deploy_dynamic_axes(self) -> None:
        """
        Input: the split det head of a model sampling one fused stage, and of a three stage model
        sampling two.
        Expected: the deploy export traces the sparse lookup with dynamic voxel counts, which needs a
        constant empty row, reads the voxel coordinates of every stage but the deepest and samples
        every decoder layer with one aggregation node.
        Check: the ONNX file passes the checker, declares the spec inputs and holds one aggregation
        node per layer with no sampling chain around it, reading int32 voxel lookups.
        """
        for enc_channels, feature_strides in [((8, 16), (1,)), ((8, 16, 32), (1, 2))]:
            with (
                self.subTest(feature_strides=feature_strides),
                tempfile.TemporaryDirectory() as tmp,
            ):
                path = Path(tmp) / "det3d_head.onnx"
                self._check_det_head_export(path, enc_channels, feature_strides)

    def _check_det_head_export(
        self, path: Path, enc_channels: tuple[int, ...], feature_strides: tuple[int, ...]
    ) -> None:
        """Export the split det head to path and check the ONNX graph."""
        device = torch.device("cuda")
        model = SparseDriveModel(
            encoder=build_ptv3_encoder(enc_channels=enc_channels),
            voxel_encoder=SweepSplitVoxelFeatureEncoder(),
            seg3d_head=build_seg3d_head(
                dec_depths=(1,) * (len(enc_channels) - 1), enc_channels=enc_channels
            ),
            det_fusion=PTv3Det3DFeaturePyramid(stage_channels=enc_channels, out_channels=32),
            bbox_head=build_sparse4d_head(
                make_anchor_recipe(), num_temp_instances=4, feature_strides=feature_strides
            ),
            log_dict_configs=LogDictConfigs(prog_bar=False),
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
            grid_size=1.0,
            point_cloud_range=POINT_CLOUD_RANGE,
        )
        model = model.to(device).eval()
        specs = model.build_export_specs(frame_inputs(device, 100.0))
        spec = specs["det3d_head"]
        self.assertEqual(list(specs), ["lidar_encoder", "seg3d_head", "det3d_head"])
        self.assertEqual(
            spec.input_names[-len(TEMPORAL_STATE_INPUT_NAMES) :], list(TEMPORAL_STATE_INPUT_NAMES)
        )
        self.assertEqual(spec.output_names, list(SPARSE4D_EXPORT_OUTPUT_NAMES))

        export_to_onnx(
            spec,
            OmegaConf.create({"opset_version": 17, "dynamo": False, "do_constant_folding": False}),
            path,
        )

        graph = onnx.load(str(path)).graph
        onnx.checker.check_model(onnx.load(str(path)))
        declared = {value.name for value in graph.input}
        self.assertLessEqual(declared, set(spec.input_names))
        self.assertNotIn(f"point_grid_coord_{len(enc_channels) - 1}", spec.input_names)
        aggregations = [
            node for node in graph.node if node.op_type == "SparseDeformableAggregation"
        ]
        self.assertEqual(len(aggregations), len(model.bbox_head.deformable))
        self.assertEqual(
            {(node.domain, len(node.input)) for node in aggregations},
            {("autoware", 2 + 2 * len(feature_strides))},
        )
        self.assertFalse(any(node.op_type == "Floor" for node in graph.node))
        inferred = onnx.shape_inference.infer_shapes(onnx.load(str(path)), strict_mode=False).graph
        types = {
            value.name: value.type.tensor_type.elem_type
            for value in (*inferred.value_info, *inferred.input, *inferred.output)
        }
        grids = {name for node in aggregations for name in node.input[2 + len(feature_strides) :]}
        self.assertEqual({types[name] for name in grids}, {onnx.TensorProto.INT32})
