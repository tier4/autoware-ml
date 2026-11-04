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

"""Tests for the dense BEV branch of the Sparse4D head."""

from __future__ import annotations

import itertools
import math
import unittest

import torch

from autoware_ml.models.detection3d.heads.sparse4d import (
    BEVDeformableAggregation,
    LidarSampling,
    Sparse4DHead,
    SparseBox3DKeyPointsGenerator,
    SparseDeformableAggregation,
    TemporalStateInputs,
)
from autoware_ml.models.detection3d.ptv3 import PTv3BEVEncoder, PTv3StageBEVPyramid
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import POINT_CLOUD_RANGE
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    forward_frame,
    gt_batch,
    head_loss,
    make_anchor_recipe,
    random_stage,
)
from autoware_ml.ops.deformable_aggregation.bev_deformable_aggregation import BEVLevel

BEV_CHANNELS = 16


def _pyramid(
    stage_strides: tuple[int, ...] = (1,),
    projected_strides: tuple[int, ...] = (1,),
    num_downsampled: int = 0,
    point_cloud_range: list[float] = POINT_CLOUD_RANGE,
    encoder_channels: int = BEV_CHANNELS,
) -> PTv3StageBEVPyramid:
    return PTv3StageBEVPyramid(
        point_cloud_range=point_cloud_range,
        grid_size=1.0,
        stage_strides=stage_strides,
        projected_strides=projected_strides,
        in_channels=32,
        channels=BEV_CHANNELS,
        bev_encoder=PTv3BEVEncoder(
            encoder_channels, encoder_channels, encoder_channels, dilations=[1]
        ),
        num_downsampled=num_downsampled,
    )


def _head(
    lidar_sampling: LidarSampling, multiscale: bool = False, **overrides: object
) -> Sparse4DHead:
    """A test head of the sampling mode.

    A BEV only head reads a halved BEV level too. A multiscale head samples two stages and
    projects both with a halved BEV level on top.
    """
    strides = (1, 2) if multiscale else (1,)
    return build_sparse4d_head(
        make_anchor_recipe(),
        lidar_sampling=lidar_sampling,
        feature_strides=strides,
        bev=None
        if lidar_sampling == LidarSampling.SPARSE
        else _pyramid(
            stage_strides=strides,
            projected_strides=strides,
            num_downsampled=int(multiscale or lidar_sampling == LidarSampling.BEV),
        ),
        **overrides,
    )


def _block(**overrides: object) -> SparseDeformableAggregation:
    kwargs: dict[str, object] = {
        "embed_dims": 32,
        "num_groups": 4,
        "num_levels": 1,
        "kps_generator": SparseBox3DKeyPointsGenerator(
            embed_dims=32, fix_scale=((0.0, 0.0, 0.0),), num_learnable_pts=0
        ),
    }
    kwargs.update(overrides)
    return SparseDeformableAggregation(**kwargs)


class TestSparse4DBEV(unittest.TestCase):
    def test_pyramid_projects_the_stages_and_halves_the_encoded_map(self) -> None:
        pyramid = _pyramid(
            stage_strides=(1, 2),
            projected_strides=(1, 2),
            num_downsampled=1,
            point_cloud_range=[0.0, 0.0, 0.0, 8.0, 8.0, 8.0],
        ).eval()
        stages = [random_stage((8, 8, 8), 32), random_stage((4, 4, 4), 16, seed=1)]

        levels = pyramid(stages)

        self.assertEqual(
            [tuple(level.feat.shape) for level in levels],
            [(1, BEV_CHANNELS, 8, 8), (1, BEV_CHANNELS, 4, 4), (1, BEV_CHANNELS, 2, 2)],
        )
        self.assertEqual([level.cell_size for level in levels], [1.0, 2.0, 4.0])
        self.assertEqual(
            [level.grid_origin for level in levels], [(0.0, 0.0), (0.0, 0.0), (-1.0, -1.0)]
        )

    def test_a_downsampled_level_reads_the_cell_its_convolution_centers_on(self) -> None:
        """
        Input: a halving convolution that copies the center tap, over a map with one hot cell.
        Expected: the coarse level reads the hot value at the metric center of the fine cell, which
          a stride 2 convolution with padding 1 centers its coarse cell on.
        Check: the sampled value at the fine cell center.
        """
        pyramid = _pyramid(num_downsampled=1, point_cloud_range=[0.0, 0.0, 0.0, 8.0, 8.0, 8.0])
        convolution = pyramid.downsample[0][0]
        with torch.no_grad():
            convolution.weight.zero_()
            convolution.weight[range(BEV_CHANNELS), range(BEV_CHANNELS), 1, 1] = 1.0
            fine = torch.zeros((1, BEV_CHANNELS, 8, 8))
            fine[..., 4, 2] = 1.0
            coarse = BEVLevel(convolution(fine), pyramid.grid_origins[1], pyramid.cell_sizes[1])

        # Fine cell (x 2, y 4) of 1 m centers at (2.5, 4.5)
        value = coarse.sample(torch.tensor([[[[2.5, 4.5, 0.0]]]]))

        self.assertTrue(torch.allclose(value, torch.ones_like(value)))

    def test_pyramid_rejects_a_stride_that_is_not_a_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "projected_strides"):
            _pyramid(stage_strides=(1, 2), projected_strides=(4,))

    def test_pyramid_rejects_an_encoder_of_another_width_than_its_maps(self) -> None:
        with self.assertRaisesRegex(ValueError, "the BEV maps have 16"):
            _pyramid(encoder_channels=8)

    def test_pyramid_rejects_stages_of_another_count_than_its_strides(self) -> None:
        """
        Input: a pyramid of the strides 1 and 2 given one stage.
        Expected: ValueError, the stage of a stride would be read from the wrong level.
        """
        pyramid = _pyramid(stage_strides=(1, 2), projected_strides=(2,))
        with self.assertRaisesRegex(ValueError, r"strides \[1, 2\], but got 1 stages"):
            pyramid([random_stage((8, 8, 2), 5)])

    def test_every_parameter_of_every_sampling_mode_trains(self) -> None:
        for lidar_sampling, multiscale in itertools.product(LidarSampling, (False, True)):
            with self.subTest(lidar_sampling=lidar_sampling, multiscale=multiscale):
                head = _head(lidar_sampling, multiscale)
                head.cls_threshold_to_reg = 1.0
                stages = [random_stage(grid.grid_dims, 48) for grid in head.level_grids]
                gt = gt_batch(
                    [torch.tensor([[4.0, 4.0, 0.0, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])],
                    [torch.tensor([1])],
                )

                losses = head_loss(head, forward_frame(head, stages, gt_detections=gt), gt, [0])
                losses["loss"].backward()

                missing = [
                    name
                    for name, parameter in head.named_parameters()
                    if parameter.requires_grad and parameter.grad is None
                ]
                self.assertEqual(missing, [])
                for block in head.deformable:
                    self.assertIs(block.weights_fc is None, lidar_sampling == LidarSampling.BEV)
                    self.assertIs(
                        block.bev_aggregation is None, lidar_sampling == LidarSampling.SPARSE
                    )

    def test_a_bev_head_exports_the_training_forward(self) -> None:
        """
        Input: heads sampling the voxels and the BEV maps or the BEV maps alone, of one stage or
          of two stages with a halved BEV level on top, run in training form and in export form
          for two frames of one scene, the export form fed the state it returned.
        Expected: the export form decodes both frames as the training form does and returns the
          state the instance bank caches.
        """
        pose1 = torch.eye(4, dtype=torch.float64)[None]
        pose2 = pose1.clone()
        pose2[0, :2, :2] = torch.tensor(
            [[math.cos(0.2), -math.sin(0.2)], [math.sin(0.2), math.cos(0.2)]]
        )
        pose2[0, :3, 3] = torch.tensor([1.0, 0.5, 0.0])
        for lidar_sampling, multiscale in itertools.product(
            (LidarSampling.SPARSE_BEV, LidarSampling.BEV), (False, True)
        ):
            with self.subTest(lidar_sampling=lidar_sampling, multiscale=multiscale):
                head = _head(lidar_sampling, multiscale, num_temp_instances=4).eval()
                export_head = head.prepare_for_export()
                stages = [random_stage(grid.grid_dims, 48) for grid in head.level_grids]

                with torch.no_grad():
                    train1 = forward_frame(head, stages, 100.0, None, ["scene"], lidar2global=pose1)
                    export1 = export_head.forward_temporal_export(
                        stages, export_head.temporal_state_export_inputs(torch.device("cpu"))
                    )
                    train2 = forward_frame(head, stages, 100.5, None, ["scene"], lidar2global=pose2)
                    export2 = export_head.forward_temporal_export(
                        stages,
                        TemporalStateInputs(
                            temp_instance_feature=export1.temp_instance_feature_out,
                            temp_anchor=export1.temp_anchor_out,
                            temp_confidence=export1.temp_confidence_out,
                            temp_valid=torch.ones(1),
                            time_interval=torch.tensor([0.5]),
                            prev2current=torch.linalg.solve(pose2, pose1).to(torch.float32),
                        ),
                    )

                for train, export in ((train1, export1), (train2, export2)):
                    for exported, trained in (
                        (export.classification, train.classification),
                        (export.prediction, train.prediction),
                        (export.quality, train.quality),
                    ):
                        torch.testing.assert_close(exported, trained[:, -1], atol=1e-5, rtol=0)
                bank = head.instance_bank
                torch.testing.assert_close(
                    export2.temp_instance_feature_out, bank._cached_feature, atol=1e-5, rtol=0
                )
                torch.testing.assert_close(
                    export2.temp_anchor_out, bank._cached_anchor, atol=1e-4, rtol=0
                )
                torch.testing.assert_close(
                    export2.temp_confidence_out, bank._cached_confidence, atol=1e-5, rtol=0
                )
                self.assertTrue(torch.equal(export2.temp_query_index_out, bank.cached_query_index))

    def test_head_rejects_bev_settings_that_do_not_go_together(self) -> None:
        cases = [
            ("bev", False, "BEV maps"),
            ("sparse", True, "BEV maps"),
            ("dense", True, "not a valid LidarSampling"),
        ]
        for lidar_sampling, with_bev, match in cases:
            with self.subTest(lidar_sampling=lidar_sampling, with_bev=with_bev):
                with self.assertRaisesRegex(ValueError, match):
                    build_sparse4d_head(
                        make_anchor_recipe(),
                        lidar_sampling=lidar_sampling,
                        bev=_pyramid() if with_bev else None,
                    )

    def test_a_block_samples_voxels_bev_maps_or_both(self) -> None:
        with self.assertRaisesRegex(ValueError, "sparse voxels, BEV maps or both"):
            _block(sample_sparse=False)

    def test_bev_branch_rejects_settings_and_inputs_it_was_not_built_for(self) -> None:
        with self.assertRaisesRegex(ValueError, "not divisible by 3 groups"):
            BEVDeformableAggregation(
                embed_dims=32, bev_channels=BEV_CHANNELS, num_groups=3, num_levels=1, num_points=1
            )
        branch = BEVDeformableAggregation(
            embed_dims=32, bev_channels=BEV_CHANNELS, num_groups=4, num_levels=2, num_points=1
        )
        with self.assertRaisesRegex(ValueError, "2 levels"):
            branch(torch.zeros(1, 1, 32), torch.zeros(1, 1, 1, 3), [])
