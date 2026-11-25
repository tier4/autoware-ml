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

"""Invariants of the StreamPETR stages the generic experiment config test does not check."""

from __future__ import annotations

import inspect
import unittest

import hydra
from omegaconf import DictConfig, OmegaConf

from autoware_ml.datamodule.streaming_data_module import StreamingDataModule
from autoware_ml.utils.config.tests.config_roots import compose_task
from autoware_ml.utils.config.tests.test_task_configs import EXPERIMENT_CONFIGS

STAGES = [config for config in EXPERIMENT_CONFIGS if "/streampetr/" in config]
# The T4 taxonomy swaps the last two classes, so stage 2 loads a permuted stage-1 checkpoint.
NUSCENES_CLASSES = ["car", "truck", "bus", "bicycle", "pedestrian", "traffic_cone", "barrier"]
T4_CLASSES = ["car", "truck", "bus", "bicycle", "pedestrian", "barrier", "traffic_cone"]


class TestStreamPETRConfigs(unittest.TestCase):
    """Invariants of the bundled StreamPETR stages."""

    configs: dict[str, DictConfig]

    @classmethod
    def setUpClass(cls) -> None:
        cls.configs = {stage: compose_task(stage) for stage in STAGES}

    @staticmethod
    def _step_names(config: DictConfig) -> list[str]:
        """Return the class names of the steps of a pipeline or of the preprocessing modules.

        Args:
            config: The list of ``_target_`` blocks.

        Returns:
            The last component of every ``_target_``, in order.
        """
        return [step["_target_"].rsplit(".", 1)[1] for step in config]

    def test_the_stages_are_bundled(self) -> None:
        """Test that the nuScenes pretrain and the T4 test leaf are the StreamPETR leaves.

        Input: the experiment configs of both config roots under tasks/detection3d/streampetr.
        Expected: the nuScenes pretrain is bundled as one leaf config, and the T4Dataset leaf of
        the test config root composes the bundled T4 base, as the T4 stages of the external
        config root do; no other leaf.
        Check: the leaf names of the StreamPETR experiments, in the order the experiment
        configs are listed.
        """
        self.assertEqual(
            [stage.rsplit("/", 1)[1] for stage in STAGES],
            [
                "vov99_flrfpn_320x800_b8_30e_nuscenes_51m_pretrain",
                "vov99_flrfpn_480x640_b16_35e_t4dataset_51m",
            ],
        )

    def test_every_stage_streams_the_seven_classes_of_its_taxonomy(self) -> None:
        """Test that every stage trains the classes of its taxonomy on streamed scenes.

        Input: the composed config of every stage.
        Expected: both heads train the seven classes in the order of the taxonomy: the nuScenes
        order in stage 1, and in the T4 stages the dynamic T4 order, the one the other T4
        detectors train and the deployed node decodes. The head carries a temporal memory
        across batches, so the datamodule is a StreamingDataModule seeded by the experiment,
        whose loaders do not shuffle and whose evaluation runs one lane per rank. The schedule
        is bound to the trainer, and the initial weights are left for the command line.
        Check: the class names of the taxonomy and of both heads, the datamodule class, its
        seed and its keys against the arguments of StreamingDataModule, the dataloader
        settings, the max_epochs of the scheduler and the missing weights.
        """
        parameters = set(inspect.signature(StreamingDataModule.__init__).parameters) - {"self"}
        for stage, cfg in self.configs.items():
            with self.subTest(stage=stage):
                expected = T4_CLASSES if "/t4dataset/" in stage else NUSCENES_CLASSES
                taxonomy = hydra.utils.instantiate(cfg.taxonomy.detection3d)
                self.assertEqual(list(taxonomy.class_names), expected)
                self.assertEqual(list(cfg.model.bbox_head.class_names), expected)
                self.assertEqual(list(cfg.model.auxiliary_2d_head.class_names), expected)
                datamodule = cfg.datamodule
                self.assertIs(hydra.utils.get_class(datamodule._target_), StreamingDataModule)
                self.assertEqual(datamodule.seed, cfg.seed)
                self.assertEqual(set(datamodule) - {"_target_", "_convert_"}, parameters)
                self.assertFalse(datamodule.train_dataloader.shuffle)
                self.assertEqual(datamodule.validation_dataloader.batch_size, 1)
                self.assertEqual(datamodule.test_dataloader.batch_size, 1)
                self.assertEqual(cfg.model.scheduler.max_epochs, cfg.trainer.max_epochs)
                self.assertTrue(OmegaConf.is_missing(cfg, "weights"))

    def test_every_pipeline_filters_the_physical_boxes_before_anything_else(self) -> None:
        """Test that BBoxesPhysicalFilter is the first step of every pipeline.

        Input: the training and test pipelines of every stage.
        Expected: BBoxesPhysicalFilter zeroes a non-finite velocity and drops the boxes that
        cannot become a target, a zero extent among them, before any other transform reads the
        boxes, so no 2D target is projected from a box it drops.
        Check: the first step of every pipeline.
        """
        for stage, cfg in self.configs.items():
            for split in ("train", "test"):
                with self.subTest(stage=stage, split=split):
                    steps = self._step_names(cfg.datasets[split].transforms.pipeline)
                    self.assertEqual(steps[0], "BBoxesPhysicalFilter")

    def test_every_dataset_projecting_2d_boxes_pads_them_to_its_3d_maximum(self) -> None:
        """Test that every dataset projecting 2D boxes pads them to its 3D maximum.

        Input: the composed datasets of every stage.
        Expected: a dataset whose pipeline projects the 3D boxes onto the cameras collates its
        2D boxes to a fixed count, without which its first batch is refused; a camera cannot
        see more 2D boxes than its frame has 3D boxes, so the 3D maximum serves as the 2D one.
        Check: the training dataset projects 2D boxes, and every dataset that does sets
        max_num_2d_gt_bboxes to its max_num_3d_gt_bboxes.
        """
        for stage, cfg in self.configs.items():
            with self.subTest(stage=stage):
                projecting = [
                    name
                    for name, dataset in cfg.datasets.items()
                    if "LoadAnnotations2DFromBoxes3D"
                    in self._step_names(dataset.transforms.pipeline)
                ]
                self.assertIn("train", projecting)
                for name in projecting:
                    dataset = cfg.datasets[name]
                    self.assertEqual(
                        dataset.max_num_2d_gt_bboxes, dataset.max_num_3d_gt_bboxes, name
                    )

    def test_every_stage_projects_the_2d_boxes_before_the_range_filter(self) -> None:
        """Test that every training pipeline projects the 2D boxes before the range filter.

        Input: the training pipeline of every stage.
        Expected: the 2D targets of the reference recipes keep the objects outside the 3D
        range, so every stage projects its 2D boxes before BBoxesRangeFilter drops those
        objects; the other way round the auxiliary 2D head would learn that they are
        background.
        Check: the positions of LoadAnnotations2DFromBoxes3D and BBoxesRangeFilter.
        """
        for stage, cfg in self.configs.items():
            with self.subTest(stage=stage):
                steps = self._step_names(cfg.datasets.train.transforms.pipeline)
                self.assertLess(
                    steps.index("LoadAnnotations2DFromBoxes3D"), steps.index("BBoxesRangeFilter")
                )

    def test_t4_stages_drop_the_low_point_boxes_after_the_2d_projection(self) -> None:
        """Test that the boxes with too few lidar points stay 2D targets in the T4 stages.

        Input: the training datasets of the T4 stages.
        Expected: the evaluation drops the boxes recorded with too few lidar points, so the 3D
        head does not train on them, while a camera can still see them; BBoxesMinPointsFilter
        runs after LoadAnnotations2DFromBoxes3D and reads the recorded counts, as the pipeline
        loads no point cloud.
        Check: the position of BBoxesMinPointsFilter after LoadAnnotations2DFromBoxes3D, and
        its use_bbox_num_lidar_points, in every T4 training dataset.
        """
        for stage, cfg in self.configs.items():
            if "/t4dataset/" not in stage:
                continue
            training = [name for name in cfg.datasets if name.endswith("train")]
            with self.subTest(stage=stage):
                self.assertIn("train", training)
            for name in training:
                with self.subTest(stage=stage, dataset=name):
                    pipeline = cfg.datasets[name].transforms.pipeline
                    steps = self._step_names(pipeline)
                    index = steps.index("BBoxesMinPointsFilter")
                    self.assertLess(steps.index("LoadAnnotations2DFromBoxes3D"), index)
                    self.assertIs(pipeline[index].use_bbox_num_lidar_points, True)

    def test_only_the_t4_training_pipelines_shuffle_the_loaded_cameras(self) -> None:
        """Test that only the T4 training pipelines shuffle the cameras, right after loading.

        Input: the training and test pipelines of every stage.
        Expected: the T4 reference recipe shuffles the camera order of every training frame,
        and never at evaluation, while the nuScenes reference recipe does not shuffle at all.
        Check: ShuffleCameraOrder directly follows LoadImagesFromFile in the T4 training
        pipelines and appears in no other pipeline.
        """
        for stage, cfg in self.configs.items():
            for split in ("train", "test"):
                with self.subTest(stage=stage, split=split):
                    steps = self._step_names(cfg.datasets[split].transforms.pipeline)
                    shuffles = split == "train" and "/t4dataset/" in stage
                    loading = steps.index("LoadImagesFromFile")
                    self.assertEqual(steps[loading + 1] == "ShuffleCameraOrder", shuffles)
                    self.assertEqual("ShuffleCameraOrder" in steps, shuffles)

    def test_t4_stages_ignore_the_cones_and_barriers_of_the_scenes_without_them(self) -> None:
        """Test that the T4 stages leave out the cones and barriers where they are unannotated.

        Input: the partial annotations of the T4 stages, instantiated from their configs.
        Expected: the T4 scenarios flag whether their traffic cones and barriers are annotated,
        and where they are not, those two classes are not supervised; the pair resolves by
        name, whichever order the taxonomy lists it in.
        Check: the class ids the partial annotations ignore, and the class names at those ids.
        """
        for stage, cfg in self.configs.items():
            if "/t4dataset/" not in stage:
                continue
            with self.subTest(stage=stage):
                partial_annotations = hydra.utils.instantiate(cfg.model.partial_annotations)
                ignored = list(partial_annotations.partial_ignore_class_ids)
                self.assertEqual(ignored, [5, 6])
                self.assertEqual(
                    {T4_CLASSES[index] for index in ignored}, {"traffic_cone", "barrier"}
                )

    def test_the_nuscenes_stage_supervises_every_class(self) -> None:
        """Test that the nuScenes stage has no partial annotations.

        Input: the composed config of the nuScenes stage.
        Expected: nuScenes annotates every class in every frame, so the stage has no partial
        annotations.
        Check: model.partial_annotations is None.
        """
        stage = next(stage for stage in STAGES if "/nuscenes/" in stage)
        self.assertIsNone(self.configs[stage].model.partial_annotations)

    def test_every_pipeline_loads_unit_range_images_for_the_normalization(self) -> None:
        """Test that every stage normalizes images that its pipelines load in the unit range.

        Input: the preprocessing modules and the training and test pipelines of every stage.
        Expected: BatchNormalizeImages holds unit-range statistics and does not check the
        images, so 0-255 pixels would train without an error; it is the first preprocessing
        module, and every pipeline loads its images once, with the unit-range default of
        LoadImagesFromFile.
        Check: the first preprocessing module, and the one LoadImagesFromFile of every pipeline
        with normalize_to_unit unset or True.
        """
        for stage, cfg in self.configs.items():
            modules = self._step_names(cfg.data_preprocessor.preprocessor_modules)
            with self.subTest(stage=stage):
                self.assertEqual(modules[0], "BatchNormalizeImages")
            for split in ("train", "test"):
                with self.subTest(stage=stage, split=split):
                    loaders = [
                        step
                        for step in cfg.datasets[split].transforms.pipeline
                        if step["_target_"].endswith("LoadImagesFromFile")
                    ]
                    self.assertEqual(len(loaders), 1)
                    self.assertIs(loaders[0].get("normalize_to_unit", True), True)
