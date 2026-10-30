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

import hydra
import pytest
from omegaconf import OmegaConf

from autoware_ml.configs.tests.test_task_configs import EXPERIMENT_CONFIGS, compose_experiment
from autoware_ml.datamodule.streaming_data_module import StreamingDataModule

STAGES = [config for config in EXPERIMENT_CONFIGS if "/streampetr/" in config]
# The nuScenes taxonomy and the dynamic T4 taxonomy are the same seven classes with the last
# two swapped. Stage 1 -> 2 therefore crosses taxonomies: `--weights` loads by shape, so the
# classification rows of the stage-1 checkpoint are permuted first (docs/models/streampetr.md).
NUSCENES_CLASSES = ["car", "truck", "bus", "bicycle", "pedestrian", "traffic_cone", "barrier"]
T4_CLASSES = ["car", "truck", "bus", "bicycle", "pedestrian", "barrier", "traffic_cone"]


def test_the_stages_are_bundled() -> None:
    """
    Input: the experiment configs bundled under tasks/detection3d/streampetr.
    Expected: the three stages of the flow are bundled, each as one leaf config and no other:
    the nuScenes pretrain, the j6gen2 fine-tune and the T4 gen1 and gen2 base.
    Check: the leaf names of the StreamPETR experiments, in the order the experiment configs are
    listed.
    """
    assert [stage.rsplit("/", 1)[1] for stage in STAGES] == [
        "vov99_cpfpn_320x800_b8_30e_nuscenes_51m_pretrain",
        "vov99_cpfpn_480x640_b16_35e_t4dataset_51m_j6gen2_base",
        "vov99_cpfpn_480x640_b8_35e_t4dataset_51m_gen1_gen2_base",
    ]


@pytest.mark.parametrize("config_name", STAGES)
def test_every_stage_streams_the_seven_classes_of_its_taxonomy(config_name: str) -> None:
    """
    Input: the composed config of every stage.
    Expected: every stage trains the seven classes in the order of its database taxonomy in both
    heads: the nuScenes order in stage 1, the dynamic T4 order, the one the other T4 detectors
    train and the deployed node decodes, in stages 2 and 3, on both databases of stage 2. The
    head carries a temporal memory across batches, so the streaming overlay makes the datamodule
    a StreamingDataModule seeded by the experiment, whose loaders do not shuffle and whose
    evaluation runs one lane per rank. The schedule is bound to the trainer, and only the
    initial weights are left for the command line.
    Check: the class names of the taxonomies and of both heads, the datamodule class, its seed
    and its keys against the arguments of StreamingDataModule, the dataloader settings, the
    max_epochs of the scheduler and the keys left missing.
    """
    cfg = compose_experiment(config_name)

    expected = T4_CLASSES if "/t4dataset/" in config_name else NUSCENES_CLASSES
    for database in (cfg.database, cfg.get("gen2_database")):
        if database is not None:
            taxonomy = hydra.utils.instantiate(database.taxonomy)
            assert list(taxonomy.detection3d.class_names) == expected
    assert list(OmegaConf.to_container(cfg.model.bbox_head.class_names)) == expected
    assert list(OmegaConf.to_container(cfg.model.img_roi_head.class_names)) == expected
    datamodule = cfg.datamodule
    assert hydra.utils.get_class(datamodule._target_) is StreamingDataModule
    assert datamodule.seed == cfg.seed
    parameters = set(inspect.signature(StreamingDataModule.__init__).parameters) - {"self"}
    assert set(datamodule) - {"_target_", "_convert_"} == parameters
    assert not cfg.datamodule.train_dataloader.shuffle
    assert cfg.datamodule.validation_dataloader.batch_size == 1
    assert cfg.datamodule.test_dataloader.batch_size == 1
    assert cfg.model.scheduler.max_epochs == cfg.trainer.max_epochs
    missing = [key for key in ("weights",) if OmegaConf.is_missing(cfg, key)]
    assert missing == ["weights"]


@pytest.mark.parametrize("config_name", STAGES)
def test_every_pipeline_filters_the_physical_boxes_before_anything_else(config_name: str) -> None:
    """
    Input: the training and test pipelines of every stage.
    Expected: a non-finite velocity or a zero extent would turn the losses NaN or infinite, so
    BBoxesPhysicalFilter drops such boxes before any other transform reads them.
    Check: the first step of every pipeline is BBoxesPhysicalFilter.
    """
    cfg = compose_experiment(config_name)
    for split in ("train", "test"):
        steps = [step["_target_"] for step in cfg.datasets[split].transforms.pipeline]
        assert steps[0].endswith("boxes3d.filters.BBoxesPhysicalFilter"), (config_name, split)


@pytest.mark.parametrize("config_name", STAGES)
def test_every_dataset_projecting_2d_boxes_pads_them_to_its_3d_maximum(config_name: str) -> None:
    """
    Input: the composed datasets of every stage.
    Expected: a dataset whose pipeline projects the 3D boxes onto the cameras collates its 2D
    boxes to a fixed count, without which its first batch is refused; a camera cannot see more
    2D boxes than its frame has 3D boxes, so the 3D maximum serves as the 2D one, on the gen2
    source of stage 2 too.
    Check: every dataset with LoadAnnotations2DFromBoxes3D in its pipeline, and at least the
    training dataset, sets max_num_2d_gt_bboxes to its max_num_3d_gt_bboxes.
    """
    cfg = compose_experiment(config_name)
    projecting = [
        name
        for name, dataset in cfg.datasets.items()
        if any(
            step["_target_"].endswith("LoadAnnotations2DFromBoxes3D")
            for step in dataset.transforms.pipeline
        )
    ]
    assert "train" in projecting
    for name in projecting:
        dataset = cfg.datasets[name]
        assert dataset.max_num_2d_gt_bboxes == dataset.max_num_3d_gt_bboxes, name


@pytest.mark.parametrize("config_name", STAGES)
def test_every_stage_projects_the_2d_boxes_before_the_range_filter(config_name: str) -> None:
    """
    Input: the training pipeline of every stage.
    Expected: the 2D targets of the reference recipes keep the objects outside the 3D range, so
    every stage projects its 2D boxes before BBoxesRangeFilter drops those objects; the other
    way round the auxiliary 2D head would learn that they are background.
    Check: LoadAnnotations2DFromBoxes3D comes before BBoxesRangeFilter in every training
    pipeline.
    """
    cfg = compose_experiment(config_name)
    steps = [step["_target_"].rsplit(".", 1)[1] for step in cfg.datasets.train.transforms.pipeline]
    assert steps.index("LoadAnnotations2DFromBoxes3D") < steps.index("BBoxesRangeFilter")


@pytest.mark.parametrize("config_name", [stage for stage in STAGES if "/t4dataset/" in stage])
def test_t4_stages_drop_the_low_point_boxes_after_the_2d_projection(config_name: str) -> None:
    """The boxes recorded with too few lidar points leave the 3D targets but stay 2D targets.

    The evaluation drops these boxes, so the 3D head does not train on them, while a camera can
    still see them. Every T4 training dataset, the gen2 source of stage 2 too, therefore runs
    BBoxesMinRecordedPointsFilter after LoadAnnotations2DFromBoxes3D.
    """
    cfg = compose_experiment(config_name)
    training = [name for name in cfg.datasets if name.endswith("train")]
    assert "train" in training
    for name in training:
        steps = [
            step["_target_"].rsplit(".", 1)[1] for step in cfg.datasets[name].transforms.pipeline
        ]
        assert steps.index("LoadAnnotations2DFromBoxes3D") < steps.index(
            "BBoxesMinRecordedPointsFilter"
        ), name


@pytest.mark.parametrize("config_name", STAGES)
def test_only_the_t4_training_pipelines_shuffle_the_loaded_cameras(config_name: str) -> None:
    """
    Input: the training and test pipelines of every stage.
    Expected: the T4 reference recipe shuffles the camera order of every training frame, and
    never at evaluation, while the nuScenes reference recipe does not shuffle at all.
    Check: ShuffleCameraOrder directly follows LoadImagesFromFile in the T4 training pipelines
    and appears in no other pipeline.
    """
    cfg = compose_experiment(config_name)
    for split in ("train", "test"):
        steps = [
            step["_target_"].rsplit(".", 1)[1] for step in cfg.datasets[split].transforms.pipeline
        ]
        shuffles = split == "train" and "/t4dataset/" in config_name
        loading = steps.index("LoadImagesFromFile")
        assert (steps[loading + 1] == "ShuffleCameraOrder") == shuffles, split
        assert ("ShuffleCameraOrder" in steps) == shuffles, split


@pytest.mark.parametrize("config_name", [stage for stage in STAGES if "/t4dataset/" in stage])
def test_t4_stages_ignore_the_cones_and_barriers_of_the_scenes_without_them(
    config_name: str,
) -> None:
    """
    Input: the partial annotations of the T4 stages, instantiated from their configs.
    Expected: the T4 scenarios flag whether their traffic cones and barriers are annotated, and
    where they are not, those two classes are not supervised; the pair resolves by name,
    whichever order the taxonomy lists it in.
    Check: the class ids the partial annotations ignore, and the class names at those ids.
    """
    cfg = compose_experiment(config_name)
    partial_annotations = hydra.utils.instantiate(cfg.model.partial_annotations)
    assert list(partial_annotations.partial_ignore_class_ids) == [5, 6]
    assert {T4_CLASSES[index] for index in partial_annotations.partial_ignore_class_ids} == {
        "traffic_cone",
        "barrier",
    }


def test_the_nuscenes_stage_supervises_every_class() -> None:
    """
    Input: the composed config of the nuScenes stage.
    Expected: nuScenes annotates every class in every frame, so the stage has no partial
    annotations.
    Check: model.partial_annotations is None.
    """
    cfg = compose_experiment([stage for stage in STAGES if "/nuscenes/" in stage][0])
    assert cfg.model.partial_annotations is None


def test_the_gen_base_stage_joins_the_gen1_and_gen2_corpora() -> None:
    """
    Input: the composed config of stage 2, the T4 gen1 and gen2 base.
    Expected: the main source reads the gen1 base corpus and a second source the gen2 base
    corpus, both through the same split datasets, each one only from its own database.
    Check: the database versions, the source names, and the transforms, cameras and root path of
    the gen2 datasets.
    """
    cfg = compose_experiment([stage for stage in STAGES if "gen1_gen2" in stage][0])
    assert "gen1_base" in cfg.database.version
    assert "gen2_base" in cfg.gen2_database.version
    assert set(cfg.datamodule.sources) == {"main", "gen2"}
    for split in ("train", "test"):
        gen2 = cfg.datasets[f"gen2_{split}"]
        assert gen2.transforms == cfg.datasets[split].transforms
        assert gen2.camera_names == cfg.datasets[split].camera_names
        assert gen2.database_root_path == cfg.gen2_database.root_path


@pytest.mark.parametrize("config_name", STAGES)
def test_every_pipeline_loads_unit_range_images_for_the_normalization(config_name: str) -> None:
    """BatchNormalizeImages trusts its input, so the loaders keep the unit-range default.

    The statistics are unit-range and the layer does not check the images: 0-255 pixels would
    train without an error. It is the first preprocessing module, and every pipeline loads its
    images with LoadImagesFromFile(normalize_to_unit=True).
    """
    cfg = compose_experiment(config_name)
    modules = [
        module["_target_"].rsplit(".", 1)[1]
        for module in cfg.data_preprocessor.preprocessor_modules
    ]
    assert modules[0] == "BatchNormalizeImages"
    for split in ("train", "test"):
        loaders = [
            step
            for step in cfg.datasets[split].transforms.pipeline
            if step["_target_"].endswith("LoadImagesFromFile")
        ]
        assert len(loaders) == 1, split
        assert loaders[0].get("normalize_to_unit", True) is True, split
