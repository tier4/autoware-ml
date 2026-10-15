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

"""Unit tests for the model builders: the data preprocessor, the model with its weight loading
options, and the ``--weights`` checkpoint path resolution."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import pytest
import torch

from autoware_ml.builders.model_builder import (
    build_data_preprocessor,
    build_model,
    build_weight_checkpoint_paths,
)
from autoware_ml.models.module_base_model import ModuleBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor


class _TinyModel(ModuleBaseModel):
    """A two-parameter model the configs instantiate through Hydra."""

    def __init__(self, data_preprocessor: DataPreprocessor, hidden_size: int) -> None:
        super().__init__(data_preprocessor=data_preprocessor)
        self.layer = torch.nn.Linear(hidden_size, 1)


_MODULE = "autoware_ml.builders.tests.test_model_builder"
_HIDDEN_SIZE = 3
_CPU = torch.device("cpu")


def _model_cfg() -> DictConfig:
    """The ``data_preprocessor`` and ``model`` nodes the training config provides."""
    return OmegaConf.create(
        {
            "data_preprocessor": {
                "_target_": "autoware_ml.preprocessing.data_preprocessor.DataPreprocessor",
                "preprocessor_modules": [],
            },
            "model": {"_target_": f"{_MODULE}._TinyModel", "hidden_size": _HIDDEN_SIZE},
        }
    )


def _save_state_dict(path: Path, state_dict: dict[str, torch.Tensor]) -> Path:
    """Write a Lightning-style checkpoint holding only ``state_dict``."""
    torch.save({"state_dict": state_dict}, path)
    return path


def _build(
    cfg: DictConfig,
    weights_path: str | Path | list[Path] | None = None,
    resume_checkpoint_path: str | None = None,
    set_eval: bool = False,
    enforce_full_coverage: bool = False,
    skip_mismatched_weights: bool = False,
) -> _TinyModel:
    """Build the tiny model the way the entrypoints do, on CPU."""
    model = build_model(
        cfg,
        data_preprocessor=build_data_preprocessor(cfg),
        device=_CPU,
        weights_path=weights_path,
        resume_checkpoint_path=resume_checkpoint_path,
        set_eval=set_eval,
        enforce_full_coverage=enforce_full_coverage,
        skip_mismatched_weights=skip_mismatched_weights,
    )
    assert isinstance(model, _TinyModel)
    return model


class TestBuildDataPreprocessor:
    """``build_data_preprocessor`` instantiates ``cfg.data_preprocessor``."""

    def test_builds_an_empty_preprocessor_from_the_default_node(self) -> None:
        """
        Input: the default ``data_preprocessor`` node with no modules.
        Expected: a :class:`DataPreprocessor` running no module.
        Check: the type and the empty module tuple.
        """
        preprocessor = build_data_preprocessor(_model_cfg())

        assert isinstance(preprocessor, DataPreprocessor)
        assert preprocessor.preprocessor_modules == ()

    def test_requires_the_data_preprocessor_key(self) -> None:
        """
        Input: a config without a ``data_preprocessor`` node.
        Expected: the builder fails on the missing key.
        Check: an AttributeError naming the key is raised.
        """
        cfg = OmegaConf.create({"model": {"_target_": f"{_MODULE}._TinyModel"}})

        with pytest.raises(AttributeError, match="data_preprocessor"):
            build_data_preprocessor(cfg)


class TestBuildModel:
    """``build_model`` instantiates ``cfg.model`` with the preprocessor and loads weights."""

    def test_builds_the_model_with_the_preprocessor_in_training_mode(self) -> None:
        """
        Input: the minimal config, no weights and no resume checkpoint.
        Expected: the model target receives the config arguments and the built preprocessor and
            stays in training mode.
        Check: the type, the layer size, the attached preprocessor and ``training``.
        """
        cfg = _model_cfg()
        preprocessor = build_data_preprocessor(cfg)

        model = build_model(
            cfg,
            data_preprocessor=preprocessor,
            device=_CPU,
            weights_path=None,
            resume_checkpoint_path=None,
        )

        assert isinstance(model, _TinyModel)
        assert model.layer.in_features == _HIDDEN_SIZE
        assert model._data_preprocessor is preprocessor
        assert model.training

    def test_requires_the_model_key(self) -> None:
        """
        Input: a config without a ``model`` node.
        Expected: the builder fails on the missing key.
        Check: an AttributeError naming the key is raised.
        """
        cfg = _model_cfg()
        del cfg["model"]

        with pytest.raises(AttributeError, match="model"):
            build_model(
                cfg,
                data_preprocessor=build_data_preprocessor(cfg),
                device=_CPU,
                weights_path=None,
                resume_checkpoint_path=None,
            )

    def test_loads_matching_weights_and_sets_eval(self, tmp_path: Path) -> None:
        """
        Input: one checkpoint covering both parameters and ``set_eval=True``.
        Expected: the tensors are copied into the model and it switches to eval mode, as the
            test and deploy entrypoints need.
        Check: the parameter values and ``training``.
        """
        weight = torch.full((1, _HIDDEN_SIZE), 2.0)
        bias = torch.full((1,), -1.0)
        weights_path = _save_state_dict(
            tmp_path / "best.ckpt", {"layer.weight": weight, "layer.bias": bias}
        )

        model = _build(_model_cfg(), weights_path=weights_path, set_eval=True)

        assert torch.equal(model.layer.weight, weight)
        assert torch.equal(model.layer.bias, bias)
        assert not model.training

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device to move to")
    def test_moves_the_model_to_the_requested_device_without_weights(self) -> None:
        """
        Input: the minimal config, no weights, ``device=cuda``.
        Expected: the built model lives on the CUDA device, so the deploy entrypoint, which reads
            ``model.device`` to place the example batch and the trace, exports on the GPU.
        Check: every parameter's device and ``model.device``.
        """
        cuda = torch.device("cuda")
        cfg = _model_cfg()

        model = build_model(
            cfg,
            data_preprocessor=build_data_preprocessor(cfg),
            device=cuda,
            weights_path=None,
            resume_checkpoint_path=None,
        )

        assert model.device.type == "cuda"
        assert all(parameter.device.type == "cuda" for parameter in model.parameters())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device to move to")
    def test_moves_the_model_to_the_requested_device_with_weights(self, tmp_path: Path) -> None:
        """
        Input: one CPU checkpoint, ``device=cuda`` and ``set_eval=True``, as the deploy
            entrypoint calls it.
        Expected: the weights are loaded and the model ends up on the CUDA device in eval mode.
        Check: the parameter devices and values, ``model.device`` and ``training``.
        """
        cuda = torch.device("cuda")
        weight = torch.full((1, _HIDDEN_SIZE), 2.0)
        bias = torch.full((1,), -1.0)
        weights_path = _save_state_dict(
            tmp_path / "best.ckpt", {"layer.weight": weight, "layer.bias": bias}
        )
        cfg = _model_cfg()

        model = build_model(
            cfg,
            data_preprocessor=build_data_preprocessor(cfg),
            device=cuda,
            weights_path=weights_path,
            resume_checkpoint_path=None,
            set_eval=True,
        )

        assert model.device.type == "cuda"
        assert all(parameter.device.type == "cuda" for parameter in model.parameters())
        assert torch.equal(model.layer.weight.cpu(), weight)
        assert torch.equal(model.layer.bias.cpu(), bias)
        assert not model.training

    def test_merges_several_checkpoints_in_order(self, tmp_path: Path) -> None:
        """
        Input: two checkpoints, each covering one parameter, and ``enforce_full_coverage=True``.
        Expected: the union of the checkpoints covers the model, so the build succeeds with both
            tensors loaded.
        Check: the parameter values.
        """
        weight = torch.full((1, _HIDDEN_SIZE), 3.0)
        bias = torch.full((1,), 0.5)
        weight_ckpt = _save_state_dict(tmp_path / "weight.ckpt", {"layer.weight": weight})
        bias_ckpt = _save_state_dict(tmp_path / "bias.ckpt", {"layer.bias": bias})

        model = _build(
            _model_cfg(), weights_path=[weight_ckpt, bias_ckpt], enforce_full_coverage=True
        )

        assert torch.equal(model.layer.weight, weight)
        assert torch.equal(model.layer.bias, bias)

    def test_full_coverage_rejects_a_checkpoint_missing_a_parameter(self, tmp_path: Path) -> None:
        """
        Input: a checkpoint holding only the weight and ``enforce_full_coverage=True``.
        Expected: the bias stays uninitialized, which the deploy contract forbids.
        Check: a RuntimeError naming the uncovered key is raised.
        """
        weights_path = _save_state_dict(
            tmp_path / "partial.ckpt", {"layer.weight": torch.ones(1, _HIDDEN_SIZE)}
        )

        with pytest.raises(RuntimeError, match="layer.bias"):
            _build(_model_cfg(), weights_path=weights_path, enforce_full_coverage=True)

    def test_partial_checkpoint_is_accepted_without_full_coverage(self, tmp_path: Path) -> None:
        """
        Input: a checkpoint holding only the weight, with the training defaults.
        Expected: the weight loads and the bias keeps its initialization.
        Check: the weight value and the unchanged bias.
        """
        weight = torch.full((1, _HIDDEN_SIZE), 4.0)
        weights_path = _save_state_dict(tmp_path / "partial.ckpt", {"layer.weight": weight})
        bias_before = _TinyModel(DataPreprocessor(preprocessor_modules=[]), _HIDDEN_SIZE)

        model = _build(_model_cfg(), weights_path=weights_path)

        assert torch.equal(model.layer.weight, weight)
        assert model.layer.bias.shape == bias_before.layer.bias.shape

    def test_mismatched_shape_raises_by_default(self, tmp_path: Path) -> None:
        """
        Input: a checkpoint whose bias has a different shape than the model's.
        Expected: without ``--skip-mismatched-weights`` the load fails loudly.
        Check: a ValueError mentioning incompatible shapes is raised.
        """
        weights_path = _save_state_dict(
            tmp_path / "mismatch.ckpt",
            {"layer.weight": torch.ones(1, _HIDDEN_SIZE), "layer.bias": torch.ones(2)},
        )

        with pytest.raises(ValueError, match="incompatible shapes"):
            _build(_model_cfg(), weights_path=weights_path)

    def test_mismatched_shape_is_skipped_with_a_warning_when_requested(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Input: the same mismatched checkpoint with ``skip_mismatched_weights=True``.
        Expected: the matching weight loads, the mismatched bias keeps its initialization and a
            warning names it.
        Check: the parameter values, shapes and the warning text.
        """
        caplog.set_level("WARNING")
        weight = torch.full((1, _HIDDEN_SIZE), 5.0)
        weights_path = _save_state_dict(
            tmp_path / "mismatch.ckpt", {"layer.weight": weight, "layer.bias": torch.ones(2)}
        )

        model = _build(_model_cfg(), weights_path=weights_path, skip_mismatched_weights=True)

        assert torch.equal(model.layer.weight, weight)
        assert model.layer.bias.shape == (1,)
        assert "layer.bias" in caplog.text

    def test_mismatched_shape_skipped_still_counts_as_uncovered(self, tmp_path: Path) -> None:
        """
        Input: the mismatched checkpoint with both ``skip_mismatched_weights`` and
            ``enforce_full_coverage``.
        Expected: the skipped bias is not covered by any checkpoint, so full coverage fails.
        Check: a RuntimeError naming the bias is raised.
        """
        weights_path = _save_state_dict(
            tmp_path / "mismatch.ckpt",
            {"layer.weight": torch.ones(1, _HIDDEN_SIZE), "layer.bias": torch.ones(2)},
        )

        with pytest.raises(RuntimeError, match="layer.bias"):
            _build(
                _model_cfg(),
                weights_path=weights_path,
                skip_mismatched_weights=True,
                enforce_full_coverage=True,
            )

    def test_weights_and_resume_checkpoint_are_mutually_exclusive(self, tmp_path: Path) -> None:
        """
        Input: both a weights path and a resume checkpoint path.
        Expected: the builder refuses the combination the CLI also rejects.
        Check: a ValueError mentioning mutual exclusion is raised.
        """
        weights_path = _save_state_dict(tmp_path / "best.ckpt", {})

        with pytest.raises(ValueError, match="mutually exclusive"):
            _build(
                _model_cfg(),
                weights_path=weights_path,
                resume_checkpoint_path=str(tmp_path / "last.ckpt"),
            )

    def test_resume_checkpoint_only_reports_progress(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Input: a resume checkpoint carrying the epoch and global step, no weights.
        Expected: the builder leaves weight restoration to the trainer and only logs where
            training continues.
        Check: the log mentions the checkpoint, the saved epoch and the next epoch, and the
            model keeps its fresh initialization in training mode.
        """
        caplog.set_level("INFO")
        resume_path = tmp_path / "last.ckpt"
        torch.save(
            {
                "epoch": 3,
                "global_step": 40,
                "state_dict": {"layer.weight": torch.full((1, _HIDDEN_SIZE), 9.0)},
            },
            resume_path,
        )

        model = _build(_model_cfg(), resume_checkpoint_path=str(resume_path))

        assert not torch.equal(model.layer.weight, torch.full((1, _HIDDEN_SIZE), 9.0))
        assert model.training
        assert f"Resuming from '{resume_path}'" in caplog.text
        assert "epoch 3 (global step 40)" in caplog.text
        assert "continues at epoch 4" in caplog.text


class TestBuildWeightCheckpointPaths:
    """``build_weight_checkpoint_paths`` resolves ``cfg.weights`` for test and deploy."""

    def test_requires_the_weights_key(self) -> None:
        """
        Input: a config without ``weights``.
        Expected: the test and deploy entrypoints cannot run without a checkpoint.
        Check: a ValueError mentioning ``--weights`` is raised.
        """
        with pytest.raises(ValueError, match="--weights"):
            build_weight_checkpoint_paths(OmegaConf.create({}))

    def test_single_string_is_both_the_weights_and_the_checkpoint(self, tmp_path: Path) -> None:
        """
        Input: ``weights`` given as one path string.
        Expected: a one-element list of paths whose last element is the lineage checkpoint.
        Check: both return values.
        """
        checkpoint = _save_state_dict(tmp_path / "best.ckpt", {})

        weight_paths, checkpoint_path = build_weight_checkpoint_paths(
            OmegaConf.create({"weights": str(checkpoint)})
        )

        assert list(weight_paths) == [checkpoint]
        assert checkpoint_path == checkpoint

    def test_last_entry_of_a_list_is_the_lineage_checkpoint(self, tmp_path: Path) -> None:
        """
        Input: ``weights`` given as the list the CLI builds from repeated ``--weights``.
        Expected: every path is kept in order and the last one names the lineage checkpoint.
        Check: both return values.
        """
        encoder = _save_state_dict(tmp_path / "encoder.ckpt", {})
        head = _save_state_dict(tmp_path / "head.ckpt", {})

        weight_paths, checkpoint_path = build_weight_checkpoint_paths(
            OmegaConf.create({"weights": [str(encoder), str(head)]})
        )

        assert list(weight_paths) == [encoder, head]
        assert checkpoint_path == head

    def test_missing_file_is_rejected(self, tmp_path: Path) -> None:
        """
        Input: a ``weights`` list whose second entry does not exist.
        Expected: the missing file is reported before any model is built.
        Check: a FileNotFoundError naming the missing path is raised.
        """
        present = _save_state_dict(tmp_path / "present.ckpt", {})
        missing = tmp_path / "missing.ckpt"

        with pytest.raises(FileNotFoundError, match="missing.ckpt"):
            build_weight_checkpoint_paths(
                OmegaConf.create({"weights": [str(present), str(missing)]})
            )
