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

"""Unit tests for the PTv3 deployment graphs."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
import onnx
import torch

from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.deployment.graphs import GraphTensor
from autoware_ml.deployment.tests.deployment_assertions import (
    assert_deployment_reproduces_forward,
    assert_specs_export_onnx_with_the_declared_io,
)
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    CLASS_NAMES,
    build_inputs,
    build_seg3d_model,
    build_trans_model,
    whole_level_patch_sizes,
)
from autoware_ml.models.segmentation3d.ptv3_base import (
    DET3D_HEAD_GRAPH,
    ENCODER_EXPORT_INPUT_NAMES,
    ENCODER_GRAPH,
    SEG3D_HEAD_GRAPH,
    SEG3D_HEAD_OUTPUT_NAMES,
    encoder_graph_input_names,
    subm_pair_names,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.utils.deploy import export_to_onnx

REQUIRES_SPARSE_CUDA = unittest.skipUnless(
    IS_SPCONV_AVAILABLE and torch.cuda.is_available(), "PTv3 export needs spconv and CUDA"
)


class TestDeclarationKeepsTheModelMode(unittest.TestCase):
    """Declaring the graphs changes nothing on the model."""

    def test_declaring_the_graphs_leaves_the_live_voxel_encoder_in_training_mode(self) -> None:
        """Test that declaring the graphs does not switch the module the graphs share with the model.

        Input: the segmentation and the detection model in training mode.
        Expected: the voxel encoder, the one live module the encoder graph wraps, is still in
            training mode after the declaration; the framework switches modes at export only.
        Check: ``voxel_encoder.training`` after ``deployment_declaration()``.
        """
        for model in (build_seg3d_model(), build_trans_model()):
            model.train()
            model.deployment_declaration()
            self.assertTrue(model.voxel_encoder.training, type(model).__name__)


@REQUIRES_SPARSE_CUDA
class TestSeg3dDeployment(unittest.TestCase):
    """The segmentation model's deployment graphs and the export specs derived from them."""

    def setUp(self) -> None:
        """Build the segmentation model and one batch on the GPU."""
        self.model = build_seg3d_model().cuda().eval()
        self.batch = build_inputs(device=torch.device("cuda"))

    def test_seg3d_declaration_and_specs(self) -> None:
        """Test that the segmentation model declares the encoder and the seg3d head graphs.

        Input: the segmentation model's deployment declaration, its export specs and one
            serialized batch.
        Expected: the graphs are the encoder and the seg3d head, named as the runtime names
            them; every graph input is a serialized tensor or an encoder output; the encoder
            spec takes the encoder graph's inputs with the voxel count dynamic, and both specs'
            modules run on their args.
        Check: the graph names, the encoder's leading input names, the head's output names, the
            input names against the serialized tensors, the spec input names, the dynamic axis
            of ``voxels``, and the shapes of the level features and of the head's labels and
            probabilities.
        """
        graphs = {graph.name: graph for graph in self.model.deployment_declaration().graphs}
        self.assertEqual(set(graphs), {ENCODER_GRAPH, SEG3D_HEAD_GRAPH})
        encoder, head = graphs[ENCODER_GRAPH], graphs[SEG3D_HEAD_GRAPH]
        self.assertEqual(encoder.input_names[:5], ENCODER_EXPORT_INPUT_NAMES)
        self.assertFalse(any(name.endswith("_cluster") for name in encoder.input_names))
        self.assertEqual(head.output_names, ("pred_labels", "pred_probs"))
        tensors = self.model.serialize_points(self.batch)
        self.assertLessEqual(set(encoder.input_names), set(tensors))
        self.assertLessEqual(set(head.input_names), set(tensors) | set(encoder.output_names))

        specs = self.model.build_export_specs(self.batch)
        self.assertEqual(set(specs), set(graphs))
        encoder_spec, head_spec = specs[ENCODER_GRAPH], specs[SEG3D_HEAD_GRAPH]
        self.assertEqual(
            encoder_spec.input_names,
            list(encoder_graph_input_names(len(self.model.encoder.stride))),
        )
        self.assertEqual(encoder_spec.dynamic_axes["voxels"], {0: "num_voxels"})
        with torch.no_grad():
            level_feats = encoder_spec.module(*encoder_spec.args)
            pred_labels, pred_probs = head_spec.module(*head_spec.args)
        num_voxels = self.batch.voxels_data.voxels.shape[0]
        self.assertEqual(len(level_feats), len(self.model.encoder.stride) + 1)
        self.assertEqual(pred_labels.shape, (num_voxels,))
        self.assertEqual(pred_probs.shape[0], num_voxels)

    def test_seg3d_specs_export_onnx_with_the_declared_io_contract(self) -> None:
        """Test that each derived spec exports an ONNX with the graph's declared names and axes.

        Input: the segmentation model's specs, exported with the base config's ONNX settings.
        Expected: each ONNX carries the graph's input and output names and its dynamic axes,
            the seg3d head's outputs included.
        Check: the seg3d head's outputs among its declared axes, then
            assert_specs_export_onnx_with_the_declared_io().
        """
        graphs = {graph.name: graph for graph in self.model.deployment_graphs()}
        self.assertLessEqual(
            set(SEG3D_HEAD_OUTPUT_NAMES), set(graphs[SEG3D_HEAD_GRAPH].onnx_dynamic_axes)
        )
        assert_specs_export_onnx_with_the_declared_io(self.model, self.batch)


@REQUIRES_SPARSE_CUDA
class TestSeg3dDeploymentReusingSubmPairs(unittest.TestCase):
    """The segmentation export in which the head reuses the encoder's index pairs."""

    def setUp(self) -> None:
        """Build one batch on the GPU."""
        self.batch = build_inputs(device=torch.device("cuda"))

    def _export(self, export_subm_pairs: bool) -> dict[str, onnx.GraphProto]:
        """Export the segmentation model's graphs with the base config's ONNX settings.

        Args:
            export_subm_pairs: Whether the export hands the encoder's index pairs to the head.

        Returns:
            The ONNX graph of every deployment graph, by name.
        """
        model = build_seg3d_model(export_subm_pairs=export_subm_pairs).cuda().eval()
        onnx_cfg = OmegaConf.create(
            {"opset_version": 17, "dynamo": False, "do_constant_folding": False}
        )
        graphs = {}
        with tempfile.TemporaryDirectory() as directory:
            for name, spec in model.build_export_specs(self.batch).items():
                path = Path(directory) / f"{name}.onnx"
                export_to_onnx(spec, onnx_cfg, path)
                graphs[name] = onnx.load(path).graph
        return graphs

    def test_the_encoder_hands_its_pairs_to_the_head(self) -> None:
        """Test that the encoder outputs the pairs the head convolves with, under one name.

        Input: the segmentation model's ONNX with and without ``export_subm_pairs``; its one
            decoder level has a block.
        Expected: the encoder outputs the level's pairs after the level features, with both
            axes of the pair table dynamic; the head takes them as its last inputs, builds no
            pairs and no longer reads the level's voxel coordinates. The default head builds
            pairs.
        Check: the ONNX output and input names, the pair table's ``dim_param`` and the
            ``GetIndicePairsImplicitGemm`` nodes of both heads.
        """
        pairs = subm_pair_names([0])
        graphs = self._export(export_subm_pairs=True)
        encoder, head = graphs[ENCODER_GRAPH], graphs[SEG3D_HEAD_GRAPH]
        self.assertEqual([tensor.name for tensor in encoder.output][-3:], pairs)
        head_inputs = {tensor.name: tensor for tensor in head.input}
        self.assertEqual(list(head_inputs)[-3:], pairs)
        self.assertNotIn("grid_coord", head_inputs)
        dims = head_inputs[pairs[0]].type.tensor_type.shape.dim
        self.assertEqual([dim.dim_param for dim in dims], [f"{pairs[0]}_rows", f"{pairs[0]}_cols"])
        builders = [node for node in head.node if node.op_type == "GetIndicePairsImplicitGemm"]
        self.assertEqual(builders, [])
        default_head = self._export(export_subm_pairs=False)[SEG3D_HEAD_GRAPH]
        default_ops = {node.op_type for node in default_head.node}
        self.assertIn("GetIndicePairsImplicitGemm", default_ops)


@REQUIRES_SPARSE_CUDA
class TestDet3dDeployment(unittest.TestCase):
    """The detection model's deployment graphs and their specs."""

    def setUp(self) -> None:
        """Build the detection model and one batch on the GPU."""
        self.model = build_trans_model().cuda().eval()
        self.batch = build_inputs(device=torch.device("cuda"))

    def test_det3d_declaration_and_specs(self) -> None:
        """Test that the detection model declares the encoder and the det3d head graphs.

        Input: the detection model's deployment declaration and its export specs.
        Expected: the graphs and the specs are the encoder and the det3d head; the head spec
            emits the configured export output names, one tensor each, the labels as int64 and
            the dense heatmap as one map per class.
        Check: the graph names, the spec names, the head spec's output names, and the count,
            the ``query_labels`` dtype and the ``dense_heatmap`` leading dims of a run.
        """
        graphs = {graph.name: graph for graph in self.model.deployment_declaration().graphs}
        self.assertEqual(set(graphs), {ENCODER_GRAPH, DET3D_HEAD_GRAPH})
        specs = self.model.build_export_specs(self.batch)
        self.assertEqual(set(specs), set(graphs))
        head_spec = specs[DET3D_HEAD_GRAPH]
        self.assertEqual(head_spec.output_names, list(self.model.bbox_head.export_output_names))
        with torch.no_grad():
            outputs = dict(zip(head_spec.output_names, head_spec.module(*head_spec.args)))
        self.assertEqual(len(outputs), len(head_spec.output_names))
        self.assertEqual(outputs["query_labels"].dtype, torch.int64)
        self.assertEqual(outputs["dense_heatmap"].shape[:2], (1, len(CLASS_NAMES)))

    def test_det3d_specs_export_onnx_with_the_declared_io_contract(self) -> None:
        """Test that the encoder and the det3d head specs export ONNX with their graphs' names.

        Input: the detection model's specs, exported with the base config's ONNX settings.
        Expected: each ONNX carries the graph's input and output names and its dynamic axes.
        Check: assert_specs_export_onnx_with_the_declared_io().
        """
        assert_specs_export_onnx_with_the_declared_io(self.model, self.batch)


@REQUIRES_SPARSE_CUDA
class TestDeploymentReproducesForward(unittest.TestCase):
    """The deployed PTv3 models against forward() on one batch."""

    def setUp(self) -> None:
        """Build one batch and the windows under which forward() attends like the export."""
        self.device = torch.device("cuda")
        self.batch = build_inputs(device=self.device)
        self.patch_sizes = whole_level_patch_sizes(self.batch)

    @staticmethod
    def _seg3d_tensors(outputs: ModelOutputs) -> dict[str, GraphTensor]:
        """Return forward()'s labels and probabilities under the seg3d head graph's names.

        Args:
            outputs: The outputs of ``forward()``.

        Returns:
            The label and the class probabilities of every voxel.
        """
        probs = torch.softmax(outputs.segmentation3d().logits, dim=1)
        return {"pred_labels": probs.argmax(dim=1), "pred_probs": probs}

    @staticmethod
    def _det3d_tensors(outputs: ModelOutputs, names: Sequence[str]) -> dict[str, GraphTensor]:
        """Return forward()'s detection head outputs under the det3d head graph's names.

        Args:
            outputs: The outputs of ``forward()``.
            names: The det3d head graph's output names.

        Returns:
            The head outputs by name.
        """
        head_outputs = outputs.detection3d().transfusion_head()
        return dict(zip(names, head_outputs.export_tensors(names)))

    def test_seg3d_deployment_reproduces_forward(self) -> None:
        """Test that the segmentation deployment reproduces forward().

        Input: a segmentation model whose window at every level is that level's voxel count,
            and one batch.
        Expected: the deployed labels equal forward()'s and the probabilities match to 1e-5.
        Check: assert_deployment_reproduces_forward().
        """
        model = build_seg3d_model(patch_sizes=self.patch_sizes).cuda().eval()
        assert_deployment_reproduces_forward(
            model, self.batch, self.device, self._seg3d_tensors, rtol=1e-5, atol=1e-5
        )

    def test_seg3d_deployment_reusing_subm_pairs_reproduces_forward(self) -> None:
        """Test that the segmentation deployment reproduces forward() when the head reuses pairs.

        Input: a segmentation model with ``export_subm_pairs`` whose window at every level is
            that level's voxel count, and one batch.
        Expected: the deployed labels equal forward()'s and the probabilities match to 1e-5.
        Check: assert_deployment_reproduces_forward().
        """
        model = build_seg3d_model(patch_sizes=self.patch_sizes, export_subm_pairs=True)
        assert_deployment_reproduces_forward(
            model.cuda().eval(), self.batch, self.device, self._seg3d_tensors, rtol=1e-5, atol=1e-5
        )

    def test_det3d_deployment_reproduces_forward(self) -> None:
        """Test that the detection deployment reproduces forward().

        Input: a detection model whose window at every level is that level's voxel count, and
            one batch.
        Expected: every head output of the deployment matches forward()'s to 1e-5.
        Check: assert_deployment_reproduces_forward().
        """
        model = build_trans_model(patch_sizes=self.patch_sizes).cuda().eval()
        graphs = {graph.name: graph for graph in model.deployment_graphs()}
        names = graphs[DET3D_HEAD_GRAPH].output_names
        assert_deployment_reproduces_forward(
            model,
            self.batch,
            self.device,
            partial(self._det3d_tensors, names=names),
            rtol=1e-5,
            atol=1e-5,
        )
