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

from pathlib import Path
import tempfile
import unittest

import onnx
import torch
from omegaconf import OmegaConf

from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import (
    build_inputs,
    build_seg3d_model,
    build_trans_model,
)
from autoware_ml.models.segmentation3d.ptv3_base import (
    DET3D_HEAD_GRAPH,
    ENCODER_EXPORT_INPUT_NAMES,
    ENCODER_GRAPH,
    SEG3D_HEAD_OUTPUT_NAMES,
    SEG3D_HEAD_GRAPH,
    build_seg3d_head_graph,
    encoder_graph_input_names,
    serialize_points,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.utils.deploy import export_to_onnx

REQUIRES_SPARSE_CUDA = unittest.skipUnless(
    IS_SPCONV_AVAILABLE and torch.cuda.is_available(), "PTv3 export needs spconv and CUDA"
)


@REQUIRES_SPARSE_CUDA
class TestSeg3dDeployment(unittest.TestCase):
    """The segmentation model's deployment graphs and the export specs derived from them."""

    def test_seg3d_declaration_names_the_runtime_modules_and_is_name_covered(self) -> None:
        """Test that the seg3d declaration names the runtime modules and every input is produced.

        Input: the segmentation model's deployment declaration and one serialized batch.
        Expected: the graphs are the encoder and the seg3d head, named and ordered as the
            runtime binds them; every graph input is a serialized tensor or an encoder output.
        Check: the graph names, the encoder's leading input names, the head's output names, and
            the input names against the serialized tensors.
        """
        model = build_seg3d_model().cuda().eval()
        graphs = model.deployment_declaration().graphs
        self.assertEqual([graph.name for graph in graphs], [ENCODER_GRAPH, SEG3D_HEAD_GRAPH])
        encoder, head = graphs
        self.assertEqual(encoder.input_names[:5], ENCODER_EXPORT_INPUT_NAMES)
        self.assertFalse(any(name.endswith("_cluster") for name in encoder.input_names))
        self.assertEqual(head.output_names, ("pred_labels", "pred_probs"))
        # The serialized points hold every graph input the encoder does not produce.
        tensors = serialize_points(
            model, build_inputs(device=torch.device("cuda")), torch.device("cuda")
        )
        self.assertLessEqual(set(encoder.input_names), set(tensors))
        self.assertLessEqual(set(head.input_names), set(tensors) | set(encoder.output_names))

    def test_seg3d_export_specs_are_derived_and_keep_the_split_contract(self) -> None:
        """Test that the export specs derived from the graphs keep the split export contract.

        Input: the segmentation model and one batch.
        Expected: an encoder spec and a seg3d head spec whose inputs follow the encoder graph's
            input names, with the voxel count dynamic, and whose modules run on their args.
        Check: the spec names and input names, the dynamic axis of ``voxels``, and the shapes
            of the level features and of the head's labels and probabilities.
        """
        model = build_seg3d_model().cuda().eval()
        batch = build_inputs(device=torch.device("cuda"))
        specs = model.build_export_specs(batch)
        self.assertEqual(list(specs), ["ptv3_encoder", "ptv3_seg3d_head"])
        encoder_spec = specs["ptv3_encoder"]
        self.assertEqual(
            encoder_spec.input_names, encoder_graph_input_names(len(model.encoder.stride))
        )
        self.assertIn("serialized_pooling_0_patch_order", encoder_spec.input_names)
        self.assertFalse(any("_cluster" in name for name in encoder_spec.input_names))
        self.assertEqual(encoder_spec.dynamic_axes["voxels"], {0: "num_voxels"})
        with torch.no_grad():
            level_feats = encoder_spec.module(*encoder_spec.args)
            pred_labels, pred_probs = specs["ptv3_seg3d_head"].module(
                *specs["ptv3_seg3d_head"].args
            )
        num_voxels = batch.voxels_data.voxels.shape[0]
        self.assertEqual(len(level_feats), len(model.encoder.stride) + 1)
        self.assertEqual(pred_labels.shape, (num_voxels,))
        self.assertEqual(pred_probs.shape[0], num_voxels)

    def test_derived_specs_export_onnx_with_the_declared_io_contract(self) -> None:
        """Test that each derived spec exports an ONNX with the graph's declared names and axes.

        Input: the segmentation model's specs, exported with the base config's ONNX settings.
        Expected: each ONNX carries the graph's input and output names and its dynamic axes.
        Check: the ONNX input and output names and the ``dim_param`` of every declared axis.
        """
        # forward() is no numeric reference: it reshuffles the serialization orders and adapts
        # the attention windows, which the export copy pins.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        tmp_path = Path(directory.name)
        model = build_seg3d_model().cuda().eval()
        batch = build_inputs(device=torch.device("cuda"))
        graphs = {graph.name: graph for graph in model.deployment_graphs()}
        onnx_cfg = OmegaConf.create(
            {"opset_version": 17, "dynamo": False, "do_constant_folding": False}
        )
        for name, spec in model.build_export_specs(batch).items():
            path = tmp_path / f"{name}.onnx"
            export_to_onnx(spec, onnx_cfg, path)
            graph = onnx.load(path).graph
            self.assertEqual(
                [tensor.name for tensor in graph.input], list(graphs[name].input_names)
            )
            self.assertEqual(
                [tensor.name for tensor in graph.output], list(graphs[name].output_names)
            )
            for tensor in graph.input:
                axes = graphs[name].onnx_dynamic_axes.get(tensor.name, {})
                dims = tensor.type.tensor_type.shape.dim
                self.assertEqual({i: dims[i].dim_param for i in axes}, dict(axes), tensor.name)

    def test_seg3d_head_graph_emits_its_output_names(self) -> None:
        """Test that the seg3d head graph declares the head's output names and their axes.

        Input: the seg3d head graph of the segmentation model.
        Expected: the outputs are ``pred_labels`` and ``pred_probs`` with dynamic axes.
        Check: the graph's output names and the keys of its dynamic axes.
        """
        model = build_seg3d_model().cuda().eval()
        graph = build_seg3d_head_graph(model, model.seg3d_head)
        self.assertEqual(graph.output_names, SEG3D_HEAD_OUTPUT_NAMES)
        self.assertEqual(SEG3D_HEAD_OUTPUT_NAMES, ("pred_labels", "pred_probs"))
        self.assertGreaterEqual(set(graph.onnx_dynamic_axes), set(SEG3D_HEAD_OUTPUT_NAMES))


@REQUIRES_SPARSE_CUDA
class TestDet3dDeployment(unittest.TestCase):
    """The detection model's deployment graphs and their specs."""

    def test_det3d_declaration_and_specs(self) -> None:
        """Test that the detection model declares the encoder and the det3d head graphs.

        Input: the detection model's deployment declaration and its export specs.
        Expected: the graphs and the specs are the encoder and the det3d head, and the head
            spec emits the configured export output names.
        Check: the graph names, the spec names and the head spec's output names.
        """
        model = build_trans_model().cuda().eval()
        graphs = model.deployment_declaration().graphs
        self.assertEqual([graph.name for graph in graphs], [ENCODER_GRAPH, DET3D_HEAD_GRAPH])
        batch = build_inputs(device=torch.device("cuda"))
        specs = model.build_export_specs(batch)
        self.assertEqual(list(specs), ["ptv3_encoder", "ptv3_det3d_head"])
        self.assertEqual(specs["ptv3_det3d_head"].output_names, list(model.export_output_names))
