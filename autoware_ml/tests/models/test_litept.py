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

"""Tests for the LitePT encoder: rotary embedding, stage gating, export contract."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.models.segmentation3d.encoders.ptv3 import (
    Block,
    LitePTEncoder,
    Point3DRoPE,
    rope_span,
)
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.tests.models.ptv3_detection_fixtures import (
    build_inputs,
    build_litept_encoder,
    build_litept_seg_model,
    build_ptv3_encoder,
    build_seg_model,
    move_batch_to_device,
)

REQUIRES_SPARSE_CUDA = pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="LitePT sparse-convolution tests require CUDA spconv",
)


def _reference_rope(
    tensor: torch.Tensor, grid_coord: torch.Tensor, chunk: int, base: float
) -> torch.Tensor:
    """Rotate the leading ``3 * chunk`` dims with an explicit split/rotate/concat."""
    outputs = []
    for axis in range(3):
        start = axis * chunk
        inv_freq = 1.0 / (base ** (torch.arange(0, chunk, 2).float() / chunk))
        angle = grid_coord[:, axis : axis + 1].float() * inv_freq.unsqueeze(0)
        angle = torch.cat((angle, angle), dim=-1).unsqueeze(1)
        part = tensor[..., start : start + chunk]
        half = chunk // 2
        rotated_half = torch.cat((-part[..., half:], part[..., :half]), dim=-1)
        outputs.append(part * angle.cos() + rotated_half * angle.sin())
    if 3 * chunk < tensor.shape[-1]:
        outputs.append(tensor[..., 3 * chunk :])
    return torch.cat(outputs, dim=-1)


@pytest.mark.parametrize(
    ("head_dim", "expected"),
    [(6, 6), (12, 12), (16, 12), (18, 18), (24, 24), (32, 30), (64, 60), (96, 96)],
)
def test_rope_span_is_the_largest_multiple_of_six(head_dim: int, expected: int) -> None:
    """Divisibility by three is not enough: an odd chunk width mispairs the halves."""
    assert rope_span(head_dim) == expected


def test_point3drope_rejects_head_dim_that_cannot_be_rotated() -> None:
    with pytest.raises(ValueError, match="at least 6"):
        Point3DRoPE(head_dim=4, base=100.0)


@pytest.mark.parametrize("head_dim", [12, 18, 32, 64])
def test_point3drope_matches_explicit_split_rotate_concat(head_dim: int) -> None:
    """The fused table form is exactly the textbook split/rotate/concat formulation."""
    torch.manual_seed(0)
    rope = Point3DRoPE(head_dim=head_dim, base=100.0)
    query = torch.randn(9, 2, head_dim)
    key = torch.randn(9, 2, head_dim)
    grid_coord = torch.randint(0, 50, (9, 3))

    rotated_query, rotated_key = rope(query, key, grid_coord)

    expected_query = _reference_rope(query, grid_coord, rope.chunk, rope.base)
    expected_key = _reference_rope(key, grid_coord, rope.chunk, rope.base)
    assert torch.allclose(rotated_query, expected_query, atol=1e-6)
    assert torch.allclose(rotated_key, expected_key, atol=1e-6)


@pytest.mark.parametrize("head_dim", [12, 18, 32, 64])
def test_point3drope_preserves_norm_and_passes_the_tail_through(head_dim: int) -> None:
    """A correct rotation keeps norms; the unrotated tail must come back untouched."""
    torch.manual_seed(1)
    rope = Point3DRoPE(head_dim=head_dim, base=100.0)
    query = torch.randn(7, 3, head_dim)
    grid_coord = torch.randint(0, 40, (7, 3))

    rotated, _ = rope(query, query, grid_coord)

    span = rope.rotated
    assert torch.allclose(
        rotated[..., :span].norm(dim=-1), query[..., :span].norm(dim=-1), atol=1e-5
    )
    assert torch.equal(rotated[..., span:], query[..., span:])


def test_point3drope_dot_product_depends_only_on_relative_position() -> None:
    """The defining RoPE property, checked on a head dimension that rotates fully."""
    torch.manual_seed(2)
    rope = Point3DRoPE(head_dim=24, base=100.0)
    query = torch.randn(1, 1, 24)
    key = torch.randn(1, 1, 24)

    def score(first: list[int], second: list[int]) -> torch.Tensor:
        rotated_query, _ = rope(query, query, torch.tensor([first]))
        _, rotated_key = rope(key, key, torch.tensor([second]))
        return (rotated_query * rotated_key).sum()

    assert torch.allclose(score([7, 3, 5], [2, 1, 4]), score([12, 5, 6], [7, 3, 5]), atol=1e-4)
    assert not torch.allclose(score([7, 3, 5], [2, 1, 4]), score([7, 3, 5], [4, 1, 4]), atol=1e-4)


def test_block_rejects_a_configuration_with_no_operation() -> None:
    with pytest.raises(ValueError, match="at least one"):
        Block(
            channels=12,
            num_heads=1,
            patch_size=2,
            mlp_ratio=2.0,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.0,
            pre_norm=True,
            order_index=0,
            cpe_indice_key="stage0",
            enable_rpe=False,
            enable_flash=False,
            upcast_attention=False,
            upcast_softmax=False,
            enable_conv=False,
            enable_attn=False,
        )


def test_litept_encoder_gates_convolution_and_attention_per_stage() -> None:
    """Early stages carry convolution only, late stages attention with RoPE only."""
    encoder = build_litept_encoder()
    stages = list(encoder.enc._modules.values())

    for stage_index, expects_conv in enumerate((True, True, False)):
        block = stages[stage_index]._modules["block0"]
        assert block.enable_conv is expects_conv
        assert block.enable_attn is (not expects_conv)
        assert (block.cpe is not None) is expects_conv
        assert (block.norm0 is not None) is (not expects_conv)
        if expects_conv:
            assert block.attn is None
            assert block.mlp is None
        else:
            assert block.attn.rope is not None
            assert block.attn.rope.base == 100.0


def test_ptv3_default_keeps_attention_everywhere_and_no_rope() -> None:
    """The PTv3 default path must be untouched by the gating and RoPE options."""
    encoder = build_ptv3_encoder()
    assert encoder.enc_attn == [True, True]
    assert encoder.enc_conv == [True, True]
    for stage in encoder.enc._modules.values():
        block = stage._modules["block0"]
        assert block.attn is not None
        assert block.attn.rope is None
        assert block.norm0 is None


def test_expand_stage_flags_rejects_a_wrong_length_sequence() -> None:
    with pytest.raises(ValueError, match="enc_attn must have 2 entries"):
        LitePTEncoder(
            in_channels=4,
            order=("z",),
            stride=(2,),
            enc_depths=(1, 1),
            enc_channels=(12, 24),
            enc_num_head=(1, 2),
            enc_patch_size=(2, 2),
            enc_conv=(True, True),
            enc_attn=(False, False, True),
        )


@REQUIRES_SPARSE_CUDA
def test_litept_is_a_drop_in_encoder_for_the_ptv3_segmentation_model() -> None:
    """The unmodified PTv3 task model trains and predicts with a LitePT encoder."""
    model = build_litept_seg_model().cuda().eval()
    batch = move_batch_to_device(build_inputs(), torch.device("cuda"))

    with torch.no_grad():
        logits = model(**batch)

    assert logits.shape == (batch["voxels"].shape[0], 3)
    assert torch.isfinite(logits).all()


@REQUIRES_SPARSE_CUDA
def test_litept_split_export_declares_and_consumes_the_ptv3_contract() -> None:
    """A gated model keeps PTv3's encoder signature and still runs on it.

    The unread tensors stay declared - the deployed runtime binds them by name -
    so the only difference from PTv3 is which of them the graph consumes.
    """
    model = build_litept_seg_model().cuda().eval()
    batch = move_batch_to_device(build_inputs(), torch.device("cuda"))

    specs = model.build_export_specs(batch)
    encoder_spec = specs["ptv3_encoder"]

    assert "serialized_code" in encoder_spec.input_param_names
    assert not any("_cluster" in name for name in encoder_spec.input_param_names)
    assert "serialized_pooling_0_serialized_order" in encoder_spec.input_param_names
    assert "serialized_pooling_1_serialized_order" in encoder_spec.input_param_names
    assert len(encoder_spec.args) == len(encoder_spec.input_param_names)

    with torch.no_grad():
        stage_feats = encoder_spec.module(*encoder_spec.args)
    assert len(stage_feats) == 3

    head_spec = specs["ptv3_seg3d_head"]
    # dec_depths is all zeros, so the head reduces to features plus clusters.
    assert head_spec.input_param_names == [
        "point_feat_0",
        "point_feat_1",
        "point_feat_2",
        "pooling_cluster_0",
        "pooling_cluster_1",
    ]
    with torch.no_grad():
        pred_labels, pred_probs = head_spec.module(*head_spec.args)
    assert pred_labels.shape == (batch["voxels"].shape[0],)
    assert pred_probs.shape == (batch["voxels"].shape[0], 3)


@REQUIRES_SPARSE_CUDA
def test_litept_monolithic_export_runs_on_its_declared_inputs() -> None:
    """The single-graph export adds clusters and otherwise matches PTv3 too."""
    model = build_litept_seg_model().cuda().eval()
    batch = move_batch_to_device(build_inputs(), torch.device("cuda"))

    spec = model.build_export_spec(batch)

    assert "serialized_code" in spec.input_param_names
    assert "serialized_pooling_0_cluster" in spec.input_param_names
    assert "serialized_pooling_0_serialized_order" in spec.input_param_names
    assert len(spec.args) == len(spec.input_param_names)

    with torch.no_grad():
        pred_labels, pred_probs = spec.module(*spec.args)
    assert pred_labels.shape == (batch["voxels"].shape[0],)
    assert pred_probs.shape == (batch["voxels"].shape[0], 3)


@REQUIRES_SPARSE_CUDA
def test_ptv3_monolithic_export_contract_still_lists_every_tensor() -> None:
    """Regression guard: the PTv3 single-graph contract is byte-for-byte what it was."""
    model = build_seg_model().cuda().eval()
    batch = move_batch_to_device(build_inputs(), torch.device("cuda"))

    spec = model.build_export_spec(batch)

    assert spec.input_param_names == [
        "voxels",
        "num_points_per_voxel",
        "grid_coord",
        "serialized_code",
        "serialized_pooling_0_indices",
        "serialized_pooling_0_indptr",
        "serialized_pooling_0_cluster",
        "serialized_pooling_0_head_indices",
        "serialized_pooling_0_grid_coord",
        "serialized_pooling_0_serialized_order",
        "serialized_pooling_0_serialized_inverse",
    ]


@REQUIRES_SPARSE_CUDA
def test_exported_encoder_graph_declares_a_subset_of_the_contract(tmp_path) -> None:
    """The exported graph must not accept anything the export did not declare.

    Asserting on ``input_param_names`` alone cannot catch this: that is what we
    declare, not what survives tracing. LitePT leaves part of PTv3's signature
    unread and the exporter prunes exactly that part, so a gated artifact declares
    a subset of the contract - which is what the deployed node accepts.
    """
    import onnx
    from omegaconf import OmegaConf

    from autoware_ml.utils.deploy import export_to_onnx

    onnx_cfg = OmegaConf.create(
        {"opset_version": 17, "dynamo": False, "do_constant_folding": False}
    )

    for tag, model in (("litept", build_litept_seg_model()), ("ptv3", build_seg_model())):
        model = model.cuda().eval()
        batch = move_batch_to_device(build_inputs(), torch.device("cuda"))
        spec = model.build_export_specs(batch)["ptv3_encoder"]
        path = tmp_path / f"{tag}_encoder.onnx"

        export_to_onnx(
            model=spec.module,
            input_sample=spec.args,
            onnx_cfg=onnx_cfg,
            input_param_names=list(spec.input_param_names),
            output_names_override=list(spec.output_names),
            dynamic_axes_override=spec.dynamic_axes,
            output_path=path,
        )

        graph = onnx.load(str(path)).graph
        declared = {value.name for value in graph.input}
        assert declared <= set(spec.input_param_names), tag
        assert {"voxels", "num_points_per_voxel"} <= declared, tag
        onnx.checker.check_model(onnx.load(str(path)))


@REQUIRES_SPARSE_CUDA
def test_litept_encoder_contract_matches_ptv3_field_for_field() -> None:
    """A gated model declares the same per-stage tensors PTv3 does, so it drops in.

    The deployed runtime declares its encoder IO statically - every pooling field for
    every stage, plus ``serialized_code`` - and rejects an engine missing any of them.
    """
    batch = move_batch_to_device(build_inputs(), torch.device("cuda"))
    litept = build_litept_seg_model().cuda().eval().build_export_specs(batch)["ptv3_encoder"]
    ptv3 = build_seg_model().cuda().eval().build_export_specs(batch)["ptv3_encoder"]

    def stage_fields(names: list[str]) -> set[str]:
        return {name.split("_", 3)[3] for name in names if name.startswith("serialized_pooling_")}

    # The fixtures differ in stage count, so compare the per-stage field structure.
    assert litept.input_param_names[:4] == [
        "voxels",
        "num_points_per_voxel",
        "grid_coord",
        "serialized_code",
    ]
    assert stage_fields(litept.input_param_names) == stage_fields(ptv3.input_param_names)
