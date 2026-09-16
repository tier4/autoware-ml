"""Unit tests for ONNX metadata stamping."""

from __future__ import annotations

import json

import onnx
import pytest
from onnx import TensorProto, helper

from autoware_ml.utils.onnx_meta import (
    MODEL_DOMAIN,
    PRODUCER_NAME,
    meta_value_to_str,
    release_to_model_version,
    stamp_onnx_meta,
)


def _tiny_model(path) -> None:
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, producer_name="pytorch", producer_version="2.4.0")
    onnx.save(model, str(path))


def test_release_encoding_is_monotonic_and_reversible() -> None:
    assert release_to_model_version(None) == 0
    assert release_to_model_version("v0.0.1") == 1
    assert release_to_model_version("v0.1.0") == 100
    assert release_to_model_version("v1.2.3") == 10203
    assert release_to_model_version("v0.0.1") < release_to_model_version("v0.0.2")
    assert release_to_model_version("v0.99.99") < release_to_model_version("v1.0.0")


@pytest.mark.parametrize("bad", ["0.0.1", "v0.1", "v1.2.3.4", "release-1", "v0.100.0"])
def test_release_encoding_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        release_to_model_version(bad)


def test_release_encoding_reserves_zero_for_unversioned() -> None:
    with pytest.raises(ValueError, match="reserved"):
        release_to_model_version("v0.0.0")


def test_release_encoding_bounds_to_int64() -> None:
    # An over-int64 encoding must fail before export instead of at onnx.save.
    assert release_to_model_version("v922337203685476.99.99") <= 2**63 - 1
    with pytest.raises(ValueError, match="int64"):
        release_to_model_version("v922337203685477.99.99")
    with pytest.raises(ValueError, match="int64"):
        release_to_model_version("v922337203685478.0.0")


def test_meta_value_serialization() -> None:
    assert meta_value_to_str("z-trans") == '"z-trans"'
    assert meta_value_to_str(True) == "true"
    assert meta_value_to_str(False) == "false"
    assert meta_value_to_str(500) == "500"
    assert meta_value_to_str(-122.88) == "-122.88"
    assert meta_value_to_str(["z", "z-trans"]) == '["z","z-trans"]'
    assert meta_value_to_str([0.12, 0.12, 0.12]) == "[0.12,0.12,0.12]"
    assert meta_value_to_str((2, 2, 2, 2)) == "[2,2,2,2]"
    assert meta_value_to_str([[1, 2], [3, 4]]) == "[[1,2],[3,4]]"
    assert meta_value_to_str({"car": 0.5, "truck": 0.4}) == '{"car":0.5,"truck":0.4}'
    assert meta_value_to_str("car,truck") == '"car,truck"'


@pytest.mark.parametrize(
    "value",
    [500, -122.88, True, "z-trans", [0.12, 0.12, 0.12], ["car", "truck"], {"car": 1}],
)
def test_meta_value_round_trips_through_json(value) -> None:
    assert json.loads(meta_value_to_str(value)) == value


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), [1.0, float("-inf")]])
def test_meta_value_serialization_rejects_non_finite(bad) -> None:
    with pytest.raises(ValueError):
        meta_value_to_str(bad)


def test_meta_value_serialization_rejects_unsupported_types() -> None:
    with pytest.raises(TypeError):
        meta_value_to_str(object())


def test_stamp_release_export_with_metainfo(tmp_path) -> None:
    path = tmp_path / "ptv3_det3d_head.onnx"
    _tiny_model(path)
    stamp_onnx_meta(
        path,
        config_name="multi/ptv3/voxel012_122m_t4dataset_j6gen2",
        module="ptv3_det3d_head",
        release="v0.0.1",
        export_git_sha="be5b967",
        metainfo={
            "class_names": ["car", "truck"],
            "num_proposals": 500,
            "has_twist": True,
            "post_center_range": [-132.88, -132.88, -5.0, 132.88, 132.88, 12.0],
        },
        tracker="mlflow",
        run_id="run-123",
    )
    model = onnx.load(str(path))
    assert model.producer_name == PRODUCER_NAME
    assert model.producer_version == "be5b967"
    assert model.domain == MODEL_DOMAIN
    assert model.model_version == 1
    assert model.doc_string == "ptv3_det3d_head v0.0.1"
    props = {p.key: p.value for p in model.metadata_props}
    assert props["release"] == "v0.0.1"
    assert props["module"] == "ptv3_det3d_head"
    assert props["class_names"] == '["car","truck"]'
    assert props["num_proposals"] == "500"
    assert props["has_twist"] == "true"
    assert props["post_center_range"] == "[-132.88,-132.88,-5.0,132.88,132.88,12.0]"
    assert props["exported_with"] == "pytorch 2.4.0"
    assert props["tracker"] == "mlflow" and props["run_id"] == "run-123"
    assert "export_date" in props
    # The export sha lives in producer_version only, and the identity is the
    # config_name itself, never split into derived keys.
    for absent in ("export_git_sha", "train_git_sha", "model_name", "task"):
        assert absent not in props


def test_stamp_unversioned_export_without_metainfo(tmp_path) -> None:
    path = tmp_path / "ptv3_encoder.onnx"
    _tiny_model(path)
    stamp_onnx_meta(
        path,
        config_name="segmentation3d/ptv3/voxel012_122m_t4dataset_j6gen2",
        module="ptv3_encoder",
        release=None,
        export_git_sha="be5b967",
    )
    model = onnx.load(str(path))
    assert model.model_version == 0
    assert model.doc_string == "ptv3_encoder unversioned"
    props = {p.key: p.value for p in model.metadata_props}
    assert props["release"] == "unversioned"
    for absent in ("tracker", "run_id"):
        assert absent not in props


def test_stamp_rejects_reserved_metainfo_keys(tmp_path) -> None:
    path = tmp_path / "ptv3_encoder.onnx"
    _tiny_model(path)
    with pytest.raises(ValueError, match="reserved"):
        stamp_onnx_meta(
            path,
            config_name="segmentation3d/ptv3/voxel012_122m_t4dataset_j6gen2",
            module="ptv3_encoder",
            release=None,
            export_git_sha="be5b967",
            metainfo={"release": "v9.9.9"},
        )
