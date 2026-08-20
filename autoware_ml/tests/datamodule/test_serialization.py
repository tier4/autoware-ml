import multiprocessing
import pickle

import numpy as np
import pytest

from autoware_ml.datamodule.common.serialization import SerializedSampleList


def _make_samples(count: int = 5) -> list[dict]:
    return [
        {
            "token": f"sample_{index}",
            "scene_token": f"scene_{index // 2}",
            "timestamp": 1700000000.0 + index,
            "instances": [{"bbox_label_3d": index, "bbox_3d": [0.1 * index] * 7}],
            "lidar_points": {"lidar_path": f"scene/{index}.pcd.bin", "num_pts_feats": 5},
            "array": np.arange(4, dtype=np.float32) + index,
        }
        for index in range(count)
    ]


class TestSerializedSampleList:
    def test_roundtrip_matches_original(self) -> None:
        samples = _make_samples()
        serialized = SerializedSampleList(samples)

        assert len(serialized) == len(samples)
        for index, original in enumerate(samples):
            restored = serialized[index]
            assert restored["token"] == original["token"]
            assert restored["instances"] == original["instances"]
            assert restored["lidar_points"] == original["lidar_points"]
            np.testing.assert_array_equal(restored["array"], original["array"])

    def test_negative_index_and_bounds(self) -> None:
        samples = _make_samples(3)
        serialized = SerializedSampleList(samples)

        assert serialized[-1]["token"] == samples[-1]["token"]
        with pytest.raises(IndexError):
            serialized[3]
        with pytest.raises(IndexError):
            serialized[-4]

    def test_iteration_visits_all_samples(self) -> None:
        samples = _make_samples(4)
        serialized = SerializedSampleList(samples)

        tokens = [sample["token"] for sample in serialized]

        assert tokens == [sample["token"] for sample in samples]

    def test_slicing_is_rejected(self) -> None:
        serialized = SerializedSampleList(_make_samples(2))

        with pytest.raises(TypeError):
            serialized[0:2]

    def test_returned_samples_are_independent_copies(self) -> None:
        serialized = SerializedSampleList(_make_samples(1))

        first = serialized[0]
        first["token"] = "mutated"

        assert serialized[0]["token"] == "sample_0"

    def test_empty_list(self) -> None:
        serialized = SerializedSampleList([])

        assert len(serialized) == 0
        with pytest.raises(IndexError):
            serialized[0]

    def test_survives_pickle(self) -> None:
        # DataLoader workers under the spawn start method pickle the dataset.
        samples = _make_samples(3)
        serialized = pickle.loads(pickle.dumps(SerializedSampleList(samples)))

        assert len(serialized) == 3
        assert serialized[2]["token"] == samples[2]["token"]

    def test_forked_child_reads_shared_buffer(self) -> None:
        samples = _make_samples(6)
        serialized = SerializedSampleList(samples)
        context = multiprocessing.get_context("fork")
        queue = context.Queue()

        def child(store, out) -> None:
            out.put([sample["token"] for sample in store])

        process = context.Process(target=child, args=(serialized, queue))
        process.start()
        tokens = queue.get(timeout=30)
        process.join(timeout=30)

        assert tokens == [sample["token"] for sample in samples]
        assert process.exitcode == 0
