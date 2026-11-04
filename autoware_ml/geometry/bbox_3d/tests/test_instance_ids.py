"""Stable integer identities of annotation instance tokens."""

from autoware_ml.geometry.bbox_3d.instance_ids import hash_instance_ids


def test_equal_tokens_map_to_one_non_negative_id() -> None:
    ids = hash_instance_ids(["a3f1", "77c0", "a3f1"])

    assert ids.dtype.name == "int64"
    assert ids[0] == ids[2] != ids[1]
    assert (ids >= 0).all()
    # Independent of the process, unlike the salted built-in hash
    assert int(ids[0]) == int(hash_instance_ids(["a3f1"])[0])
