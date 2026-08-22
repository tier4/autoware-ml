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

"""ONNX metadata stamping for deployed artifacts.

Every exported module carries its provenance and inference parameters inside
the file itself: producer, release, config, export date and the per-module
``metainfo`` declared in the deploy config. Any ONNX consumer can read them
with no side channel. The stamper is tracker-agnostic: whichever experiment
tracker is active reduces to the generic ``tracker`` / ``run_id`` values the caller
passes in.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import onnx

PRODUCER_NAME = "Autoware-ML"
MODEL_DOMAIN = "jp.tier4.autoware-ml"
UNVERSIONED = "unversioned"

_RELEASE_PATTERN = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

# ONNX model_version is a protobuf int64. A larger encoding would only fail at
# onnx.save, after the export already ran.
_MODEL_VERSION_MAX = 2**63 - 1

# Property keys the stamper owns. User metainfo must not redefine them.
RESERVED_META_KEYS = frozenset(
    {
        "release",
        "module",
        "config_name",
        "export_date",
        "exported_with",
        "tracker",
        "run_id",
    }
)


def release_to_model_version(release: str | None) -> int:
    """Encode a ``vMAJOR.MINOR.PATCH`` release into ONNX's int64 model_version.

    The encoding is monotonic and reversible: ``major * 10000 + minor * 100 +
    patch``, so ``v0.0.1`` encodes to 1, ``v0.1.0`` to 100 and ``v1.2.3`` to
    10203. ``None`` (an unversioned dev export) encodes to 0. ``v0.0.0`` is
    rejected so 0 stays unambiguous, and an encoding beyond the int64
    ``model_version`` field is rejected too. A malformed release raises, a typo
    must never ship as a mis-stamped artifact, and every reject fires before
    any export work.
    """
    if release is None:
        return 0
    match = _RELEASE_PATTERN.match(release)
    if match is None:
        raise ValueError(f"release {release!r} must match vMAJOR.MINOR.PATCH (e.g. v0.0.1).")
    major, minor, patch = (int(group) for group in match.groups())
    if minor >= 100 or patch >= 100:
        raise ValueError(
            f"release {release!r} exceeds the model_version encoding (minor/patch < 100)."
        )
    version = major * 10000 + minor * 100 + patch
    if version == 0:
        raise ValueError(
            "release 'v0.0.0' encodes to model_version 0, which is reserved for "
            "unversioned exports."
        )
    if version > _MODEL_VERSION_MAX:
        raise ValueError(
            f"release {release!r} encodes to {version}, exceeding the int64 "
            "model_version ONNX can store."
        )
    return version


def meta_value_to_str(value: Any) -> str:
    """Serialize one metainfo value into an ONNX metadata string.

    Scalars map directly. ``bool`` becomes ``"true"`` or ``"false"`` and is
    checked before ``int`` since ``bool`` subclasses it. Flat sequences join
    their serialized elements with commas. Anything that would make the
    encoding ambiguous or lossy raises.
    """
    if isinstance(value, str):
        if "," in value:
            raise ValueError(
                f"metainfo string {value!r} contains a comma, so the comma-separated "
                "encoding would be ambiguous."
            )
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, Sequence):
        elements = []
        for element in value:
            if isinstance(element, Sequence) and not isinstance(element, str):
                raise ValueError(
                    f"metainfo sequence {value!r} is nested. Only flat sequences "
                    "serialize to an unambiguous comma-separated string."
                )
            elements.append(meta_value_to_str(element))
        return ",".join(elements)
    raise ValueError(
        f"metainfo value {value!r} of type {type(value).__name__} is not serializable. "
        "Supported: str, bool, int, float, flat sequences thereof."
    )


def stamp_onnx_meta(
    onnx_path: Path | str,
    *,
    config_name: str,
    module: str,
    release: str | None,
    export_git_sha: str,
    metainfo: Mapping[str, Any] | None = None,
    tracker: str | None = None,
    run_id: str | None = None,
) -> None:
    """Stamp one exported module in place with its identity and provenance.

    Args:
        onnx_path: The exported (and possibly graph-modified) module file.
        config_name: Config name the deploy ran with, stamped as provenance.
        module: Export module name (``ptv3_encoder``, ``ptv3_det3d_head``, ...).
        release: ``vMAJOR.MINOR.PATCH`` or ``None`` for an unversioned export.
        export_git_sha: Repository revision performing the export, recorded as
            ``producer_version``.
        metainfo: Per-module inference parameters declared in the deploy config
            (``deploy.onnx.modules.<module>.metainfo``), each value serialized
            with :func:`meta_value_to_str`. ``None`` stamps no extra props.
        tracker / run_id: The active experiment tracker and its deploy run,
            omitted from the stamp when no tracker is enabled.
    """
    model = onnx.load(str(onnx_path))
    exported_with = f"{model.producer_name} {model.producer_version}".strip()

    model.producer_name = PRODUCER_NAME
    model.producer_version = export_git_sha
    model.domain = MODEL_DOMAIN
    model.model_version = release_to_model_version(release)
    model.doc_string = f"{module} {release or UNVERSIONED}"

    props = {
        "release": release or UNVERSIONED,
        "module": module,
        "config_name": config_name,
        "export_date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "exported_with": exported_with,
    }
    if tracker is not None:
        props["tracker"] = tracker
    if run_id is not None:
        props["run_id"] = run_id
    for key, value in (metainfo or {}).items():
        if key in RESERVED_META_KEYS:
            raise ValueError(
                f"metainfo key {key!r} collides with an automatically stamped property. "
                f"reserved keys: {sorted(RESERVED_META_KEYS)}."
            )
        props[key] = meta_value_to_str(value)
    onnx.helper.set_model_props(model, props)

    onnx.save(model, str(onnx_path))
