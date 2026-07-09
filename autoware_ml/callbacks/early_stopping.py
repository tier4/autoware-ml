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

"""Early stopping with config-authoritative restore semantics."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from lightning.pytorch.callbacks import EarlyStopping as LightningEarlyStopping

logger = logging.getLogger(__name__)


class ConfigAuthoritativeStateMixin:
    """Keep configured values when restoring callback state from a checkpoint.

    Lightning restores a callback's full ``state_dict`` on resume, which
    silently reverts configuration changes made between runs. This mixin
    splits the state by a general rule: a state key that is also a
    constructor parameter is configuration and the instantiated value wins;
    every other key is runtime progress and is restored unchanged. Each
    overridden value is logged.
    """

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore runtime state while keeping configured values.

        Args:
            state_dict: Callback state loaded from a checkpoint.
        """
        config_keys = state_dict.keys() & inspect.signature(type(self).__init__).parameters.keys()
        state = dict(state_dict)
        for key in sorted(config_keys):
            configured = getattr(self, key)
            if state[key] != configured:
                logger.warning(
                    "%s.%s: checkpoint value %r overridden by configured value %r.",
                    type(self).__name__,
                    key,
                    state[key],
                    configured,
                )
            state[key] = configured
        super().load_state_dict(state)


class EarlyStopping(ConfigAuthoritativeStateMixin, LightningEarlyStopping):
    """Early stopping whose configuration survives checkpoint resumes."""
