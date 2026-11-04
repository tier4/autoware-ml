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

"""Unit tests for the motion head outputs."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.dataclasses.models.motion.head_outputs import MotionHeadOutputs


class TestMotionHeadOutputs(unittest.TestCase):
    def test_select_samples_keeps_the_selected_sample_in_every_field(self) -> None:
        """
        Input: outputs of three samples whose values name their sample, the mask [F, T, F].
        Expected: the mode scores and the trajectories of sample 1 only.
        """
        sample = torch.arange(3, dtype=torch.float32)
        outputs = MotionHeadOutputs(
            agent_mode_logits=sample.view(-1, 1, 1).expand(3, 4, 6).clone(),
            agent_trajectories=sample.view(-1, 1, 1, 1, 1).expand(3, 4, 6, 5, 2).clone(),
        )

        selected = outputs.select_samples(torch.tensor([False, True, False]))

        for tensor in (selected.agent_mode_logits, selected.agent_trajectories):
            self.assertEqual(tensor.shape[0], 1)
            self.assertTrue((tensor == 1).all())

    def test_trajectories_of_another_mode_count_are_refused(self) -> None:
        """
        Input: mode scores of six modes and trajectories of five.
        Expected: ValueError naming both shapes.
        """
        with self.assertRaisesRegex(ValueError, "must lead with the shape of agent_mode_logits"):
            MotionHeadOutputs(
                agent_mode_logits=torch.zeros(1, 4, 6),
                agent_trajectories=torch.zeros(1, 4, 5, 4, 2),
            )
