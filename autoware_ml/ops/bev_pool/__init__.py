# Copyright 2025 TIER IV, Inc.
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

"""BEV pooling operator exports used by camera-to-BEV projection models.

This package re-exports the custom BEV pooling kernels and wrappers used by
camera-to-BEV projection architectures.
"""

from autoware_ml.ops.bev_pool.bev_pool import bev_pool

__all__ = ["bev_pool"]
