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

"""Setup script for Autoware-ML custom operations (dev-only shortcut)."""

from setuptools import setup

from autoware_ml.ops.build import get_cmdclass, get_ext_modules


def get_packages():
    """Derive packages from extension modules."""
    ext_modules = get_ext_modules()
    return list({".".join(ext.name.split(".")[:-1]) for ext in ext_modules})


if __name__ == "__main__":
    setup(
        name="autoware_ml",
        packages=get_packages(),
        ext_modules=get_ext_modules(),
        cmdclass=get_cmdclass(),
        zip_safe=False,
    )
