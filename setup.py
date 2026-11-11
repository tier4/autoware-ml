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

"""Setup entrypoint for Autoware-ML with compiled extensions."""

from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
from setuptools import setup

# Add the project root directory to the Python path, so that the build script can be found
build_path = Path(__file__).parent / "autoware_ml" / "ops" / "build.py"
spec = spec_from_file_location("autoware_ml_setup_build", build_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load build spec from {build_path}")

module = module_from_spec(spec)
spec.loader.exec_module(module)

setup(
    ext_modules=module.get_ext_modules(),
    cmdclass=module.get_cmdclass(),
)
