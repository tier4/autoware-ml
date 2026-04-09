"""Setup entrypoint for Autoware-ML with compiled extensions."""

from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
from setuptools import setup

# Add the project root directory to the Python path, so that the build script can be found
build_path = Path(__file__).parent / "autoware_ml" / "ops" / "build.py"
spec = spec_from_file_location("autoware_ml_setup_build", build_path)
module = module_from_spec(spec)

assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)

setup(
    ext_modules=module.get_ext_modules(),
    cmdclass=module.get_cmdclass(),
)
