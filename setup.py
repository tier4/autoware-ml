"""Setup entrypoint for Autoware-ML with compiled extensions."""

import sys
import os
from setuptools import setup

# Add the project root directory to the Python path, so that the build script can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from autoware_ml.ops.build import get_cmdclass, get_ext_modules

setup(
    ext_modules=get_ext_modules(),
    cmdclass=get_cmdclass(),
)
