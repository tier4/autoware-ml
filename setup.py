"""Setup entrypoint for Autoware-ML with compiled extensions."""

from setuptools import setup

from autoware_ml.ops.build import get_cmdclass, get_ext_modules

setup(
    ext_modules=get_ext_modules(),
    cmdclass=get_cmdclass(),
)
