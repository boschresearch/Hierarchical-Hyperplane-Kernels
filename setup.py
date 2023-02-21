from setuptools import find_packages, setup

from alef import __version__

name = "hhk"
version = __version__
description = "Hierarchical-Hyperplane-Kernel"
url = ""

setup(name=name, version=version, packages=find_packages(exclude=["tests"]), description=description, url=url)
