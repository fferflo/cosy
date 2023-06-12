#!/usr/bin/env python3

from setuptools import setup

setup(
    name="cosy1",
    version="0.1.0",
    description="Library for managing coordinate systems and transformations",
    author="Florian Fervers",
    author_email="florian.fervers@gmail.com",
    packages=["cosy"],
    package_data={"cosy": [
        "*.so",
        "proj_data/*",
    ]},
    license="MIT",
    install_requires=[
        "numpy",
        "pyquaternion",
    ],
    url="https://github.com/fferflo/cosy",
)
