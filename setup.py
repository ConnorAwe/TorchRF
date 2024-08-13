# Copyright (c) 2023 SRI International.  Use of this material is subject
# to the terms specified in the license located at /LICENSE.txt
#
# Reference: https://packaging.python.org/tutorials/packaging-projects/

import setuptools

with open('README.md') as f:
    readme = f.read()

with open('LICENSE.txt') as f:
    license = f.read()

with open('VERSION') as f:
    version = f.read()

with open('requirements.txt') as f:
    requirements = f.read()


setuptools.setup(
    name='torchrf',
    version=version,
    description='PyTorch Based Open-Source RF Simulator',
    long_description=readme,
    url='https://sri.com',
    author='Connor Awe, Aravind Sundaresan, Sam Austin, Samual McCallum, and Paul Titterton',
    install_requires=requirements,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    license=license,
    platforms='any'
)
