#!/usr/bin/env python3

import re
from setuptools import (
    setup,
    find_packages,
)

# Parses version number: https://stackoverflow.com/a/7071358
VERSIONFILE = 'plotify/_version.py'
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    VERSION = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in %s.' % (VERSIONFILE,))

# Installs the package
setup(
    name='plotify',
    packages=find_packages(),
    version=VERSION,
    description='Beautiful plots, fast.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Seb Arnold',
    author_email='smr.arnold@gmail.com',
    url='https://learnables.github.com/plotify',
    download_url='https://github.com/seba-1511/plotify/archive/' + str(VERSION) + '.zip',
    license='License :: OSI Approved :: Apache Software License',
    classifiers=[],
    scripts=[
        'bin/plotify',
    ],
    install_requires=[
        'numpy',
        'matplotlib>=3.3.0',
        'Pillow',
        'plotly',
        'scipy',
        'wandb',
        'ttf-opensans',
        'simple-parsing',
        'pyyaml',
    ],
)
