#!/usr/bin/env python

from distutils.core import setup
from setuptools import (
    setup as install,
    find_packages,
)

VERSION = '0.1.0'

setup(
    name='plotify',
)

install(
    name='plotify',
    version=VERSION,
    description="Make beautiful plots, fast.",
    long_description=open('README.md').read(),
    author='Seb Arnold',
    author_email='smr.arnold@gmail.com',
    url='http://www.seba1511.com',
    download_url='https://github.com/seba-1511/plotify/archive/0.1.0.zip',
    license='License :: OSI Approved :: Apache Software License',
    packages=find_packages(exclude=["tests"]),
    classifiers=[
        'Tools',
        'Productivity',
    ]
)
