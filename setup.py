#!/usr/bin/env python

import os

from setuptools import (
    setup as install,
    find_packages,
)

dir_path = os.path.dirname(os.path.realpath(__file__))

VERSION = '0.1.7'

install(
    name='plotify',
    version=VERSION,
    description="Make beautiful plots, fast.",
#    long_description=open('README.md').read(),
    author='Seb Arnold',
    author_email='smr.arnold@gmail.com',
    url='http://www.seba1511.com',
    download_url='https://github.com/seba-1511/plotify/archive/' + str(VERSION) + '.zip',
    license='License :: OSI Approved :: Apache Software License',
    packages=find_packages(exclude=["tests"]),
    classifiers=[],
    install_requires=[open(os.path.join(dir_path, 'requirements.txt')).read().split('\n')],
)
