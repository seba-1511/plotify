#!/usr/bin/env python

from setuptools import (
    setup as install,
    find_packages,
)

VERSION = '0.1.6'

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
    install_requires=[open('requirements.txt').read().split('\n')],
)
