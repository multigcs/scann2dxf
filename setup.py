#!/usr/bin/env python3
#
#

import os
from setuptools import setup

setup(
    name='scann2dxf',
    version='0.1.0',
    author='Oliver Dippel',
    author_email='o.dippel@gmx.de',
    packages=['scann2dxf'],
    scripts=['bin/scann2dxf'],
    url='https://github.com/multigcs/scann2dxf/',
    license='LICENSE',
    description='simple python based scann to dxf converter',
    long_description=open('README.md').read(),
    install_requires=['ezdxf', 'numpy', 'opencv-python', 'Pillow', 'scipy', 'python-sane'],
    include_package_data=True,
    data_files = []
)

