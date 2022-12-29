#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import codecs
import os.path
from setuptools import setup
import re

# The directory containing this file
#HERE = pathlib.Path(__file__).resolve().parent

# The text of the README file is used as a description
#README = (HERE / "README.md").read_text()


here = os.path.abspath(os.path.dirname(__file__))
readme_md = os.path.join(here, 'README.md')
version_py = os.path.join(here, 'dtwParallel', '_version.py')

# Get the package description from the README.md file
with codecs.open(readme_md, encoding='utf-8') as f:
    long_description = f.read()

with codecs.open(version_py, 'r', encoding='utf-8') as fd:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
                        fd.read(), re.MULTILINE).group(1)

# This call to setup() does all the work
setup(
    name="dtwParallel",
    version=version,
    description="Python implementation of Dynamic Time Warping (DTW), which allows computing the dtw distance between one-dimensional and multidimensional time series, with the possibility of visualisation (one-dimensional case) and parallelisation (multidimensional case).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/oscarescuderoarnanz/dtwParallel",
    author="oscarescuderoarnanz",
    author_email="escuderoarnanzoscar@gmail.com",
    license="BSD 2-clauses",
    packages=[
       "dtwParallel"
    ],
    namespace_packages=[
      "dtwParallel",
    ],
    keywords="Dynamic Time Warping Parallelization Visualisation Distance",
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3'
    ],

    include_package_data = True,
    package_data = {
    # If any package contains *.ini files, include them
    '': ['*.ini'],
    },
    py_modules={
        "utils",
        "dtw_functions",
        "error_control",
        "utils_visualizations"
    },
    entry_points={
        'console_scripts': [
        'dtwParallel=dtwParallel.dtwParallel:main'
        ]
    }
)
