"""Setup script for realpython-reader"""

# Standard library imports
import pathlib

# Third party imports
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).resolve().parent

# The text of the README file is used as a description
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="dtwParallel",
    version="0.0.8",
    description="Python implementation of Dynamic Time Warping (DTW), which allows computing the dtw distance between one-dimensional and multidimensional time series, with the possibility of visualisation (one-dimensional case) and parallelisation (multidimensional case).",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/oscarescuderoarnanz/dtwParallel",
    author="oscarescuderoarnanz",
    author_email="escuderoarnanzoscar@gmail.com",
    license="BSD 2-clauses",
    packages=["dtwParallel"],
    keywords="Dynamic Time Warping Parallelization Visualisation Distance",
    #classifiers=[
    #    'Intended Audience :: Developers/Science/Reasearch',
    #    'Topic :: Scientific/Engineering/Software Development',
    #    'License :: OSI Approved :: BSD License',
    #    'Programming Language :: Python :: 3'
    #]
)
