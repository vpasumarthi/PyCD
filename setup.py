#!/usr/bin/env python
"""
This is a setup script to install PyCD
"""

import setuptools

from PyCD import __version__


if __name__ == "__main__":
    setuptools.setup(
        name='PyCD',
        version=__version__,
        description='Python-based mesoscale model for Charge Transport',
        author='Viswanath Pasumarthi',
        author_email='pasumart@buffalo.edu',
        url="https://github.com/vpasumarthi/PyCD",
        packages=setuptools.find_packages(),
        install_requires=[
            'scipy',
            'numpy',
            'matplotlib',
            'pyyaml'
        ],
        extras_require={
            'docs': [
                'sphinx==1.2.3',  # autodoc was broken in 1.3.1
                'sphinxcontrib-napoleon',
                'sphinx_rtd_theme',
                'numpydoc',
            ],
            'tests': [
                'pytest',
                'pytest-cov',
                'pytest-pep8',
                'tox',
            ],
        },

        tests_require=[
            'pytest',
            'pytest-cov',
            'pytest-pep8',
            'tox',
        ],

        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=True)
