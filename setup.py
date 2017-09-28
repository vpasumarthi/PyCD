#!/usr/bin/env python
"""
This is a setup script to install Kinetic Model
"""

import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name='KineticModel',
        version="0.1.1",
        description='KMC Model to simulate charge transport in crystal lattice',
        author='Viswanath Pasumarthi',
        author_email='pasumart@buffalo.edu',
        url="https://github.com/vpasumarthi/KineticModel",
        license='BSD-3C',
        packages=setuptools.find_packages(),
        install_requires=[
            'numpy>=1.7',
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
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=True,
)
