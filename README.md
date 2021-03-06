Python-based Charge Dynamics (PyCD)
===================================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/vpasumarthi/PyCD.svg?branch=master)](https://travis-ci.com/vpasumarthi/PyCD)
[![License](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Version](https://img.shields.io/badge/version-v4.0-blue)](https://github.com/vpasumarthi/PyCD/tree/v4.0)
[![python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)](https://github.com/vpasumarthi/PyCD)
[![Documentation Status](https://readthedocs.org/projects/pycd/badge/?version=latest)](https://pycd.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/90578885.svg)](https://zenodo.org/badge/latestdoi/90578885)

Open-source, cross-platform application supporting lattice-based kinetic Monte Carlo simulations in crystalline systems

Documentation available at [`pycd.readthedocs.io`](https://pycd.readthedocs.io/en/latest/).

## Package Installation

### For production:

```bash
# Clone this repository
$ git clone https://github.com/vpasumarthi/PyCD.git

# Navigate to root directory
$ cd PyCD

# Perform a local installation of package with Pip 
$ pip install -e .
```

### For development:

```bash
# Create a new conda environment
$ conda create -n pycd-env python=3.6

# Activate the new conda environment
$ source activate pycd-env

# Clone this repository
$ git clone https://github.com/vpasumarthi/PyCD.git

# Navigate to root directory
$ cd PyCD

# Install requirements
$ pip install -r requirements.txt

# Perform a local installation of package with Pip 
$ pip install -e .
```

### Copyright

Copyright (c) 2020, Viswanath Pasumarthi


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.


