"""
PyCD
Open-source, cross-platform application supporting lattice-based kinetic Monte Carlo simulations in crystalline systems
"""

# Add imports here
from .core import *
from .material_setup import material_setup
try:
    from .hdf5_io import *
except ImportError:
    pass  # HDF5 support optional

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
