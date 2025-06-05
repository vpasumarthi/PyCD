Running Example Simulations
===========================

This page explains how to set up an environment for PyCD and how to
run the example simulations contained in the ``examples`` directory.
It also provides a short description of the required input files and the
``simulation_parameters.yml`` file.

Configuring the environment
---------------------------

A simple approach is to create a fresh conda environment and install
PyCD in editable mode:

.. code-block:: bash

   # create a conda environment
   conda create -n pycd-env python=3.6

   # activate the environment
   conda activate pycd-env

   # clone the repository and install dependencies
   git clone https://github.com/vpasumarthi/PyCD.git
   cd PyCD
   pip install -r requirements.txt
   pip install -e .

Running the examples
--------------------

After installing the package, navigate to one of the example directories
and execute ``Run.py``.  For instance, to run the ``Hematite`` example:

.. code-block:: bash

   cd examples/Hematite
   python Run.py

Similarly the ``examples/BVO`` directory can be run in the same way.
During execution PyCD will generate a folder named ``traj1`` (or
``traj2`` if a previous run exists) containing the simulation outputs.

Required input files
--------------------

Each example directory contains an ``InputFiles`` folder with several
precomputed data files:

``POSCAR``
    Unit cell coordinates of the material.
``sys_config.yml``
    Material specific parameters such as charges and dielectric
    constants.
``hop_neighbor_list.npy``
    Neighbour list used for the kinetic Monte Carlo steps.
``pairwise_min_image_vector_data.npy``
    Precomputed pairwise displacement information for Ewald summations.
``precomputed_array.npy``
    Table of transition probabilities used by the KMC algorithm.

A ``simulation_parameters.yml`` file sits next to ``Run.py`` and
contains all the settings controlling the run.  Important fields
include:

``system_size``
    Size of the simulation supercell (number of unit cells in ``x``,
    ``y`` and ``z``).
``pbc``
    Flags enabling periodic boundary conditions in each dimension.
``temp``
    Simulation temperature in Kelvin.
``t_final`` and ``time_interval``
    Final simulation time and interval between KMC steps.
``n_traj``
    Number of trajectories (used mainly in parallel runs).
``output_data``
    Dictionary specifying which arrays to write to disk.

Interpreting the output
-----------------------

After a run finishes, the ``traj*`` directory contains at least three
files:

``unwrapped_traj.npy``
    Cartesian trajectory of the charge carrier for every KMC step.
``time_data.npy``
    Simulation time corresponding to each step in ``unwrapped_traj.npy``.
``initial_rnd_state.dump``
    Pickled state of the random number generator used to reproduce the
    run.

The trajectory and time data can be loaded with ``numpy`` for further
analysis or plotting.
