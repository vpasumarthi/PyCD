#!/usr/bin/env python

import os.path

import numpy as np

from PyCT.materialSetup import materialSetup

# Input parameters:
systemSize = np.array([2, 2, 1])
pbc = np.array([1, 1, 1])
generateObjectFiles = 1
generateHopNeighborList = 1
generateCumDispList = 1
alpha = 0.18
nmax = 0
kmax = 4
generatePrecomputedArray = 1
systemDirectoryPath = os.path.dirname(os.path.realpath(__file__))

materialSetup(systemDirectoryPath, systemSize, pbc, generateHopNeighborList,
              generateCumDispList, alpha, nmax, kmax, generatePrecomputedArray)
