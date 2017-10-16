#!/usr/bin/env python

import os.path

import numpy as np

from PyCT.materialSetup import materialSetup

# Frequently used parameters:
fileFormatIndex = 1  # fileFormatIndex: 0=VASP; 1=VESTA

# Input parameters:
systemSize = np.array([2, 2, 1])
pbc = np.array([1, 1, 1])
generateHopNeighborList = 1
generateCumDispList = 1
generatePrecomputedArray = 1
systemDirectoryPath = os.path.dirname(os.path.realpath(__file__))

materialSetup(systemDirectoryPath, fileFormatIndex, systemSize, pbc,
              generateHopNeighborList, generateCumDispList,
              generatePrecomputedArray)
