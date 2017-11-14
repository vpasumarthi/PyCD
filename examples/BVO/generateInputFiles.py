#!/usr/bin/env python

from pathlib import Path

import numpy as np

from PyCT.materialSetup import materialSetup

# Input parameters:
systemSize = np.array([2, 2, 1])
pbc = np.array([1, 1, 1])
generateHopNeighborList = 1
generateCumDispList = 1
generatePrecomputedArray = 1
systemDirectoryPath = Path.cwd()
inputFileDirectoryName = 'InputFiles'
inputDirectoryPath = systemDirectoryPath.joinpath(inputFileDirectoryName)

materialSetup(inputDirectoryPath, systemSize, pbc, generateHopNeighborList,
              generateCumDispList, generatePrecomputedArray)
