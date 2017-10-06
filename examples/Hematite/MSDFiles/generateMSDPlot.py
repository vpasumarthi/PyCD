#!/usr/bin/env python

import os.path

import numpy as np

from PyCT.materialMSD import materialMSD

# Frequently modified input parameters:
nElectrons = 1
tFinal = 1.00E-08
nTraj = 1.00E+02
timeInterval = 1.00E-10
msdTFinal = 5  # in units of reprTime
trimLength = 1

# Input parameters:
nDim = 3
Temp = 300  # K
nHoles = 0
speciesCount = np.array([nElectrons, nHoles])
displayErrorBars = 1
reprTime = 'ns'
reprDist = 'angstrom'
report = 1
systemDirectoryPath = os.path.dirname(
                                os.path.dirname(os.path.realpath(__file__)))

materialMSD(systemDirectoryPath, nDim, Temp, speciesCount,
            tFinal, nTraj, timeInterval, msdTFinal, trimLength,
            displayErrorBars, reprTime, reprDist, report)
