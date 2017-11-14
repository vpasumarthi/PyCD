#!/usr/bin/env python

from pathlib import Path

import numpy as np

from PyCT.materialMSD import materialMSD

# Frequently modified input parameters:
nElectrons = 1
tFinal = 1.00E-07
nTraj = 1.00E+02
timeInterval = 1.00E-08
msdTFinal = 60  # in units of reprTime
trimLength = 1
ionChargeType = 'full'
speciesChargeType = 'full'

# Input parameters:
nDim = 3
Temp = 300  # K
nHoles = 0
speciesCount = np.array([nElectrons, nHoles])
displayErrorBars = 1
reprTime = 'ns'
reprDist = 'angstrom'
report = 1
dstPath = Path.cwd()
nLevelUp = 0
systemDirectoryPath = dstPath.resolve().parents[nLevelUp]
inputFileDirectoryName = 'InputFiles'
inputDirectoryPath = systemDirectoryPath.joinpath(inputFileDirectoryName)

materialMSD(inputDirectoryPath, dstPath, nDim, speciesCount, tFinal, nTraj,
            timeInterval, msdTFinal, trimLength, displayErrorBars, reprTime,
            reprDist, report)
