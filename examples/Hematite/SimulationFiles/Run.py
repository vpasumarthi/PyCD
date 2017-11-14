#!/usr/bin/env python

from pathlib import Path

import numpy as np

from PyCT.materialRun import materialRun

# Frequently modified input parameters:
nElectrons = 1
tFinal = 1.00E-08
nTraj = 1.00E+02
timeInterval = 1.00E-10
ionChargeType = 'full'
speciesChargeType = 'full'

# Input parameters:
systemSize = np.array([2, 2, 1])
pbc = np.array([1, 1, 1])
Temp = 300  # K
nHoles = 0
speciesCount = np.array([nElectrons, nHoles])
randomSeed = 2
report = 1
overWrite = 1

# input/output directories
dstPath = Path.cwd()
nLevelUp = 0
systemDirectoryPath = dstPath.resolve().parents[nLevelUp]
inputFileDirectoryName = 'InputFiles'
inputDirectoryPath = systemDirectoryPath.joinpath(inputFileDirectoryName)

materialRun(inputDirectoryPath, dstPath, systemSize, pbc, Temp, ionChargeType,
            speciesChargeType, speciesCount, tFinal, nTraj, timeInterval,
            randomSeed, report, overWrite)
