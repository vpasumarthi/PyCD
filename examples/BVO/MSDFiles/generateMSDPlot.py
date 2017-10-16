#!/usr/bin/env python

import os.path

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
systemSize = np.array([2, 2, 1])
pbc = np.array([1, 1, 1])
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
# fileFormatIndex: 0=VASP; 1=VESTA
fileFormatIndex = 1

materialMSD(systemDirectoryPath, fileFormatIndex, systemSize, pbc, nDim, Temp,
            ionChargeType, speciesChargeType, speciesCount, tFinal, nTraj,
            timeInterval, msdTFinal, trimLength, displayErrorBars, reprTime,
            reprDist, report)
