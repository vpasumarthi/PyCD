#!/usr/bin/env python

import os.path

import numpy as np

from PyCT.materialRun import materialRun

# Frequently modified input parameters:
nElectrons = 1
tFinal = 1.00E-08
nTraj = 1.00E+02
timeInterval = 1.00E-10

# Input parameters:
systemSize = np.array([2, 2, 1])
pbc = np.array([1, 1, 1])
Temp = 300  # K
nHoles = 0
speciesCount = np.array([nElectrons, nHoles])
randomSeed = 2
report = 1
overWrite = 1
systemDirectoryPath = os.path.dirname(
                                os.path.dirname(os.path.realpath(__file__)))
# fileFormatIndex: 0=VASP; 1=VESTA
fileFormatIndex = 1

materialRun(systemDirectoryPath, fileFormatIndex, systemSize, pbc, Temp,
            speciesCount, tFinal, nTraj, timeInterval, randomSeed, report,
            overWrite)
