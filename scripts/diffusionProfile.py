#!/usr/bin/env python

import os
import platform
from textwrap import wrap

import numpy as np
import matplotlib.pyplot as plt

from bvoSetup import bvoSetup

directorySeparator = '\\' if platform.uname()[0] == 'Windows' else '/'

# Frequently modified input parameters:
nElectronsList = range(1, 15) + [20, 30, 50]
speciesType = 'Electron'
nHoles = 0
tFinal = 1.00E-04
nTraj = 1.00E+02
Temp = 300
timeInterval = 1.00E-08
msdTFinal = 600  # in units of reprTime
reprTime = 'ns'
nSpecies = len(nElectronsList)
diffusivityProfileData = np.zeros((nSpecies, 2))
diffusivityProfileData[:, 0] = nElectronsList

cwd = os.path.dirname(os.path.realpath(__file__))
nLevelUp = 2
systemDirectoryPath = directorySeparator.join(cwd.split(
                                            directorySeparator)[:-nLevelUp])
simulationFileDirectoryName = 'SimulationFiles'
simulationFileDirectoryPath = (systemDirectoryPath
                               + directorySeparator
                               + simulationFileDirectoryName)

for speciesIndex, nElectrons in enumerate(nElectronsList):
    parentDir1 = (str(nElectrons)
                  + ('electron' if nElectrons == 1 else 'electrons') + ', ' +
                  str(nHoles) + ('hole' if nHoles == 1 else 'holes'))
    parentDir2 = str(Temp) + 'K'
    workDir = (('%1.2E' % tFinal) + 'SEC,' + ('%1.2E' % timeInterval)
               + 'TimeInterval,' + ('%1.2E' % nTraj) + 'Traj')
    workDirPath = (simulationFileDirectoryPath + directorySeparator
                   + directorySeparator.join([parentDir1, parentDir2,
                                              workDir]))

    fileName = '%1.2E%s' % (msdTFinal, reprTime)
    msdAnalysisLogFileName = ('MSD_Analysis' + ('_' if fileName else '')
                              + fileName + '.log')
    msdAnalysisLogFilePath = (workDirPath + directorySeparator
                              + msdAnalysisLogFileName)

    with open(msdAnalysisLogFilePath, 'r') as msdAnalysisLogFile:
        firstLine = msdAnalysisLogFile.readline()
    diffusivityProfileData[speciesIndex, 1] = float(firstLine[-13:-6])

plt.switch_backend('Agg')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(diffusivityProfileData[:, 0], diffusivityProfileData[:, 1], 'o-',
        color='blue', markerfacecolor='blue', markeredgecolor='black')
ax.set_xlabel('Number of electrons')
ax.set_ylabel('Diffusivity (${{\mu}}m^2/s$)')
figureTitle = 'Diffusion coefficient as a function of number of electrons'
ax.set_title('\n'.join(wrap(figureTitle, 60)))
filename = (str(speciesType) + 'DiffusionProfile_' + str(nElectronsList[0])
            + '-' + str(nElectronsList[-1]))
figureName = filename + '.jpg'
figurePath = cwd + directorySeparator + figureName
plt.savefig(figurePath)

dataFileName = filename + '.txt'
dataFilePath = cwd + directorySeparator + dataFileName
np.savetxt(dataFilePath, diffusivityProfileData)
