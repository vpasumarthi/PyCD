#!/usr/bin/env python

import os.path
from textwrap import wrap
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt


def diffusionProfile(outdir, systemDirectoryPath, profilingSpeciesTypeIndex,
                     speciesCountList, ionChargeType, speciesChargeType, Temp,
                     tFinal, timeInterval, nTraj, msdTFinal, reprTime):
    profilingSpeciesList = speciesCountList[profilingSpeciesTypeIndex]
    profileLength = len(profilingSpeciesList)
    diffusivityProfileData = np.zeros((profileLength, 2))
    diffusivityProfileData[:, 0] = profilingSpeciesList
    speciesType = 'electron' if profilingSpeciesTypeIndex == 0 else 'hole'

    if profilingSpeciesTypeIndex == 0:
        nHoles = speciesCountList[1][0]
    else:
        nElectrons = speciesCountList[0][0]

    parentDir1 = 'SimulationFiles'
    parentDir2 = ('ionChargeType=' + ionChargeType
                  + '; speciesChargeType=' + speciesChargeType)

    fileName = '%1.2E%s' % (msdTFinal, reprTime)
    msdAnalysisLogFileName = ('MSD_Analysis' + ('_' if fileName else '')
                              + fileName + '.log')

    for speciesIndex, nSpecies in enumerate(profilingSpeciesList):
        # Change to working directory
        if profilingSpeciesTypeIndex == 0:
            nElectrons = nSpecies
        else:
            nHoles = nSpecies
        parentDir3 = (str(nElectrons)
                      + ('electron' if nElectrons == 1 else 'electrons') + ', '
                      + str(nHoles) + ('hole' if nHoles == 1 else 'holes'))
        parentDir4 = str(Temp) + 'K'
        workDir = (('%1.2E' % tFinal) + 'SEC,' + ('%1.2E' % timeInterval)
                   + 'TimeInterval,' + ('%1.2E' % nTraj) + 'Traj')
        msdAnalysisLogFilePath = os.path.join(
                    systemDirectoryPath, parentDir1, parentDir2, parentDir3,
                    parentDir4, workDir, msdAnalysisLogFileName)

        with open(msdAnalysisLogFilePath, 'r') as msdAnalysisLogFile:
            firstLine = msdAnalysisLogFile.readline()
        diffusivityProfileData[speciesIndex, 1] = float(firstLine[-13:-6])

    plt.switch_backend('Agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(diffusivityProfileData[:, 0], diffusivityProfileData[:, 1], 'o-',
            color='blue', markerfacecolor='blue', markeredgecolor='black')
    ax.set_xlabel('Number of ' + speciesType + 's')
    ax.set_ylabel('Diffusivity (${{\mu}}m^2/s$)')
    figureTitle = ('Diffusion coefficient as a function of number of '
                   + speciesType + 's')
    ax.set_title('\n'.join(wrap(figureTitle, 60)))
    filename = (str(speciesType) + 'DiffusionProfile_'
                + str(profilingSpeciesList[0]) + '-'
                + str(profilingSpeciesList[-1]))
    figureName = filename + '.png'
    figurePath = os.path.join(outdir, figureName)
    plt.savefig(figurePath)

    dataFileName = filename + '.txt'
    dataFilePath = os.path.join(outdir, dataFileName)
    np.savetxt(dataFilePath, diffusivityProfileData)


def runtimeProfile(outdir, systemDirectoryPath, profilingSpeciesTypeIndex,
                   speciesCountList, ionChargeType, speciesChargeType, Temp,
                   tFinal, timeInterval, nTraj, msdTFinal, reprTime):
    profilingSpeciesList = speciesCountList[profilingSpeciesTypeIndex]
    profileLength = len(profilingSpeciesList)
    elapsedSecondsData = np.zeros((profileLength, 2))
    elapsedSecondsData[:, 0] = profilingSpeciesList
    speciesType = 'electron' if profilingSpeciesTypeIndex == 0 else 'hole'

    if profilingSpeciesTypeIndex == 0:
        nHoles = speciesCountList[1][0]
    else:
        nElectrons = speciesCountList[0][0]

    parentDir1 = 'SimulationFiles'
    parentDir2 = ('ionChargeType=' + ionChargeType
                  + '; speciesChargeType=' + speciesChargeType)
    runLogFileName = 'Run.log'

    for speciesIndex, nSpecies in enumerate(profilingSpeciesList):
        # Change to working directory
        if profilingSpeciesTypeIndex == 0:
            nElectrons = nSpecies
        else:
            nHoles = nSpecies
        parentDir3 = (str(nElectrons)
                      + ('electron' if nElectrons == 1 else 'electrons') + ', '
                      + str(nHoles) + ('hole' if nHoles == 1 else 'holes'))
        parentDir4 = str(Temp) + 'K'
        workDir = (('%1.2E' % tFinal) + 'SEC,' + ('%1.2E' % timeInterval)
                   + 'TimeInterval,' + ('%1.2E' % nTraj) + 'Traj')
        runLogFilePath = os.path.join(
                    systemDirectoryPath, parentDir1, parentDir2, parentDir3,
                    parentDir4, workDir, runLogFileName)

        with open(runLogFilePath, 'r') as runLogFile:
            firstLine = runLogFile.readline()
        if 'days' in firstLine:
            numDays = float(firstLine[14:16])
            numHours = float(firstLine[23:25])
            numMinutes = float(firstLine[33:35])
            numSeconds = float(firstLine[45:47])
        else:
            numDays = 0
            numHours = float(firstLine[14:16])
            numMinutes = float(firstLine[24:26])
            numSeconds = float(firstLine[36:38])
        elapsedTime = timedelta(days=numDays, hours=numHours,
                                minutes=numMinutes, seconds=numSeconds)
        elapsedSecondsData[speciesIndex, 1] = int(elapsedTime.total_seconds())

    plt.switch_backend('Agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(elapsedSecondsData[:, 0], elapsedSecondsData[:, 1], 'o-',
            color='blue', markerfacecolor='blue', markeredgecolor='black')
    ax.set_xlabel('Number of ' + speciesType + 's')
    ax.set_ylabel('Run Time (sec)')
    figureTitle = ('Simulation run time as a function of number of '
                   + speciesType + 's')
    ax.set_title('\n'.join(wrap(figureTitle, 60)))
    filename = (str(speciesType) + 'RunTimeProfile_'
                + str(profilingSpeciesList[0]) + '-'
                + str(profilingSpeciesList[-1]))
    figureName = filename + '.png'
    figurePath = os.path.join(outdir, figureName)
    plt.savefig(figurePath)

    dataFileName = filename + '.txt'
    dataFilePath = os.path.join(outdir, dataFileName)
    np.savetxt(dataFilePath, elapsedSecondsData)
