#!/usr/bin/env python

import os
import platform
import pickle

import numpy as np

from PyCT.core import system, run

directorySeparator = '\\' if platform.uname()[0] == 'Windows' else '/'


def materialRun(systemSize, pbc, Temp, speciesCount, tFinal, nTraj,
                timeInterval, randomSeed, report, overWrite, gui):

    # Determine path for system directory
    cwd = os.path.dirname(os.path.realpath(__file__))
    directorySeparator = '\\' if platform.uname()[0] == 'Windows' else '/'
    nLevelUp = 3 if platform.uname()[0] == 'Linux' else 3
    systemDirectoryPath = directorySeparator.join(
            cwd.split(directorySeparator)[:-nLevelUp]
            + ['PyCTSimulations', 'BVO', ('PBC' if pbc else 'NoPBC'),
               ('SystemSize' + str(systemSize).replace(' ', ''))])

    # Change to working directory
    parentDir1 = 'SimulationFiles'
    nElectrons = speciesCount[0]
    nHoles = speciesCount[1]
    parentDir2 = (str(nElectrons)
                  + ('electron' if nElectrons == 1 else 'electrons') + ', '
                  + str(nHoles) + ('hole' if nHoles == 1 else 'holes'))
    parentDir3 = str(Temp) + 'K'
    workDir = (('%1.2E' % tFinal) + 'SEC,' + ('%1.2E' % timeInterval)
               + 'TimeInterval,' + ('%1.2E' % nTraj) + 'Traj')
    workDirPath = (systemDirectoryPath + directorySeparator
                   + directorySeparator.join([parentDir1, parentDir2,
                                              parentDir3, workDir]))
    if not os.path.exists(workDirPath):
        os.makedirs(workDirPath)
    os.chdir(workDirPath)

    fileExists = 0
    if os.path.exists('Run.log') and os.path.exists('Time.dat'):
        fileExists = 1
    if not fileExists or overWrite:
        # Determine path for input files
        inputFileDirectoryName = 'InputFiles'
        inputFileDirectoryPath = (systemDirectoryPath
                                  + directorySeparator
                                  + inputFileDirectoryName)

        # Build path for material and neighbors object files
        materialName = 'bvo'
        tailName = '.obj'
        objectFileDirectoryName = 'ObjectFiles'
        objectFileDirPath = (inputFileDirectoryPath
                             + directorySeparator
                             + objectFileDirectoryName)
        materialFileName = (objectFileDirPath + directorySeparator
                            + materialName + tailName)
        neighborsFileName = (objectFileDirPath + directorySeparator
                             + materialName + 'Neighbors' + tailName)

        # Load material object
        file_bvo = open(materialFileName, 'r')
        bvo = pickle.load(file_bvo)
        file_bvo.close()

        # Load neighbors object
        file_bvoNeighbors = open(neighborsFileName, 'r')
        bvoNeighbors = pickle.load(file_bvoNeighbors)
        file_bvoNeighbors.close()

        # Load input files to instantiate system class
        os.chdir(inputFileDirectoryPath)
        hopNeighborListFileName = (inputFileDirectoryPath
                                   + directorySeparator
                                   + 'hopNeighborList.npy')
        hopNeighborList = np.load(hopNeighborListFileName)[()]
        cumulativeDisplacementListFilePath = (
                                            inputFileDirectoryPath
                                            + directorySeparator
                                            + 'cumulativeDisplacementList.npy')
        cumulativeDisplacementList = np.load(
                                            cumulativeDisplacementListFilePath)
        ewaldParametersFilePath = (inputFileDirectoryPath
                                   + directorySeparator
                                   + 'ewaldParameters.npy')
        ewaldParameters = np.load(ewaldParametersFilePath)[()]
        alpha = ewaldParameters['alpha']
        nmax = ewaldParameters['nmax']
        kmax = ewaldParameters['kmax']
        bvoSystem = system(bvo, bvoNeighbors, hopNeighborList,
                           cumulativeDisplacementList, speciesCount,
                           alpha, nmax, kmax)

        # Load precomputed array to instantiate run class
        precomputedArrayFilePath = (inputFileDirectoryPath
                                    + directorySeparator
                                    + 'precomputedArray.npy')
        precomputedArray = np.load(precomputedArrayFilePath)
        bvoRun = run(bvoSystem, precomputedArray, Temp, nTraj,
                     tFinal, timeInterval, gui)

        bvoRun.doKMCSteps(workDirPath, report, randomSeed)
    else:
        print ('Simulation files already exists in '
               + 'the destination directory')
