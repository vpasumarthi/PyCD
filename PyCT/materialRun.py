#!/usr/bin/env python

import os
import pickle

import numpy as np

from PyCT.core import system, run


def materialRun(systemDirectoryPath, Temp, speciesCount, tFinal, nTraj,
                timeInterval, randomSeed, report, overWrite):

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
    workDirPath = os.path.join(systemDirectoryPath, parentDir1, parentDir2,
                               parentDir3, workDir)
    if not os.path.exists(workDirPath):
        os.makedirs(workDirPath)
    os.chdir(workDirPath)

    fileExists = 0
    if os.path.exists('Run.log') and os.path.exists('Time.dat'):
        fileExists = 1
    if not fileExists or overWrite:
        # Determine path for input files
        inputFileDirectoryName = 'InputFiles'
        inputFileDirectoryPath = os.path.join(systemDirectoryPath,
                                              inputFileDirectoryName)

        # Build path for material and neighbors object files
        tailName = '.obj'
        objectFileDirectoryName = 'ObjectFiles'
        objectFileDirPath = os.path.join(inputFileDirectoryPath,
                                         objectFileDirectoryName)
        materialFileName = (os.path.join(objectFileDirPath, 'material')
                            + tailName)
        neighborsFileName = (os.path.join(objectFileDirPath, 'neighbors')
                             + tailName)

        # Load material object
        file_material = open(materialFileName, 'r')
        materialInfo = pickle.load(file_material)
        file_material.close()

        # Load neighbors object
        file_materialNeighbors = open(neighborsFileName, 'r')
        materialNeighbors = pickle.load(file_materialNeighbors)
        file_materialNeighbors.close()

        # Load input files to instantiate system class
        os.chdir(inputFileDirectoryPath)
        hopNeighborListFileName = os.path.join(inputFileDirectoryPath,
                                               'hopNeighborList.npy')
        hopNeighborList = np.load(hopNeighborListFileName)[()]
        cumulativeDisplacementListFilePath = os.path.join(
                                            inputFileDirectoryPath,
                                            'cumulativeDisplacementList.npy')
        cumulativeDisplacementList = np.load(
                                            cumulativeDisplacementListFilePath)
        ewaldParametersFilePath = os.path.join(inputFileDirectoryPath,
                                               'ewaldParameters.npy')
        ewaldParameters = np.load(ewaldParametersFilePath)[()]
        alpha = ewaldParameters['alpha']
        nmax = ewaldParameters['nmax']
        kmax = ewaldParameters['kmax']
        materialSystem = system(materialInfo, materialNeighbors,
                                hopNeighborList, cumulativeDisplacementList,
                                speciesCount, alpha, nmax, kmax)

        # Load precomputed array to instantiate run class
        precomputedArrayFilePath = os.path.join(inputFileDirectoryPath,
                                                'precomputedArray.npy')
        precomputedArray = np.load(precomputedArrayFilePath)
        materialRun = run(materialSystem, precomputedArray, Temp, nTraj,
                          tFinal, timeInterval)

        materialRun.doKMCSteps(workDirPath, report, randomSeed)
    else:
        print ('Simulation files already exists in '
               + 'the destination directory')
