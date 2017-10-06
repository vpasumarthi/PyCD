#!/usr/bin/env python

import os

import numpy as np
import yaml

from PyCT.core import material, neighbors, system, run


def materialRun(systemDirectoryPath, systemSize, pbc, Temp, speciesCount,
                tFinal, nTraj, timeInterval, randomSeed, report, overWrite):

    # Load material parameters
    configDirName = 'ConfigurationFiles'
    configFileName = 'sysconfig.yml'
    configFilePath = os.path.join(systemDirectoryPath, configDirName,
                                  configFileName)
    with open(configFilePath, 'r') as stream:
        try:
            params = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    inputCoordinateFileName = 'POSCAR'
    inputCoorFileLocation = os.path.join(systemDirectoryPath, configDirName,
                                         inputCoordinateFileName)
    params.update({'inputCoorFileLocation': inputCoorFileLocation})
    materialParameters = returnValues(params)

    # Build material object files
    materialInfo = material(materialParameters)

    # Build neighbors object files
    materialNeighbors = neighbors(materialInfo, systemSize, pbc)

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


class returnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, inputdict):
        for key, value in inputdict.items():
            setattr(self, key, value)
