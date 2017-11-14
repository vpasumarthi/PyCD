#!/usr/bin/env python

from pathlib import Path

import numpy as np
import yaml

from PyCT.core import material, neighbors, system, run


def materialRun(inputDirectoryPath, systemSize, pbc, Temp, ionChargeType,
                speciesChargeType, speciesCount, tFinal, nTraj, timeInterval,
                randomSeed, report, overWrite):

    # Load material parameters
    configFileName = 'sysconfig.yml'
    configFilePath = inputDirectoryPath.joinpath(configFileName)
    with open(configFilePath, 'r') as stream:
        try:
            params = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    inputCoordinateFileName = 'POSCAR'
    inputCoorFileLocation = inputDirectoryPath.joinpath(
                                                    inputCoordinateFileName)
    params.update({'inputCoorFileLocation': inputCoorFileLocation})
    configParams = returnValues(params)

    # Build material object files
    materialInfo = material(configParams)

    # Build neighbors object files
    materialNeighbors = neighbors(materialInfo, systemSize, pbc)

    # Change to working directory
    parentDir1 = 'SimulationFiles'
    parentDir2 = ('ionChargeType=' + ionChargeType
                  + '; speciesChargeType=' + speciesChargeType)
    nElectrons = speciesCount[0]
    nHoles = speciesCount[1]
    parentDir3 = (str(nElectrons)
                  + ('electron' if nElectrons == 1 else 'electrons') + ', '
                  + str(nHoles) + ('hole' if nHoles == 1 else 'holes'))
    parentDir4 = str(Temp) + 'K'
    workDir = (('%1.2E' % tFinal) + 'SEC,' + ('%1.2E' % timeInterval)
               + 'TimeInterval,' + ('%1.2E' % nTraj) + 'Traj')
    systemDirectoryPath = inputDirectoryPath.resolve().parent
    workDirPath = systemDirectoryPath.joinpath(parentDir1, parentDir2,
                                               parentDir3, parentDir4, workDir)
    Path.mkdir(workDirPath, parents=True, exist_ok=True)

    fileExists = 0
    if workDirPath.joinpath('Run.log').exists():
        fileExists = 1
    if not fileExists or overWrite:
        # Load input files to instantiate system class
        hopNeighborListFileName = inputDirectoryPath.joinpath(
                                                        'hopNeighborList.npy')
        hopNeighborList = np.load(hopNeighborListFileName)[()]
        cumulativeDisplacementListFilePath = inputDirectoryPath.joinpath(
                                            'cumulativeDisplacementList.npy')
        cumulativeDisplacementList = np.load(
                                            cumulativeDisplacementListFilePath)
        alpha = configParams.alpha
        nmax = configParams.nmax
        kmax = configParams.kmax

        materialSystem = system(materialInfo, materialNeighbors,
                                hopNeighborList, cumulativeDisplacementList,
                                speciesCount, alpha, nmax, kmax)

        # Load precomputed array to instantiate run class
        precomputedArrayFilePath = inputDirectoryPath.joinpath(
                                                        'precomputedArray.npy')
        precomputedArray = np.load(precomputedArrayFilePath)
        materialRun = run(materialSystem, precomputedArray, Temp,
                          ionChargeType, speciesChargeType, nTraj, tFinal,
                          timeInterval)
        materialRun.doKMCSteps(workDirPath, report, randomSeed)
    else:
        print ('Simulation files already exists in '
               + 'the destination directory')
    return None


class returnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, inputdict):
        for key, value in inputdict.items():
            setattr(self, key, value)
