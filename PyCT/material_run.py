#!/usr/bin/env python

import numpy as np
import yaml

from PyCT.core import material, neighbors, system, run


def materialRun(dstPath):
    # Load simulation parameters
    simParamFileName = 'simulationParameters.yml'
    simParamFilePath = dstPath / simParamFileName
    with open(simParamFilePath, 'r') as stream:
        try:
            simParams = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # data type conversion:
    simParams['systemSize'] = np.asarray(simParams['systemSize'])
    simParams['pbc'] = np.asarray(simParams['pbc'])
    simParams['speciesCount'] = np.asarray(simParams['speciesCount'])

    # Load material parameters
    configFileName = 'sysconfig.yml'
    inputDirectoryPath = (
                    dstPath.resolve().parents[simParams['workDirDepth'] - 1]
                    / simParams['inputFileDirectoryName'])
    configFilePath = inputDirectoryPath / configFileName
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
    materialNeighbors = neighbors(materialInfo, simParams['systemSize'],
                                  simParams['pbc'])

    fileExists = 0
    if dstPath.joinpath('Run.log').exists():
        fileExists = 1
    if not fileExists or simParams['overWrite']:
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
                                simParams['speciesCount'], alpha, nmax, kmax)

        # Load precomputed array to instantiate run class
        precomputedArrayFilePath = inputDirectoryPath.joinpath(
                                                        'precomputedArray.npy')
        precomputedArray = np.load(precomputedArrayFilePath)
        materialRun = run(materialSystem, precomputedArray, simParams['Temp'],
                          simParams['ionChargeType'],
                          simParams['speciesChargeType'], simParams['nTraj'],
                          simParams['tFinal'], simParams['timeInterval'])
        materialRun.doKMCSteps(dstPath, simParams['report'],
                               simParams['randomSeed'])
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
