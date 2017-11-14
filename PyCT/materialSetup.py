#!/usr/bin/env python

import numpy as np
import yaml

from PyCT.core import material, neighbors, system


def materialSetup(inputDirectoryPath, fileFormatIndex, systemSize, pbc,
                  generateHopNeighborList, generateCumDispList,
                  generatePrecomputedArray):
    """Prepare material class object file, neighborlist and \
        saves to the provided destination path"""

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
    params.update({'fileFormatIndex': fileFormatIndex})
    configParams = returnValues(params)

    # Build material object files
    materialInfo = material(configParams)

    # Build neighbors object files
    materialNeighbors = neighbors(materialInfo, systemSize, pbc)

    # generate neighbor list
    if generateHopNeighborList:
        materialNeighbors.generateNeighborList(inputDirectoryPath)

    # generate cumulative displacement list
    if generateCumDispList:
        materialNeighbors.generateCumulativeDisplacementList(
                                                            inputDirectoryPath)

    # Build precomputed array and save to disk
    if generatePrecomputedArray:
        # Load input files to instantiate system class
        hopNeighborListFileName = inputDirectoryPath.joinpath(
                                                        'hopNeighborList.npy')
        hopNeighborList = np.load(hopNeighborListFileName)[()]
        cumulativeDisplacementListFilePath = inputDirectoryPath.joinpath(
                                            'cumulativeDisplacementList.npy')
        cumulativeDisplacementList = np.load(
                                            cumulativeDisplacementListFilePath)

        # Note: speciesCount doesn't effect the precomputedArray,
        # however it is essential to instantiate system class
        # So, any non-zero value of speciesCount will do.
        nElectrons = 1
        nHoles = 0
        speciesCount = np.array([nElectrons, nHoles])

        alpha = configParams.alpha
        nmax = configParams.nmax
        kmax = configParams.kmax

        materialSystem = system(materialInfo, materialNeighbors,
                                hopNeighborList, cumulativeDisplacementList,
                                speciesCount, alpha, nmax, kmax)
        precomputedArray = materialSystem.ewaldSumSetup(inputDirectoryPath)
        precomputedArrayFilePath = inputDirectoryPath.joinpath(
                                                        'precomputedArray.npy')
        np.save(precomputedArrayFilePath, precomputedArray)
    return None


class returnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, inputdict):
        for key, value in inputdict.items():
            setattr(self, key, value)
