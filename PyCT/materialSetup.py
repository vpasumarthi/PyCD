#!/usr/bin/env python

import os

import numpy as np
import yaml

from PyCT.core import material, neighbors, system


def materialSetup(systemDirectoryPath, systemSize, pbc, generateObjectFiles,
                  generateHopNeighborList, generateCumDispList, alpha, nmax,
                  kmax, generatePrecomputedArray):
    """Prepare material class object file, neighborlist and \
        saves to the provided destination path"""

    # Load material parameters
    configDirName = 'ConfigurationFiles'
    configFileName = 'sysconfig.yml'
    configFilePath = os.path.join(systemDirectoryPath, configDirName,
                                  configFileName)
    with open(configFilePath, 'r') as stream:
        try:
            params = yaml.load(stream)
        except yaml.YAMLError as exc:
            print exc

    inputCoordinateFileName = 'POSCAR'
    inputCoorFileLocation = os.path.join(systemDirectoryPath, configDirName,
                                         inputCoordinateFileName)
    params.update({'inputCoorFileLocation': inputCoorFileLocation})
    materialParameters = returnValues(params)

    # Build material object files
    bvo = material(materialParameters)
    materialName = bvo.name

    # Build neighbors object files
    bvoNeighbors = neighbors(bvo, systemSize, pbc)

    # Determine path for input files
    inputFileDirectoryName = 'InputFiles'
    inputFileDirectoryPath = os.path.join(systemDirectoryPath,
                                          inputFileDirectoryName)

    # Build path for material and neighbors object files
    tailName = '.obj'
    objectFileDirectoryName = 'ObjectFiles'
    objectFileDirPath = os.path.join(inputFileDirectoryPath,
                                     objectFileDirectoryName)
    if not os.path.exists(objectFileDirPath):
        os.makedirs(objectFileDirPath)

    materialFilePath = os.path.join(objectFileDirPath, materialName) + tailName
    neighborsFilePath = (os.path.join(objectFileDirPath, materialName)
                         + 'Neighbors' + tailName)

    if generateObjectFiles:
        bvo.generateMaterialFile(bvo, materialFilePath)
        bvoNeighbors.generateNeighborsFile(bvoNeighbors, neighborsFilePath)

    # generate neighbor list
    if generateHopNeighborList:
        bvoNeighbors.generateNeighborList(inputFileDirectoryPath,
                                          generateCumDispList)

    # Build precomputed array and save to disk
    precomputedArrayFilePath = os.path.join(inputFileDirectoryPath,
                                            'precomputedArray.npy')
    if generatePrecomputedArray:
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

        # Note: speciesCount doesn't effect the precomputedArray,
        # however it is essential to instantiate system class
        # So, any non-zero value of speciesCount will do.
        nElectrons = 1
        nHoles = 0
        speciesCount = np.array([nElectrons, nHoles])

        bvoSystem = system(bvo, bvoNeighbors, hopNeighborList,
                           cumulativeDisplacementList, speciesCount,
                           alpha, nmax, kmax)
        precomputedArray = bvoSystem.ewaldSumSetup(inputFileDirectoryPath)
        np.save(precomputedArrayFilePath, precomputedArray)

        ewaldParametersFilePath = os.path.join(inputFileDirectoryPath,
                                               'ewaldParameters.npy')
        ewaldParameters = {'alpha': alpha,
                           'nmax': nmax,
                           'kmax': kmax}
        np.save(ewaldParametersFilePath, ewaldParameters)


class returnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, inputdict):
        for key, value in inputdict.items():
            setattr(self, key, value)
