#!/usr/bin/env python

import os
import platform

import numpy as np

from PyCT.materialParameters import materialParameters
from PyCT.core import material, neighbors, system

directorySeparator = '\\' if platform.uname()[0] == 'Windows' else '/'


def materialSetup(systemSize, pbc, generateObjectFiles, generateHopNeighborList,
             generateCumDispList, alpha, nmax, kmax, generatePrecomputedArray):
    """Prepare material class object file, neighborlist and \
        saves to the provided destination path"""

    # Determine path for system directory
    cwd = os.path.dirname(os.path.realpath(__file__))

    nLevelUp = 3 if platform.uname()[0] == 'Linux' else 3
    systemDirectoryPath = directorySeparator.join(
        cwd.split(directorySeparator)[:-nLevelUp]
        + ['PyCTSimulations', 'BVO',
           ('PBC' if np.all(pbc) else 'NoPBC'),
           ('SystemSize[' + ','.join(['%i' % systemSize[i]
                                      for i in range(len(systemSize))]) + ']')]
                                                  )

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
    if not os.path.exists(objectFileDirPath):
        os.makedirs(objectFileDirPath)
    
    materialFileName = (objectFileDirPath + directorySeparator
                        + materialName + tailName)
    neighborsFileName = (objectFileDirPath + directorySeparator + materialName
                         + 'Neighbors' + tailName)

    # Build material object files
    bvo = material(materialParameters())

    # Build neighbors object files
    bvoNeighbors = neighbors(bvo, systemSize, pbc)

    if generateObjectFiles:
        bvo.generateMaterialFile(bvo, materialFileName)
        bvoNeighbors.generateNeighborsFile(bvoNeighbors, neighborsFileName)

    # generate neighbor list
    if generateHopNeighborList:
        bvoNeighbors.generateNeighborList(inputFileDirectoryPath,
                                          generateCumDispList)

    # Build precomputed array and save to disk
    precomputedArrayFilePath = (inputFileDirectoryPath
                                + directorySeparator
                                + 'precomputedArray.npy')
    if generatePrecomputedArray:
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

        ewaldParametersFilePath = (inputFileDirectoryPath
                                   + directorySeparator
                                   + 'ewaldParameters.npy')
        ewaldParameters = {'alpha': alpha,
                           'nmax': nmax,
                           'kmax': kmax}
        np.save(ewaldParametersFilePath, ewaldParameters)
