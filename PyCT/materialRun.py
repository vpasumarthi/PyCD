#!/usr/bin/env python

import os

import numpy as np
import yaml

from PyCT.core import material, neighbors, system, run


def materialRun(inputDirectoryPath, fileFormatIndex, systemSize, pbc, Temp,
                ionChargeType, speciesChargeType, speciesCount, tFinal, nTraj,
                timeInterval, randomSeed, report, overWrite):

    # Load material parameters
    configFileName = 'sysconfig.yml'
    configFilePath = os.path.join(inputDirectoryPath, configFileName)
    with open(configFilePath, 'r') as stream:
        try:
            params = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    inputCoordinateFileName = 'POSCAR'
    inputCoorFileLocation = os.path.join(inputDirectoryPath,
                                         inputCoordinateFileName)
    params.update({'inputCoorFileLocation': inputCoorFileLocation})
    params.update({'fileFormatIndex': fileFormatIndex})
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
    workDirPath = os.path.join(inputDirectoryPath, '..', parentDir1, parentDir2,
                               parentDir3, parentDir4, workDir)
    if not os.path.exists(workDirPath):
        os.makedirs(workDirPath)
    os.chdir(workDirPath)

    fileExists = 0
    if os.path.exists('Run.log') and os.path.exists('Time.dat'):
        fileExists = 1
    if not fileExists or overWrite:
        # Load input files to instantiate system class
        os.chdir(inputDirectoryPath)
        hopNeighborListFileName = os.path.join(inputDirectoryPath,
                                               'hopNeighborList.npy')
        hopNeighborList = np.load(hopNeighborListFileName)[()]
        cumulativeDisplacementListFilePath = os.path.join(
                                            inputDirectoryPath,
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
        precomputedArrayFilePath = os.path.join(inputDirectoryPath,
                                                'precomputedArray.npy')
        precomputedArray = np.load(precomputedArrayFilePath)
        materialRun = run(materialSystem, precomputedArray, Temp,
                          ionChargeType, speciesChargeType, nTraj, tFinal,
                          timeInterval)
        #print(workDirPath)
        #import pdb; pdb.set_trace()
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
