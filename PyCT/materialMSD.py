#!/usr/bin/env python

import os

import yaml

from PyCT.core import material, analysis


def materialMSD(systemDirectoryPath, fileFormatIndex, systemSize, pbc, nDim,
                Temp, speciesCount, tFinal, nTraj, timeInterval, msdTFinal,
                trimLength, displayErrorBars, reprTime, reprDist, report):

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
    params.update({'fileFormatIndex': fileFormatIndex})
    materialParameters = returnValues(params)

    # Build material object files
    materialInfo = material(materialParameters)

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
        print('Simulation files do not exist. Aborting.')
    else:
        os.chdir(workDirPath)

        materialAnalysis = analysis(materialInfo, nDim, speciesCount,
                                    nTraj, tFinal, timeInterval, msdTFinal,
                                    trimLength, reprTime, reprDist)

        msdAnalysisData = materialAnalysis.computeMSD(workDirPath, report)
        msdData = msdAnalysisData.msdData
        stdData = msdAnalysisData.stdData
        speciesTypes = msdAnalysisData.speciesTypes
        fileName = msdAnalysisData.fileName
        materialAnalysis.generateMSDPlot(msdData, stdData, displayErrorBars,
                                         speciesTypes, fileName, workDirPath)


class returnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, inputdict):
        for key, value in inputdict.items():
            setattr(self, key, value)
