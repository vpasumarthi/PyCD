#!/usr/bin/env python

import numpy as np
import yaml

from PyCT.core import material, analysis


def materialMSD(dstPath):
    # Load simulation parameters
    simParamFileName = 'simulationParameters.yml'
    simParamFilePath = dstPath / simParamFileName
    with open(simParamFilePath, 'r') as stream:
        try:
            simParams = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # data type conversion:
    simParams['speciesCount'] = np.asarray(simParams['speciesCount'])

    # Load material parameters
    configFileName = 'sysconfig.yml'
    inputDirectoryPath = (
                    dstPath.resolve().parents[simParams['workDirDepth'] - 1]
                    / simParams['inputFileDirectoryName'])
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
    materialParameters = returnValues(params)

    # Build material object files
    materialInfo = material(materialParameters)

    materialAnalysis = analysis(
        materialInfo, simParams['nDim'], simParams['speciesCount'],
        simParams['nTraj'], simParams['tFinal'], simParams['timeInterval'],
        simParams['msdTFinal'], simParams['trimLength'], simParams['reprTime'],
        simParams['reprDist'])

    msdAnalysisData = materialAnalysis.computeMSD(dstPath, simParams['report'])
    msdData = msdAnalysisData.msdData
    stdData = msdAnalysisData.stdData
    speciesTypes = msdAnalysisData.speciesTypes
    fileName = msdAnalysisData.fileName
    materialAnalysis.generateMSDPlot(msdData, stdData,
                                     simParams['displayErrorBars'],
                                     speciesTypes, fileName, dstPath)
    return None


class returnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, inputdict):
        for key, value in inputdict.items():
            setattr(self, key, value)
