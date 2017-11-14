#!/usr/bin/env python

import yaml

from PyCT.core import material, analysis


def materialMSD(inputDirectoryPath, dstPath, nDim, speciesCount, tFinal, nTraj,
                timeInterval, msdTFinal, trimLength, displayErrorBars,
                reprTime, reprDist, report):

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
    materialParameters = returnValues(params)

    # Build material object files
    materialInfo = material(materialParameters)

    materialAnalysis = analysis(materialInfo, nDim, speciesCount,
                                nTraj, tFinal, timeInterval, msdTFinal,
                                trimLength, reprTime, reprDist)

    msdAnalysisData = materialAnalysis.computeMSD(dstPath, report)
    msdData = msdAnalysisData.msdData
    stdData = msdAnalysisData.stdData
    speciesTypes = msdAnalysisData.speciesTypes
    fileName = msdAnalysisData.fileName
    materialAnalysis.generateMSDPlot(msdData, stdData, displayErrorBars,
                                     speciesTypes, fileName, dstPath)
    return None


class returnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, inputdict):
        for key, value in inputdict.items():
            setattr(self, key, value)
