#!/usr/bin/env python

import yaml

from PyCT.core import material, analysis


def materialMSD(inputDirectoryPath, fileFormatIndex, systemSize, pbc, nDim,
                Temp, ionChargeType, speciesChargeType, speciesCount, tFinal,
                nTraj, timeInterval, msdTFinal, trimLength, displayErrorBars,
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
    params.update({'fileFormatIndex': fileFormatIndex})
    materialParameters = returnValues(params)

    # Build material object files
    materialInfo = material(materialParameters)

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

    if not workDirPath.exists():
        print('Simulation files do not exist. Aborting.')
    else:
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
    return None


class returnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, inputdict):
        for key, value in inputdict.items():
            setattr(self, key, value)
