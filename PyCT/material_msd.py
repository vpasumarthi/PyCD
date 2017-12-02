#!/usr/bin/env python

import numpy as np
import yaml

from PyCT.core import Material, Analysis


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
    simParams['species_count'] = np.asarray(simParams['species_count'])

    # Load material parameters
    config_file_name = 'sysconfig.yml'
    input_directory_path = (
                    dstPath.resolve().parents[simParams['workDirDepth'] - 1]
                    / simParams['inputFileDirectoryName'])
    config_file_path = input_directory_path.joinpath(config_file_name)
    with open(config_file_path, 'r') as stream:
        try:
            params = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    input_coordinate_file_name = 'POSCAR'
    input_coord_file_location = input_directory_path.joinpath(
                                                    input_coordinate_file_name)
    params.update({'input_coord_file_location': input_coord_file_location})
    materialParameters = ReturnValues(params)

    # Build material object files
    material_info = Material(materialParameters)

    materialAnalysis = Analysis(
        material_info, simParams['nDim'], simParams['species_count'],
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


class ReturnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)
