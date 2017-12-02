#!/usr/bin/env python

import os

import yaml

from PyCT.core import Material, Analysis


def materialCOCMSD(systemDirectoryPath, fileFormatIndex, system_size, pbc, nDim,
                   temp, ion_charge_type, species_charge_type, species_count,
                   t_final, n_traj, time_interval, msdTFinal, trimLength,
                   displayErrorBars, reprTime, reprDist, report):

    # Load material parameters
    configDirName = 'ConfigurationFiles'
    config_file_name = 'sys_config.yml'
    config_file_path = os.path.join(systemDirectoryPath, configDirName,
                                  config_file_name)
    with open(config_file_path, 'r') as stream:
        try:
            params = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    input_coordinate_file_name = 'POSCAR'
    input_coord_file_location = os.path.join(systemDirectoryPath, configDirName,
                                         input_coordinate_file_name)
    params.update({'input_coord_file_location': input_coord_file_location})
    params.update({'fileFormatIndex': fileFormatIndex})
    materialParameters = ReturnValues(params)

    # Build material object files
    material_info = Material(materialParameters)

    # Change to working directory
    parentDir1 = 'SimulationFiles'
    parentDir2 = ('ion_charge_type=' + ion_charge_type
                  + '; species_charge_type=' + species_charge_type)
    n_electrons = species_count[0]
    n_holes = species_count[1]
    parentDir3 = (str(n_electrons)
                  + ('electron' if n_electrons == 1 else 'electrons') + ', '
                  + str(n_holes) + ('hole' if n_holes == 1 else 'holes'))
    parentDir4 = str(temp) + 'K'
    workDir = (('%1.2E' % t_final) + 'SEC,' + ('%1.2E' % time_interval)
               + 'TimeInterval,' + ('%1.2E' % n_traj) + 'Traj')
    workDirPath = os.path.join(systemDirectoryPath, parentDir1, parentDir2,
                               parentDir3, parentDir4, workDir)

    if not os.path.exists(workDirPath):
        print('Simulation files do not exist. Aborting.')
    else:
        os.chdir(workDirPath)

        materialAnalysis = Analysis(material_info, nDim, species_count,
                                    n_traj, t_final, time_interval, msdTFinal,
                                    trimLength, reprTime, reprDist)

        msdAnalysisData = materialAnalysis.computeCOCMSD(workDirPath, report)
        msdData = msdAnalysisData.msdData
        stdData = msdAnalysisData.stdData
        speciesTypes = msdAnalysisData.speciesTypes
        fileName = msdAnalysisData.fileName
        materialAnalysis.generateCOCMSDPlot(msdData, stdData, displayErrorBars,
                                            speciesTypes, fileName,
                                            workDirPath)


class ReturnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)
