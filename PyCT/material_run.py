#!/usr/bin/env python

import numpy as np
import yaml

from PyCT.core import Material, Neighbors, System, Run


def materialRun(dstPath):
    # Load simulation parameters
    simParamFileName = 'simulationParameters.yml'
    simParamFilePath = dstPath / simParamFileName
    with open(simParamFilePath, 'r') as stream:
        try:
            simParams = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # data type conversion:
    simParams['system_size'] = np.asarray(simParams['system_size'])
    simParams['pbc'] = np.asarray(simParams['pbc'])
    simParams['species_count'] = np.asarray(simParams['species_count'])

    # Load material parameters
    config_file_name = 'sysconfig.yml'
    input_directory_path = (
                    dstPath.resolve().parents[simParams['workDirDepth'] - 1]
                    / simParams['inputFileDirectoryName'])
    config_file_path = input_directory_path / config_file_name
    with open(config_file_path, 'r') as stream:
        try:
            params = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    input_coordinate_file_name = 'POSCAR'
    input_coord_file_location = input_directory_path.joinpath(
                                                    input_coordinate_file_name)
    params.update({'input_coord_file_location': input_coord_file_location})
    config_params = ReturnValues(params)

    # Build material object files
    material_info = Material(config_params)

    # Build neighbors object files
    material_neighbors = Neighbors(material_info, simParams['system_size'],
                                  simParams['pbc'])

    fileExists = 0
    if dstPath.joinpath('Run.log').exists():
        fileExists = 1
    if not fileExists or simParams['overWrite']:
        # Load input files to instantiate system class
        hop_neighbor_list_file_name = input_directory_path.joinpath(
                                                        'hop_neighbor_list.npy')
        hop_neighbor_list = np.load(hop_neighbor_list_file_name)[()]
        cumulative_displacement_list_file_path = input_directory_path.joinpath(
                                            'cumulative_displacement_list.npy')
        cumulative_displacement_list = np.load(
                                            cumulative_displacement_list_file_path)
        alpha = config_params.alpha
        n_max = config_params.n_max
        k_max = config_params.k_max

        material_system = System(material_info, material_neighbors,
                                hop_neighbor_list, cumulative_displacement_list,
                                simParams['species_count'], alpha, n_max, k_max)

        # Load precomputed array to instantiate run class
        precomputed_array_file_path = input_directory_path.joinpath(
                                                        'precomputed_array.npy')
        precomputed_array = np.load(precomputed_array_file_path)
        materialRun = Run(material_system, precomputed_array, simParams['Temp'],
                          simParams['ionChargeType'],
                          simParams['speciesChargeType'], simParams['nTraj'],
                          simParams['tFinal'], simParams['timeInterval'])
        materialRun.doKMCSteps(dstPath, simParams['report'],
                               simParams['randomSeed'])
    else:
        print ('Simulation files already exists in '
               + 'the destination directory')
    return None


class ReturnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)
