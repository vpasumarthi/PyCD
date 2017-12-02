#!/usr/bin/env python

import numpy as np
import yaml

from PyCT.core import Material, Neighbors, System, Run


def material_run(dst_path):
    # Load simulation parameters
    sim_param_file_name = 'simulation_parameters.yml'
    sim_param_file_path = dst_path / sim_param_file_name
    with open(sim_param_file_path, 'r') as stream:
        try:
            sim_params = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # data type conversion:
    sim_params['system_size'] = np.asarray(sim_params['system_size'])
    sim_params['pbc'] = np.asarray(sim_params['pbc'])
    sim_params['species_count'] = np.asarray(sim_params[
                                                    'species_count'])

    # Load material parameters
    config_file_name = 'sys_config.yml'
    input_directory_path = (
        dst_path.resolve().parents[sim_params['work_dir_depth'] - 1]
        / sim_params['input_file_directory_name'])
    config_file_path = input_directory_path / config_file_name
    with open(config_file_path, 'r') as stream:
        try:
            params = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    input_coordinate_file_name = 'POSCAR'
    input_coord_file_location = input_directory_path.joinpath(
                                            input_coordinate_file_name)
    params.update({'input_coord_file_location':
                   input_coord_file_location})
    config_params = ReturnValues(params)

    # Build material object files
    material_info = Material(config_params)

    # Build neighbors object files
    material_neighbors = Neighbors(material_info,
                                   sim_params['system_size'],
                                   sim_params['pbc'])

    file_exists = 0
    if dst_path.joinpath('Run.log').exists():
        file_exists = 1
    if not file_exists or sim_params['over_write']:
        # Load input files to instantiate system class
        hop_neighbor_list_file_name = input_directory_path.joinpath(
                                            'hop_neighbor_list.npy')
        hop_neighbor_list = np.load(hop_neighbor_list_file_name)[()]
        cumulative_displacement_list_file_path = (
                            input_directory_path.joinpath(
                                'cumulative_displacement_list.npy'))
        cumulative_displacement_list = np.load(
                                cumulative_displacement_list_file_path)
        alpha = config_params.alpha
        n_max = config_params.n_max
        k_max = config_params.k_max

        material_system = System(
            material_info, material_neighbors, hop_neighbor_list,
            cumulative_displacement_list, sim_params['species_count'],
            alpha, n_max, k_max)

        # Load precomputed array to instantiate run class
        precomputed_array_file_path = input_directory_path.joinpath(
                                            'precomputed_array.npy')
        precomputed_array = np.load(precomputed_array_file_path)
        material_run = Run(
            material_system, precomputed_array, sim_params['temp'],
            sim_params['ion_charge_type'],
            sim_params['species_charge_type'], sim_params['n_traj'],
            sim_params['t_final'], sim_params['time_interval'])
        material_run.do_kmc_steps(dst_path, sim_params['report'],
                                  sim_params['random_seed'])
    else:
        print('Simulation files already exists in '
              + 'the destination directory')
    return None


class ReturnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)
