#!/usr/bin/env python

import numpy as np
import yaml

from PyCT.core import Material, Neighbors, System


def material_setup(input_directory_path, system_size, pbc,
                   generate_hop_neighbor_list, generate_cum_disp_list,
                   generate_precomputed_array):
    """Prepare material class object file, neighbor list and \
        saves to the provided destination path"""

    # Load material parameters
    config_file_name = 'sys_config.yml'
    config_file_path = input_directory_path.joinpath(config_file_name)
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
    material_neighbors = Neighbors(material_info, system_size, pbc)

    # generate neighbor list
    if generate_hop_neighbor_list:
        material_neighbors.generate_neighbor_list(input_directory_path)

    # generate cumulative displacement list
    if generate_cum_disp_list:
        material_neighbors.generate_cumulative_displacement_list(
                                                input_directory_path)

    # Build precomputed array and save to disk
    if generate_precomputed_array:
        # Load input files to instantiate system class
        hop_neighbor_list_file_name = input_directory_path.joinpath(
                                            'hop_neighbor_list.npy')
        hop_neighbor_list = np.load(hop_neighbor_list_file_name)[()]
        cumulative_displacement_list_file_path = (
                            input_directory_path.joinpath(
                                'cumulative_displacement_list.npy'))
        cumulative_displacement_list = np.load(
                                cumulative_displacement_list_file_path)

        # Note: species_count doesn't effect the precomputed_array,
        # however it is essential to instantiate system class
        # So, any non-zero value of species_count will do.
        n_electrons = 1
        n_holes = 0
        species_count = np.array([n_electrons, n_holes])

        alpha = config_params.alpha
        n_max = config_params.n_max
        k_max = config_params.k_max

        material_system = System(
            material_info, material_neighbors, hop_neighbor_list,
            cumulative_displacement_list, species_count, alpha, n_max,
            k_max)
        precomputed_array = material_system.ewald_sum_setup(
                                                input_directory_path)
        precomputed_array_file_path = input_directory_path.joinpath(
                                            'precomputed_array.npy')
        np.save(precomputed_array_file_path, precomputed_array)
    return None


class ReturnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)
