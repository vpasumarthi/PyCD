# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import yaml

from PyCT.core import Material, Neighbors, System


def material_setup(input_directory_path, system_size, pbc,
                   generate_hop_neighbor_list, generate_pairwise_min_image_vector_data,
                   generate_precomputed_array, compute_energy_contributions,
                   return_k_vector_data):
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
        local_system_size = np.array([3, 3, 3])
        material_neighbors.generate_neighbor_list(input_directory_path,
                                                  local_system_size)

    # generate cumulative displacement list
    if generate_pairwise_min_image_vector_data:
        material_neighbors.get_pairwise_min_image_vector_data(
                                                input_directory_path)

    # Build precomputed array and save to disk
    if generate_precomputed_array:
        # Load input files to instantiate system class
        hop_neighbor_list_file_name = input_directory_path.joinpath(
                                            'hop_neighbor_list.npy')
        hop_neighbor_list = np.load(hop_neighbor_list_file_name)[()]
        pairwise_min_image_vector_data_file_path = (
                            input_directory_path.joinpath(
                                'pairwise_min_image_vector_data.npy'))
        pairwise_min_image_vector_data = np.load(
                                pairwise_min_image_vector_data_file_path)

        alpha = config_params.alpha
        r_cut = config_params.r_cut
        k_cut = config_params.k_cut
        precision_parameters = config_params.precision_parameters

        # dummy variables
        step_system_size_array = []
        step_hop_neighbor_master_list = []

        material_system = System(
            material_info, material_neighbors, hop_neighbor_list,
            pairwise_min_image_vector_data, alpha, r_cut, k_cut, precision_parameters,
            step_system_size_array, step_hop_neighbor_master_list)
        (precomputed_array, output_dir) = material_system.get_precomputed_array(
                                                input_directory_path,
                                                compute_energy_contributions,
                                                return_k_vector_data)
        precomputed_array_file_path = output_dir.joinpath(
                                            'precomputed_array.npy')
        np.save(precomputed_array_file_path, precomputed_array)
    return None


class ReturnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)
