# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from datetime import datetime

import numpy as np
import yaml

from PyCT.core import Material


def generate_transition_prob_matrix(neighbor_system_element_indices, dst_path,
                                    report=1):
    start_time = datetime.now()

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
    system_size = sim_params['system_size']

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

    num_cells = system_size.prod()

    element_type_index = 0
    num_neighbors = len(neighbor_system_element_indices[0])
    num_basal_neighbors = 3
    # num_c_neighbors = 1
    temp = 300 * material_info.K2AUTEMP

    hop_element_type = 'Fe:Fe'
    k_list = np.zeros(num_neighbors)
    delg_0 = 0
    for neighbor_index in range(num_neighbors):
        if neighbor_index < num_basal_neighbors:
            hop_dist_type = 0
        else:
            hop_dist_type = 1
        lambda_value = material_info.lambda_values[hop_element_type][
            hop_dist_type]
        v_ab = material_info.v_ab[hop_element_type][hop_dist_type]
        delg_s = (((lambda_value + delg_0) ** 2 / (4 * lambda_value))
                  - v_ab)
        k_list[neighbor_index] = material_info.vn * np.exp(-delg_s / temp)

    k_total = np.sum(k_list)
    prob_list = k_list / k_total

    system_element_index_offset_array = np.repeat(
        np.arange(0, (material_info.total_elements_per_unit_cell
                      * num_cells),
                  material_info.total_elements_per_unit_cell),
        material_info.n_elements_per_unit_cell[element_type_index])
    neighbor_site_se_indices = (
            np.tile(material_info.n_elements_per_unit_cell[
                                                    :element_type_index].sum()
                    + np.arange(0, material_info.n_elements_per_unit_cell[
                                                        element_type_index]),
                    num_cells)
            + system_element_index_offset_array)

    num_element_type_sites = len(neighbor_system_element_indices)
    transition_prob_matrix = np.zeros((num_element_type_sites,
                                       num_element_type_sites))
    for center_site_index in range(num_element_type_sites):
        for neighbor_index in range(num_neighbors):
            neighbor_site_index = np.where(
                neighbor_site_se_indices
                == neighbor_system_element_indices[
                    center_site_index][neighbor_index])[0][0]
            transition_prob_matrix[center_site_index][
                neighbor_site_index] = prob_list[neighbor_index]
    file_name = 'transition_prob_matrix.npy'
    transition_prob_matrix_file_path = dst_path.joinpath(file_name)
    np.save(transition_prob_matrix_file_path, transition_prob_matrix)
    if report:
        generate_transition_prob_matrix_list_report(dst_path, start_time)
    return None


def generate_transition_prob_matrix_list_report(dst_path, start_time):
    """Generates a neighbor list and prints out a report to the
        output directory"""
    transition_prob_matrix_log_name = 'transition_prob_matrix.log'
    transition_prob_matrix_log_path = dst_path.joinpath(
        transition_prob_matrix_log_name)
    report = open(transition_prob_matrix_log_path, 'w')
    end_time = datetime.now()
    time_elapsed = end_time - start_time
    report.write('Time elapsed: ' + ('%2d days, ' % time_elapsed.days
                                     if time_elapsed.days else '')
                 + ('%2d hours' % ((time_elapsed.seconds // 3600) % 24))
                 + (', %2d minutes' % ((time_elapsed.seconds // 60) % 60))
                 + (', %2d seconds' % (time_elapsed.seconds % 60)))
    report.close()
    return None


class ReturnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)
