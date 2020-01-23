# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from datetime import datetime

import numpy as np
import yaml

from PyCT.core import Material, Neighbors


def generate_hematite_neighbor_se_indices(dst_path, report=1):
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

    # Build neighbors object files
    material_neighbors = Neighbors(material_info,
                                   sim_params['system_size'],
                                   sim_params['pbc'])

    num_cells = system_size.prod()

    offset_list = np.array(
        [[[-1, 0, -1], [0, 0, -1], [0, 1, -1], [0, 0, -1]],
         [[-1, -1, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, -1]],
         [[0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0]],
         [[-1, -1, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0]],
         [[0, -1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]],
         [[-1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]],
         [[-1, -1, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]],
         [[-1, -1, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0]],
         [[0, -1, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1]]])
    element_type_index = 0
    basal_neighbor_element_site_indices = np.array(
        [11, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 0])
    c_neighbor_element_site_indices = np.array(
        [9, 4, 11, 6, 1, 8, 3, 10, 5, 0, 7, 2])
    num_basal_neighbors = 3
    num_c_neighbors = 1
    num_neighbors = num_basal_neighbors + num_c_neighbors
    n_elements_per_unit_cell = material_info.n_elements_per_unit_cell[
        element_type_index]
    neighbor_element_site_indices = np.zeros(
        (n_elements_per_unit_cell, 4), int)
    for i_neighbor in range(num_neighbors):
        if i_neighbor < num_basal_neighbors:
            neighbor_element_site_indices[:, i_neighbor] = \
                basal_neighbor_element_site_indices
        else:
            neighbor_element_site_indices[:, i_neighbor] = \
                c_neighbor_element_site_indices
    system_element_index_offset_array = (
        np.repeat(np.arange(
            0, (material_info.total_elements_per_unit_cell
                * num_cells),
            material_info.total_elements_per_unit_cell),
            material_info.n_elements_per_unit_cell[
                element_type_index]))
    center_site_se_indices = (
            np.tile(material_info.n_elements_per_unit_cell[
                    :element_type_index].sum()
                    + np.arange(
                0, material_info.n_elements_per_unit_cell[
                    element_type_index]),
                    num_cells)
            + system_element_index_offset_array)
    num_center_site_elements = len(center_site_se_indices)
    neighbor_system_element_indices = np.zeros(
        (num_center_site_elements, num_neighbors))

    for center_site_index, center_site_se_index in enumerate(
            center_site_se_indices):
        center_site_quantum_indices = \
            material_neighbors.generate_quantum_indices(system_size,
                                                        center_site_se_index)
        center_site_unit_cell_indices = center_site_quantum_indices[:3]
        center_site_element_site_index = center_site_quantum_indices[-1:][
            0]
        for neighbor_index in range(num_neighbors):
            neighbor_unit_cell_indices = (
                    center_site_unit_cell_indices
                    + offset_list[center_site_element_site_index][
                        neighbor_index])
            for index, neighbor_unit_cell_index in enumerate(
                    neighbor_unit_cell_indices):
                if neighbor_unit_cell_index < 0:
                    neighbor_unit_cell_indices[index] += \
                        system_size[index]
                elif neighbor_unit_cell_index >= system_size[index]:
                    neighbor_unit_cell_indices[index] -= \
                        system_size[index]
                neighbor_quantum_indices = np.hstack((
                    neighbor_unit_cell_indices, element_type_index,
                    neighbor_element_site_indices[
                        center_site_element_site_index][neighbor_index]))
                neighbor_se_index = \
                    material_neighbors.generate_system_element_index(
                                        system_size, neighbor_quantum_indices)
                neighbor_system_element_indices[center_site_index][
                    neighbor_index] = neighbor_se_index

    file_name = 'neighbor_system_element_indices.npy'
    neighbor_system_element_indices_file_path = dst_path.joinpath(
        file_name)
    np.save(neighbor_system_element_indices_file_path,
            neighbor_system_element_indices)
    if report:
        generate_hematite_neighbor_se_indices_report(dst_path, start_time)
    return None


def generate_hematite_neighbor_se_indices_report(dst_path, start_time):
    """Generates a neighbor list and prints out a report to the output
        directory"""
    neighbor_system_element_indices_log_name = \
        'neighbor_system_element_indices.log'
    neighbor_system_element_indices_log_path = dst_path.joinpath(
        neighbor_system_element_indices_log_name)
    report = open(neighbor_system_element_indices_log_path, 'w')
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
