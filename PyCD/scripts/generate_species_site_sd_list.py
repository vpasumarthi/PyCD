# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from datetime import datetime

import numpy as np
import yaml

from PyCT.core import Material, Neighbors


def generate_species_site_sd_list(center_site_quantum_indices, dst_path,
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

    # Build neighbors object files
    material_neighbors = Neighbors(material_info,
                                   sim_params['system_size'],
                                   sim_params['pbc'])

    num_cells = system_size.prod()

    element_type_index = center_site_quantum_indices[3]
    center_site_se_index = material_neighbors.generate_system_element_index(
        system_size,
        center_site_quantum_indices)
    system_element_index_offset_array = np.repeat(
        np.arange(0, (material_info.total_elements_per_unit_cell
                      * num_cells),
                  material_info.total_elements_per_unit_cell),
        material_info.n_elements_per_unit_cell[element_type_index])
    neighbor_site_se_indices = (
            np.tile(material_info.n_elements_per_unit_cell[
                    :element_type_index].sum()
                    + np.arange(0,
                                material_info.n_elements_per_unit_cell[
                                    element_type_index]),
                    num_cells) + system_element_index_offset_array)
    species_site_sd_list = np.zeros(len(neighbor_site_se_indices))
    for neighbor_site_index, neighbor_site_se_index in enumerate(
            neighbor_site_se_indices):
        species_site_sd_list[neighbor_site_index] = (
                material_neighbors.compute_distance(
                                            system_size, center_site_se_index,
                                            neighbor_site_se_index) ** 2)
    species_site_sd_list /= material_info.ANG2BOHR ** 2
    file_name = 'species_site_sd_list.npy'
    species_site_sd_list_file_path = dst_path.joinpath(file_name)
    np.save(species_site_sd_list_file_path, species_site_sd_list)
    if report:
        generate_species_site_sd_list_report(dst_path, start_time)
    return None


def generate_species_site_sd_list_report(dst_path, start_time):
    """Generates a neighbor list and prints out a report to the
        output directory"""
    species_site_sd_list_log_name = 'species_site_sd_list.log'
    species_site_sd_list_log_path = dst_path.joinpath(
        species_site_sd_list_log_name)
    report = open(species_site_sd_list_log_path, 'w')
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
