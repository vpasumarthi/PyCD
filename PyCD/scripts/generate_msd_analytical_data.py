# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from datetime import datetime

import numpy as np
import yaml

from PyCT.core import Material, Neighbors


def generate_msd_analytical_data(
                            transition_prob_matrix, species_site_sd_list,
                            center_site_quantum_indices, analytical_t_final,
                            analytical_time_interval, dst_path, report=1):
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

    file_name = '%1.2Ens' % analytical_t_final
    msd_analytical_data_file_name = ('MSD_Analytical_Data_' + file_name
                                     + '.dat')
    msd_analytical_data_file_path = dst_path.joinpath(
        msd_analytical_data_file_name)
    open(msd_analytical_data_file_path, 'w').close()

    element_type_index = 0
    num_data_points = (int(analytical_t_final / analytical_time_interval)
                       + 1)
    msd_data = np.zeros((num_data_points, 2))
    msd_data[:, 0] = np.arange(
        0, analytical_t_final + analytical_time_interval,
        analytical_time_interval)

    system_element_index_offset_array = np.repeat(
        np.arange(0, (material_info.total_elements_per_unit_cell
                      * num_cells),
                  material_info.total_elements_per_unit_cell),
        material_info.n_elements_per_unit_cell[element_type_index])
    center_site_se_indices = (
            np.tile(material_info.n_elements_per_unit_cell[
                    :element_type_index].sum()
                    + np.arange(
                0, material_info.n_elements_per_unit_cell[
                    element_type_index]),
                    num_cells)
            + system_element_index_offset_array)

    center_site_se_index = material_neighbors.generate_system_element_index(
        system_size, center_site_quantum_indices)
    num_basal_neighbors = 3
    num_c_neighbors = 1
    num_neighbors = num_basal_neighbors + num_c_neighbors
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
    time_step = (material_info.SEC2NS / k_total / material_info.SEC2AUTIME)

    sim_time = 0
    start_index = 0
    row_index = np.where(center_site_se_indices == center_site_se_index)
    new_transition_prob_matrix = np.copy(transition_prob_matrix)
    with open(msd_analytical_data_file_path, 'a') as \
            msd_analytical_data_file:
        np.savetxt(msd_analytical_data_file, msd_data[start_index, :][
                                             None, :])
    while True:
        new_transition_prob_matrix = np.dot(new_transition_prob_matrix,
                                            transition_prob_matrix)
        sim_time += time_step
        end_index = int(sim_time / analytical_time_interval)
        if end_index >= start_index + 1:
            msd_data[end_index, 1] = np.dot(
                new_transition_prob_matrix[row_index],
                species_site_sd_list)
            with open(msd_analytical_data_file_path, 'a') as \
                    msd_analytical_data_file:
                np.savetxt(msd_analytical_data_file,
                           msd_data[end_index, :][None, :])
            start_index += 1
            if end_index == num_data_points - 1:
                break

    if report:
        generate_msd_analytical_data_report(file_name, dst_path, start_time)
    return_msd_data = ReturnValues({'msd_data': msd_data})
    return return_msd_data


def generate_msd_analytical_data_report(file_name, dst_path, start_time):
    """Generates a neighbor list and prints out a report to the
        output directory"""
    msd_analytical_data_log_name = ('MSD_Analytical_Data_' + file_name
                                    + '.log')
    msd_analytical_data_log_path = dst_path.joinpath(
        msd_analytical_data_log_name)
    report = open(msd_analytical_data_log_path, 'w')
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
