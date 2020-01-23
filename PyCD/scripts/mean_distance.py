# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from datetime import datetime

import numpy as np
import yaml

from PyCT.core import Material, Analysis


def compute_mean_distance(dst_path, mean=1, plot=1, report=1):
    """
    Add combType as one of the inputs
    combType = 0  # combType = 0: like-like; 1: like-unlike; 2: both
    if combType == 0:
        numComb = sum(
            [material_analysis.species_count[index] *
            (material_analysis.species_count[index] - 1)
             for index in len(material_analysis.species_count)])
    elif combType == 1:
        numComb = np.prod(material_analysis.species_count)
    elif combType == 2:
        numComb = (np.prod(material_analysis.species_count)
                   + sum([material_analysis.species_count[index]
                          * (material_analysis.species_count[index] - 1)
                          for index in len(material_analysis.species_count)]))
    """
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

    material_analysis = Analysis(
        material_info, sim_params['n_dim'], sim_params['species_count'],
        sim_params['n_traj'], sim_params['t_final'],
        sim_params['time_interval'], sim_params['msd_t_final'],
        sim_params['trim_length'], sim_params['repr_time'],
        sim_params['repr_dist'])

    position_array = (np.loadtxt(dst_path.joinpath('wrapped_traj.dat'))
                      * material_analysis.dist_conversion)
    # TODO: Currently assuming only electrons exist and coding accordingly.
    # Need to change according to combType
    pbc = [1, 1, 1]  # change to generic
    n_electrons = material_analysis.species_count[0]  # change to generic
    x_range = range(-1, 2) if pbc[0] == 1 else [0]
    y_range = range(-1, 2) if pbc[1] == 1 else [0]
    z_range = range(-1, 2) if pbc[2] == 1 else [0]
    # Initialization
    system_translational_vector_list = np.zeros((3 ** sum(pbc), 3))
    index = 0
    for x_offset in x_range:
        for y_offset in y_range:
            for z_offset in z_range:
                system_translational_vector_list[index] = np.dot(
                    np.multiply(np.array([x_offset, y_offset, z_offset]),
                                system_size),
                    (material_info.lattice_matrix
                     * material_analysis.dist_conversion))
                index += 1
    if mean:
        mean_distance = np.zeros((material_analysis.n_traj,
                                  material_analysis.num_path_steps_per_traj))
    else:
        inter_distance_array = np.zeros(
                        (material_analysis.n_traj,
                         material_analysis.num_path_steps_per_traj, n_electrons
                         * (n_electrons - 1) / 2))
    inter_distance_list = np.zeros(n_electrons * (n_electrons - 1) / 2)
    for traj_index in range(material_analysis.n_traj):
        head_start = traj_index * material_analysis.num_path_steps_per_traj
        for step in range(material_analysis.num_path_steps_per_traj):
            index = 0
            for i in range(n_electrons):
                for j in range(i + 1, n_electrons):
                    neighbor_image_coords = (
                            system_translational_vector_list
                            + position_array[head_start + step, j])
                    neighbor_image_displacement_vectors = (
                            neighbor_image_coords
                            - position_array[head_start + step, i])
                    neighbor_image_displacements = np.linalg.norm(
                        neighbor_image_displacement_vectors, axis=1)
                    displacement = np.min(neighbor_image_displacements)
                    inter_distance_list[index] = displacement
                    index += 1
            if mean:
                mean_distance[traj_index, step] = np.mean(
                    inter_distance_list)
                mean_distance_over_traj = np.mean(mean_distance, axis=0)
            else:
                inter_distance_array[traj_index, step] = np.copy(
                    inter_distance_list)

    inter_distance_array_over_traj = np.mean(inter_distance_array, axis=0)
    kmc_steps = range(0, material_analysis.num_path_steps_per_traj
                      * int(material_analysis.stepInterval),
                      int(material_analysis.stepInterval))
    if mean:
        mean_distance_array = np.zeros(
                                (material_analysis.num_path_steps_per_traj, 2))
        mean_distance_array[:, 0] = kmc_steps
        mean_distance_array[:, 1] = mean_distance_over_traj
        mean_distance_file_name = 'mean_distance_data.npy'
        mean_distance_file_path = dst_path.joinpath(mean_distance_file_name)
        np.save(mean_distance_file_path, mean_distance_array)
    else:
        inter_species_distance_array = np.zeros(
            (material_analysis.num_path_steps_per_traj,
             n_electrons * (n_electrons - 1) / 2 + 1))
        inter_species_distance_array[:, 0] = kmc_steps
        inter_species_distance_array[:, 1:] = (
            inter_distance_array_over_traj)
        inter_species_distance_file_name = 'inter_species_distance.npy'
        inter_species_distance_file_path = dst_path.joinpath(
            inter_species_distance_file_name)
        np.save(inter_species_distance_file_path,
                inter_species_distance_array)

    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure()
        if mean:
            plt.plot(mean_distance_array[:, 0], mean_distance_array[:, 1])
            plt.title('Mean Distance between species along '
                      'simulation length')
            plt.xlabel('KMC Step')
            plt.ylabel('Distance (' + material_analysis.repr_dist + ')')
            figure_name = 'MeanDistanceOverTraj.jpg'
            figure_path = dst_path.joinpath(figure_name)
            plt.savefig(figure_path)
        else:
            legend_list = []
            for i in range(n_electrons):
                for j in range(i + 1, n_electrons):
                    legend_list.append('r_' + str(i) + ':' + str(j))
            line_objects = plt.plot(inter_species_distance_array[:, 0],
                                    inter_species_distance_array[:, 1:])
            plt.title('Inter-species Distances along simulation length')
            plt.xlabel('KMC Step')
            plt.ylabel('Distance (' + material_analysis.repr_dist + ')')
            lgd = plt.legend(line_objects, legend_list, loc='center left',
                             bbox_to_anchor=(1, 0.5))
            figure_name = 'inter_species_distance.jpg'
            figure_path = dst_path.joinpath(figure_name)
            plt.savefig(figure_path, bbox_extra_artists=(lgd,),
                        bbox_inches='tight')
    if report:
        generate_mean_displacement_analysis_log_report(dst_path, start_time)
    output = mean_distance_array if mean else inter_species_distance_array
    return output


def generate_mean_displacement_analysis_log_report(dst_path, start_time):
    """Generates an log report of the MSD Analysis and outputs to the
        working directory"""
    mean_displacement_analysis_log_file_name = (
        'mean_displacement_analysis.log')
    mean_displacement_analysis_log_file_path = dst_path.joinpath(
        mean_displacement_analysis_log_file_name)
    report = open(mean_displacement_analysis_log_file_path, 'w')
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
