# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from datetime import datetime

import numpy as np
import yaml

from PyCT.core import Material, Analysis


def compute_coc_msd(dst_path, report=1):
    """Returns the squared displacement of the trajectories"""
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

    assert dst_path, 'Please provide the destination path where ' \
                     'MSD output files needs to be saved'
    num_existent_species = 0
    for species_type_index in range(material_info.num_species_types):
        if material_analysis.species_count[species_type_index] != 0:
            num_existent_species += 1

    position_array = np.loadtxt(dst_path.joinpath('unwrapped_traj.dat'))
    num_traj_recorded = int(len(position_array)
                            / material_analysis.num_path_steps_per_traj)
    position_array = (
                position_array[
                    :num_traj_recorded
                    * material_analysis.num_path_steps_per_traj + 1].reshape(
                    (num_traj_recorded
                     * material_analysis.num_path_steps_per_traj,
                     material_analysis.total_species, 3))
                * material_analysis.dist_conversion)
    coc_position_array = np.mean(position_array, axis=1)
    np.savetxt('coc_position_array.txt', coc_position_array)
    file_name = 'center_of_charge'
    plot_coc_dispvector(material_analysis, coc_position_array, file_name,
                        dst_path)
    coc_position_array = coc_position_array[:, np.newaxis, :]
    sd_array = np.zeros((num_traj_recorded,
                         material_analysis.num_msd_steps_per_traj,
                         num_existent_species))
    for traj_index in range(num_traj_recorded):
        head_start = traj_index * material_analysis.num_path_steps_per_traj
        for time_step in range(1, material_analysis.num_msd_steps_per_traj):
            num_disp = material_analysis.num_path_steps_per_traj - time_step
            add_on = np.arange(num_disp)
            pos_diff = (coc_position_array[head_start + time_step + add_on]
                        - coc_position_array[head_start + add_on])
            sd_array[traj_index, time_step, :] = np.mean(
                np.einsum('ijk,ijk->ij', pos_diff, pos_diff), axis=0)
    species_avg_sd_array = np.zeros(
        (num_traj_recorded, material_analysis.num_msd_steps_per_traj,
         material_info.num_species_types
         - list(material_analysis.species_count).count(0)))
    start_index = 0
    num_non_existent_species = 0
    non_existent_species_indices = []
    for species_type_index in range(material_info.num_species_types):
        if material_analysis.species_count[species_type_index] != 0:
            end_index = start_index + material_analysis.species_count[
                                                            species_type_index]
            species_avg_sd_array[
                :, :, (species_type_index - num_non_existent_species)] = \
                np.mean(sd_array[:, :, start_index:end_index], axis=2)
            start_index = end_index
        else:
            num_non_existent_species += 1
            non_existent_species_indices.append(species_type_index)

    msd_data = np.zeros((material_analysis.num_msd_steps_per_traj,
                         (material_info.num_species_types + 1
                          - list(material_analysis.species_count).count(0))))
    time_array = (np.arange(material_analysis.num_msd_steps_per_traj)
                  * material_analysis.time_interval
                  * material_analysis.time_conversion)
    msd_data[:, 0] = time_array
    msd_data[:, 1:] = np.mean(species_avg_sd_array, axis=0)
    std_data = np.std(species_avg_sd_array, axis=0)
    file_name = (('%1.2E' % (material_analysis.msd_t_final
                             * material_analysis.time_conversion))
                 + str(material_analysis.repr_time)
                 + (',n_traj: %1.2E' % num_traj_recorded
                 if num_traj_recorded != material_analysis.n_traj else ''))
    msd_file_name = 'coc_msd_data_' + file_name + '.npy'
    msd_file_path = dst_path.joinpath(msd_file_name)
    species_types = [species_type for index, species_type in enumerate(
                                                material_info.species_types)
                     if index not in non_existent_species_indices]
    np.save(msd_file_path, msd_data)

    if report:
        generate_coc_msd_analysis_log_report(material_info, material_analysis,
                                             msd_data, species_types,
                                             file_name, dst_path, start_time)

    return_msd_data = ReturnValues(msd_data=msd_data,
                                   std_data=std_data,
                                   species_types=species_types,
                                   file_name=file_name)
    return return_msd_data


def plot_coc_dispvector(material_analysis, coc_position_array, file_name,
                        dst_path):
    """Returns a line plot of the MSD data"""
    assert dst_path, 'Please provide the destination path where MSD Plot ' \
                     'files needs to be saved'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    num_traj_recorded = int(len(coc_position_array)
                            / material_analysis.num_path_steps_per_traj)
    x_min = y_min = z_min = 10
    x_max = y_max = z_max = -10
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, num_traj_recorded)]
    disp_vector_list = np.zeros((num_traj_recorded, 6))
    for traj_index in range(num_traj_recorded):
        start_pos = coc_position_array[
                        traj_index * material_analysis.num_path_steps_per_traj]
        end_pos = coc_position_array[
            (traj_index + 1) * material_analysis.num_path_steps_per_traj - 1]
        disp_vector_list[traj_index, :3] = start_pos
        disp_vector_list[traj_index, 3:] = end_pos
        pos_stack = np.vstack((start_pos, end_pos))
        ax.plot(pos_stack[:, 0], pos_stack[:, 1], pos_stack[:, 2],
                color=colors[traj_index])
        x_min = min(x_min, start_pos[0], end_pos[0])
        y_min = min(y_min, start_pos[1], end_pos[1])
        z_min = min(z_min, start_pos[2], end_pos[2])
        x_max = max(x_max, start_pos[0], end_pos[0])
        y_max = max(y_max, start_pos[1], end_pos[1])
        z_max = max(z_max, start_pos[2], end_pos[2])
    np.savetxt('displacement_vector_list.txt', disp_vector_list)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([x_min - 0.2 * abs(x_min), x_max + 0.2 * abs(x_max)])
    ax.set_ylim([y_min - 0.2 * abs(y_min), y_max + 0.2 * abs(y_max)])
    ax.set_zlim([z_min - 0.2 * abs(z_min), z_max + 0.2 * abs(z_max)])
    ax.set_title('trajectory-wise center of charge displacement vectors '
                 '\n$N_{{%s}}$=' % 'species'
                 + str(material_analysis.total_species))
    plt.show()  # temp change
    figure_name = ('coc_disp_vectors_' + file_name + '.png')
    figure_path = dst_path.joinpath(figure_name)
    plt.savefig(figure_path)
    return None


def generate_coc_msd_plot(material_info, material_analysis, msd_data, std_data,
                          display_error_bars, species_types, file_name,
                          dst_path):
    """Returns a line plot of the MSD data"""
    assert dst_path, 'Please provide the destination path where ' \
                     'MSD Plot files needs to be saved'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnchoredText
    from textwrap import wrap
    fig = plt.figure()
    ax = fig.add_subplot(111)
    from scipy.stats import linregress
    for species_index, species_type in enumerate(species_types):
        ax.plot(msd_data[:, 0], msd_data[:, species_index + 1], 'o',
                markerfacecolor='blue', markeredgecolor='black',
                label=species_type)
        if display_error_bars:
            ax.errorbar(msd_data[:, 0], msd_data[:, species_index + 1],
                        yerr=std_data[:, species_index], fmt='o',
                        capsize=3, color='blue', markerfacecolor='none',
                        markeredgecolor='none')
        slope, intercept, r_value, _, _ = \
            linregress(msd_data[material_analysis.trim_length:
                                -material_analysis.trim_length, 0],
                       msd_data[material_analysis.trim_length:
                                -material_analysis.trim_length,
                       species_index + 1])
        species_diff = (slope * material_info.ANG2UM ** 2
                        * material_info.SEC2NS / (2 * material_analysis.n_dim))
        ax.add_artist(
            AnchoredText('Est. $D_{{%s}}$ = %4.3f' % (species_type,
                                                      species_diff)
                         + '  ${{\mu}}m^2/s$; $r^2$=%4.3e' % (r_value ** 2),
                         loc=4))
        ax.plot(msd_data[material_analysis.trim_length:
                         -material_analysis.trim_length, 0],
                intercept + slope * msd_data[
                                    material_analysis.trim_length:
                                    -material_analysis.trim_length, 0],
                'r', label=species_type + '-fitted')
    ax.set_xlabel('Time (' + material_analysis.repr_time + ')')
    ax.set_ylabel('MSD (' + ('$\AA^2$'
                  if material_analysis.repr_dist == 'angstrom'
                  else (material_analysis.repr_dist + '^2')) + ')')
    figure_title = 'MSD_' + file_name
    ax.set_title('\n'.join(wrap(figure_title, 60)))
    plt.legend()
    plt.show()  # temp change
    figure_name = ('coc_msd_plot_' + file_name + '_trim='
                   + str(material_analysis.trim_length) + '.png')
    figure_path = dst_path.joinpath(figure_name)
    plt.savefig(figure_path)
    return None


def generate_coc_msd_analysis_log_report(material_info, material_analysis,
                                         msd_data, species_types, file_name,
                                         dst_path, start_time):
    """Generates an log report of the MSD Analysis and
        outputs to the working directory"""
    msd_analysis_log_file_name = (
            'coc_msd_analysis' + ('_' if file_name else '')
            + file_name + '.log')
    msd_log_file_path = dst_path.joinpath(msd_analysis_log_file_name)
    report = open(msd_log_file_path, 'w')
    end_time = datetime.now()
    time_elapsed = end_time - start_time
    from scipy.stats import linregress
    for species_index, species_type in enumerate(species_types):
        slope, _, _, _, _ = \
            linregress(msd_data[material_analysis.trim_length:
                                -material_analysis.trim_length, 0],
                       msd_data[material_analysis.trim_length:
                                -material_analysis.trim_length,
                       species_index + 1])
        species_diff = (slope * material_info.ANG2UM ** 2
                        * material_info.SEC2NS / (2 * material_analysis.n_dim))
        report.write('Estimated value of {:s} diffusivity is: '
                     '{:4.3f} um2/s\n'.format(species_type, species_diff))
    report.write('Time elapsed: ' + ('%2d days, ' % time_elapsed.days
                                     if time_elapsed.days else '')
                 + ('%2d hours' % ((time_elapsed.seconds // 3600) % 24))
                 + (', %2d minutes' % ((time_elapsed.seconds // 60) % 60))
                 + (', %2d seconds' % (time_elapsed.seconds % 60)))
    report.close()
    return None


class ReturnValues(object):
    """dummy class to return objects from methods defined inside
        other classes"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
