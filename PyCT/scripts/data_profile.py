#!/usr/bin/env python

import os.path
from textwrap import wrap
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt


def diffusion_profile(out_dir, system_directory_path, species_count_list,
                      ion_charge_type, species_charge_type, plot_error_bars,
                      temp, t_final, time_interval, n_traj, msd_t_final,
                      external_field, repr_time):
    if len(species_count_list[0]) == 1:
        profiling_species_type_index = 1
    else:
        profiling_species_type_index = 0
    profiling_species_list = species_count_list[profiling_species_type_index]
    profile_length = len(profiling_species_list)
    diffusivity_profile_data = np.zeros((profile_length, 3))
    diffusivity_profile_data[:, 0] = profiling_species_list
    species_type = 'electron' if profiling_species_type_index == 0 else 'hole'
    if profiling_species_type_index == 0:
        n_holes = species_count_list[1][0]
    else:
        n_electrons = species_count_list[0][0]

    parent_dir1 = 'SimulationFiles'
    parent_dir2 = ('IonChargeType=' + ion_charge_type
                   + ';SpeciesChargeType=' + species_charge_type)

    file_name = '%1.2E%s' % (msd_t_final, repr_time)
    msd_analysis_log_file_name = ('MSD_Analysis' + ('_' if file_name else '')
                                  + file_name + '.log')

    for species_index, n_species in enumerate(profiling_species_list):
        # Change to working directory
        if profiling_species_type_index == 0:
            n_electrons = n_species
        else:
            n_holes = n_species
        parent_dir3 = (str(n_electrons)
                       + ('electron' if n_electrons == 1 else 'electrons')
                       + ',' + str(n_holes)
                       + ('hole' if n_holes == 1 else 'holes'))
        parent_dir4 = str(temp) + 'K'
        parent_dir5 = (('%1.2E' % t_final) + 'SEC,' + ('%1.2E' % time_interval)
                       + 'TimeInterval,' + ('%1.2E' % n_traj) + 'Traj')
        ld_tag = 'ld_' if external_field['electric']['ld'] else ''
        field_tag = (
                'ef_' + ld_tag
                + str(external_field['electric']['dir']).replace(' ','') + '_'
                + ('%1.2E' % external_field['electric']['mag']))
        work_dir = (field_tag
                    if external_field['electric']['active'] else 'no_field')
        msd_analysis_log_file_path = os.path.join(
                system_directory_path, parent_dir1, parent_dir2, parent_dir3,
                parent_dir4, parent_dir5, work_dir, msd_analysis_log_file_name)

        with open(msd_analysis_log_file_path, 'r') as msd_analysis_log_file:
            first_line = msd_analysis_log_file.readline()
            second_line = msd_analysis_log_file.readline()
        diffusivity_profile_data[species_index, 1] = float(first_line[-13:-6])
        diffusivity_profile_data[species_index, 2] = float(second_line[-13:-6])

    plt.switch_backend('Agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(diffusivity_profile_data[:, 0], diffusivity_profile_data[:, 1],
            'o-', color='blue', markerfacecolor='blue', markeredgecolor='black'
            )
    if plot_error_bars:
        ax.errorbar(diffusivity_profile_data[:, 0],
                    diffusivity_profile_data[:, 1],
                    yerr=diffusivity_profile_data[:, 2], fmt='o', capsize=3,
                    color='blue', markerfacecolor='none',
                    markeredgecolor='none')
    ax.set_xlabel('Number of ' + species_type + 's')
    ax.set_ylabel('Diffusivity (${{\mu}}m^2/s$)')
    figure_title = ('Diffusion coefficient as a function of number of '
                    + species_type + 's')
    ax.set_title('\n'.join(wrap(figure_title, 60)))
    filename = (str(species_type) + '_diffusion_profile_' + ion_charge_type[0]
                + species_charge_type[0] + '_' + str(profiling_species_list[0])
                + '-' + str(profiling_species_list[-1]) + '_' + work_dir)
    figure_name = filename + '.png'
    figure_path = os.path.join(out_dir, figure_name)
    plt.tight_layout()
    plt.savefig(figure_path)

    data_file_name = filename + '.txt'
    data_file_path = os.path.join(out_dir, data_file_name)
    np.savetxt(data_file_path, diffusivity_profile_data)


def runtime_profile(out_dir, system_directory_path, species_count_list,
                    ion_charge_type, species_charge_type, temp, t_final,
                    time_interval, n_traj, external_field):
    if len(species_count_list[0]) == 1:
        profiling_species_type_index = 1
    else:
        profiling_species_type_index = 0
    profiling_species_list = species_count_list[profiling_species_type_index]
    profile_length = len(profiling_species_list)
    elapsed_seconds_data = np.zeros((profile_length, 2))
    elapsed_seconds_data[:, 0] = profiling_species_list
    species_type = 'electron' if profiling_species_type_index == 0 else 'hole'

    if profiling_species_type_index == 0:
        n_holes = species_count_list[1][0]
    else:
        n_electrons = species_count_list[0][0]

    parent_dir1 = 'SimulationFiles'
    parent_dir2 = ('IonChargeType=' + ion_charge_type
                   + ';SpeciesChargeType=' + species_charge_type)
    run_log_file_name = 'Run.log'

    for species_index, n_species in enumerate(profiling_species_list):
        # Change to working directory
        if profiling_species_type_index == 0:
            n_electrons = n_species
        else:
            n_holes = n_species
        parent_dir3 = (str(n_electrons)
                       + ('electron' if n_electrons == 1 else 'electrons')
                       + ',' + str(n_holes)
                       + ('hole' if n_holes == 1 else 'holes'))
        parent_dir4 = str(temp) + 'K'
        parent_dir5 = (('%1.2E' % t_final) + 'SEC,' + ('%1.2E' % time_interval)
                       + 'TimeInterval,' + ('%1.2E' % n_traj) + 'Traj')
        ld_tag = 'ld_' if external_field['electric']['ld'] else ''
        field_tag = (
                'ef_' + ld_tag
                + str(external_field['electric']['dir']).replace(' ','') + '_'
                + ('%1.2E' % external_field['electric']['mag']))
        work_dir = (field_tag
                    if external_field['electric']['active'] else 'no_field')
        run_log_file_path = os.path.join(
                system_directory_path, parent_dir1, parent_dir2, parent_dir3,
                parent_dir4, parent_dir5, work_dir, run_log_file_name)

        with open(run_log_file_path, 'r') as run_log_file:
            first_line = run_log_file.readline()
        if 'days' in first_line:
            num_days = float(first_line[14:16])
            num_hours = float(first_line[23:25])
            num_minutes = float(first_line[33:35])
            num_seconds = float(first_line[45:47])
        else:
            num_days = 0
            num_hours = float(first_line[14:16])
            num_minutes = float(first_line[24:26])
            num_seconds = float(first_line[36:38])
        elapsed_time = timedelta(days=num_days, hours=num_hours,
                                 minutes=num_minutes, seconds=num_seconds)
        elapsed_seconds_data[species_index, 1] = int(
                                                elapsed_time.total_seconds())

    plt.switch_backend('Agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(elapsed_seconds_data[:, 0], elapsed_seconds_data[:, 1], 'o-',
            color='blue', markerfacecolor='blue', markeredgecolor='black')
    ax.set_xlabel('Number of ' + species_type + 's')
    ax.set_ylabel('Run Time (sec)')
    figure_title = ('Simulation run time as a function of number of '
                   + species_type + 's')
    ax.set_title('\n'.join(wrap(figure_title, 60)))
    filename = (str(species_type) + '_run_time_profile_' + ion_charge_type[0]
                + species_charge_type[0] + '_' + str(profiling_species_list[0])
                + '-' + str(profiling_species_list[-1]) + '_' + work_dir)
    figure_name = filename + '.png'
    figure_path = os.path.join(out_dir, figure_name)
    plt.tight_layout()
    plt.savefig(figure_path)

    data_file_name = filename + '.txt'
    data_file_path = os.path.join(out_dir, data_file_name)
    np.savetxt(data_file_path, elapsed_seconds_data)
