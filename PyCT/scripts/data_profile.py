#!/usr/bin/env python

from textwrap import wrap
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt


class DataProfile(object):
    def __init__(self, dst_path, system_directory_path,
                 profiling_species_type_index, non_var_species_count,
                 var_species_count_list, ion_charge_type, species_charge_type,
                 temp, t_final, time_interval, n_traj, external_field):
        self.dst_path = dst_path
        self.system_directory_path = system_directory_path
        self.profiling_species_type_index = profiling_species_type_index
        self.var_species_type = (
            'electron' if self.profiling_species_type_index == 0 else 'hole')
        self.non_var_species_count = non_var_species_count
        self.var_species_count_list = var_species_count_list
        self.profile_length = len(self.var_species_count_list)
        self.species_count_list = np.zeros((self.profile_length, 2), int)
        if self.profiling_species_type_index == 0:
            self.species_count_list[:, 0] = self.var_species_count_list
            self.species_count_list[:, 1] = self.non_var_species_count
        else:
            self.species_count_list[:, 0] = self.non_var_species_count
            self.species_count_list[:, 1] = self.var_species_count_list
        self.ion_charge_type = ion_charge_type
        self.species_charge_type = species_charge_type
        self.temp = temp
        self.t_final = t_final
        self.time_interval = time_interval
        self.n_traj = n_traj
        self.external_field = external_field
        ld_tag = 'ld_' if self.external_field['electric']['ld'] else ''
        ef_field_tag = (
            'ef_' + ld_tag
            + str(self.external_field['electric']['dir']).replace(' ','') + '_'
            + ('%1.2E' % self.external_field['electric']['mag']))
        self.field_tag = (
                ef_field_tag
                if self.external_field['electric']['active'] else 'no_field')
        return None

    def generate_work_dir_path(self, species_count):
        (n_electrons, n_holes) = species_count
        parent_dir1 = 'SimulationFiles'
        parent_dir2 = ('IonChargeType=' + self.ion_charge_type
                       + ';SpeciesChargeType=' + self.species_charge_type)
        parent_dir3 = (str(n_electrons)
                       + ('electron' if n_electrons == 1 else 'electrons')
                       + ',' + str(n_holes)
                       + ('hole' if n_holes == 1 else 'holes'))
        parent_dir4 = str(self.temp) + 'K'
        parent_dir5 = (('%1.2E' % self.t_final) + 'SEC,'
                       + ('%1.2E' % self.time_interval) + 'TimeInterval,'
                       + ('%1.2E' % self.n_traj) + 'Traj')
        work_dir_path = (self.system_directory_path / parent_dir1 / parent_dir2
                         / parent_dir3 / parent_dir4 / parent_dir5
                         / self.field_tag)
        return work_dir_path

    def generate_profile_plot(self, profile_data, y_label, figure_title,
                              profiling_quantity, plot_error_bars):
        plt.switch_backend('Agg')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(profile_data[:, 0], profile_data[:, 1], 'o-', color='blue',
                markerfacecolor='blue', markeredgecolor='black')
        if plot_error_bars:
            ax.errorbar(profile_data[:, 0], profile_data[:, 1],
                        yerr=profile_data[:, 2], fmt='o', capsize=3,
                        color='blue', markerfacecolor='none',
                        markeredgecolor='none')
        ax.set_xlabel('Number of ' + self.var_species_type + 's')
        ax.set_ylabel(y_label)
        ax.set_title('\n'.join(wrap(figure_title, 60)))
        filename = (
            self.var_species_type + '_' + profiling_quantity + '_profile_'
            + self.ion_charge_type[0] + self.species_charge_type[0]
            + '_' + str(self.var_species_count_list[0])
            + '-' + str(self.var_species_count_list[-1])
            + '_' + self.field_tag)
        figure_name = filename + '.png'
        figure_path = self.dst_path / figure_name
        plt.tight_layout()
        plt.savefig(str(figure_path))
        return filename

    def diffusion_profile(self, plot_error_bars, msd_t_final, repr_time):
        diffusivity_profile_data = np.zeros((self.profile_length, 3))
        diffusivity_profile_data[:, 0] = np.copy(self.var_species_count_list)
        file_name = '%1.2E%s' % (msd_t_final, repr_time)
        msd_analysis_log_file_name = (
            'MSD_Analysis' + ('_' if file_name else '') + file_name + '.log')

        for index in range(self.profile_length):
            species_count = self.species_count_list[index, :]
            work_dir_path = self.generate_work_dir_path(species_count)
            msd_analysis_log_file_path = (work_dir_path
                                          / msd_analysis_log_file_name)
            with open(msd_analysis_log_file_path, 'r') as msd_analysis_log_file:
                first_line = msd_analysis_log_file.readline()
                second_line = msd_analysis_log_file.readline()
            diffusivity_profile_data[index, 1] = float(first_line[-13:-6])
            diffusivity_profile_data[index, 2] = float(second_line[-13:-6])

        y_label = 'Diffusivity (${{\mu}}m^2/s$)'
        figure_title = ('Diffusion coefficient as a function of number of '
                        + self.var_species_type + 's')
        profiling_quantity = 'diffusion'
        filename = self.generate_profile_plot(
                            diffusivity_profile_data, y_label, figure_title,
                            profiling_quantity, plot_error_bars)
        data_file_name = filename + '.dat'
        data_file_path = self.dst_path / data_file_name
        np.savetxt(data_file_path, diffusivity_profile_data)
        return None

    def drift_mobility_profile(self, plot_error_bars):
        drift_mobility_profile_data = np.zeros((self.profile_length, 3))
        drift_mobility_profile_data[:, 0] = np.copy(self.var_species_count_list)
        run_log_file_name = 'Run.log'

        for index in range(self.profile_length):
            species_count = self.species_count_list[index, :]
            work_dir_path = self.generate_work_dir_path(species_count)
            msd_analysis_log_file_path = work_dir_path / run_log_file_name
            with open(msd_analysis_log_file_path, 'r') as msd_analysis_log_file:
                first_line = msd_analysis_log_file.readline()
                second_line = msd_analysis_log_file.readline()
            drift_mobility_profile_data[index, 1] = float(first_line[-19:-10])
            drift_mobility_profile_data[index, 2] = float(second_line[-19:-10])

        y_label = 'Drift mobility ($cm^2/V.s$)'
        figure_title = ('Drift mobility as a function of number of '
                        + self.var_species_type + 's')
        profiling_quantity = 'drift_mobility'
        filename = self.generate_profile_plot(
                            drift_mobility_profile_data, y_label, figure_title,
                            profiling_quantity, plot_error_bars)
        data_file_name = filename + '.dat'
        data_file_path = self.dst_path / data_file_name
        np.savetxt(data_file_path, drift_mobility_profile_data)
        return None

    def runtime_profile(self):
        elapsed_seconds_data = np.zeros((self.profile_length, 2))
        elapsed_seconds_data[:, 0] = np.copy(self.var_species_count_list)
        run_log_file_name = 'Run.log'

        for index in range(self.profile_length):
            species_count = self.species_count_list[index, :]
            work_dir_path = self.generate_work_dir_path(species_count)
            run_log_file_path = work_dir_path / run_log_file_name
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
            elapsed_seconds_data[index, 1] = int(elapsed_time.total_seconds())

        y_label = 'Run Time (sec)'
        figure_title = ('Simulation run time as a function of number of '
                       + self.var_species_type + 's')
        profiling_quantity = 'run_time'
        plot_error_bars = 0
        filename = self.generate_profile_plot(
                            elapsed_seconds_data, y_label, figure_title,
                            profiling_quantity, plot_error_bars)
        data_file_name = filename + '.dat'
        data_file_path = self.dst_path / data_file_name
        np.savetxt(data_file_path, elapsed_seconds_data)
        return None

