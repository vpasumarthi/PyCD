# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os

import yaml

from PyCT.core import Material, Analysis


def materialCOCMSD(systemDirectoryPath, fileFormatIndex, system_size, pbc, n_dim,
                   temp, ion_charge_type, species_charge_type, species_count,
                   t_final, n_traj, time_interval, msd_t_final, trim_length,
                   display_error_bars, repr_time, repr_dist, report):

    # Load material parameters
    configDirName = 'ConfigurationFiles'
    config_file_name = 'sys_config.yml'
    config_file_path = os.path.join(systemDirectoryPath, configDirName,
                                  config_file_name)
    with open(config_file_path, 'r') as stream:
        try:
            params = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    input_coordinate_file_name = 'POSCAR'
    input_coord_file_location = os.path.join(systemDirectoryPath, configDirName,
                                         input_coordinate_file_name)
    params.update({'input_coord_file_location': input_coord_file_location})
    params.update({'fileFormatIndex': fileFormatIndex})
    material_parameters = ReturnValues(params)

    # Build material object files
    material_info = Material(material_parameters)

    # Change to working directory
    parentDir1 = 'SimulationFiles'
    parentDir2 = ('ion_charge_type=' + ion_charge_type
                  + '; species_charge_type=' + species_charge_type)
    n_electrons = species_count[0]
    n_holes = species_count[1]
    parentDir3 = (str(n_electrons)
                  + ('electron' if n_electrons == 1 else 'electrons') + ', '
                  + str(n_holes) + ('hole' if n_holes == 1 else 'holes'))
    parentDir4 = str(temp) + 'K'
    workDir = (('%1.2E' % t_final) + 'SEC,' + ('%1.2E' % time_interval)
               + 'TimeInterval,' + ('%1.2E' % n_traj) + 'Traj')
    workDirPath = os.path.join(systemDirectoryPath, parentDir1, parentDir2,
                               parentDir3, parentDir4, workDir)

    if not os.path.exists(workDirPath):
        print('Simulation files do not exist. Aborting.')
    else:
        os.chdir(workDirPath)

        material_analysis = Analysis(material_info, n_dim, species_count,
                                    n_traj, t_final, time_interval, msd_t_final,
                                    trim_length, repr_time, repr_dist)

        msd_analysis_data = material_analysis.compute_coc_msd(workDirPath, report)
        msd_data = msd_analysis_data.msd_data
        std_data = msd_analysis_data.std_data
        species_types = msd_analysis_data.species_types
        file_name = msd_analysis_data.file_name
        material_analysis.generate_coc_msd_plot(msd_data, std_data, display_error_bars,
                                            species_types, file_name,
                                            workDirPath)


class ReturnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)
