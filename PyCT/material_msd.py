#!/usr/bin/env python

import numpy as np
import yaml

from PyCT.core import Material, Analysis


def material_msd(dst_path):
    # Load simulation parameters
    sim_param_file_name = 'simulation_parameters.yml'
    sim_param_file_path = dst_path / sim_param_file_name
    with open(sim_param_file_path, 'r') as stream:
        try:
            sim_params = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # data type conversion:
    sim_params['species_count'] = np.asarray(
                                        sim_params['species_count'])

    # Load material parameters
    config_file_name = 'sys_config.yml'
    input_directory_path = (
        dst_path.resolve().parents[sim_params['work_dir_depth'] - 1]
        / sim_params['input_file_directory_name'])
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
    material_parameters = ReturnValues(params)

    # Build material object files
    material_info = Material(material_parameters)

    material_analysis = Analysis(
        material_info, sim_params['n_dim'], sim_params['species_count'],
        sim_params['n_traj'], sim_params['t_final'],
        sim_params['time_interval'], sim_params['msd_t_final'],
        sim_params['trim_length'], sim_params['repr_time'],
        sim_params['repr_dist'])

    msd_analysis_data = material_analysis.compute_msd(
                                        dst_path, sim_params['report'])
    msd_data = msd_analysis_data.msd_data
    std_data = msd_analysis_data.std_data
    species_types = msd_analysis_data.species_types
    file_name = msd_analysis_data.file_name
    material_analysis.generate_msd_plot(msd_data, std_data,
                                        sim_params['display_error_bars'],
                                        species_types, file_name,
                                        dst_path)
    return None


class ReturnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            setattr(self, key, value)