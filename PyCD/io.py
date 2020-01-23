# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from datetime import datetime

import numpy as np


def read_poscar(input_file_path):
    input_file = open(input_file_path, 'r')
    element_types_line_number = 6
    for line_index, line in enumerate(input_file):
        line_number = line_index + 1
        if line_number == element_types_line_number:
            prospective_element_types = line[:-1].split()
            element_types_exist = prospective_element_types[0].isalpha()
            break
    input_file.close()

    lattice_matrix = np.zeros((3, 3))
    lattice_parameter_index = 0
    lattice_parameters_line_range = range(3, 6)
    element_types_line_number = 6 * element_types_exist
    num_elements_line_number = 6 + element_types_exist
    coordinate_type_number = 7 + element_types_exist
    coord_start_line_number = 8 + element_types_exist
    input_file = open(input_file_path, 'r')
    for line_index, line in enumerate(input_file):
        line_number = line_index + 1
        if line_number == 1 and not element_types_exist:
            element_types = line.split()
        elif line_number in lattice_parameters_line_range:
            lattice_matrix[lattice_parameter_index, :] = (
                                        np.fromstring(line, sep=' '))
            lattice_parameter_index += 1
        elif (line_number == element_types_line_number
              and element_types_exist):
            element_types = line.split()
        elif line_number == num_elements_line_number:
            num_elements = np.fromstring(line, dtype=int, sep=' ')
            total_elements = num_elements.sum()
            coordinates = np.zeros((total_elements, 3))
        elif line_number == coordinate_type_number:
            coordinate_type = line.split()[0]
        elif line_number == coord_start_line_number:
            element_index = 0
            coordinates[element_index, :] = np.fromstring(line, sep=' ')
            coordinate_string_length = len(line.split()[0])
            if coordinate_string_length == 18:
                file_format = 'VASP'
            elif coordinate_string_length == 11:
                file_format = 'VESTA'
            else:
                file_format = 'unknown'
        elif ((line_number > coord_start_line_number)
              and (element_index < total_elements - 1)):
            element_index += 1
            coordinates[element_index, :] = np.fromstring(line, sep=' ')
    input_file.close()
    poscar_info_keys = {'lattice_matrix', 'element_types', 'num_elements',
                        'total_elements', 'coordinate_type', 'coordinates',
                        'file_format'}
    poscar_info = {}
    for key in poscar_info_keys:
        poscar_info[key] = locals()[key]
    return poscar_info


def write_poscar(src_file_path, dst_file_path, file_format,
                 element_types_cluster, num_elements_cluster, coordinate_type,
                 coordinates_cluster):
    unmodified_line_number_limit = 5
    src_file = open(src_file_path, 'r')
    open(dst_file_path, 'w').close()
    dst_file = open(dst_file_path, 'a')
    for line_index, line in enumerate(src_file):
        line_number = line_index + 1
        if line_number <= unmodified_line_number_limit:
            dst_file.write(line)
        else:
            break
    src_file.close()

    element_types_line = (' ' * 3 + (' ' * 4).join(element_types_cluster)
                          + '\n')
    dst_file.write(element_types_line)
    num_elements_line = (
            ' ' * 3 + (' ' * 4).join(map(str, num_elements_cluster)) + '\n')
    dst_file.write(num_elements_line)
    dst_file.write(coordinate_type + '\n')
    for element_coordinates in coordinates_cluster:
        if file_format == 'VASP' or file_format == 'unknown':
            line = (
                ''.join([
                    ' ' * 1,
                    [' ', '-'][element_coordinates[0] < 0],
                    f'{abs(element_coordinates[0]):18.16f}',
                    ' ' * 1,
                    [' ', '-'][element_coordinates[1] < 0],
                    f'{abs(element_coordinates[1]):18.16f}',
                    ' ' * 1,
                    [' ', '-'][element_coordinates[2] < 0],
                    f'{abs(element_coordinates[2]):18.16f}'])
                + '\n')
        elif file_format == 'VESTA':
            line = (
                ''.join([
                    ' ' * 4,
                    [' ', '-'][element_coordinates[0] < 0],
                    f'{abs(element_coordinates[0]):11.9f}',
                    ' ' * 8,
                    [' ', '-'][element_coordinates[1] < 0],
                    f'{abs(element_coordinates[1]):11.9f}',
                    ' ' * 8,
                    [' ', '-'][element_coordinates[2] < 0],
                    f'{abs(element_coordinates[2]):11.9f}'])
                + '\n')
        dst_file.write(line)
    dst_file.close()
    return None


def generate_report(start_time, dst_path, file_name, print_time_elapsed,
                    prefix=None):
    """Generates a report file to the output directory"""
    report_file_name = file_name + '.log'
    report_file_path = dst_path / report_file_name
    report = open(report_file_path, 'w')
    end_time = datetime.now()
    time_elapsed = end_time - start_time
    if prefix:
        report.write(prefix)
    if print_time_elapsed:
        report.write('Time elapsed: ' + ('%2d days, ' % time_elapsed.days
                                         if time_elapsed.days else '')
                     + ('%2d hours' % ((time_elapsed.seconds // 3600) % 24))
                     + (', %2d minutes' % ((time_elapsed.seconds // 60) % 60))
                     + (', %2d seconds' % (time_elapsed.seconds % 60)))
    report.close()
    return None
