#!/usr/bin/env python

from datetime import datetime

import numpy as np


def read_poscar(input_file_path):
    lattice_matrix = np.zeros((3, 3))
    lattice_parameter_index = 0
    lattice_parameters_line_range = range(3, 6)
    input_file = open(input_file_path, 'r')
    for line_index, line in enumerate(input_file):
        line_number = line_index + 1
        if line_number in lattice_parameters_line_range:
            lattice_matrix[lattice_parameter_index, :] = (
                                        np.fromstring(line, sep=' '))
            lattice_parameter_index += 1
        elif line_number == 6:
            element_types = line.split()
        elif line_number == 7:
            n_elements = np.fromstring(line, dtype=int, sep=' ')
            total_elements = n_elements.sum()
            coordinates = np.zeros((total_elements, 3))
            element_index = 0
        elif line_number == 8:
            coordinate_type = line.split()[0]
        elif (line_number > 8
              and element_index < total_elements):
            coordinates[element_index, :] = np.fromstring(line, sep=' ')
            element_index += 1
    input_file.close()
    poscar_info = np.array(
        [lattice_matrix, element_types, n_elements, total_elements,
         coordinate_type, coordinates], dtype=object)
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
                    ' ' * 2,
                    '%18.16f' % element_coordinates[0],
                    ' ' * 2,
                    '%18.16f' % element_coordinates[1],
                    ' ' * 2,
                    '%18.16f' % element_coordinates[2]])
                + '\n')
        elif file_format == 'VESTA':
            line = (
                ''.join([
                    ' ' * 5,
                    '%11.9f' % element_coordinates[0],
                    ' ' * 9,
                    '%11.9f' % element_coordinates[1],
                    ' ' * 9,
                    '%11.9f' % element_coordinates[2]])
                + '\n')
        dst_file.write(line)
    dst_file.close()
    return None


def generate_report(start_time, dst_path, file_name, prefix=None):
    """Generates a report file to the output directory"""
    report_file_name = file_name + '.log'
    report_file_path = dst_path / report_file_name
    report = open(report_file_path, 'w')
    end_time = datetime.now()
    time_elapsed = end_time - start_time
    if prefix:
        report.write(prefix)
    report.write('Time elapsed: ' + ('%2d days, ' % time_elapsed.days
                                     if time_elapsed.days else '')
                 + ('%2d hours' % ((time_elapsed.seconds // 3600) % 24))
                 + (', %2d minutes' % ((time_elapsed.seconds // 60) % 60))
                 + (', %2d seconds' % (time_elapsed.seconds % 60)))
    report.close()
    return None
