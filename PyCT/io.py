#!/usr/bin/env python

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
            n_elements_per_unit_cell = np.fromstring(
                                            line, dtype=int, sep=' ')
            total_elements_per_unit_cell = (
                                        n_elements_per_unit_cell.sum())
            fractional_unit_cell_coords = np.zeros(
                                    (total_elements_per_unit_cell, 3))
            element_index = 0
        elif (line_number > 8
              and element_index < total_elements_per_unit_cell):
            fractional_unit_cell_coords[element_index, :] = (
                                        np.fromstring(line, sep=' '))
            element_index += 1
    input_file.close()
    poscar_info = np.array(
        [lattice_matrix, element_types, n_elements_per_unit_cell,
         total_elements_per_unit_cell, fractional_unit_cell_coords],
                           dtype=object)
    return poscar_info
