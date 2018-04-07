#!/usr/bin/env python
"""
kMC model to run kinetic Monte Carlo simulations and compute mean
square displacement of random walk of charge carriers on 3D lattice
systems
"""
from pathlib import Path
from datetime import datetime
import random as rnd
from collections import defaultdict
import itertools
import pdb

import numpy as np
from scipy.special import erfc
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from textwrap import wrap

from PyCT.io import read_poscar, generate_report
from PyCT import constants


class Material(object):
    """Defines the properties and structure of working material
    :param str name: A string representing the material name
    :param list element_types: list of chemical elements
    :param dict species_to_element_type_map: list of charge carrier species
    :param unit_cell_coords: positions of all elements in the unit cell
    :type unit_cell_coords: np.array (nx3)
    :param element_type_index_list: list of element types for all unit cell
            coordinates
    :type element_type_index_list: np.array (n)
    :param dict charge_types: types of atomic charges considered for the
            working material
    :param list lattice_parameters: list of three lattice constants in
            angstrom and three angles between them in degrees
    :param float vn: typical frequency for nuclear motion
    :param dict lambda_values: Reorganization energies
    :param dict v_ab: Electronic coupling matrix element
    :param dict neighbor_cutoff_dist: List of neighbors and their respective
            cutoff distances in angstrom
    :param float neighbor_cutoff_dist_tol: Tolerance value in angstrom for
            neighbor cutoff distance
    :param str element_type_delimiter: Delimiter between element types
    :param float epsilon: Dielectric constant of the material

    The additional attributes are:
        * **n_elements_per_unit_cell** (np.array (n)): element-type wise total
            number of elements in a unit cell
        * **element_type_to_species_map** (dict): dictionary of element to
                species mapping
        * **hop_element_types** (dict): dictionary of species to
            hopping element types separated by element_type_delimiter
        * **lattice_matrix** (np.array (3x3): lattice cell matrix
    """

    def __init__(self, material_parameters):
        """

        :param material_parameters:
        """
        # Read Input POSCAR
        poscar_info = (
                    read_poscar(material_parameters.input_coord_file_location))
        self.lattice_matrix = poscar_info['lattice_matrix']
        self.element_types = poscar_info['element_types']
        self.n_elements_per_unit_cell = poscar_info['num_elements']
        self.total_elements_per_unit_cell = poscar_info['total_elements']
        coordinate_type = poscar_info['coordinate_type']
        unit_cell_coords = poscar_info['coordinates']

        if coordinate_type == 'Direct':
            fractional_unit_cell_coords = unit_cell_coords
        elif coordinate_type == 'Cartesian':
            fractional_unit_cell_coords = np.dot(
                        unit_cell_coords, np.linalg.inv(self.lattice_matrix))
        self.lattice_matrix *= constants.ANG2BOHR
        self.num_element_types = len(self.element_types)
        self.element_type_index_list = np.repeat(
            np.arange(self.num_element_types), self.n_elements_per_unit_cell)
        self.name = material_parameters.name
        self.species_types = material_parameters.species_types[:]
        self.num_species_types = len(self.species_types)
        self.species_charge_list = material_parameters.species_charge_list
        self.species_to_element_type_map = {
                    key: values
                    for key, values in
                    material_parameters.species_to_element_type_map.items()}

        # Initialization
        self.fractional_unit_cell_coords = np.zeros(
                                            fractional_unit_cell_coords.shape)
        self.unit_cell_class_list = []
        start_index = 0
        # Reorder element-wise unit cell coordinates in ascending order
        # of z-coordinate
        for element_type_index in range(self.num_element_types):
            element_fract_unit_cell_coords = fractional_unit_cell_coords[
                            self.element_type_index_list == element_type_index]
            end_index = (start_index
                         + self.n_elements_per_unit_cell[element_type_index])
            self.fractional_unit_cell_coords[start_index:end_index] = (
                        element_fract_unit_cell_coords[
                            element_fract_unit_cell_coords[:, 2].argsort()])
            element_type = self.element_types[element_type_index]
            self.unit_cell_class_list.extend(
                [material_parameters.class_list[element_type][index] - 1
                 for index in element_fract_unit_cell_coords[:, 2].argsort()])
            start_index = end_index

        self.cartesian_unit_cell_coords = np.dot(
                        self.fractional_unit_cell_coords, self.lattice_matrix)
        self.charge_types = material_parameters.charge_types

        self.vn = material_parameters.vn / constants.SEC2AUTIME

        self.lambda_values = {
                key: [[value * constants.EV2HARTREE for value in values]
                      for values in class_values]
                for key, class_values in material_parameters.lambda_values.items()}

        self.v_ab = {
                key: [[value * constants.EV2HARTREE for value in values]
                      for values in class_values]
                for key, class_values in material_parameters.v_ab.items()}

        self.neighbor_cutoff_dist = {
            key: [[(value * constants.ANG2BOHR) if value else None
                  for value in values] for values in class_values]
            for key, class_values in material_parameters.neighbor_cutoff_dist.items()
            }

        self.neighbor_cutoff_dist_tol = {
            key: [[(value * constants.ANG2BOHR) if value else None
                  for value in values] for values in class_values]
            for key, class_values in material_parameters.neighbor_cutoff_dist_tol.items()
            }

        self.num_unique_hopping_distances = {
                    key: [len(values) for values in class_values]
                    for key, class_values in (self.neighbor_cutoff_dist.items())}

        self.element_type_delimiter = \
            material_parameters.element_type_delimiter
        self.dielectric_constant = material_parameters.dielectric_constant

        self.num_classes = [
                        len(set(material_parameters.class_list[element_type]))
                        for element_type in self.element_types]

        self.element_type_to_species_map = defaultdict(list)
        for element_type in self.element_types:
            for species_type in self.species_types:
                if element_type in (self.species_to_element_type_map[
                                                                species_type]):
                    self.element_type_to_species_map[element_type].append(
                                                                species_type)

        self.hop_element_types = {
            species_type: [self.element_type_delimiter.join(comb)
                           for comb in list(itertools.product(
                               self.species_to_element_type_map[species_type],
                               repeat=2))]
            for species_type in self.species_types}

    def generate_sites(self, element_type_indices, cell_size):
        """Returns system_element_indices and coordinates of specified elements
            in a cell of size *cell_size*

        :param str element_type_indices: element type indices
        :param cell_size: size of the cell
        :type cell_size: np.array (3x1)
        :return: an object with following attributes:

            * **cell_coordinates** (np.array (nx3)):
            * **quantum_index_list** (np.array (nx5)):
            * **system_element_index_list** (np.array (n)):

        :raises ValueError: if the input cell_size is less than or equal to 0.
        """
        assert all(size > 0 for size in cell_size), 'Input size should always \
                                                    be greater than 0'
        extract_indices = np.in1d(self.element_type_index_list,
                                  element_type_indices).nonzero()[0]
        unit_cell_element_coords = self.cartesian_unit_cell_coords[
                                                            extract_indices]
        num_cells = cell_size.prod()
        n_sites_per_unit_cell = self.n_elements_per_unit_cell[
                                                    element_type_indices].sum()
        unit_cell_element_index_list = np.arange(n_sites_per_unit_cell)
        unit_cell_element_type_index = np.reshape(
                    np.concatenate((np.asarray(
                        [[element_type_index]
                         * self.n_elements_per_unit_cell[element_type_index]
                         for element_type_index in element_type_indices]))),
                    (n_sites_per_unit_cell, 1))
        unit_cell_element_type_element_index_list = np.reshape(
                        np.concatenate(([np.arange(
                            self.n_elements_per_unit_cell[element_type_index])
                                         for element_type_index in (
                                             element_type_indices)])),
                        (n_sites_per_unit_cell, 1))
        # Initialization
        cell_coordinates = np.zeros((num_cells * n_sites_per_unit_cell, 3))
        # Definition format of Quantum Indices
        # quantum_index = [unit_cell_index, element_type_index, element_index]
        quantum_index_list = np.zeros((num_cells * n_sites_per_unit_cell, 5),
                                      dtype=int)
        system_element_index_list = np.zeros(num_cells * n_sites_per_unit_cell,
                                             dtype=int)
        i_unit_cell = 0
        for x_index in range(cell_size[0]):
            for y_index in range(cell_size[1]):
                for z_index in range(cell_size[2]):
                    start_index = i_unit_cell * n_sites_per_unit_cell
                    end_index = start_index + n_sites_per_unit_cell
                    translation_vector = np.dot([x_index, y_index, z_index],
                                                self.lattice_matrix)
                    cell_coordinates[start_index:end_index] = (
                                unit_cell_element_coords + translation_vector)
                    system_element_index_list[start_index:end_index] = (
                                            i_unit_cell * n_sites_per_unit_cell
                                            + unit_cell_element_index_list)
                    quantum_index_list[start_index:end_index] = np.hstack(
                                (np.tile(np.array([x_index, y_index, z_index]),
                                         (n_sites_per_unit_cell, 1)),
                                 unit_cell_element_type_index,
                                 unit_cell_element_type_element_index_list))
                    i_unit_cell += 1

        return_sites = ReturnValues(
                        cell_coordinates=cell_coordinates,
                        quantum_index_list=quantum_index_list,
                        system_element_index_list=system_element_index_list)
        return return_sites


class Neighbors(object):
    """Returns the neighbor list file
    :param system_size: size of the super cell in terms of number of
                        unit cell in three dimensions
    :type system_size: np.array (3x1)
    """

    def __init__(self, material, system_size, pbc):
        """

        :param material:
        :param system_size:
        :param pbc:
        """
        self.start_time = datetime.now()
        self.material = material
        self.system_size = system_size
        self.n_dim = len(system_size)
        self.pbc = pbc[:]

        # total number of unit cells
        self.num_cells = self.system_size.prod()
        self.num_system_elements = (
                self.num_cells * self.material.total_elements_per_unit_cell)

        # generate all sites in the system
        self.element_type_indices = range(self.material.num_element_types)
        self.bulk_sites = self.material.generate_sites(
                                self.element_type_indices, self.system_size)

        x_range = range(-1, 2) if self.pbc[0] == 1 else [0]
        y_range = range(-1, 2) if self.pbc[1] == 1 else [0]
        z_range = range(-1, 2) if self.pbc[2] == 1 else [0]
        # Initialization
        self.system_translational_vector_list = np.zeros((3**sum(self.pbc), 3))
        index = 0
        for x_offset in x_range:
            for y_offset in y_range:
                for z_offset in z_range:
                    self.system_translational_vector_list[index] = (
                        np.dot(np.multiply(
                                np.array([x_offset, y_offset, z_offset]),
                                system_size), self.material.lattice_matrix))
                    index += 1

    def generate_system_element_index(self, system_size, quantum_indices):
        """Returns the system_element_index of the element
        :param system_size:
        :param quantum_indices:
        :return:
        """
        # assert type(system_size) is np.ndarray, \
        #     'Please input system_size as a numpy array'
        # assert type(quantum_indices) is np.ndarray, \
        #     'Please input quantum_indices as a numpy array'
        # assert np.all(system_size > 0), \
        #     'System size should be positive in all dimensions'
        # assert all(quantum_index >= 0
        #            for quantum_index in quantum_indices), \
        #            'Quantum Indices cannot be negative'
        # assert quantum_indices[-1] < (
        #     self.material.n_elements_per_unit_cell[
        #         quantum_indices[-2]]), \
        #         'Element Index exceed number of \
        #         elements of the specified element type'
        # assert np.all(
        #             quantum_indices[:3] < system_size), \
        #             'Unit cell indices exceed the given system size'
        unit_cell_index = np.copy(quantum_indices[:3])
        [element_type_index, element_index] = quantum_indices[-2:]
        system_element_index = (element_index
                                + self.material.n_elements_per_unit_cell[
                                                :element_type_index].sum())
        n_dim = len(system_size)
        for index in range(n_dim):
            if index == 0:
                system_element_index += (
                                    self.material.total_elements_per_unit_cell
                                    * unit_cell_index[n_dim-1-index])
            else:
                system_element_index += (
                                    self.material.total_elements_per_unit_cell
                                    * unit_cell_index[n_dim-1-index]
                                    * system_size[-index:].prod())
        return system_element_index

    def generate_quantum_indices(self, system_size, system_element_index):
        """Returns the quantum indices of the element
        :param system_size:
        :param system_element_index:
        :return:
        """
        # assert system_element_index >= 0, \
        #     'System Element Index cannot be negative'
        # assert system_element_index < (
        #             system_size.prod()
        #             * self.material.total_elements_per_unit_cell), \
        # 'System Element Index out of range for the given system size'
        quantum_indices = np.zeros(5, dtype=int)  # [0] * 5
        unit_cell_element_index = (
                                system_element_index
                                % self.material.total_elements_per_unit_cell)
        quantum_indices[3] = np.where(
                                self.material.n_elements_per_unit_cell.cumsum()
                                >= (unit_cell_element_index + 1))[0][0]
        quantum_indices[4] = (unit_cell_element_index
                              - self.material.n_elements_per_unit_cell[
                                            :quantum_indices[3]].sum())
        n_filled_unit_cells = ((system_element_index - unit_cell_element_index)
                               / self.material.total_elements_per_unit_cell)
        for index in range(3):
            quantum_indices[index] = (n_filled_unit_cells
                                      / system_size[index+1:].prod())
            n_filled_unit_cells -= (quantum_indices[index]
                                    * system_size[index+1:].prod())
        return quantum_indices

    def compute_coordinates(self, system_size, system_element_index):
        """Returns the coordinates in atomic units of the given
            system element index for a given system size
            :param system_size:
            :param system_element_index:
            :return: """
        quantum_indices = self.generate_quantum_indices(system_size,
                                                        system_element_index)
        unit_cell_translation_vector = np.dot(quantum_indices[:3],
                                              self.material.lattice_matrix)
        coordinates = (unit_cell_translation_vector
                       + self.material.cartesian_unit_cell_coords[
                                   quantum_indices[4]
                                   + self.material.n_elements_per_unit_cell[
                                       :quantum_indices[3]].sum()])
        return coordinates

    def compute_distance(self, system_size, system_element_index_1,
                         system_element_index_2):
        """Returns the distance in atomic units between the two system element
            indices for a given system size
            :param system_size:
            :param system_element_index_1:
            :param system_element_index_2:
            :return: """
        center_coord = self.compute_coordinates(system_size,
                                                system_element_index_1)
        neighbor_coord = self.compute_coordinates(system_size,
                                                  system_element_index_2)

        neighbor_image_coords = (self.system_translational_vector_list
                                 + neighbor_coord)
        neighbor_image_displacement_vectors = (neighbor_image_coords
                                               - center_coord)
        neighbor_image_displacements = np.linalg.norm(
                                neighbor_image_displacement_vectors, axis=1)
        displacement = np.min(neighbor_image_displacements)
        return displacement

    def hop_neighbor_sites(self, bulk_sites, center_site_indices,
                           neighbor_site_indices, cutoff_dist_limits,
                           cutoff_dist_key):
        """Returns system_element_index_map and distances between center sites
            and its neighbor sites within cutoff distance
            :param bulk_sites:
            :param center_site_indices:
            :param neighbor_site_indices:
            :param cutoff_dist_limits:
            :param cutoff_dist_key:
            :return: """
        neighbor_site_coords = bulk_sites.cell_coordinates[
                                                        neighbor_site_indices]
        neighbor_site_system_element_index_list = (
                                        bulk_sites.system_element_index_list[
                                                        neighbor_site_indices])
        center_site_coords = bulk_sites.cell_coordinates[center_site_indices]

        neighbor_system_element_indices = np.empty(len(center_site_coords),
                                                   dtype=object)
        displacement_vector_list = np.empty(len(center_site_coords),
                                            dtype=object)
        num_neighbors = np.array([], dtype=int)

        if cutoff_dist_key == 'Fe:Fe':
            quick_test = 0  # commit reference: 1472bb4
        else:
            quick_test = 0

        for center_site_index, center_coord in enumerate(center_site_coords):
            i_neighbor_site_index_list = []
            i_displacement_vectors = []
            i_num_neighbors = 0
            if quick_test:
                displacement_list = np.zeros(len(neighbor_site_coords))
            for neighbor_site_index, neighbor_coord in enumerate(
                                                        neighbor_site_coords):
                neighbor_image_coords = (self.system_translational_vector_list
                                         + neighbor_coord)
                neighbor_image_displacement_vectors = (neighbor_image_coords
                                                       - center_coord)
                neighbor_image_displacements = np.linalg.norm(
                                        neighbor_image_displacement_vectors,
                                        axis=1)
                [displacement, image_index] = [
                                    np.min(neighbor_image_displacements),
                                    np.argmin(neighbor_image_displacements)]
                if quick_test:
                    displacement_list[neighbor_site_index] = displacement
                if (cutoff_dist_limits[0] < displacement
                        <= cutoff_dist_limits[1]):
                    i_neighbor_site_index_list.append(neighbor_site_index)
                    i_displacement_vectors.append(
                            neighbor_image_displacement_vectors[image_index])
                    i_num_neighbors += 1
            neighbor_system_element_indices[center_site_index] = (
                                    neighbor_site_system_element_index_list[
                                                i_neighbor_site_index_list])
            displacement_vector_list[center_site_index] = np.asarray(
                                                        i_displacement_vectors)
            num_neighbors = np.append(num_neighbors, i_num_neighbors)
            if quick_test == 1:
                print(np.sort(displacement_list)[:10] / constants.ANG2BOHR)
                pdb.set_trace()
            elif quick_test == 2:
                for cutoff_dist in range(2, 7):
                    cutoff = cutoff_dist * constants.ANG2BOHR
                    print(cutoff_dist)
                    print(displacement_list[displacement_list < cutoff].shape)
                    print(np.unique(np.sort(np.round(
                                displacement_list[displacement_list < cutoff]
                                / constants.ANG2BOHR, 4))).shape)
                    print(np.unique(np.sort(np.round(
                                displacement_list[displacement_list < cutoff]
                                / constants.ANG2BOHR, 3))).shape)
                    print(np.unique(np.sort(np.round(
                                displacement_list[displacement_list < cutoff]
                                / constants.ANG2BOHR, 2))).shape)
                    print(np.unique(np.sort(np.round(
                                displacement_list[displacement_list < cutoff]
                                / constants.ANG2BOHR, 1))).shape)
                    print(np.unique(np.sort(np.round(
                                displacement_list[displacement_list < cutoff]
                                / constants.ANG2BOHR, 0))).shape)
                pdb.set_trace()

        return_neighbors = ReturnValues(
            neighbor_system_element_indices=neighbor_system_element_indices,
            displacement_vector_list=displacement_vector_list,
            num_neighbors=num_neighbors)
        return return_neighbors

    def generate_cumulative_displacement_list(self, dst_path):
        """Returns cumulative displacement list for the given system size
            printed out to disk
            :param dst_path:
            :return: """
        cumulative_displacement_list = np.zeros((self.num_system_elements,
                                                 self.num_system_elements, 3))
        for center_site_index, center_coord in enumerate(
                                            self.bulk_sites.cell_coordinates):
            cumulative_system_translational_vector_list = np.tile(
                                        self.system_translational_vector_list,
                                        (self.num_system_elements, 1, 1))
            cumulative_neighbor_image_coords = (
                cumulative_system_translational_vector_list
                + np.tile(self.bulk_sites.cell_coordinates[:, np.newaxis, :],
                          (1, len(self.system_translational_vector_list), 1)))
            cumulative_neighbor_image_displacement_vectors = (
                            cumulative_neighbor_image_coords - center_coord)
            cumulative_neighbor_image_displacements = np.linalg.norm(
                                cumulative_neighbor_image_displacement_vectors,
                                axis=2)
            cumulative_displacement_list[center_site_index] = \
                cumulative_neighbor_image_displacement_vectors[
                    np.arange(self.num_system_elements),
                    np.argmin(cumulative_neighbor_image_displacements, axis=1)]
        cumulative_displacement_list_file_path = dst_path.joinpath(
                                            'cumulative_displacement_list.npy')
        np.save(cumulative_displacement_list_file_path,
                cumulative_displacement_list)
        return None

    def generate_neighbor_list(self, dst_path, local_system_size):
        """Adds the neighbor list to the system object and returns the
            neighbor list
            :param dst_path:
            :param local_system_size:
            :return: """
        assert dst_path, \
            'Please provide the path to the parent directory of ' \
            'neighbor list files'
        assert all(size >= 3 for size in local_system_size), \
            'Local system size in all dimensions should always be ' \
            'greater than or equal to 3'

        Path.mkdir(dst_path, parents=True, exist_ok=True)
        hop_neighbor_list_file_path = dst_path.joinpath(
                                                    'hop_neighbor_list.npy')

        hop_neighbor_list = {}
        tol_dist = self.material.neighbor_cutoff_dist_tol
        element_types = self.material.element_types[:]

        for cutoff_dist_key in self.material.neighbor_cutoff_dist:
            cutoff_dist_list = self.material.neighbor_cutoff_dist[
                                                            cutoff_dist_key][:]
            neighbor_list_cutoff_dist_key = []
            [center_element_type, _] = cutoff_dist_key.split(
                                        self.material.element_type_delimiter)
            center_site_element_type_index = element_types.index(
                                                        center_element_type)
            local_bulk_sites = self.material.generate_sites(
                                                    self.element_type_indices,
                                                    self.system_size)
            system_element_index_offset_array = np.repeat(
                        np.arange(0,
                                  (self.material.total_elements_per_unit_cell
                                   * self.num_cells),
                                  self.material.total_elements_per_unit_cell),
                        self.material.n_elements_per_unit_cell[
                                            center_site_element_type_index])
            center_site_indices = neighbor_site_indices = (
                        np.tile((self.material.n_elements_per_unit_cell[
                                        :center_site_element_type_index].sum()
                                 + np.arange(
                                     0,
                                     self.material.n_elements_per_unit_cell[
                                            center_site_element_type_index])),
                                self.num_cells)
                        + system_element_index_offset_array)

            for class_index, class_cutoff_dist_list in enumerate(cutoff_dist_list):
                class_neighbor_list_cutoff_dist_key = []
                for index, cutoff_dist in enumerate(class_cutoff_dist_list):
                    cutoff_dist_limits = (
                        [(cutoff_dist - tol_dist[cutoff_dist_key][class_index][index]),
                         (cutoff_dist + tol_dist[cutoff_dist_key][class_index][index])])

                    class_neighbor_list_cutoff_dist_key.append(
                        self.hop_neighbor_sites(local_bulk_sites,
                                                center_site_indices,
                                                neighbor_site_indices,
                                                cutoff_dist_limits,
                                                cutoff_dist_key))
                neighbor_list_cutoff_dist_key.append(
                                            class_neighbor_list_cutoff_dist_key[:])
            hop_neighbor_list[cutoff_dist_key] = (
                [class_neighbor_list_cutoff_dist_key[:]
                 for class_neighbor_list_cutoff_dist_key in neighbor_list_cutoff_dist_key])
        np.save(hop_neighbor_list_file_path, hop_neighbor_list)

        file_name = 'neighbor_list'
        generate_report(self.start_time, dst_path, file_name)
        return None


class System(object):
    """defines the system we are working on

    Attributes:
    size: An array (3 x 1) defining the system size in multiple of
    unit cells
    """
    def __init__(self, material_info, material_neighbors,
                 hop_neighbor_list, cumulative_displacement_list, alpha, n_max,
                 k_max):
        """Return a system object whose size is *size*
        :param material_info:
        :param material_neighbors:
        :param hop_neighbor_list:
        :param cumulative_displacement_list:
        :param species_count:
        :param alpha:
        :param n_max:
        :param k_max:
        """
        self.start_time = datetime.now()

        self.material = material_info
        self.neighbors = material_neighbors
        self.hop_neighbor_list = hop_neighbor_list

        self.pbc = self.neighbors.pbc

        # total number of unit cells
        self.system_size = self.neighbors.system_size
        self.num_cells = self.system_size.prod()

        self.cumulative_displacement_list = cumulative_displacement_list

        # variables for ewald sum
        self.translational_matrix = np.multiply(
                        self.system_size, self.material.lattice_matrix)
        self.system_volume = abs(
                        np.dot(self.translational_matrix[0],
                               np.cross(self.translational_matrix[1],
                                        self.translational_matrix[2])))
        self.reciprocal_lattice_matrix = (
                2 * np.pi / self.system_volume
                * np.array([np.cross(self.translational_matrix[1],
                                     self.translational_matrix[2]),
                            np.cross(self.translational_matrix[2],
                                     self.translational_matrix[0]),
                            np.cross(self.translational_matrix[0],
                                     self.translational_matrix[1])]))
        self.translational_vector_length = np.linalg.norm(
                                    self.translational_matrix, axis=1)
        self.reciprocal_lattice_vector_length = np.linalg.norm(
                                self.reciprocal_lattice_matrix, axis=1)

        # class list
        self.system_class_index_list = (
                np.tile(self.material.unit_cell_class_list, self.num_cells))

        # species-wise number of nearest neighbors
        self.num_neighbors = np.zeros(self.material.num_species_types, int)
        for species_type_index, species_type in enumerate(
                                                self.material.species_types):
            # NOTE: Used [0] at the end of the statement assuming all 
            # hop_element_types have equivalent number of nearest neighbors
            hop_element_type = self.material.hop_element_types[species_type][0]
            if hop_element_type in self.material.neighbor_cutoff_dist:
                # NOTE: Assuming number of neighbors is identical to all class indices
                class_index = 0
                num_hop_dist_types = len(self.material.neighbor_cutoff_dist[
                                                hop_element_type][class_index])
            else:
                num_hop_dist_types = 0
            for hop_dist_type in range(num_hop_dist_types):
                # NOTE: Assuming number of neighbors is identical to all class indices
                class_index = 0
                self.num_neighbors[species_type_index] += (
                        self.hop_neighbor_list[hop_element_type][class_index][
                                                hop_dist_type].num_neighbors[0])

        # ewald parameters:
        self.alpha = alpha
        self.n_max = n_max
        self.k_max = k_max

    def ewald_sum_setup(self, dst_path):
        """

        :param dst_path:
        :return:
        """
        sqrt_alpha = np.sqrt(self.alpha)
        alpha4 = 4 * self.alpha
        fourier_sum_coeff = (2 * np.pi) / self.system_volume
        precomputed_array = np.zeros((self.neighbors.num_system_elements,
                                      self.neighbors.num_system_elements))

        for i in range(-self.n_max, self.n_max+1):
            for j in range(-self.n_max, self.n_max+1):
                for k in range(-self.n_max, self.n_max+1):
                    temp_array = np.linalg.norm(
                                        (self.cumulative_displacement_list
                                         + np.dot(np.array([i, j, k]),
                                                  self.translational_matrix)),
                                        axis=2)
                    precomputed_array += erfc(sqrt_alpha * temp_array) / 2

                    if np.all(np.array([i, j, k]) == 0):
                        for a in range(self.neighbors.num_system_elements):
                            for b in range(self.neighbors.num_system_elements):
                                if a != b:
                                    precomputed_array[a][b] /= temp_array[a][b]
                    else:
                        precomputed_array /= temp_array

        for i in range(-self.k_max, self.k_max+1):
            for j in range(-self.k_max, self.k_max+1):
                for k in range(-self.k_max, self.k_max+1):
                    if not np.all(np.array([i, j, k]) == 0):
                        k_vector = np.dot(np.array([i, j, k]),
                                          self.reciprocal_lattice_matrix)
                        k_vector_2 = np.dot(k_vector, k_vector)
                        precomputed_array += (
                                        fourier_sum_coeff
                                        * np.exp(-k_vector_2 / alpha4)
                                        * np.cos(np.tensordot(
                                            self.cumulative_displacement_list,
                                            k_vector, axes=([2], [0])))
                                        / k_vector_2)

        precomputed_array /= self.material.dielectric_constant

        file_name = 'precomputed_array'
        generate_report(self.start_time, dst_path, file_name)
        return precomputed_array


class Run(object):
    """defines the subroutines for running Kinetic Monte Carlo and
        computing electrostatic interaction energies"""
    def __init__(self, system, precomputed_array, temp, ion_charge_type,
                 species_charge_type, n_traj, t_final, time_interval,
                 species_count, relative_energies, external_field, doping):
        """Returns the PBC condition of the system
        :param system:
        :param precomputed_array:
        :param temp:
        :param ion_charge_type:
        :param species_charge_type:
        :param n_traj:
        :param t_final:
        :param time_interval:
        """
        self.start_time = datetime.now()

        self.system = system
        self.material = self.system.material
        self.neighbors = self.system.neighbors
        self.precomputed_array = precomputed_array
        self.temp = temp * constants.K2AUTEMP
        self.ion_charge_type = ion_charge_type
        self.species_charge_type = species_charge_type
        self.n_traj = int(n_traj)
        self.t_final = t_final * constants.SEC2AUTIME
        self.time_interval = time_interval * constants.SEC2AUTIME
        self.species_count = species_count
        self.relative_energies = relative_energies

        # relative energies
        unit_cell_relative_energies = np.zeros(self.material.total_elements_per_unit_cell)
        start_index = end_index = 0
        for element_index in range(self.material.num_element_types):
            end_index = start_index + self.material.n_elements_per_unit_cell[element_index]
            if self.material.num_classes[element_index] != 1:
                element_type = self.material.element_types[element_index]
                unit_cell_relative_energies[start_index:end_index] += [
                    relative_energies['class_index'][element_type][class_index] * constants.EV2HARTREE
                    for class_index in self.material.unit_cell_class_list[start_index:end_index]]
            start_index = end_index

        self.system_relative_energies = (
                np.tile(unit_cell_relative_energies, self.system.num_cells))

        self.num_shells_dopant = {}
        self.max_shells = 0
        for element_type, element_relative_energies in self.relative_energies['doping'].items():
            self.num_shells_dopant[element_type] = [
                len(dopant_element_relative_energies)
                for dopant_element_relative_energies in element_relative_energies]
            for dopant_element_relative_energies in element_relative_energies:
                self.max_shells = max(self.max_shells, len(dopant_element_relative_energies))  

        # electric field
        electric_field = external_field['electric']
        self.electric_field_ld = electric_field['ld']
        self.electric_field_mag = electric_field['mag']
        if electric_field['active']:
            self.electric_field_active = 1
            field_dir = np.asarray(electric_field['dir'])
            if self.electric_field_ld == 1:
                field_dir = np.dot(field_dir, self.material.lattice_matrix)
            self.electric_field = (self.electric_field_mag
                                   * (field_dir / np.linalg.norm(field_dir)))
        else:
            self.electric_field_active = 0
            self.electric_field = np.zeros(self.neighbors.n_dim)

        # doping
        self.doping = doping
        if np.any(doping['num_dopants']):
            self.doping_active = 1
            self.dopant_species_types = []
            self.dopant_element_types = []
            self.substitution_element_types = []
            self.dopant_to_substitution_element_type_map = {}
            for i_doping_element_map in self.doping['doping_element_map']:
                [substitution_element_type, dopant_element_type] = (
                    i_doping_element_map.split(self.material.element_type_delimiter))
                self.dopant_element_types.append(dopant_element_type)
                self.substitution_element_types.append(substitution_element_type)
                self.dopant_species_types.append(
                    self.material.element_type_to_species_map[substitution_element_type][0])
                self.dopant_to_substitution_element_type_map[dopant_element_type] = substitution_element_type
            self.num_dopant_element_types = len(self.dopant_element_types)
        else:
            self.doping_active = 0

        self.system_size = self.system.system_size

        # number of kinetic processes
        self.n_proc = np.dot(self.species_count, self.system.num_neighbors)

        # n_elements_per_unit_cell
        self.head_start_n_elements_per_unit_cell_cum_sum = [
                                self.material.n_elements_per_unit_cell[
                                            :site_element_type_index].sum()
                                for site_element_type_index in (
                                        self.neighbors.element_type_indices)]

        # species_type_list
        self.species_type_list = [self.material.species_types[index]
                                  for index, value in enumerate(self.species_count)
                                  for _ in range(value)]
        self.species_type_index_list = [index
                                        for index, value in enumerate(self.species_count)
                                        for _ in range(value)]
        self.species_charge_list = [
            self.material.species_charge_list[self.species_charge_type][index]
            for index in self.species_type_index_list]
        self.hop_element_type_list = [
                            self.material.hop_element_types[species_type][0]
                            for species_type in self.species_type_list]
        # NOTE: Assuming number of neighbors is identical to all class indices
        class_index = 0
        self.len_hop_dist_type_list = [
                    len(self.material.neighbor_cutoff_dist[hop_element_type][class_index])
                    for hop_element_type in self.hop_element_type_list]

        class_based_sample_site_indices = {}
        for species_type_index, species_type in enumerate(
                                                self.material.species_types):
            element_type = self.material.species_to_element_type_map[
                                                            species_type][0]
            element_type_index = self.material.element_types.index(
                                                                element_type)
            class_based_sample_site_indices[element_type] = []
            start_index = self.material.n_elements_per_unit_cell[
                                                    :element_type_index].sum()
            end_index = start_index + self.material.n_elements_per_unit_cell[
                                                            element_type_index]
            for class_index in range(self.material.num_classes[
                                                        element_type_index]):
                sample_site_index = self.material.unit_cell_class_list[
                                    start_index:end_index].index(class_index)
                class_based_sample_site_indices[element_type].append(
                                                            sample_site_index)

        self.n_proc_species_index_list = []
        # NOTE: doesn't work with doping.
        self.n_proc_hop_element_type_list = []
        # NOTE: doesn't work with doping.
        self.n_proc_site_element_type_index_list = []
        self.n_proc_hop_dist_type_list = {}
        self.n_proc_neighbor_index_list = {}
        self.n_proc_lambda_value_list = {}
        self.n_proc_v_ab_list = {}
        for species_type_index, species_type in enumerate(
                                                self.material.species_types):
            species_type_species_count = self.species_count[species_type_index]
            hop_element_type = self.material.hop_element_types[species_type][0]
            element_type = self.material.species_to_element_type_map[
                                                            species_type][0]
            element_type_index = self.material.element_types.index(
                                                                element_type)
            self.n_proc_species_index_list.extend(
                                np.repeat(range(species_type_species_count),
                                self.system.num_neighbors[species_type_index]))
            self.n_proc_hop_element_type_list.extend(
                            [hop_element_type] * species_type_species_count
                            * self.system.num_neighbors[species_type_index])
            self.n_proc_site_element_type_index_list.extend(
                            [element_type_index] * species_type_species_count
                            * self.system.num_neighbors[species_type_index])
            if species_type_species_count != 0:
                self.n_proc_hop_dist_type_list[hop_element_type] = []
                self.n_proc_neighbor_index_list[hop_element_type] = []
                self.n_proc_lambda_value_list[hop_element_type] = []
                self.n_proc_v_ab_list[hop_element_type] = []
                for class_index in range(self.material.num_classes[
                                                        element_type_index]):
                    sample_site_index = class_based_sample_site_indices[
                                                    element_type][class_index]
                    local_num_neighbors = []
                    len_local_num_neighbors = len(
                        self.material.neighbor_cutoff_dist[hop_element_type][class_index])
                    for hop_dist_type in range(len_local_num_neighbors):
                        local_num_neighbors.append(
                            self.system.hop_neighbor_list[hop_element_type][class_index][
                                hop_dist_type].num_neighbors[sample_site_index]
                            )
                    self.n_proc_hop_dist_type_list[hop_element_type].append(
                        np.repeat(range(len_local_num_neighbors),
                                  local_num_neighbors))
                    self.n_proc_neighbor_index_list[hop_element_type].append(
                                [index
                                 for value in np.unique(
                                     self.n_proc_hop_dist_type_list[
                                         hop_element_type][class_index],
                                     return_counts=True)[1]
                                 for index in range(value)])
                self.n_proc_lambda_value_list[hop_element_type].extend(
                    [[self.material.lambda_values[hop_element_type][
                                                class_index][hop_dist_type]
                     for hop_dist_type in self.n_proc_hop_dist_type_list[
                                        hop_element_type][class_index]]
                    for class_index in range(self.material.num_classes[element_type_index])])
                self.n_proc_v_ab_list[hop_element_type].extend(
                    [[self.material.v_ab[hop_element_type][class_index][hop_dist_type]
                     for hop_dist_type in self.n_proc_hop_dist_type_list[
                                        hop_element_type][class_index]]
                    for class_index in range(self.material.num_classes[element_type_index])])

        self.n_proc_species_proc_list = [
            species_proc_index
            for index in self.species_type_index_list
            for species_proc_index in range(self.system.num_neighbors[index])]

        # system coordinates
        self.system_coordinates = self.neighbors.bulk_sites.cell_coordinates

        # total number of species
        self.total_species = self.species_count.sum()

    def get_element_type_element_index(self, site_element_type_index,
                                       system_element_index):
        element_index = (
            system_element_index % self.material.total_elements_per_unit_cell
            - self.head_start_n_elements_per_unit_cell_cum_sum[site_element_type_index])
        element_type_element_index = (
            system_element_index // self.material.total_elements_per_unit_cell
            * self.material.n_elements_per_unit_cell[site_element_type_index]
            + element_index)
        return (element_type_element_index, element_index)

    def get_process_attributes(self, occupancy):
        i_proc = i_proc_old = 0
        old_site_system_element_index_list = np.zeros(self.n_proc, dtype=int)
        new_site_system_element_index_list = np.zeros(self.n_proc, dtype=int)
        element_type_element_index_list = np.zeros(self.n_proc, dtype=int)
        for species_site_system_element_index in occupancy:
            species_index = self.n_proc_species_index_list[i_proc]
            hop_element_type = self.n_proc_hop_element_type_list[i_proc]
            site_element_type_index = self.n_proc_site_element_type_index_list[i_proc]
            (element_type_element_index, element_index) = (
                                self.get_element_type_element_index(
                                    site_element_type_index,
                                    species_site_system_element_index))
            site_element_type_index = self.n_proc_site_element_type_index_list[
                                                                        i_proc]
            class_index = self.material.unit_cell_class_list[
                self.material.n_elements_per_unit_cell[:site_element_type_index].sum()
                + element_index]
            for hop_dist_type in range(self.len_hop_dist_type_list[
                                                            species_index]):
                local_neighbor_site_system_element_index_list = (
                            self.system.hop_neighbor_list[hop_element_type][
                                            class_index][hop_dist_type]
                            .neighbor_system_element_indices[
                                                element_type_element_index])
                num_neighbors = len(
                                local_neighbor_site_system_element_index_list)
                # TODO: Introduce If condition if neighbor_system_element_index
                # not in current_state_occupancy: commit 898baa8
                new_site_system_element_index_list[
                    i_proc:i_proc+num_neighbors] = \
                        local_neighbor_site_system_element_index_list
                i_proc += num_neighbors
            old_site_system_element_index_list[i_proc_old:i_proc] = \
                                            species_site_system_element_index
            element_type_element_index_list[i_proc_old:i_proc] = \
                                                    element_type_element_index
            i_proc_old = i_proc
        process_attributes = (old_site_system_element_index_list,
                              new_site_system_element_index_list,
                              element_type_element_index_list)
        return process_attributes

    def get_process_rates(self, process_attributes, charge_config):
        nproc_delg_0_array = np.zeros(self.n_proc)
        nproc_hop_vector_array = np.zeros((self.n_proc, self.neighbors.n_dim))
        k_list = np.zeros(self.n_proc)
        (old_site_system_element_index_list,
         new_site_system_element_index_list,
         element_type_element_index_list) = process_attributes

        for i_proc in range(self.n_proc):
            species_site_system_element_index = \
                                    old_site_system_element_index_list[i_proc]
            neighbor_site_system_element_index = \
                                    new_site_system_element_index_list[i_proc]
            species_index = self.n_proc_species_index_list[i_proc]
            species_proc_index = self.n_proc_species_proc_list[i_proc]
            term01 = 2 * np.dot(
                charge_config[:, 0],
                (self.precomputed_array[neighbor_site_system_element_index, :]
                 - self.precomputed_array[species_site_system_element_index, :]
                 ))
            term02 = (
                self.species_charge_list[species_index]
                * (self.precomputed_array[species_site_system_element_index,
                                          species_site_system_element_index]
                   + self.precomputed_array[neighbor_site_system_element_index,
                                            neighbor_site_system_element_index]
                   - 2 * self.precomputed_array[
                                       species_site_system_element_index,
                                       neighbor_site_system_element_index]))

            delg_0_ewald = (self.species_charge_list[species_index]
                            * (term01 + term02))
            class_index = self.system.system_class_index_list[
                                            species_site_system_element_index]
            hop_element_type = self.n_proc_hop_element_type_list[i_proc]
            hop_dist_type = self.n_proc_hop_dist_type_list[hop_element_type][
                                            class_index][species_proc_index]
            delg_0_shift = (
                self.system_relative_energies[neighbor_site_system_element_index]
                - self.system_relative_energies[species_site_system_element_index])
            delg_0 = (delg_0_ewald + delg_0_shift)

            nproc_delg_0_array[i_proc] = delg_0
            lambda_value = self.n_proc_lambda_value_list[hop_element_type][
                                            class_index][species_proc_index]
            v_ab = self.n_proc_v_ab_list[hop_element_type][class_index][
                                                            species_proc_index]
            element_type_element_index = element_type_element_index_list[
                                                                        i_proc]
            neighbor_index = self.n_proc_neighbor_index_list[hop_element_type][
                                            class_index][species_proc_index]
            hop_vector = (self.system.hop_neighbor_list[hop_element_type][
                            class_index][hop_dist_type].displacement_vector_list[
                                element_type_element_index][neighbor_index])
            nproc_hop_vector_array[i_proc] = hop_vector
            if self.electric_field_active:
                delg_s_shift = 0.5 * np.dot(self.electric_field, hop_vector)
            else:
                delg_s_shift = 0
            delg_s = (((lambda_value + delg_0) ** 2
                       / (4 * lambda_value)) - v_ab) - delg_s_shift
            k_list[i_proc] = self.material.vn * np.e**(-delg_s / self.temp)
        process_rate_info = (k_list, nproc_delg_0_array,
                             nproc_hop_vector_array)
        return process_rate_info

    def compute_drift_mobility(self, drift_velocity_array, dst_path,
                               prefix_list):
        drift_mobility_au = (np.dot(drift_velocity_array, self.electric_field)
                             / self.electric_field_mag**2)
        # mobility in cm2/V.s.
        drift_mobility_array = (drift_mobility_au * (constants.BOHR2CM**2
                                                     * constants.SEC2AUTIME
                                                     * constants.V2AUPOT))
        # write drift mobility data to file
        output_file_name = dst_path.joinpath('drift_mobility.dat')
        with open(output_file_name, 'wb') as output_file:
            np.savetxt(output_file, drift_mobility_array)
        # compute average drift mobiltiy and standard error of mean
        start_species_index = 0
        for species_index, num_species in enumerate(self.species_count):
            end_species_index = start_species_index + num_species
            if num_species != 0:
                species_type = self.material.species_types[species_index]
                species_drift_mobility = drift_mobility_array[
                                    :, start_species_index:end_species_index]
                species_avg_drift_mobility = np.mean(species_drift_mobility,
                                                     axis=1)
                mean_drift_mobility = np.mean(species_avg_drift_mobility)
                sem_drift_mobility = (np.std(species_avg_drift_mobility)
                                      / np.sqrt(self.n_traj))
                prefix_list.append(
                        'Estimated value of {:s} drift mobility is: '
                        '{:4.3e} cm2/V.s.\n'.format(species_type,
                                                    mean_drift_mobility))
                prefix_list.append(
                    'Standard error of mean in {:s} drift mobility is: '
                    '{:4.3e} cm2/V.s.\n'.format(species_type,
                                                sem_drift_mobility))
            start_species_index = end_species_index
        return prefix_list

    def get_doping_distribution(self):
        dopant_site_indices = {}
        for map_index in range(self.num_dopant_element_types):
            num_dopants = self.doping['num_dopants'][map_index]
            if num_dopants != 0:
                insertion_type = self.doping['insertion_type'][map_index]
                dopant_element_type = self.dopant_element_types[map_index]
                if insertion_type == 'manual':
                    dopant_site_indices[dopant_element_type] = (
                        self.doping['dopant_site_indices'][map_index][:num_dopants])
                elif insertion_type == 'random':
                    substitution_element_type = self.substitution_element_types[map_index]
                    substitution_element_type_index = self.material.element_types.index(
                                                                substitution_element_type)
                    system_element_index_offset_array = np.repeat(
                                np.arange(
                                    0, (self.material.total_elements_per_unit_cell
                                        * self.system.num_cells),
                                    self.material.total_elements_per_unit_cell),
                                self.material.n_elements_per_unit_cell[
                                                substitution_element_type_index])
                    site_indices = (
                        np.tile(self.material.n_elements_per_unit_cell[
                                    :substitution_element_type_index].sum()
                                + np.arange(0,
                                            self.material.n_elements_per_unit_cell[
                                                substitution_element_type_index]),
                                self.system.num_cells)
                        + system_element_index_offset_array).tolist()
                    dopant_site_indices[dopant_element_type] = rnd.sample(site_indices,
                                                                          num_dopants)[:]
        return dopant_site_indices

    def get_shell_based_neighbors(self, dopant_site_indices):
        shell_based_neighbors = {}
        for dopant_element_type, site_indices in dopant_site_indices.items():
            map_index = self.dopant_element_types.index(dopant_element_type)
            substitution_element_type = self.substitution_element_types[map_index]
            site_element_type_index = self.material.element_types.index(
                                                    substitution_element_type)
            hop_element_type = self.material.element_type_delimiter.join(
                                                [substitution_element_type] * 2)
            shell_based_neighbors[dopant_element_type] = []
            for site_index in site_indices:
                site_index_shell_neighbors = []
                for shell_index in range(self.num_shells_dopant[
                                        substitution_element_type][map_index]):
                    current_shell_elements = []
                    if shell_index == 0:
                        current_shell_elements.extend([site_index])
                        current_shell_neighbors = current_shell_elements
                    else:
                        inner_shell_neighbor_indices = site_index_shell_neighbors[shell_index - 1]
                        for system_element_index in inner_shell_neighbor_indices:
                            class_index = self.system.system_class_index_list[system_element_index]
                            (element_type_element_index, _) = (
                                self.get_element_type_element_index(
                                    site_element_type_index, system_element_index))
                            local_neighbor_site_system_element_index_list = []
                            for hop_dist_type_object in self.system.hop_neighbor_list[hop_element_type][class_index]:
                                local_neighbor_site_system_element_index_list.extend(
                                    hop_dist_type_object.neighbor_system_element_indices[
                                        element_type_element_index].tolist())
                            current_shell_elements.extend(
                                local_neighbor_site_system_element_index_list)
                        # avoid duplication of inner_shell_neighbor_indices or dopant_site_index
                        current_shell_neighbors = [
                            current_shell_element
                            for current_shell_element in current_shell_elements
                            if (current_shell_element not in inner_shell_neighbor_indices)
                            and (current_shell_element not in site_index_shell_neighbors[0])]
                    # avoid duplication within the shell
                    current_shell_neighbors = list(set(current_shell_neighbors))
                    site_index_shell_neighbors.append(current_shell_neighbors)
                shell_based_neighbors[dopant_element_type].append(site_index_shell_neighbors)
        return shell_based_neighbors

    def inspect_shell_overlap(self, shell_based_neighbors, allow_overlap, prefix_list):
        cumulative_neighbors = [
                system_element_index
                for _, dopant_shell_neighbors in shell_based_neighbors.items()
                for dopant_site_shell_neighbors in dopant_shell_neighbors
                for shell_index_neighbors in dopant_site_shell_neighbors
                for system_element_index in shell_index_neighbors]
        (unique_neighbors, counts) = np.unique(cumulative_neighbors, return_counts=True)
        num_unique_neighbors = len(unique_neighbors)
        if np.all(counts == 1):
            prefix_list.append(
                            'All shell based neighbor sites are independent\n')
        else:
            prefix_list.append(
                        'All shell based neighbor sites are NOT independent\n')

            overlap_sites = {}
            for index in range(num_unique_neighbors):
                if counts[index] != 1:
                    neighbor_site_index = unique_neighbors[index]
                    overlap_sites[neighbor_site_index] = []
                    prefix_list.append(
                        f'Site index {neighbor_site_index} belongs to shell ')
                    num_instances = 0
                    for dopant_element_type, dopant_shell_neighbors in shell_based_neighbors.items():
                        for dopant_site_index, dopant_site_shell_neighbors in enumerate(dopant_shell_neighbors):
                            for shell_index, shell_index_neighbors in enumerate(dopant_site_shell_neighbors):
                                if neighbor_site_index in shell_index_neighbors:
                                    overlap_sites[neighbor_site_index].append([dopant_element_type, dopant_site_index, shell_index])
                                    if num_instances != 0:
                                        prefix_list.append(', ')
                                    prefix_list.append(
                                        f'{shell_index} of {dopant_element_type}{dopant_site_index+1}')
                                    num_instances += 1
                    prefix_list.append('.\n')
            
            if not allow_overlap:
                for neighbor_site_index, neighbor_overlap_sites in overlap_sites.items():
                    min_shell_index = self.max_shells
                    for overlap_index, neighbor_overlap_site in enumerate(neighbor_overlap_sites):
                        shell_index = neighbor_overlap_site[2]
                        if min_shell_index > shell_index:
                            retain_overlap_index = overlap_index
                            min_shell_index = shell_index
                    for overlap_index, neighbor_overlap_site in enumerate(neighbor_overlap_sites):
                        if overlap_index != retain_overlap_index:
                            dopant_element_type = neighbor_overlap_site[0]
                            dopant_site_index = neighbor_overlap_site[1]
                            shell_index = neighbor_overlap_site[2]
                            shell_based_neighbors[dopant_element_type][dopant_site_index][shell_index].remove(neighbor_site_index)

                cumulative_neighbors = [
                        system_element_index
                        for _, dopant_shell_neighbors in shell_based_neighbors.items()
                        for dopant_site_shell_neighbors in dopant_shell_neighbors
                        for shell_index_neighbors in dopant_site_shell_neighbors
                        for system_element_index in shell_index_neighbors]
                (unique_neighbors, counts) = np.unique(cumulative_neighbors, return_counts=True)

                if sum(counts != 1) == 0:
                    prefix_list.append('SOLVED all overlap conflicts.\n')
                else:
                    prefix_list.append('WARNING: Not able to solve all overlap conflicts.\n')

        prefix_list.append('\n')
        return (shell_based_neighbors, prefix_list)

    def generate_initial_occupancy(self, dopant_site_indices,
                                   site_charge_initiation_active):
        """generates initial occupancy list based on species count
        :param species_count:
        :return:
        """
        occupancy = []
        for species_type_index, num_species in enumerate(self.species_count):
            species_type = self.material.species_types[species_type_index]
            if site_charge_initiation_active:
                for map_index, dopant_species_type in enumerate(
                                                    self.dopant_species_types):
                    num_dopant_sites = self.doping['num_dopants'][map_index]
                    if (self.doping['site_charge_initiation'][map_index] == 'yes'
                        and dopant_species_type == species_type
                        and num_dopant_sites and num_species):
                        dopant_element_type = self.dopant_element_types[map_index]
                        occupancy.extend(dopant_site_indices[dopant_element_type][:num_species])
                        num_species -= len(dopant_site_indices[dopant_element_type][:num_species])

            if num_species:
                species_site_element_list = (
                        self.material.species_to_element_type_map[species_type])
                species_site_element_type_index_list = [
                            self.material.element_types.index(species_site_element)
                            for species_site_element in species_site_element_list]
                species_site_indices = []
                for species_site_element_type_index in (
                                            species_site_element_type_index_list):
                    system_element_index_offset_array = np.repeat(
                                np.arange(
                                    0, (self.material.total_elements_per_unit_cell
                                        * self.system.num_cells),
                                    self.material.total_elements_per_unit_cell),
                                self.material.n_elements_per_unit_cell[
                                                species_site_element_type_index])
                    site_indices = (
                        np.tile(self.material.n_elements_per_unit_cell[
                                    :species_site_element_type_index].sum()
                                + np.arange(0,
                                            self.material.n_elements_per_unit_cell[
                                                species_site_element_type_index]),
                                self.system.num_cells)
                        + system_element_index_offset_array)
                    species_site_indices.extend(list(site_indices))
                    species_site_indices = [index
                                            for index in species_site_indices
                                            if index not in occupancy]
                occupancy.extend(rnd.sample(species_site_indices, num_species)[:])
        return occupancy

    def charge_config(self, occupancy, dopant_site_indices):
        """Returns charge distribution of the current configuration
        :param occupancy:
        :param ion_charge_type:
        :param species_charge_type:
        :return:
        """

        # generate lattice charge list
        unit_cell_charge_list = np.array(
            [self.material.charge_types[self.ion_charge_type][
                 self.material.element_types[element_type_index]]
             for element_type_index in self.material.element_type_index_list])
        charge_list = np.tile(unit_cell_charge_list, self.system.num_cells)[
                                                                :, np.newaxis]

        if self.doping_active:
            for dopant_element_type, site_indices in dopant_site_indices.items():
                dopant_site_charge = self.doping['charge'][
                                    self.ion_charge_type][dopant_element_type]
                charge_list[site_indices] = dopant_site_charge

        for species_type_index in range(self.material.num_species_types):
            start_index = 0 + self.species_count[:species_type_index].sum()
            end_index = start_index + self.species_count[species_type_index]
            center_site_system_element_indices = occupancy[
                                                    start_index:end_index][:]
            charge_list[center_site_system_element_indices] += (
                    self.material.species_charge_list[self.species_charge_type][
                                                        species_type_index])
        return charge_list

    def do_kmc_steps(self, dst_path, random_seed, output_data):
        """Subroutine to run the KMC simulation by specified number
        of steps
        :param dst_path:
        :param random_seed:
        :return: """
        assert dst_path, 'Please provide the destination path where \
                          simulation output files needs to be saved'

        rnd.seed(random_seed)
        num_path_steps_per_traj = int(self.t_final / self.time_interval) + 1
        # Initialize data arrays
        for output_data_type, output_attributes in output_data.items():
            if output_attributes['write']:
                output_file_name = dst_path.joinpath(output_attributes[
                                                                'file_name'])
                open(output_file_name, 'wb').close()
                if output_data_type == 'unwrapped_traj':
                    unwrapped_position_array = np.zeros(
                            (num_path_steps_per_traj, self.total_species * 3))
                elif output_data_type == 'wrapped_traj':
                    wrapped_position_array = np.zeros((num_path_steps_per_traj,
                                                       self.total_species * 3))
                elif output_data_type == 'energy':
                    energy_array = np.zeros(num_path_steps_per_traj)
                elif output_data_type == 'delg_0':
                    delg_0_array = np.zeros(num_path_steps_per_traj)
                elif output_data_type == 'potential':
                    potential_array = np.zeros((num_path_steps_per_traj,
                                                self.total_species))
        if self.electric_field_active:
            drift_velocity_array = np.zeros((self.n_traj,
                                             self.total_species, 3))

        prefix_list = []
        system_charge = np.dot(self.species_count,
                               self.material.species_charge_list[
                                            self.species_charge_type])
        ewald_neut = - (np.pi * (system_charge**2)
                        / (2 * self.system.system_volume * self.system.alpha))
        for traj_index in range(self.n_traj):
            if self.doping_active:
                dopant_site_indices = self.get_doping_distribution()
                site_charge_initiation_active = 1
                # update system_relative_energies
                allow_overlap = 0
                shell_based_neighbors = self.get_shell_based_neighbors(dopant_site_indices)
                prefix_list.append(f'Trajectory {traj_index+1}:\n')
                (shell_based_neighbors, prefix_list) = self.inspect_shell_overlap(
                                    shell_based_neighbors, allow_overlap, prefix_list)

                for dopant_element_type, dopant_shell_based_neighbors in shell_based_neighbors.items():
                    map_index = self.dopant_element_types.index(dopant_element_type)
                    for dopant_site_shell_based_neighbors in dopant_shell_based_neighbors:
                        for shell_index, i_shell_based_neighbors in enumerate(
                                            dopant_site_shell_based_neighbors):
                            substitution_element_type = self.substitution_element_types[map_index]
                            self.system_relative_energies[
                                i_shell_based_neighbors] += self.relative_energies[
                                    'doping'][substitution_element_type][
                                        map_index][shell_index] * constants.EV2HARTREE
            else:
                dopant_site_indices = {}
                site_charge_initiation_active = 0

            current_state_occupancy = self.generate_initial_occupancy(
                            dopant_site_indices, site_charge_initiation_active)
            current_state_charge_config = self.charge_config(
                                current_state_occupancy, dopant_site_indices)
            current_state_charge_config_prod = np.multiply(
                                    current_state_charge_config.transpose(),
                                    current_state_charge_config)
            ewald_self = - (
                        np.sqrt(self.system.alpha / np.pi)
                        * np.einsum('ii', current_state_charge_config_prod))
            # TODO: How helpful is recording precomputed_array?
            current_state_energy = (
                        ewald_neut + ewald_self
                        + np.sum(np.multiply(current_state_charge_config_prod,
                                             self.precomputed_array)))
            start_path_index = end_path_index = 1
            if output_data['energy']['write']:
                energy_array[0] = current_state_energy
            species_displacement_vector_list = np.zeros(
                                                (1, self.total_species * 3))
            sim_time = 0
            while end_path_index < num_path_steps_per_traj:
                process_attributes = self.get_process_attributes(
                                                    current_state_occupancy)
                new_site_system_element_index_list = process_attributes[1]
                process_rate_info = self.get_process_rates(
                            process_attributes, current_state_charge_config)
                (k_list, nproc_delg_0_array,
                 nproc_hop_vector_array) = process_rate_info

                k_total = sum(k_list)
                k_cum_sum = (k_list / k_total).cumsum()
                # Randomly choose a kinetic process
                rand1 = rnd.random()
                proc_index = np.where(k_cum_sum > rand1)[0][0]
                # Update simulation time
                rand2 = rnd.random()
                sim_time -= np.log(rand2) / k_total
                end_path_index = int(sim_time / self.time_interval)

                # Update data arrays at each kmc step
                if output_data['delg_0']['write']:
                    delg_0_array[start_path_index:end_path_index] = \
                        nproc_delg_0_array[proc_index]
                species_index = self.n_proc_species_index_list[proc_index]
                old_site_system_element_index = current_state_occupancy[
                                                                species_index]
                new_site_system_element_index = (
                                new_site_system_element_index_list[proc_index])
                current_state_occupancy[species_index] = \
                    new_site_system_element_index
                species_displacement_vector_list[
                    0, species_index * 3:(species_index + 1) * 3] += \
                        nproc_hop_vector_array[proc_index]
                if self.electric_field_active:
                    drift_velocity_array[traj_index, species_index, :] += (
                                            nproc_hop_vector_array[proc_index]
                                            * k_list[proc_index])
                current_state_energy += nproc_delg_0_array[proc_index]
                current_state_charge_config[old_site_system_element_index] -= \
                    self.species_charge_list[species_index]
                current_state_charge_config[new_site_system_element_index] += \
                    self.species_charge_list[species_index]
                # Update data arrays for each path step
                if end_path_index >= start_path_index + 1:
                    if end_path_index >= num_path_steps_per_traj:
                        end_path_index = num_path_steps_per_traj
                    unwrapped_position_array[start_path_index:end_path_index] \
                        = (unwrapped_position_array[start_path_index-1]
                           + species_displacement_vector_list)
                    if output_data['energy']['write']:
                        energy_array[start_path_index:end_path_index] = \
                            current_state_energy
                    species_displacement_vector_list = np.zeros(
                                                (1, self.total_species * 3))
                    start_path_index = end_path_index

            # Write output data arrays to disk
            for output_data_type, output_attributes in output_data.items():
                if output_attributes['write']:
                    output_file_name = dst_path.joinpath(output_attributes[
                                                                'file_name'])
                    with open(output_file_name, 'ab') as output_file:
                        if output_data_type == 'unwrapped_traj':
                            np.savetxt(output_file, unwrapped_position_array)
                        elif output_data_type == 'wrapped_traj':
                            np.savetxt(output_file, wrapped_position_array)
                        elif output_data_type == 'energy':
                            np.savetxt(output_file, energy_array)
                        elif output_data_type == 'delg_0':
                            np.savetxt(output_file, delg_0_array)
                        elif output_data_type == 'potential':
                            np.savetxt(output_file, potential_array)

        if self.electric_field_active:
            prefix_list = self.compute_drift_mobility(drift_velocity_array,
                                                      dst_path, prefix_list)

        file_name = 'Run'
        prefix = ''.join(prefix_list)
        generate_report(self.start_time, dst_path, file_name, prefix)
        return None


class Analysis(object):
    """Post-simulation analysis methods"""
    def __init__(self, material_info, n_dim, species_count, n_traj, t_final,
                 time_interval, msd_t_final, trim_length, repr_time='ns',
                 repr_dist='Angstrom'):
        """
        :param material_info:
        :param n_dim:
        :param species_count:
        :param n_traj:
        :param t_final:
        :param time_interval:
        :param msd_t_final:
        :param trim_length:
        :param repr_time:
        :param repr_dist:
        """
        self.start_time = datetime.now()

        self.material = material_info
        self.n_dim = n_dim
        self.species_count = species_count
        self.total_species = self.species_count.sum()
        self.n_traj = int(n_traj)
        self.t_final = t_final * constants.SEC2AUTIME
        self.time_interval = time_interval * constants.SEC2AUTIME
        self.trim_length = trim_length
        self.num_path_steps_per_traj = (int(self.t_final / self.time_interval)
                                        + 1)
        self.repr_time = repr_time
        self.repr_dist = repr_dist

        if repr_time == 'ns':
            self.time_conversion = constants.AUTIME2NS
        elif repr_time == 'ps':
            self.time_conversion = constants.AUTIME2PS
        elif repr_time == 'fs':
            self.time_conversion = constants.AUTIME2FS
        elif repr_time == 's':
            self.time_conversion = 1E+00 / constants.SEC2AUTIME

        if repr_dist == 'm':
            self.dist_conversion = constants.BOHR
        elif repr_dist == 'um':
            self.dist_conversion = constants.BOHR2UM
        elif repr_dist == 'angstrom':
            self.dist_conversion = 1E+00 / constants.ANG2BOHR

        self.msd_t_final = msd_t_final / self.time_conversion
        self.num_msd_steps_per_traj = int(self.msd_t_final
                                          / self.time_interval) + 1

    def compute_msd(self, dst_path):
        """Returns the squared displacement of the trajectories
        :param dst_path:
        :return:
        """
        assert dst_path, 'Please provide the destination path where MSD ' \
                         'output files needs to be saved'
        position_array = np.loadtxt(dst_path.joinpath('unwrapped_traj.dat'))
        num_traj_recorded = int(len(position_array)
                                / self.num_path_steps_per_traj)
        position_array = (
            position_array[
                :num_traj_recorded * self.num_path_steps_per_traj + 1].reshape(
                (num_traj_recorded * self.num_path_steps_per_traj,
                 self.total_species, 3))
            * self.dist_conversion)
        sd_array = np.zeros((num_traj_recorded, self.num_msd_steps_per_traj,
                             self.total_species))
        for traj_index in range(num_traj_recorded):
            head_start = traj_index * self.num_path_steps_per_traj
            for time_step in range(1, self.num_msd_steps_per_traj):
                num_disp = self.num_path_steps_per_traj - time_step
                add_on = np.arange(num_disp)
                pos_diff = (position_array[head_start + time_step + add_on]
                            - position_array[head_start + add_on])
                sd_array[traj_index, time_step, :] = np.mean(
                        np.einsum('ijk,ijk->ij', pos_diff, pos_diff),
                        axis=0)
        num_existent_species = (self.material.num_species_types
                                - list(self.species_count).count(0))
        species_avg_sd_array = np.zeros((num_traj_recorded,
                                         self.num_msd_steps_per_traj,
                                         num_existent_species))
        start_index = 0
        num_non_existent_species = 0
        non_existent_species_indices = []
        for species_type_index in range(self.material.num_species_types):
            if self.species_count[species_type_index] != 0:
                end_index = start_index + self.species_count[
                                                            species_type_index]
                species_avg_sd_array[
                    :, :, (species_type_index - num_non_existent_species)] = \
                    np.mean(sd_array[:, :, start_index:end_index], axis=2)
                start_index = end_index
            else:
                num_non_existent_species += 1
                non_existent_species_indices.append(species_type_index)

        msd_data = np.zeros((self.num_msd_steps_per_traj,
                             num_existent_species + 1))
        time_array = (np.arange(self.num_msd_steps_per_traj)
                      * self.time_interval
                      * self.time_conversion)
        msd_data[:, 0] = time_array
        msd_data[:, 1:] = np.mean(species_avg_sd_array, axis=0)
        sem_data = (np.std(species_avg_sd_array, axis=0)
                    / np.sqrt(num_traj_recorded))
        file_name = (('%1.2E' % (self.msd_t_final * self.time_conversion))
                     + str(self.repr_time)
                     + (',n_traj: %1.2E' % num_traj_recorded
                        if num_traj_recorded != self.n_traj else ''))
        msd_file_name = ''.join(['MSD_Data_', file_name, '.npy'])
        msd_file_path = dst_path.joinpath(msd_file_name)
        species_types = [species_type
                         for index, species_type in enumerate(
                                            self.material.species_types)
                         if index not in non_existent_species_indices]
        np.save(msd_file_path, msd_data)

        report_file_name = ''.join(['MSD_Analysis',
                             ('_' if file_name else ''), file_name])
        slope_data = np.zeros((num_traj_recorded, num_existent_species))
        prefix_list = []
        for species_index, species_type in enumerate(species_types):
            for traj_index in range(num_traj_recorded):
                slope_data[traj_index, species_index], _, _, _, _ = \
                    linregress(msd_data[self.trim_length:-self.trim_length, 0],
                               species_avg_sd_array[traj_index,
                                                    self.trim_length:-self.trim_length,
                                                    species_index])
            slope = np.mean(slope_data[:, species_index])
            species_diff = (slope * constants.ANG2UM ** 2
                            * constants.SEC2NS / (2 * self.n_dim))
            prefix_list.append(
                        'Estimated value of {:s} diffusivity is: '
                        '{:4.3f} um2/s\n'.format(species_type, species_diff))
            slope_sem = (np.std(slope_data[:, species_index])
                         / np.sqrt(num_traj_recorded))
            species_diff_sem = (slope_sem * constants.ANG2UM ** 2
                                * constants.SEC2NS / (2 * self.n_dim))
            prefix_list.append(
                'Standard error of mean in {:s} diffusivity is: '
                '{:4.3f} um2/s\n'.format(species_type, species_diff_sem))
        prefix = ''.join(prefix_list)
        generate_report(self.start_time, dst_path, report_file_name, prefix)

        return_msd_data = ReturnValues(msd_data=msd_data,
                                       sem_data=sem_data,
                                       species_types=species_types,
                                       file_name=file_name)
        return return_msd_data

    def generate_msd_plot(self, msd_data, sem_data, display_error_bars,
                          species_types, file_name, dst_path):
        """Returns a line plot of the MSD data
        :param msd_data:
        :param std_data:
        :param display_error_bars:
        :param species_types:
        :param file_name:
        :param dst_path:
        :return:
        """
        assert dst_path, 'Please provide the destination path where MSD Plot ' \
                        'files needs to be saved'
        plt.switch_backend('Agg')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for species_index, species_type in enumerate(species_types):
            ax.plot(msd_data[:, 0], msd_data[:, species_index + 1], 'o',
                    markerfacecolor='blue', markeredgecolor='black',
                    label=species_type)
            if display_error_bars:
                ax.errorbar(msd_data[:, 0], msd_data[:, species_index + 1],
                            yerr=sem_data[:, species_index], fmt='o',
                            capsize=3, color='blue', markerfacecolor='none',
                            markeredgecolor='none')
            slope, intercept, r_value, _, _ = \
                linregress(msd_data[self.trim_length:-self.trim_length, 0],
                           msd_data[self.trim_length:-self.trim_length,
                           species_index + 1])
            species_diff = (slope * constants.ANG2UM**2
                            * constants.SEC2NS / (2 * self.n_dim))
            ax.add_artist(
                AnchoredText('Est. $D_{{%s}}$ = %4.3f' % (species_type,
                                                          species_diff)
                             + '  ${{\mu}}m^2/s$; $r^2$=%4.3e' % (r_value**2),
                             loc=4))
            ax.plot(msd_data[self.trim_length:-self.trim_length, 0],
                    intercept + slope * msd_data[
                                        self.trim_length:-self.trim_length, 0],
                    'r', label=species_type+'-fitted')
        ax.set_xlabel(''.join(['Time (', self.repr_time, ')']))
        ax.set_ylabel('MSD (' + ('$\AA^2$' if self.repr_dist == 'angstrom'
                                 else (self.repr_dist + '^2')) + ')')
        figure_title = 'MSD_' + file_name
        ax.set_title('\n'.join(wrap(figure_title, 60)))
        plt.legend()
        plt.show()  # temp change
        figure_name = ''.join(['MSD_Plot_', file_name + '_trim='
                               + str(self.trim_length) + '.png'])
        figure_path = dst_path.joinpath(figure_name)
        plt.savefig(str(figure_path))
        return None


class ReturnValues(object):
    """dummy class to return objects from methods defined inside
        other classes"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
