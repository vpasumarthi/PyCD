# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
from scipy.special import erfc, binom
from scipy.stats import linregress
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from textwrap import wrap
import pickle

from PyCD.io import read_poscar, generate_report
from PyCD import constants

plt.switch_backend('Agg')


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

    def get_system_element_index(self, system_size, quantum_indices):
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

    def get_quantum_indices(self, system_size, system_element_index):
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

    def get_coordinates(self, system_size, system_element_index):
        """Returns the coordinates in atomic units of the given
            system element index for a given system size
            :param system_size:
            :param system_element_index:
            :return: """
        quantum_indices = self.get_quantum_indices(system_size,
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
        center_coord = self.get_coordinates(system_size,
                                            system_element_index_1)
        neighbor_coord = self.get_coordinates(system_size,
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

    def get_pairwise_min_image_vector_data(self, dst_path):
        """Returns cumulative displacement list for the given system size
            printed out to disk
            :param dst_path:
            :return: """
        pairwise_min_image_vector_data = np.zeros((self.num_system_elements,
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
            pairwise_min_image_vector_data[center_site_index] = \
                cumulative_neighbor_image_displacement_vectors[
                    np.arange(self.num_system_elements),
                    np.argmin(cumulative_neighbor_image_displacements, axis=1)]
        pairwise_min_image_vector_data_file_path = dst_path.joinpath(
                                            'pairwise_min_image_vector_data.npy')
        np.save(pairwise_min_image_vector_data_file_path,
                pairwise_min_image_vector_data)
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
        print_time_elapsed = 1
        generate_report(self.start_time, dst_path, file_name, print_time_elapsed)
        return None


class System(object):
    """defines the system we are working on

    Attributes:
    size: An array (3 x 1) defining the system size in multiple of
    unit cells
    """
    def __init__(self, material_info, material_neighbors,
                 hop_neighbor_list, pairwise_min_image_vector_data, alpha, r_cut,
                 k_cut, precision_parameters, step_system_size_array, step_hop_neighbor_master_list):
        """Return a system object whose size is *size*
        :param material_info:
        :param material_neighbors:
        :param hop_neighbor_list:
        :param pairwise_min_image_vector_data:
        :param species_count:
        :param alpha:
        :param n_max:
        :param k_max:
        """
        self.start_time = datetime.now()

        self.material = material_info
        self.neighbors = material_neighbors
        self.hop_neighbor_list = hop_neighbor_list
        self.step_system_size_array = step_system_size_array
        self.step_hop_neighbor_master_list = step_hop_neighbor_master_list
        self.num_unique_step_systems = len(self.step_hop_neighbor_master_list)

        self.pbc = self.neighbors.pbc

        # total number of unit cells
        self.system_size = self.neighbors.system_size
        self.num_cells = self.system_size.prod()

        self.pairwise_min_image_vector_data = pairwise_min_image_vector_data

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

        # step system class list
        self.step_system_class_index_master_list = []
        if self.num_unique_step_systems == 1:
            step_system_size = self.step_system_size_array
            num_cells = step_system_size.prod()
            self.step_system_class_index_master_list.append(
                        np.tile(self.material.unit_cell_class_list, num_cells))
        else:
            for unique_step_system_index in range(self.num_unique_step_systems):
                step_system_size = self.step_system_size_array[unique_step_system_index, :]
                num_cells = step_system_size.prod()
                self.step_system_class_index_master_list.append(
                            np.tile(self.material.unit_cell_class_list, num_cells))

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
        if np.isreal(alpha):
            self.alpha = alpha / constants.ANG2BOHR
        else:
            self.alpha = alpha

        if np.isreal(r_cut):
            self.r_cut = r_cut * constants.ANG2BOHR
        else:
            self.r_cut = r_cut

        # k_cut: cutoff radius in k-space
        if isinstance(k_cut, list):
            self.k_cut = k_cut
        elif np.isreal(k_cut):
            self.k_cut = k_cut / constants.ANG2BOHR
        else:
            self.k_cut = k_cut

        self.lower_bound_real = precision_parameters['lower_bound_real']
        self.lower_bound_rcut = precision_parameters['lower_bound_rcut']
        self.upper_bound_rcut = precision_parameters['upper_bound_rcut']
        self.lower_bound_kcut = precision_parameters['lower_bound_kcut']
        self.upper_bound_kcut = precision_parameters['upper_bound_kcut']
        self.threshold_fraction = precision_parameters['threshold_fraction']
        self.num_data_points_low = int(precision_parameters['num_data_points_low'])
        self.num_data_points_high = int(precision_parameters['num_data_points_high'])
        self.precise_r_cut = precision_parameters['precise_r_cut']
        self.err_tol = precision_parameters['err_tol'] * constants.EV2HARTREE
        self.step_increase_tol = precision_parameters['step_increase_tol'] * constants.EV2HARTREE
        self.step_change_data_points = precision_parameters['step_change_data_points']

    def pot_r_ewald(self, alpha, r_cut):
        """Generates precomputed array with potential energy contributions from
           real-space confined to simulation cell i.e. n_max=[0, 0, 0]"""
        precomputed_array = np.zeros((self.neighbors.num_system_elements,
                                      self.neighbors.num_system_elements))

        sqrt_alpha = np.sqrt(alpha)
        dr_translated = np.linalg.norm((self.pairwise_min_image_vector_data), axis=2)
        cutoff_neighbor_pairs = dr_translated < r_cut
        precomputed_array[cutoff_neighbor_pairs] += erfc(sqrt_alpha * dr_translated[cutoff_neighbor_pairs]) / 2

        # avoid division for diagonal elements for original simulation cell
        num_neighbor_pairs = cutoff_neighbor_pairs.sum()
        np.fill_diagonal(dr_translated, 1)
        precomputed_array[cutoff_neighbor_pairs] /= dr_translated[cutoff_neighbor_pairs]
        return (precomputed_array, num_neighbor_pairs)

    def get_effective_k_vectors(self, k_max):
        k_vector_list = []
        exclude_list = []
        for i in range(-k_max[0], k_max[0]+1):
            for j in range(-k_max[1], k_max[1]+1):
                for k in range(-k_max[2], k_max[2]+1):
                    if [i, j, k] not in exclude_list:
                        k_vector_list.append([i, j, k])
                        exclude_list.append([-i, -j, -k])
        k_vector_list.remove([0, 0, 0])
        k_vector_data = np.asarray(k_vector_list)
        return k_vector_data

    def get_cosine_data(self, k_max):
        max_k_max = max(k_max)
        unit_k_vector = np.dot(np.ones(self.neighbors.n_dim),
                               self.reciprocal_lattice_matrix)
        unit_cosine_data = np.cos(np.tensordot(self.pairwise_min_image_vector_data, unit_k_vector, axes=([2], [0])))
        unit_sine_data = np.sin(np.tensordot(self.pairwise_min_image_vector_data, unit_k_vector, axes=([2], [0])))
        cosine_data_shape = (max_k_max, unit_cosine_data.shape[0], unit_cosine_data.shape[1])
        cosine_data = np.zeros(cosine_data_shape)
        for n_index in range(1, max_k_max+1):
            for k_index in range(0, n_index, 2):
                cosine_data[n_index] += (-1)**(k_index / 2) * binom(n_index, k_index) * unit_cosine_data**(n_index - k_index) * unit_sine_data**k_index
        return cosine_data

    def pot_k_ewald(self, k_max, alpha, k_cut):
        """Updates precomputed array with potential energy contributions from
           reciprocal-space"""
        precomputed_array = np.zeros((self.neighbors.num_system_elements,
                                      self.neighbors.num_system_elements))

        alpha4 = 4 * alpha
        fourier_sum_coeff = (2 * np.pi) / self.system_volume
        k_cut_2 = k_cut**2

        k_vector_data = self.get_effective_k_vectors(k_max)
        for k_vector_value in k_vector_data:
            k_vector = np.dot(k_vector_value,
                              self.reciprocal_lattice_matrix)
            k_vector_2 = np.dot(k_vector, k_vector)
            if k_vector_2 < k_cut_2:
                precomputed_array += (
                                fourier_sum_coeff
                                * np.exp(-k_vector_2 / alpha4)
                                * np.cos(np.tensordot(
                                    self.pairwise_min_image_vector_data,
                                    k_vector, axes=([2], [0])))
                                / k_vector_2)
        # effective k_vectors only include half of all possible k_vectors
        precomputed_array *= 2
        return precomputed_array

    def pot_k_ewald_with_k_vector_data(self, charge_list_prod, k_max, alpha, k_cut):
        """Updates precomputed array with potential energy contributions from
           reciprocal-space"""
        precomputed_array = np.zeros((self.neighbors.num_system_elements,
                                      self.neighbors.num_system_elements))

        alpha4 = 4 * alpha
        fourier_sum_coeff = (2 * np.pi) / self.system_volume
        k_cut_2 = k_cut**2

        k_vector_data = self.get_effective_k_vectors(k_max)
        num_vectors = len(k_vector_data)
        energy_contribution_data = np.zeros(num_vectors)
        for k_vector_index, k_vector_value in enumerate(k_vector_data):
            k_vector = np.dot(np.asarray(k_vector_value),
                              self.reciprocal_lattice_matrix)
            k_vector_2 = np.dot(k_vector, k_vector)
            if k_vector_2 < k_cut_2:
                k_vector_precomputed_array = (
                                fourier_sum_coeff
                                * np.exp(-k_vector_2 / alpha4)
                                * np.cos(np.tensordot(
                                    self.pairwise_min_image_vector_data,
                                    k_vector, axes=([2], [0])))
                                / k_vector_2)
                energy_contribution_data[k_vector_index] = np.sum(np.multiply(charge_list_prod, k_vector_precomputed_array))
                precomputed_array += k_vector_precomputed_array
            else:
                energy_contribution_data[k_vector_index] = 0
        # effective k_vectors only include half of all possible k_vectors
        precomputed_array *= 2
        return (precomputed_array, k_vector_data, energy_contribution_data)

    def benchmark_ewald(self, num_repeats, benchmark_parameters):
        k_max = benchmark_parameters['k_max']
        alpha = benchmark_parameters['alpha']
        r_cut = benchmark_parameters['r_cut']
        k_cut = benchmark_parameters['k_cut']

        start_time_r = datetime.now()
        for _ in range(num_repeats):
            self.pot_r_ewald(alpha, r_cut)
        end_time_r = datetime.now()
        time_elapsed_r = end_time_r - start_time_r
        time_elapsed_r_seconds = time_elapsed_r.total_seconds()
        num_neighbor_pairs = self.pot_r_ewald(alpha, r_cut)[1]
        tau_r = time_elapsed_r_seconds / num_repeats / num_neighbor_pairs

        start_time_f = datetime.now()
        for _ in range(num_repeats):
            self.pot_k_ewald(k_max, alpha, k_cut)
        end_time_f = datetime.now()
        time_elapsed_f = end_time_f - start_time_f
        time_elapsed_f_seconds = time_elapsed_f.total_seconds()
        num_k_vectors = np.ceil(np.prod(2 * k_max + 1) * np.pi / 6 - 1).astype(int)
        tau_f = time_elapsed_f_seconds / num_repeats / self.neighbors.num_system_elements**2 / num_k_vectors

        tau_ratio = tau_r / tau_f
        time_ratio = time_elapsed_r_seconds / time_elapsed_f_seconds
        return (tau_ratio, time_ratio)

    def base_charge_config_for_accuracy_analysis(self, ion_charge_type):
        # generate lattice charge list
        unit_cell_charge_list = np.array(
            [self.material.charge_types[ion_charge_type][
                 self.material.element_types[element_type_index]]
             for element_type_index in self.material.element_type_index_list])
        charge_list = np.tile(unit_cell_charge_list, self.num_cells)[:, np.newaxis]
        return charge_list

    def minimize_real_space_cutoff_error(self, charge_list_einsum, real_space_parameters, x_real_initial_guess):
        if 'alpha' in real_space_parameters:
            alpha = real_space_parameters['alpha']
            real_space_cutoff_error = lambda r_cut: charge_list_einsum * np.sqrt(r_cut / (2 * self.system_volume)) * (np.exp(-(alpha * r_cut)**2) / (alpha * r_cut)**2) - self.err_tol
            r_cut0 = x_real_initial_guess / alpha 
            real_space_parameters['r_cut'] = fsolve(real_space_cutoff_error, r_cut0)[0]
        else:
            r_cut = real_space_parameters['r_cut']
            real_space_cutoff_error = lambda alpha: charge_list_einsum * np.sqrt(r_cut / (2 * self.system_volume)) * (np.exp(-(alpha * r_cut)**2) / (alpha * r_cut)**2) - self.err_tol
            alpha0 = x_real_initial_guess / r_cut
            real_space_parameters['alpha'] = fsolve(real_space_cutoff_error, alpha0)[0]
        return real_space_parameters

    def minimize_fourier_space_cutoff_error(self, charge_list_einsum, volume_derived_length, fourier_space_parameters, x_fourier_initial_guess):
        if 'alpha' in fourier_space_parameters:
            alpha = fourier_space_parameters['alpha']
            fourier_space_cutoff_error = lambda n_cut: charge_list_einsum * np.sqrt(n_cut) / (alpha * volume_derived_length**2) * (np.exp(-(np.pi * n_cut / (alpha * volume_derived_length))**2) / (np.pi * n_cut / (alpha * volume_derived_length))**2) - self.err_tol
            n_cut0 = x_fourier_initial_guess * volume_derived_length * alpha / np.pi 
            fourier_space_parameters['k_cut'] = 2 * np.pi / volume_derived_length * fsolve(fourier_space_cutoff_error, n_cut0)[0]
        else:
            k_cut = fourier_space_parameters['k_cut']
            n_cut = k_cut * volume_derived_length / (2 * np.pi)
            fourier_space_cutoff_error = lambda alpha: charge_list_einsum * np.sqrt(n_cut) / (alpha * volume_derived_length**2) * (np.exp(-(np.pi * n_cut / (alpha * volume_derived_length))**2) / (np.pi * n_cut / (alpha * volume_derived_length))**2) - self.err_tol
            alpha0 = np.pi * n_cut / (x_fourier_initial_guess * volume_derived_length)
            fourier_space_parameters['alpha'] = fsolve(fourier_space_cutoff_error, alpha0)[0]
        return fourier_space_parameters

    def compute_cutoff_errors(self, charge_list_einsum, alpha, r_cut, k_cut, volume_derived_length, prefix_list):
        real_space_cutoff_error = charge_list_einsum * np.sqrt(r_cut / (2 * self.system_volume)) * (np.exp(-(alpha * r_cut)**2) / (alpha * r_cut)**2)
        n_cut = k_cut * volume_derived_length / (2 * np.pi)
        fourier_space_cutoff_error = charge_list_einsum * np.sqrt(n_cut) / (alpha * volume_derived_length**2) * (np.exp(-(np.pi * n_cut / (alpha * volume_derived_length))**2) / (np.pi * n_cut / (alpha * volume_derived_length))**2)

        prefix_list.append(f'Real-space cutoff error: {real_space_cutoff_error:.3e}\n')
        prefix_list.append(f'Fourier-space cutoff error: {fourier_space_cutoff_error:.3e}\n\n')
        return prefix_list

    def convergence_check_with_r_cut(self, charge_list_prod, alpha, r_cut_max, lower_bound, upper_bound):
        r_cut_lower = lower_bound * r_cut_max
        r_cut_upper = upper_bound * r_cut_max
        precomputed_array_real = self.get_precomputed_array_real(alpha, r_cut_lower)
        real_space_energy_lower = np.sum(np.multiply(charge_list_prod, precomputed_array_real))

        precomputed_array_real = self.get_precomputed_array_real(alpha, r_cut_upper)
        real_space_energy_upper = np.sum(np.multiply(charge_list_prod, precomputed_array_real))
        if abs(real_space_energy_lower - real_space_energy_upper) < self.err_tol:
            convergence_status = 1
        else:
            convergence_status = 0
        return convergence_status

    def get_energy_profile_with_r_cut(self, charge_list_prod, alpha, r_cut_max,
                                      lower_bound, upper_bound, num_data_points):
        # compute real-space energy by varying r_cut
        r_cut_lower = lower_bound * r_cut_max
        r_cut_upper = upper_bound * r_cut_max
        r_cut_data = np.linspace(r_cut_lower, r_cut_upper, num_data_points)
        real_space_energy_data = np.zeros(int(num_data_points))
        for r_cut_index, r_cut in enumerate(r_cut_data):
            precomputed_array_real = self.get_precomputed_array_real(alpha, r_cut)
            real_space_energy_data[r_cut_index] = np.sum(np.multiply(charge_list_prod, precomputed_array_real))
        return (r_cut_data, real_space_energy_data)

    def convergence_check_with_k_cut(self, charge_list_prod, alpha, k_cut_lower, k_cut_upper):
        precomputed_array_fourier = self.get_precomputed_array_fourier(alpha, k_cut_lower)[0]
        fourier_space_energy_lower = np.sum(np.multiply(charge_list_prod, precomputed_array_fourier))

        precomputed_array_fourier = self.get_precomputed_array_fourier(alpha, k_cut_upper)[0]
        fourier_space_energy_upper = np.sum(np.multiply(charge_list_prod, precomputed_array_fourier))

        energy_difference = abs(fourier_space_energy_lower - fourier_space_energy_upper)
        if energy_difference < self.err_tol:
            convergence_status = 1
        else:
            convergence_status = 0
        return convergence_status

    def get_energy_profile_with_k_cut(self, charge_list_prod, alpha, k_cut_lower,
                                      k_cut_upper, num_data_points):
        # compute fourier-space energy by varying r_cut
        k_cut_data = np.linspace(k_cut_lower, k_cut_upper, num_data_points)
        fourier_space_energy_data = np.zeros(int(num_data_points))
        for k_cut_index, k_cut in enumerate(k_cut_data):
            precomputed_array_fourier = self.get_precomputed_array_fourier(alpha, k_cut)[0]
            fourier_space_energy_data[k_cut_index] = np.sum(np.multiply(charge_list_prod, precomputed_array_fourier))
        return (k_cut_data, fourier_space_energy_data)

    def check_for_k_cut_step_energy_convergence(self, step_energy_k_cut_data, energy_changes, k_cut_lower, k_cut_upper):
        energy_change_between_bounds = energy_changes[(step_energy_k_cut_data >= k_cut_lower) & (step_energy_k_cut_data <= k_cut_upper)]
        total_energy_change = energy_change_between_bounds.sum()
        max_energy_change = abs(energy_change_between_bounds).max()
        
        if total_energy_change < self.err_tol and max_energy_change < self.err_tol:
            convergence_status = 1
        else:
            convergence_status = 0
        return convergence_status

    def get_convergence_rcut(self, charge_list_prod, alpha, r_cut_max, lower_bound, upper_bound):

        (r_cut_data, real_space_energy_data) = self.get_energy_profile_with_r_cut(
            charge_list_prod, alpha, r_cut_max, lower_bound, upper_bound, self.num_data_points_low)

        real_space_energy_deviation = np.abs(real_space_energy_data - real_space_energy_data[-1])
        indices_of_non_convergence = np.where(real_space_energy_deviation > self.err_tol)[0]
        if len(indices_of_non_convergence) == 0:
            r_cut_convergence = r_cut_data[0]
        else:
            r_cut_convergence = r_cut_data[indices_of_non_convergence.max() + 1]
        return r_cut_convergence

    def get_simulation_cell_real_space_parameters(self, charge_list_prod, charge_list_einsum, real_space_parameters, x_real_initial_guess, dst_path):
        r_cut_max = min(self.translational_vector_length) / 2
        initial_fractional_r_cut = 0.75
        real_space_parameters['r_cut'] = initial_fractional_r_cut * r_cut_max
        # optimize real-space cutoff error for alpha
        real_space_parameters = self.minimize_real_space_cutoff_error(charge_list_einsum, real_space_parameters, x_real_initial_guess)
        alpha = real_space_parameters['alpha']

        alpha_percent_increase = 10
        print(f'Attempting to find best alpha towards converging real-space energy within the simulation cell:\n')
        print(f'Starting with an estimate for alpha={alpha * constants.ANG2BOHR:.3e} / angstrom')
        while not self.convergence_check_with_r_cut(charge_list_prod, alpha, r_cut_max,
                                                   self.threshold_fraction, self.upper_bound_rcut):
            alpha = (1 + alpha_percent_increase / 100) * alpha
            print(f'Couldn\'t find real-space energy convergence within simulation cell. Re-attempting with {alpha_percent_increase} % increased alpha={alpha * constants.ANG2BOHR:.3e} / angstrom')

        print(f'Preliminary convergence in real-space energy achieved at alpha={alpha * constants.ANG2BOHR:.3e} / angstrom\n')
        alpha_convergence = alpha
        r_cut_convergence = self.get_convergence_rcut(charge_list_prod, alpha_convergence, r_cut_max, self.lower_bound_real, self.upper_bound_rcut)
        print(f'alpha={alpha_convergence * constants.ANG2BOHR:.3e} / angstrom; r_cut={r_cut_convergence / r_cut_max:.3f} max L/2')
        alpha_vs_fraction_r_cut_convergence = []
        alpha_vs_fraction_r_cut_convergence.append([alpha_convergence, r_cut_convergence / r_cut_max])

        alpha_percent_decrease = 5
        print(f'Attempting to achieve convergence above {self.threshold_fraction * 100:.1f} % of max L/2:')
        while r_cut_convergence / r_cut_max < self.threshold_fraction:
            alpha_new = (1 - alpha_percent_decrease / 100) * alpha_convergence
            r_cut_new = self.get_convergence_rcut(charge_list_prod, alpha_new, r_cut_max, self.lower_bound_real, self.upper_bound_rcut)
            if r_cut_new / r_cut_max < self.upper_bound_rcut:
                r_cut_convergence = r_cut_new
                alpha_convergence = alpha_new
                print(f'alpha={alpha_convergence * constants.ANG2BOHR:.3e} / angstrom; r_cut={r_cut_convergence / r_cut_max:.3f} max L/2')
                alpha_vs_fraction_r_cut_convergence.append([alpha_convergence, r_cut_convergence / r_cut_max])
            else:
                break

        real_space_parameters['r_cut'] = r_cut_convergence
        real_space_parameters['alpha'] = alpha_convergence
        alpha_vs_fraction_r_cut_convergence = np.asarray(alpha_vs_fraction_r_cut_convergence)
        print(f'Convergence in real-space energy achieved at alpha={alpha_convergence * constants.ANG2BOHR:.3e} / angstrom with r_cut={r_cut_convergence / r_cut_max:.3f} max L/2\n')

        lower_bound = 0.7500
        print(f'Generating energy profile between {lower_bound:.3f} and {self.upper_bound_rcut:.3f} fractions of max L/2')
        (r_cut_data, real_space_energy_data) = self.get_energy_profile_with_r_cut(
            charge_list_prod, alpha_convergence, r_cut_max, lower_bound, self.upper_bound_rcut, self.num_data_points_high)

        fig1 = plt.figure()        
        ax = fig1.add_subplot(111)
        ax.plot(r_cut_data / r_cut_max, real_space_energy_data / constants.EV2HARTREE, 'o-', color='#2ca02c', mec='black')
        ax.set_xlabel('Fraction of $max(r_{{cut}})$')
        ax.set_ylabel(f'Energy (eV)')
        ax.set_title('Real-space energy convergence in $r_{{cut}}$')
        figure_name = 'Real-space energy convergence with r_cut.png'
        figure_path = dst_path.joinpath(figure_name)
        plt.savefig(str(figure_path))
        print(f'Generated energy profile\n')

        fig2 = plt.figure()        
        ax = fig2.add_subplot(111)
        ax.plot(alpha_vs_fraction_r_cut_convergence[:, 0] * constants.ANG2BOHR, alpha_vs_fraction_r_cut_convergence[:, 1], 'o-', color='#2ca02c', mec='black')
        ax.set_xlabel('alpha (1/$\AA$)')
        ax.set_ylabel(f'Fraction of $max(r_{{cut}})$')
        ax.set_title('Convergence in fractional $r_{{cut}}$ with alpha')
        figure_name = 'Convergence in fractional r_cut with alpha.png'
        figure_path = dst_path.joinpath(figure_name)
        plt.savefig(str(figure_path))
        return real_space_parameters

    def get_step_change_analysis_with_k_cut(self, k_cut_data, fourier_space_energy_data):
        fourier_space_energy_data_diff = np.diff(fourier_space_energy_data)
        indices0_of_step_change = np.where(fourier_space_energy_data_diff > self.step_increase_tol)[0]
        indices1_of_step_change = indices0_of_step_change + 1

        k_cut0_of_step_change = k_cut_data[indices0_of_step_change]
        k_cut1_of_step_change = k_cut_data[indices1_of_step_change]
        energy_changes = fourier_space_energy_data[indices1_of_step_change] - fourier_space_energy_data[indices0_of_step_change]
        return (k_cut0_of_step_change, k_cut1_of_step_change, energy_changes)

    def plot_energy_profile_in_bounded_k_cut(self, k_cut_data, fourier_space_energy_data, title_suffix, dst_path):
        fig1 = plt.figure()        
        ax = fig1.add_subplot(111)
        ax.plot(k_cut_data * constants.ANG2BOHR, fourier_space_energy_data / constants.EV2HARTREE, 'o-', color='#2ca02c', mec='black')
        ax.set_xlabel('$k_{{cut}}$ (1/$\AA$)')
        ax.set_ylabel(f'Energy (eV)')
        plt.title('Fourier-space energy convergence in $k_{{cut}}$', y=1.08)
        figure_name = f'Fourier-space energy convergence with k_cut{title_suffix}.png'
        figure_path = dst_path.joinpath(figure_name)
        plt.tight_layout()
        plt.savefig(str(figure_path))
        plt.close()
        return None

    def get_new_k_vectors(self, k_cut0, k_cut1):
        k_cut0_2 = k_cut0**2
        k_cut1_2 = k_cut1**2
        k_max = np.ceil(k_cut1 / self.reciprocal_lattice_vector_length).astype(int)
        new_k_vectors = []
        for i in range(-k_max[0], k_max[0]+1):
            for j in range(-k_max[1], k_max[1]+1):
                for k in range(-k_max[2], k_max[2]+1):
                    k_vector = np.dot(np.array([i, j, k]),
                                      self.reciprocal_lattice_matrix)
                    k_vector_2 = np.dot(k_vector, k_vector)
                    if k_vector_2 >= k_cut0_2 and k_vector_2 < k_cut1_2:
                        new_k_vectors.append([i, j, k])
        return new_k_vectors

    def get_k_vector_energy_contribution(self, charge_list_prod, alpha, k_vector):
        alpha4 = 4 * alpha
        fourier_sum_coeff = (2 * np.pi) / self.system_volume

        k_vector_exact = np.dot(k_vector, self.reciprocal_lattice_matrix)
        k_vector_exact_2 = np.dot(k_vector_exact, k_vector_exact)
        precomputed_array = (fourier_sum_coeff * np.exp(-k_vector_exact_2 / alpha4)
                             * np.cos(np.tensordot(
                                 self.pairwise_min_image_vector_data,
                                 k_vector_exact, axes=([2], [0])))
                             / k_vector_exact_2)

        energy_contribution = np.sum(np.multiply(charge_list_prod, precomputed_array))
        return energy_contribution

    def get_k_vector_based_energy_contribution(self, charge_list_prod, alpha, k_cut0_of_step_change, k_cut1_of_step_change, prefix_list):
        num_steps = len(k_cut0_of_step_change)
        new_k_vectors_list = []
        num_new_k_vectors = np.zeros(num_steps, int)
        for step_index in range(num_steps):
            k_cut0 = k_cut0_of_step_change[step_index]
            k_cut1 = k_cut1_of_step_change[step_index]
            new_k_vectors_list.append(self.get_new_k_vectors(k_cut0, k_cut1))
            num_new_k_vectors[step_index] = len(new_k_vectors_list[-1])

        new_k_vectors_consolidated = np.asarray([k_vector for new_k_vectors in new_k_vectors_list for k_vector in new_k_vectors])
        num_new_k_vectors_consolidated = len(new_k_vectors_consolidated)
        print(f'Identified a total of {num_new_k_vectors_consolidated} k-vectors contributing towards energy changes')
        energy_contribution_data = np.zeros(num_new_k_vectors_consolidated)
        for k_vector_index in range(num_new_k_vectors_consolidated):
            k_vector = new_k_vectors_consolidated[k_vector_index]
            energy_contribution_data[k_vector_index] = self.get_k_vector_energy_contribution(charge_list_prod, alpha, k_vector)

        # sorting in descending order
        sort_indices = np.argsort(energy_contribution_data)[::-1]
        sorted_new_k_vectors_consolidated = new_k_vectors_consolidated[sort_indices]
        sorted_energy_contribution_data = energy_contribution_data[sort_indices]
        prefix_list.append(f'k-vectors sorted in the decreasing order of their energy contributions\n')
        for k_vector_index in range(num_new_k_vectors_consolidated):
            k_vector = sorted_new_k_vectors_consolidated[k_vector_index]
            energy_contribution = sorted_energy_contribution_data[k_vector_index]
            prefix_list.append(f'{k_vector[0]:4d} {k_vector[1]:4d} {k_vector[2]:4d}: {energy_contribution / constants.EV2HARTREE:.3e} eV\n')
        return prefix_list

    def get_precise_step_change_data(self, charge_list_prod, alpha,
                                     k_cut_lower, k_cut_upper, dst_path,
                                     sub_prefix_list):
        k_max_lower = np.ceil(k_cut_lower / self.reciprocal_lattice_vector_length).astype(int)
        num_k_vectors_lower = np.ceil(np.prod(2 * k_max_lower + 1) * np.pi / 6 - 1).astype(int)
        k_max_upper = np.ceil(k_cut_upper / self.reciprocal_lattice_vector_length).astype(int)
        num_k_vectors_upper = np.ceil(np.prod(2 * k_max_upper + 1) * np.pi / 6 - 1).astype(int)
        print(f'k_max vary from [{",".join(str(element) for element in k_max_lower)}] to [{",".join(str(element) for element in k_max_upper)}]')
        print(f'Maximum number of k-vectors vary from {num_k_vectors_lower} to {num_k_vectors_upper}')

        (k_cut_data, fourier_space_energy_data) = self.get_energy_profile_with_k_cut(
                    charge_list_prod, alpha, k_cut_lower, k_cut_upper, self.num_data_points_high)

        title_suffix = f'_{int(self.lower_bound_kcut)}x-{int(self.upper_bound_kcut)}x k_estimate'
        self.plot_energy_profile_in_bounded_k_cut(k_cut_data, fourier_space_energy_data, title_suffix, dst_path)
        print(f'Generated energy profile\n')
        converged_fourier_energy = fourier_space_energy_data[-1]

        print(f'Estimating the step energy changes in Fourier-space energy within this k_cut range:')
        (k_cut0_estimated, k_cut1_estimated) = self.get_step_change_analysis_with_k_cut(k_cut_data, fourier_space_energy_data)[:-1]

        k_cut0_of_step_change = []
        k_cut1_of_step_change = []
        energy_changes = []
        num_steps = len(k_cut0_estimated)
        print(f'Identified {num_steps} preliminary step changes\n')
        max_divergent_k_cut = 0
        print(f'Analyzing each step energy change in detail:')
        for step_index in range(num_steps):
            print(f'Analyzing step-change {step_index+1}')
            k_cut_lower = k_cut0_estimated[step_index]
            k_cut_upper = k_cut1_estimated[step_index]
            k_max_lower = np.ceil(k_cut_lower / self.reciprocal_lattice_vector_length).astype(int)
            num_k_vectors_lower = np.ceil(np.prod(2 * k_max_lower + 1) * np.pi / 6 - 1).astype(int)
            k_max_upper = np.ceil(k_cut_upper / self.reciprocal_lattice_vector_length).astype(int)
            num_k_vectors_upper = np.ceil(np.prod(2 * k_max_upper + 1) * np.pi / 6 - 1).astype(int)
            print(f'k_max vary from [{",".join(str(element) for element in k_max_lower)}] to [{",".join(str(element) for element in k_max_upper)}]')
            print(f'Maximum number of k-vectors vary from {num_k_vectors_lower} to {num_k_vectors_upper}')
            (step_k_cut_data, step_fourier_space_energy_data) = self.get_energy_profile_with_k_cut(
                        charge_list_prod, alpha, k_cut_lower, k_cut_upper, self.step_change_data_points)
            title_suffix = f'_step{step_index+1}'
            self.plot_energy_profile_in_bounded_k_cut(step_k_cut_data, step_fourier_space_energy_data, title_suffix, dst_path)

            divergent_k_cut = step_k_cut_data[abs(step_fourier_space_energy_data - converged_fourier_energy) > self.err_tol]
            if len(divergent_k_cut) != 0:
                max_divergent_k_cut = max(divergent_k_cut)
            (k_cut0_of_step_change_temp, k_cut1_of_step_change_temp, energy_changes_temp) = self.get_step_change_analysis_with_k_cut(step_k_cut_data, step_fourier_space_energy_data)
            k_cut0_of_step_change.extend(k_cut0_of_step_change_temp.tolist())
            k_cut1_of_step_change.extend(k_cut1_of_step_change_temp.tolist())
            energy_changes.extend(energy_changes_temp.tolist())

        k_cut0_of_step_change = np.asarray(k_cut0_of_step_change)
        k_cut1_of_step_change = np.asarray(k_cut1_of_step_change)
        energy_changes = np.asarray(energy_changes)
        sub_prefix_list.append(f'Number of step changes in Fourier-space energy with varying k_cut: {len(energy_changes)}\n')
        print(f'Identified a total of {len(energy_changes)} step changes\n')

        fig = plt.figure()
        import matplotlib.ticker as mtick
        ax = fig.add_subplot(111)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax.plot(k_cut0_of_step_change * constants.ANG2BOHR, energy_changes / constants.EV2HARTREE, 'o-', color='#2ca02c', mec='black')
        ax.set_xlabel('$k_{{cut}}$ (1/$\AA$)')
        ax.set_ylabel(f'Energy (eV)')
        plt.title('Magnitude of step change in Fourier-space energy with increase in $k_{{cut}}$', y=1.08)
        figure_name = f'Step change convergence with k_cut.png'
        figure_path = dst_path.joinpath(figure_name)
        plt.tight_layout()
        plt.savefig(str(figure_path))

        return (k_cut_data, k_cut0_of_step_change, k_cut1_of_step_change,
                energy_changes, max_divergent_k_cut, sub_prefix_list)

    def get_k_cut_choices(
            self, k_cut_data, k_cut0_of_step_change, k_cut1_of_step_change,
            energy_changes, max_divergent_k_cut, k_cut_estimate, sub_prefix_list):
        if max_divergent_k_cut > 0:
            k_cut_gentle = k_cut_data[k_cut_data > max_divergent_k_cut][0]
        else:
            k_cut_gentle = 0
        sub_prefix_list.append(f'k_cut (gentle): {k_cut_gentle * constants.ANG2BOHR:.3e} / angstrom\n')
        k_max_gentle = np.ceil(k_cut_gentle / self.reciprocal_lattice_vector_length).astype(int)
        num_k_vectors_gentle = np.ceil(np.prod(2 * k_max_gentle + 1) * np.pi / 6 - 1).astype(int)
        print(f'k_cut (gentle): {k_cut_gentle * constants.ANG2BOHR:.3e} / angstrom; k_max: [{",".join(str(element) for element in k_max_gentle)}]; num_k-vectors: {num_k_vectors_gentle}')

        # check for step energy change convergence
        k_cut_stringent = k_cut1_of_step_change[-1]
        sub_prefix_list.append(f'k_cut (stringent): {k_cut_stringent * constants.ANG2BOHR:.3e} / angstrom\n')
        k_max_stringent = np.ceil(k_cut_stringent / self.reciprocal_lattice_vector_length).astype(int)
        num_k_vectors_stringent = np.ceil(np.prod(2 * k_max_stringent + 1) * np.pi / 6 - 1).astype(int)
        print(f'k_cut (stringent): {k_cut_stringent * constants.ANG2BOHR:.3e} / angstrom; k_max: [{",".join(str(element) for element in k_max_stringent)}]; num_k-vectors: {num_k_vectors_stringent}\n')

        factor_of_increase_from_estimation = k_cut_stringent / k_cut_estimate
        sub_prefix_list.append(f'Factor of increase in the value of converged k_cut from estimation: {factor_of_increase_from_estimation:.3e}\n')

        print(f'Analyzing convergence in step energy change:')
        k_cut_lower = self.threshold_fraction * k_cut_stringent
        k_cut_upper = k_cut_stringent
        step_energy_convergence_status = self.check_for_k_cut_step_energy_convergence(
            k_cut0_of_step_change, energy_changes, k_cut_lower, k_cut_upper)
        convergence_keyword = 'NOT ' if not step_energy_convergence_status else ''
        sub_prefix_list.append(f'Step energy changes have {convergence_keyword}converged\n')
        print(f'Step energy changes have {convergence_keyword}converged\n')
        return (k_cut_stringent, sub_prefix_list)

    def get_optimized_r_cut(self, charge_list_prod, alpha, choice_parameters,
                            dst_path, prefix_list):
        r_cut_max = min(self.translational_vector_length) / 2
        
        (r_cut_data, real_space_energy_data) = self.get_energy_profile_with_r_cut(
            charge_list_prod, alpha, r_cut_max, self.lower_bound_rcut, self.upper_bound_rcut, self.num_data_points_high)

        # check for energy-convergence with r_cut at user-specified alpha between 0 to L/2
        if self.convergence_check_with_r_cut(charge_list_prod, alpha, r_cut_max, self.threshold_fraction, self.upper_bound_rcut):
            converged_real_space_energy = real_space_energy_data[-1]
            # get more precise r_cut by looking at the convergence point.
            if self.precise_r_cut:
                lower_bound = r_cut_data[abs(real_space_energy_data - converged_real_space_energy) > self.err_tol][-1] / r_cut_max
                upper_bound = r_cut_data[abs(real_space_energy_data - converged_real_space_energy) < self.err_tol][0] / r_cut_max
    
                (r_cut_data_local, real_space_energy_data_local) = self.get_energy_profile_with_r_cut(
                    charge_list_prod, alpha, r_cut_max, lower_bound, upper_bound, self.num_data_points_high)
                r_cut = r_cut_data_local[abs(real_space_energy_data_local - converged_real_space_energy) < self.err_tol][0]
            else:
                r_cut = r_cut_data[abs(real_space_energy_data - converged_real_space_energy) < self.err_tol][0]
        else:
            # mention that r_cut is over L/2 and the simulation code doesn't support the user-specified alpha.
            prefix_list.append(f'')

            print(f'This code doesn\'t support user-specified alpha={alpha * constants.ANG2BOHR:.3e} as r_cut is over L/2. Please re-run at a suitable alpha value.')

            prefix_list.append(f'alpha: {alpha * constants.ANG2BOHR:.3e} / angstrom ({choice_parameters["r_cut"]})\n')
            prefix_list.append(f'This code doesn\'t support user-specified alpha={alpha * constants.ANG2BOHR:.3e} as r_cut is over L/2. Please re-run at a suitable alpha value.\n\n')

            file_name = 'precomputed_array'
            print_time_elapsed = 1
            prefix = ''.join(prefix_list)
            generate_report(self.start_time, dst_path, file_name, print_time_elapsed, prefix)

            exit()
        return r_cut

    def get_cutoff_parameters(self, tau_ratio, dst_path, prefix_list):
        real_space_parameters = {}
        fourier_space_parameters = {}
        if np.isreal(self.alpha):
            alpha_choice = 'user-specified'
            alpha = real_space_parameters['alpha'] = self.alpha
            fourier_space_parameters['alpha'] = self.alpha
        else:
            alpha_choice = 'optimal'

        if np.isreal(self.r_cut):
            r_cut_choice = 'user-specified'
            r_cut = real_space_parameters['r_cut'] = self.r_cut
        elif self.r_cut == 'simulation_cell':
            r_cut_choice = 'simulation_cell'
            k_cut_choice = 'imported'
        else:
            r_cut_choice = 'optimal'

        if self.r_cut != 'simulation_cell':
            if isinstance(self.k_cut, list):
                k_cut = 1.10 * max(np.asarray(self.k_cut) * self.reciprocal_lattice_vector_length)
                print(f'At the user-specified k_max=[{",".join(str(element) for element in self.k_cut)}], k_cut={k_cut * constants.ANG2BOHR:.3e} / angstrom')
                num_k_vectors = np.ceil(np.prod(2 * np.asarray(self.k_cut) + 1) * np.pi / 6 - 1).astype(int)
                print(f'Maximum number of k-vectors: {num_k_vectors}')
                k_cut_choice = 'user-specified (k_max)'
            elif np.isreal(self.k_cut):
                k_cut_choice = 'user-specified'
                k_cut = fourier_space_parameters['k_cut'] = self.k_cut
            elif self.k_cut == 'converge' and np.array_equal(self.system_size, np.ones(self.neighbors.n_dim, int)):
                k_cut_choice = 'converge'
            else:
                k_cut_choice = 'optimal'

        choice_parameters = {'alpha': alpha_choice,
                             'r_cut': r_cut_choice,
                             'k_cut': k_cut_choice}

        # Assumption for the accuracy analysis
        ion_charge_type = 'full'
        charge_list = self.base_charge_config_for_accuracy_analysis(ion_charge_type)
        charge_list_prod = np.multiply(charge_list.transpose(), charge_list)
        charge_list_einsum = np.einsum('ii', charge_list_prod)

        x_real_initial_guess = 0.5
        x_fourier_initial_guess = 0.5
        volume_derived_length = np.power(self.system_volume, 1/3)
        # real space contribution confined to the simulation cell
        if self.r_cut == 'simulation_cell':
            real_space_parameters = self.get_simulation_cell_real_space_parameters(charge_list_prod, charge_list_einsum, real_space_parameters, x_real_initial_guess, dst_path)
            alpha = real_space_parameters['alpha']
            r_cut = real_space_parameters['r_cut']

            # use k_cut convergence information from unit cell
            k_cut_convergence_system_size = np.ones(self.neighbors.n_dim, int)
            input_file_directory_name = dst_path.parts[-1]
            k_cut_convergence_system_directory_path = (
                dst_path.resolve().parents[1]
                / ('SystemSize[' + ','.join(str(element) for element in k_cut_convergence_system_size) + ']'))
            k_cut_convergence_input_directory_path = (k_cut_convergence_system_directory_path
                                                      / input_file_directory_name)
            k_cut_convergence_alpha_directory_path = (k_cut_convergence_input_directory_path
                                                      / f'alpha={alpha * constants.ANG2BOHR:.3e}')

            if not k_cut_convergence_alpha_directory_path.exists():
                print(f'Please re-run after converging k_cut at alpha={alpha * constants.ANG2BOHR:.3e} for system size [{",".join(str(element) for element in k_cut_convergence_system_size)}]')

                prefix_list.append(f'alpha: {alpha * constants.ANG2BOHR:.3e} / angstrom ({choice_parameters["r_cut"]})\n')
                prefix_list.append(f'r_cut: {r_cut / constants.ANG2BOHR:.3e} angstrom ({choice_parameters["r_cut"]})\n')
                n_max = np.round(r_cut / self.translational_vector_length).astype(int)
                prefix_list.append(f'n_max: [{n_max[0]}, {n_max[1]}, {n_max[2]}]\n')
                prefix_list.append(f'Please re-run after converging k_cut at alpha={alpha * constants.ANG2BOHR:.3e} for system size [{",".join(str(element) for element in k_cut_convergence_system_size)}]\n\n')

                file_name = 'precomputed_array'
                print_time_elapsed = 1
                prefix = ''.join(prefix_list)
                generate_report(self.start_time, dst_path, file_name, print_time_elapsed, prefix)

                exit()
            else:
                k_cut_convergence_log_file_path = k_cut_convergence_alpha_directory_path.joinpath('k_cut_convergence.log')
                k_cut_convergence_log_file = open(k_cut_convergence_log_file_path, 'r')
                for line_index, line in enumerate(k_cut_convergence_log_file):
                    # stringent k_cut
                    if line_index == 3:
                        k_cut = float(line[19:28]) / constants.ANG2BOHR
            # re-initialize fourier_space_parameters
            fourier_space_parameters = {}
            fourier_space_parameters['alpha'] = alpha
            # optimize fourier-space cutoff error for k_cut
            fourier_space_parameters = self.minimize_fourier_space_cutoff_error(charge_list_einsum, volume_derived_length, fourier_space_parameters, x_fourier_initial_guess)
            k_cut_estimate = fourier_space_parameters['k_cut']
        elif self.k_cut == 'converge' and np.array_equal(self.system_size, np.ones(self.neighbors.n_dim, int)):
            sub_prefix_list_01 = []
            print(f'Attempting to converge k_cut for user-specified alpha={alpha * constants.ANG2BOHR:.3e} / angstrom:\n')
            output_dir_path = dst_path.joinpath(f'alpha={alpha * constants.ANG2BOHR:.3e}')
            Path.mkdir(output_dir_path, parents=True, exist_ok=True)

            # optimize fourier-space cutoff error for k_cut
            fourier_space_parameters = self.minimize_fourier_space_cutoff_error(charge_list_einsum, volume_derived_length, fourier_space_parameters, x_fourier_initial_guess)
            k_cut_estimate = fourier_space_parameters['k_cut']
            print(f'Starting with an estimate for k_cut={k_cut_estimate * constants.ANG2BOHR:.3e} / angstrom')
            percent_increase_in_k_cut_upper = 10
            print(f'Exploring convergence in Fourier-space energy between {int(self.lower_bound_kcut)}x and {int(self.upper_bound_kcut)}x of estimated k_cut')

            k_cut_upper = self.upper_bound_kcut * k_cut_estimate
            k_cut_threshold = self.threshold_fraction * k_cut_upper
            # check for convergence in the absolute value of energy with k_cut
            while not self.convergence_check_with_k_cut(charge_list_prod, alpha, k_cut_threshold, k_cut_upper):
                k_cut_upper = (1 + percent_increase_in_k_cut_upper / 100) * k_cut_upper
                print(f'Could not find convergence in given k_cut range. Re-attempting with upper bound increased by {percent_increase_in_k_cut_upper:.3f} %')
            sub_prefix_list_01.append(f'Preliminary convergence in Fourier-space energy achieved at k_cut: {k_cut_upper * constants.ANG2BOHR:.3e} / angstrom\n')
            print(f'Preliminary convergence in absolute value of Fourier-space energy achieved within k_cut={k_cut_upper * constants.ANG2BOHR:.3e} / angstrom\n')

            print(f'Attempting to identify precise k_cut:')
            dst_path = output_dir_path
            # get step energy data
            k_cut_lower = self.lower_bound_kcut * k_cut_estimate
            k_cut_upper = self.upper_bound_kcut * k_cut_estimate
            print(f'Generating energy profile between {int(self.lower_bound_kcut)}x and {int(self.upper_bound_kcut)}x of estimated k_cut')
            (k_cut_data, k_cut0_of_step_change, k_cut1_of_step_change,
             energy_changes, max_divergent_k_cut, sub_prefix_list_01
             ) = self.get_precise_step_change_data(
                 charge_list_prod, alpha, k_cut_lower, k_cut_upper,
                 output_dir_path, sub_prefix_list_01)

            # NOTE: k_cut outputted below is the k_cut_stringent
            (k_cut, sub_prefix_list_01) = self.get_k_cut_choices(
                k_cut_data, k_cut0_of_step_change, k_cut1_of_step_change,
                energy_changes, max_divergent_k_cut, k_cut_estimate, sub_prefix_list_01)

            print(f'Analyzing energy contributions of individual k-vectors:')
            # analyze the k-vectors and their energy contributions towards Fourier-space energy
            sub_prefix_list_02 = []
            sub_prefix_list_02 = self.get_k_vector_based_energy_contribution(
                                charge_list_prod, alpha, k_cut0_of_step_change,
                                k_cut1_of_step_change, sub_prefix_list_02)

            file_name = 'k_vector_energy_contribution'
            print_time_elapsed = 0
            sub_prefix_02 = ''.join(sub_prefix_list_02)
            generate_report(self.start_time, dst_path, file_name, print_time_elapsed, sub_prefix_02)
            print('Finished k-vector analysis')

            file_name = 'k_cut_convergence'
            print_time_elapsed = 0
            sub_prefix_01 = ''.join(sub_prefix_list_01)
            generate_report(self.start_time, output_dir_path, file_name, print_time_elapsed, sub_prefix_01)
        elif not (np.isreal(self.alpha) and np.isreal(self.r_cut) and ((np.isreal(self.k_cut) or isinstance(self.k_cut, list)))):
            if np.isreal(self.alpha) & np.isreal(self.r_cut):
                # optimize fourier-space cutoff error for k_cut
                fourier_space_parameters = self.minimize_fourier_space_cutoff_error(charge_list_einsum, volume_derived_length, fourier_space_parameters, x_fourier_initial_guess)
                k_cut = fourier_space_parameters['k_cut']
            elif np.isreal(self.alpha) & (np.isreal(self.k_cut) or isinstance(self.k_cut, list)):
                r_cut = self.get_optimized_r_cut(
                            charge_list_prod, alpha, choice_parameters,
                            dst_path, prefix_list)
            elif np.isreal(self.r_cut) and (np.isreal(self.k_cut) or isinstance(self.k_cut, list)):
                # optimize real-space cutoff error for alpha
                real_space_parameters = self.minimize_real_space_cutoff_error(charge_list_einsum, real_space_parameters, x_real_initial_guess)
                alpha = real_space_parameters['alpha']
            elif np.isreal(self.alpha):
                r_cut = self.get_optimized_r_cut(
                            charge_list_prod, alpha, choice_parameters,
                            dst_path, prefix_list)
                print(f'Convergence in real-space energy achieved at alpha={alpha * constants.ANG2BOHR:.3e} / angstrom with r_cut:{r_cut / constants.ANG2BOHR:.3e}\n')

                # optimize fourier-space cutoff error for k_cut
                fourier_space_parameters = self.minimize_fourier_space_cutoff_error(charge_list_einsum, volume_derived_length, fourier_space_parameters, x_fourier_initial_guess)
                k_cut = fourier_space_parameters['k_cut']
            elif np.isreal(self.r_cut):
                # optimize real-space cutoff error for alpha
                real_space_parameters = self.minimize_real_space_cutoff_error(charge_list_einsum, real_space_parameters, x_real_initial_guess)
                alpha = fourier_space_parameters['alpha'] = real_space_parameters['alpha']
                # optimize fourier-space cutoff error for k_cut
                fourier_space_parameters = self.minimize_fourier_space_cutoff_error(charge_list_einsum, volume_derived_length, fourier_space_parameters, x_fourier_initial_guess)
                k_cut = fourier_space_parameters['k_cut']
            elif np.isreal(self.k_cut) or isinstance(self.k_cut, list):
                # optimize fourier-space cutoff error for alpha
                fourier_space_parameters = self.minimize_fourier_space_cutoff_error(charge_list_einsum, volume_derived_length, fourier_space_parameters, x_fourier_initial_guess)
                alpha = real_space_parameters['alpha'] = fourier_space_parameters['alpha']

                # explore real-space convergence for r_cut
                r_cut = self.get_optimized_r_cut(
                            charge_list_prod, alpha, choice_parameters,
                            dst_path, prefix_list)
            else:
                # current implementation of pot_k_ewald has O(N^2) complexity resulting in N-independt expression for alpha 
                alpha = (tau_ratio * np.pi**3 / (self.system_volume)**2)**(1/6)
                real_space_parameters['alpha'] = alpha
                fourier_space_parameters['alpha'] = alpha

                # explore real-space convergence for r_cut
                r_cut = self.get_optimized_r_cut(
                            charge_list_prod, alpha, choice_parameters,
                            dst_path, prefix_list)

                # optimize fourier-space cutoff error for k_cut
                fourier_space_parameters = self.minimize_fourier_space_cutoff_error(charge_list_einsum, volume_derived_length, fourier_space_parameters, x_fourier_initial_guess)
                k_cut = fourier_space_parameters['k_cut']
        return (alpha, r_cut, k_cut, choice_parameters, charge_list_einsum, volume_derived_length, prefix_list, dst_path)

    def get_ewald_parameters(self, prefix_list, dst_path):

        # real-space calculation limited to original simulation cell
        n_max_benchmark = np.zeros(self.pbc.shape, int)
        # k_max = 1 on all dimensions making (27 - 1) = 26 k-vectors in total
        k_max_benchmark = np.ones(self.pbc.shape, int)
        alpha_benchmark = 0.5
        # slightly less than half of minimum box length
        r_cut_benchmark = min(self.translational_vector_length) / 2.1
        # maximum reciprocal box length
        k_cut_benchmark = max(self.reciprocal_lattice_vector_length)
        benchmark_parameters = {'n_max': n_max_benchmark,
                                'k_max': k_max_benchmark,
                                'alpha': alpha_benchmark,
                                'r_cut': r_cut_benchmark,
                                'k_cut': k_cut_benchmark}
        num_repeats = int(1E+00)

        (tau_ratio, time_ratio) = self.benchmark_ewald(num_repeats, benchmark_parameters)
        prefix_list.append(f'tau_ratio, (tau_r/tau_f): {tau_ratio:.3e}\n')
        prefix_list.append(f'time_ratio, (time_r/time_f): {time_ratio:.3e}\n\n')

        (alpha, r_cut, k_cut, choice_parameters, charge_list_einsum, volume_derived_length, prefix_list, dst_path) = self.get_cutoff_parameters(tau_ratio, dst_path, prefix_list)

        prefix_list.append(f'alpha: {alpha * constants.ANG2BOHR:.3e} / angstrom ({choice_parameters["alpha"]})\n')
        prefix_list.append(f'r_cut: {r_cut / constants.ANG2BOHR:.3e} angstrom ({choice_parameters["r_cut"]})\n')
        prefix_list.append(f'k_cut: {k_cut * constants.ANG2BOHR:.3e} / angstrom ({choice_parameters["k_cut"]})\n')

        prefix_list = self.compute_cutoff_errors(charge_list_einsum, alpha, r_cut, k_cut, volume_derived_length, prefix_list)

        ewald_parameters = {'alpha': alpha,
                            'r_cut': r_cut,
                            'k_cut': k_cut}
        return (ewald_parameters, prefix_list, dst_path)

    def get_precomputed_array_real(self, alpha, r_cut):
        precomputed_array_real = self.pot_r_ewald(alpha, r_cut)[0] / self.material.dielectric_constant
        return precomputed_array_real

    def get_precomputed_array_fourier(self, alpha, k_cut):
        k_max = np.ceil(k_cut / self.reciprocal_lattice_vector_length).astype(int)  # max number of multiples of reciprocal lattice length vectors
        num_k_vectors = np.ceil(np.prod(2 * k_max + 1) * np.pi / 6 - 1).astype(int)
        precomputed_array_fourier = self.pot_k_ewald(k_max, alpha, k_cut) / self.material.dielectric_constant
        return (precomputed_array_fourier, k_max, num_k_vectors)

    def get_precomputed_array_fourier_with_k_vector_data(self, charge_list_prod, alpha, k_cut):
        k_max = np.ceil(k_cut / self.reciprocal_lattice_vector_length).astype(int)  # max number of multiples of reciprocal lattice length vectors
        num_k_vectors = np.ceil(np.prod(2 * k_max + 1) * np.pi / 6 - 1).astype(int)
        (precomputed_array_fourier, k_vector_data, energy_contribution_data) = self.pot_k_ewald_with_k_vector_data(charge_list_prod, k_max, alpha, k_cut)
        precomputed_array_fourier /= self.material.dielectric_constant
        return (precomputed_array_fourier, k_max, num_k_vectors, k_vector_data, energy_contribution_data)

    def get_precomputed_array(self, dst_path, compute_energy_contributions,
                              return_k_vector_data):
        """

        :param dst_path:
        :return:
        """
        prefix_list = []
        (ewald_parameters, prefix_list, dst_path) = self.get_ewald_parameters(prefix_list, dst_path)
        alpha = ewald_parameters['alpha']
        r_cut = ewald_parameters['r_cut']
        k_cut = ewald_parameters['k_cut']

        precomputed_array_real = self.get_precomputed_array_real(alpha, r_cut)

        if return_k_vector_data or compute_energy_contributions:
            # Assumption for the accuracy analysis
            ion_charge_type = 'full'
            charge_list = self.base_charge_config_for_accuracy_analysis(ion_charge_type)
            charge_list_prod = np.multiply(charge_list.transpose(), charge_list)

        if return_k_vector_data:
            (precomputed_array_fourier, k_max, num_k_vectors, k_vector_data,
             energy_contribution_data) = self.get_precomputed_array_fourier_with_k_vector_data(charge_list_prod, alpha, k_cut)
            prefix_list.append(f'k_max: [{k_max[0]}, {k_max[1]}, {k_max[2]}]\n')
            prefix_list.append(f'number of k-vectors: {num_k_vectors}\n\n')

            # sorting in descending order
            sort_indices = np.argsort(energy_contribution_data)[::-1]
            sorted_k_vector_data = k_vector_data[sort_indices]
            sorted_energy_contribution_data = energy_contribution_data[sort_indices]
            sub_prefix_list_01 = []
            sub_prefix_list_01.append(f'k-vectors sorted in the decreasing order of their energy contributions\n')
            for k_vector_index in range(len(sorted_k_vector_data)):
                k_vector = sorted_k_vector_data[k_vector_index]
                energy_contribution = sorted_energy_contribution_data[k_vector_index]
                sub_prefix_list_01.append(f'{k_vector[0]:4d} {k_vector[1]:4d} {k_vector[2]:4d}: {energy_contribution / constants.EV2HARTREE:.3e} eV\n')

            file_name = 'k_vector_energy_contribution_within_user_specified_k_cut'
            print_time_elapsed = 0
            sub_prefix_01 = ''.join(sub_prefix_list_01)
            generate_report(self.start_time, dst_path, file_name, print_time_elapsed, sub_prefix_01)
            print('Finished k-vector analysis')
        else:
            (precomputed_array_fourier, k_max, num_k_vectors) = self.get_precomputed_array_fourier(alpha, k_cut)
            prefix_list.append(f'k_max: [{k_max[0]}, {k_max[1]}, {k_max[2]}]\n')
            prefix_list.append(f'number of k-vectors: {num_k_vectors}\n\n')

        precomputed_array_self = - np.eye(self.neighbors.num_system_elements) * np.sqrt(alpha / np.pi) / self.material.dielectric_constant

        precomputed_array = precomputed_array_real + precomputed_array_fourier + precomputed_array_self

        if compute_energy_contributions:
            real_space_energy = np.sum(np.multiply(charge_list_prod, precomputed_array_real))
            prefix_list.append(f'Energy contribution from Real space: {real_space_energy/ constants.EV2HARTREE} eV\n')

            fourier_space_energy = np.sum(np.multiply(charge_list_prod, precomputed_array_fourier))
            prefix_list.append(f'Energy contribution from Fourier-space: {fourier_space_energy / constants.EV2HARTREE} eV\n')

            self_interaction_energy = np.sum(np.multiply(charge_list_prod, precomputed_array_self))
            prefix_list.append(f'Energy contribution from self-interactions: {self_interaction_energy / constants.EV2HARTREE} eV\n')

            total_system_energy = real_space_energy + fourier_space_energy + self_interaction_energy
            prefix_list.append(f'Total system energy (neutral): {total_system_energy / constants.EV2HARTREE} eV\n\n')
        file_name = 'precomputed_array'
        print_time_elapsed = 1
        prefix = ''.join(prefix_list)
        generate_report(self.start_time, dst_path, file_name, print_time_elapsed, prefix)
        return (precomputed_array, dst_path)


class Run(object):
    """defines the subroutines for running Kinetic Monte Carlo and
        computing electrostatic interaction energies"""
    def __init__(self, system, precomputed_array, temp, ion_charge_type,
                 species_charge_type, n_traj, t_final, time_interval,
                 species_count, initial_occupancy, relative_energies,
                 external_field, doping):
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
        self.initial_occupancy = initial_occupancy
        self.system_size = self.system.system_size
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

        self.undoped_system_relative_energies = (
                np.tile(unit_cell_relative_energies, self.system.num_cells))
        self.system_relative_energies = np.copy(self.undoped_system_relative_energies)

        self.num_shells = {}
        for element_type, element_relative_energies in self.relative_energies['doping'].items():
            self.num_shells[element_type] = [
                (len(dopant_element_relative_energies) - 1)
                for dopant_element_relative_energies in element_relative_energies]

        # number of kinetic processes
        self.n_proc = np.dot(self.species_count, self.system.num_neighbors)

        # n_elements_per_unit_cell
        self.head_start_n_elements_per_unit_cell_cum_sum = [
                                self.material.n_elements_per_unit_cell[
                                            :site_element_type_index].sum()
                                for site_element_type_index in (
                                        self.neighbors.element_type_indices)]

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

        self.max_neighbor_shells = {}
        self.dist_based_shell_index_lookup = {}
        tol_dist = 0.5 # angstrom
        for element_type_key in self.material.neighbor_cutoff_dist:
            element_type = element_type_key.split(self.material.element_type_delimiter)[0]
            element_type_index = self.material.element_types.index(element_type)
            sample_site_quantum_indices = [0, 0, 0, element_type_index, 0]
            sample_site_index = self.neighbors.get_system_element_index(
                self.system_size, sample_site_quantum_indices)
            pairwise_dist_list = []
            shell_index_list = []
            num_shells = 0
            while True:
                shell_based_neighbors = self.get_shell_based_neighbors(
                                sample_site_index, num_shells, self.system_size,
                                self.system.system_class_index_list,
                                self.system.hop_neighbor_list)
                outer_shell_neighbors = shell_based_neighbors[-1]
                if len(outer_shell_neighbors):
                    for outer_shell_neighbor in outer_shell_neighbors:
                        dist_value = self.neighbors.compute_distance(self.system.system_size,
                                                                     sample_site_index,
                                                                     outer_shell_neighbor) / constants.ANG2BOHR
                        pairwise_dist_list.append(dist_value)
                        shell_index_list.append(num_shells)
                    num_shells += 1
                else:
                    break
            sort_indices = np.argsort(pairwise_dist_list)
            sorted_pairwise_dist_array = np.asarray(pairwise_dist_list)[sort_indices]
            sorted_shell_index_array = np.asarray(shell_index_list)[sort_indices]
            temp_bins = sorted_pairwise_dist_array[:-1] + (sorted_pairwise_dist_array[1:] - sorted_pairwise_dist_array[:-1]) / 2
            dist_bins = np.concatenate([[sorted_pairwise_dist_array[0] - tol_dist],
                                        temp_bins,
                                        [sorted_pairwise_dist_array[-1] + tol_dist]])
            self.dist_based_shell_index_lookup[element_type_key] = {}
            self.dist_based_shell_index_lookup[element_type_key]['dist_bins'] = dist_bins
            self.dist_based_shell_index_lookup[element_type_key]['shell_index_list'] = sorted_shell_index_array
            self.max_neighbor_shells[element_type] = num_shells - 1

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
        self.n_proc_hop_element_type_list = []
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
            term01 = np.dot(
                charge_config[:, 0],
                (self.precomputed_array[neighbor_site_system_element_index, :]
                 - self.precomputed_array[species_site_system_element_index, :]
                 ))
            term02 = (
                self.species_charge_list[species_index]
                * (self.precomputed_array[species_site_system_element_index,
                                          species_site_system_element_index]
                   - self.precomputed_array[species_site_system_element_index,
                                            neighbor_site_system_element_index]))

            delg_0_ewald = (2 * self.species_charge_list[species_index]
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
                        f'Estimated value of {species_type} drift mobility is: {mean_drift_mobility:4.3e} cm2/V.s.\n')
                prefix_list.append(
                    f'Standard error of mean in {species_type} drift mobility is: {sem_drift_mobility:4.3e} cm2/V.s.\n')
            start_species_index = end_species_index
        return prefix_list

    def generate_random_doping_distribution(self, system_size,
                                            system_class_index_list,
                                            hop_neighbor_list,
                                            available_site_indices, map_index,
                                            num_dopants):
        dopant_type_dopant_site_indices = []
        num_dopant_sites_inserted = 0
        while (num_dopants - num_dopant_sites_inserted) and available_site_indices:
            dopant_site_index = rnd.choice(available_site_indices)
            dopant_type_dopant_site_indices.append(dopant_site_index)
            num_dopant_sites_inserted += 1
            num_shells_discard = self.doping['min_shell_separation'][map_index]
            long_neighbor_shell_indices = self.get_shell_based_neighbors(
                            dopant_site_index, num_shells_discard, system_size,
                            system_class_index_list, hop_neighbor_list)
            combined_long_neighbor_shell_indices = [
                    system_element_index
                    for shell_neighbors in long_neighbor_shell_indices
                    for system_element_index in shell_neighbors]
            available_site_indices = [
                site_index
                for site_index in available_site_indices
                if site_index not in combined_long_neighbor_shell_indices]
        return (dopant_type_dopant_site_indices, available_site_indices)

    def get_doping_distribution(self):
        dopant_site_indices = {}
        dopant_types_inserted = 0
        for map_index, num_dopants in enumerate(self.doping['num_dopants']):
            insertion_type = self.doping['insertion_type'][map_index]
            dopant_element_type = self.dopant_element_types[map_index]
            if insertion_type != 'gradient' and num_dopants:
                if insertion_type == 'manual':
                    dopant_site_indices[dopant_element_type] = (
                        self.doping['dopant_site_indices'][map_index][:num_dopants])
                elif insertion_type == 'random':
                    if dopant_types_inserted == 0:
                        substitution_element_type_index_list = []
                        available_site_indices = []
                    system_size = self.system.system_size
                    num_cells = system_size.prod()

                    substitution_element_type = self.substitution_element_types[map_index]
                    substitution_element_type_index = self.material.element_types.index(
                                                                substitution_element_type)
                    if substitution_element_type_index not in substitution_element_type_index_list:
                        system_element_index_offset_array = np.repeat(
                                    np.arange(
                                        0, (self.material.total_elements_per_unit_cell
                                            * num_cells),
                                        self.material.total_elements_per_unit_cell),
                                    self.material.n_elements_per_unit_cell[
                                                    substitution_element_type_index])
                        site_indices = (
                            np.tile(self.material.n_elements_per_unit_cell[
                                        :substitution_element_type_index].sum()
                                    + np.arange(0,
                                                self.material.n_elements_per_unit_cell[
                                                    substitution_element_type_index]),
                                    num_cells)
                            + system_element_index_offset_array).tolist()
                        available_site_indices.extend(site_indices[:])
                    substitution_element_type_index_list.append(substitution_element_type_index)

                    (dopant_type_dopant_site_indices,
                     available_site_indices) = (
                         self.generate_random_doping_distribution(
                            system_size, self.system.system_class_index_list,
                            self.system.hop_neighbor_list,
                            available_site_indices, map_index, num_dopants))
                    dopant_site_indices[dopant_element_type] = dopant_type_dopant_site_indices
                elif insertion_type == 'pairwise':
                    system_size = self.system.system_size
                    num_cells = system_size.prod()
                    substitution_element_type = self.substitution_element_types[map_index]
                    substitution_element_type_index = self.material.element_types.index(
                                                                substitution_element_type)
                    system_element_index_offset_array = np.repeat(
                                np.arange(
                                    0, (self.material.total_elements_per_unit_cell
                                        * num_cells),
                                    self.material.total_elements_per_unit_cell),
                                self.material.n_elements_per_unit_cell[
                                                substitution_element_type_index])
                    site_indices = (
                        np.tile(self.material.n_elements_per_unit_cell[
                                    :substitution_element_type_index].sum()
                                + np.arange(0,
                                            self.material.n_elements_per_unit_cell[
                                                substitution_element_type_index]),
                                num_cells)
                        + system_element_index_offset_array)
                    num_site_indices = len(site_indices)
                    pair_wise_distance_vector_array = np.zeros((num_site_indices, num_site_indices, self.neighbors.n_dim))
                    for index, site_index in enumerate(site_indices):
                        pair_wise_distance_vector_array[index, :] = self.system.pairwise_min_image_vector_data[site_index][site_indices]
                    pair_wise_distance_array = np.linalg.norm(pair_wise_distance_vector_array, axis=2)
                    intra_pair_distance_ang = self.doping['pairwise'][map_index]['intra_pair_distance']
                    intra_pair_distance = intra_pair_distance_ang * constants.ANG2BOHR
                    rounding_digits = len(str(intra_pair_distance_ang).split(".")[1])
                    desired_pair_internal_indices_temp = np.where(pair_wise_distance_array.round(rounding_digits) == np.round(intra_pair_distance, rounding_digits))
                    desired_pair_internal_indices = np.hstack((desired_pair_internal_indices_temp[0][:, None], desired_pair_internal_indices_temp[1][:, None]))

                    # avoiding duplicate pairs
                    desired_pair_internal_indices = desired_pair_internal_indices[desired_pair_internal_indices[:, 1] > desired_pair_internal_indices[:, 0]]
                    desired_pair_indices = site_indices[desired_pair_internal_indices]
                    num_pairs = len(desired_pair_indices)

                    # arrange pairs in plane_of_arrangement
                    plane_of_arrangement = self.doping['pairwise'][map_index]['plane_of_arrangement']
                    cumulative_pair_indices = desired_pair_indices.flatten()
                    site_positions = np.zeros((2 * num_pairs, self.neighbors.n_dim))
                    for index, site_index in enumerate(cumulative_pair_indices):
                        site_positions[index] = self.neighbors.get_coordinates(system_size, site_index)
                    site_positions = np.dot(site_positions,
                                            np.linalg.inv(self.material.lattice_matrix * system_size))
                    plane_contributions = np.zeros(2 * num_pairs)
                    plane_contributions_max = 0
                    for dim_index in range(self.neighbors.n_dim):
                        if plane_of_arrangement[dim_index] != 0:
                            plane_contributions += site_positions[:, dim_index] / plane_of_arrangement[dim_index]
                            plane_contributions_max += 1
                    sort_indices_plane_contributions = np.argsort(plane_contributions)
                    sorted_plane_contributions = plane_contributions[sort_indices_plane_contributions]
                    sorted_pair_indices = cumulative_pair_indices[sort_indices_plane_contributions]
                    rounding_digits_for_plane_contributions = 6
                    rounded_plane_contributions = sorted_plane_contributions.round(rounding_digits_for_plane_contributions)
                    unique_plane_contributions = np.unique(rounded_plane_contributions)
                    num_unique_plane_contributions = len(unique_plane_contributions)
                    coupled_plane_contributions = np.reshape(unique_plane_contributions, (int(num_unique_plane_contributions / 2), 2))
                    atoms_sorted_by_plane = np.empty(int(num_unique_plane_contributions / 2), dtype=object)
                    num_planes = 2 * (system_size[0] + system_size[1])  # plane intersects a and b axis at half-unit cell length
                    num_atoms_by_plane = np.zeros(num_planes)
                    for plane_index, plane_contribution in enumerate(coupled_plane_contributions):
                        atoms_sorted_by_plane[plane_index] = (
                            np.append(sorted_pair_indices[rounded_plane_contributions == plane_contribution[0]],
                                      sorted_pair_indices[rounded_plane_contributions == plane_contribution[1]]))
                        num_atoms_by_plane[plane_index] = len(atoms_sorted_by_plane[plane_index])
                    inter_plane_spacing = self.doping['pairwise'][map_index]['inter_plane_spacing']
                    starting_plane_index = 0
                    selected_plane_indices = range(starting_plane_index, num_planes, inter_plane_spacing)
                    pair_atoms_in_selected_planes = np.hstack(atoms_sorted_by_plane[selected_plane_indices])
                    if num_dopants == len(pair_atoms_in_selected_planes):
                        dopant_site_indices[dopant_element_type] = pair_atoms_in_selected_planes
                    else:
                        print(f'Total number of pair atoms available with user-specified inter-plane spacing: {len(pair_atoms_in_selected_planes)}\n')
                        print(f'Number of pair-wise substitutions ({num_dopants}) did not match with total number of pair atoms available.\n')
                        print(f'Please re-run by changing number of pair-wise substitutions to {len(pair_atoms_in_selected_planes)}\n')
                        exit()
                dopant_types_inserted += 1
            elif insertion_type == 'gradient':
                # NOTE: 'available_site_indices' is populated based on an isolated step system size.
                # NOTE: This implementation is inefficient as overlap between doping regions across interface
                # are not avoided and can only be eliminated through multiple attempts of doping distribution.
                # NOTE: However, this implementation is chosen because of its simplicity.
                gradient_params = self.doping['gradient'][map_index]
                ld = gradient_params['ld']
                step_length_ratio = gradient_params['step_length_ratio']
                stepwise_num_dopants = gradient_params['stepwise_num_dopants']
                sum_step_length_ratio = sum(step_length_ratio)
                assert (self.system.system_size[ld] % sum_step_length_ratio == 0), 'step system size must be an integer multiple of unit cell'
                num_steps = len(step_length_ratio)
                if map_index == 0:
                    stepwise_substitution_element_type_index_list = [[] for _ in range(num_steps)]
                    stepwise_available_site_indices = [[] for _ in range(num_steps)]
                num_unit_cells_translated = 0
                for step_index in range(num_steps):
                    num_dopants = stepwise_num_dopants[step_index]
                    step_system_size = np.copy(self.system.system_size)
                    step_system_size[ld] *= step_length_ratio[step_index] / sum_step_length_ratio
                    num_cells = step_system_size.prod()
                    if num_dopants:
                        available_site_indices = stepwise_available_site_indices[step_index]
                        substitution_element_type = self.substitution_element_types[map_index]
                        substitution_element_type_index = self.material.element_types.index(
                                                                    substitution_element_type)
                        if substitution_element_type_index not in stepwise_substitution_element_type_index_list[step_index]:
                            system_element_index_offset_array = np.repeat(
                                        np.arange(
                                            0, (self.material.total_elements_per_unit_cell
                                                * num_cells),
                                            self.material.total_elements_per_unit_cell),
                                        self.material.n_elements_per_unit_cell[
                                                        substitution_element_type_index])
                            site_indices = (
                                np.tile(self.material.n_elements_per_unit_cell[
                                            :substitution_element_type_index].sum()
                                        + np.arange(0,
                                                    self.material.n_elements_per_unit_cell[
                                                        substitution_element_type_index]),
                                        num_cells)
                                + system_element_index_offset_array).tolist()
                            available_site_indices.extend(site_indices[:])
                        stepwise_substitution_element_type_index_list[step_index].append(substitution_element_type_index)

                        if self.system.num_unique_step_systems == 1:
                            lookup_index = 0
                        else:
                            lookup_index = np.where((self.system.step_system_size_array == step_system_size).all(axis=1))[0][0]
                        step_system_hop_neighbor_list = self.system.step_hop_neighbor_master_list[lookup_index]
                        step_system_class_index_list = self.system.step_system_class_index_master_list[lookup_index]
                        (step_system_dopant_type_dopant_site_indices,
                         available_site_indices) = (
                             self.generate_random_doping_distribution(
                                step_system_size, step_system_class_index_list,
                                step_system_hop_neighbor_list,
                                available_site_indices, map_index, stepwise_num_dopants[step_index]))
                        full_system_dopant_type_dopant_site_indices = []
                        for index in step_system_dopant_type_dopant_site_indices:
                            step_system_quantum_indices = self.neighbors.get_quantum_indices(step_system_size, index)
                            full_system_quantum_indices = np.copy(step_system_quantum_indices)
                            full_system_quantum_indices[ld] += num_unit_cells_translated
                            full_system_se_index = self.neighbors.get_system_element_index(self.system.system_size, full_system_quantum_indices)
                            full_system_dopant_type_dopant_site_indices.append(full_system_se_index)
                        if dopant_element_type in dopant_site_indices:
                            dopant_site_indices[dopant_element_type].extend(full_system_dopant_type_dopant_site_indices)
                        else:
                            dopant_site_indices[dopant_element_type] = full_system_dopant_type_dopant_site_indices
                    num_unit_cells_translated += step_system_size[ld]
        return (dopant_site_indices)

    def get_doping_analysis(self, dopant_site_indices, prefix_list):
        min_shell_separation = self.doping['min_shell_separation'][:]
        for dopant_element_type, dopant_type_dopant_site_indices in dopant_site_indices.items():
            map_index = self.dopant_element_types.index(dopant_element_type)
            substitution_element_type = self.substitution_element_types[map_index]
            substitution_element_type_key = self.material.element_type_delimiter.join([substitution_element_type] * 2)
            num_dopant_sites_inserted = len(dopant_type_dopant_site_indices)
            tail = 's' if num_dopant_sites_inserted > 1 else ''
            prefix_list.append(f'Inserted {num_dopant_sites_inserted} site{tail} of dopant element type {dopant_element_type}\n')
            dopant_index_precision = int(np.ceil(np.log10(num_dopant_sites_inserted + 1)))
            dopant_type_precision = len(dopant_element_type)
            entry_list = ['site1', 'site2', 'pairwise distance (ang.)', 'shells apart']
            entry_width_list = [len(entry) for entry in entry_list]
            num_decimals = 2
            stat_width = 7
            stat_decimals = 3
            if num_dopant_sites_inserted > 1:
                prefix_list.append(f'{entry_list[0]}\t{entry_list[1]}\t{entry_list[2]}\t{entry_list[3]}\n')
            for index1, dopant_site_index_1 in enumerate(dopant_type_dopant_site_indices):
                for index2, dopant_site_index_2 in enumerate(dopant_type_dopant_site_indices[index1+1:]):
                    inter_dopant_dist = self.neighbors.compute_distance(self.system.system_size,
                                                                        dopant_site_index_1,
                                                                        dopant_site_index_2) / constants.ANG2BOHR
                    lookup_index = np.digitize(inter_dopant_dist,
                                               self.dist_based_shell_index_lookup[substitution_element_type_key]['dist_bins']) - 1
                    # number of shells in between
                    shell_separation = self.dist_based_shell_index_lookup[substitution_element_type_key]['shell_index_list'][lookup_index] - 1
                    prefix_list.append(f'{dopant_element_type:>{entry_width_list[0]-dopant_index_precision}}{index1+1:0{dopant_type_precision}}\t{dopant_element_type:>{entry_width_list[1]-dopant_index_precision}}{index1+index2+2:0{dopant_type_precision}}\t{inter_dopant_dist:>{entry_width_list[2]}.{num_decimals}f}\t{shell_separation:{entry_width_list[3]}}\n')
                    if index1 == index2 == 0:
                        min_separation = {'site1': dopant_site_index_1,
                                          'site2': dopant_site_index_2,
                                          'dist': inter_dopant_dist,
                                          'shell_separation': shell_separation}
                    else:
                        if shell_separation < min_separation['shell_separation']:
                            min_separation = {'site1': dopant_site_index_1,
                                              'site2': dopant_site_index_2,
                                              'dist': inter_dopant_dist,
                                              'shell_separation': shell_separation}
            if num_dopant_sites_inserted > 1:
                min_shell_separation[map_index] = min_separation["shell_separation"]
                prefix_list.append(f'All dopant sites of element type \'{dopant_element_type}\' are separated by at least {min_shell_separation[map_index]} shells\n')

                site_coords = np.zeros((num_dopant_sites_inserted, self.neighbors.n_dim))
                for index, dopant_site_index in enumerate(dopant_type_dopant_site_indices):
                    site_coords[index, :] = self.neighbors.get_coordinates(self.system.system_size,
                                                                           dopant_site_index)
                mean_center = np.mean(site_coords, axis=0) / constants.ANG2BOHR
                std_dist_dev = np.std(site_coords, axis=0) / constants.ANG2BOHR
                prefix_list.append(f'Mean center of dopant sites of element type \'{dopant_element_type}\' is: [' + "".join(f'{val:{stat_width}.{stat_decimals}f}' for val in mean_center) + ']\n')
                prefix_list.append(f'Standard distance deviation of dopant sites of element type \'{dopant_element_type}\' is: [' + "".join(f'{val:{stat_width}.{stat_decimals}f}' for val in std_dist_dev) + ']\n')
            else:
                min_shell_separation = self.doping['min_shell_separation']
        return (prefix_list, min_shell_separation)

    def get_shell_based_neighbors(self, site_system_element_index, num_shells,
                                  system_size, system_class_index_list,
                                  hop_neighbor_list):
        shell_based_neighbors = []
        inner_shell_neighbor_indices = []
        site_element_type_index = self.neighbors.get_quantum_indices(
                                    system_size, site_system_element_index)[3]
        substitution_element_type = self.material.element_types[site_element_type_index]
        hop_element_type = self.material.element_type_delimiter.join(
                                            [substitution_element_type] * 2)
        for shell_index in range(num_shells+1):
            current_shell_elements = []
            if shell_index == 0:
                current_shell_elements.extend([site_system_element_index])
                current_shell_neighbors = current_shell_elements
            else:
                for system_element_index in inner_shell_neighbor_indices:
                    class_index = system_class_index_list[system_element_index]
                    (element_type_element_index, _) = (
                        self.get_element_type_element_index(
                            site_element_type_index, system_element_index))
                    local_neighbor_site_system_element_index_list = []
                    for hop_dist_type_object in hop_neighbor_list[hop_element_type][class_index]:
                        local_neighbor_site_system_element_index_list.extend(
                            hop_dist_type_object.neighbor_system_element_indices[
                                element_type_element_index].tolist())
                    current_shell_elements.extend(
                        local_neighbor_site_system_element_index_list)
                # avoid duplication of inner_shell_neighbor_indices or dopant_site_index
                current_shell_neighbors = [
                    current_shell_element
                    for current_shell_element in current_shell_elements
                    if current_shell_element not in inner_shell_neighbor_indices]
            # avoid duplication within the shell
            current_shell_neighbors = list(set(current_shell_neighbors))
            inner_shell_neighbor_indices.extend(current_shell_neighbors)
            shell_based_neighbors.append(current_shell_neighbors)
        return shell_based_neighbors

    def get_system_shell_based_neighbors(self, dopant_site_indices):
        system_shell_based_neighbors = {}
        dopant_site_element_types = {}
        for dopant_element_type, dopant_element_type_site_indices in dopant_site_indices.items():
            dopant_element_type_index = self.dopant_element_types.index(dopant_element_type)
            substitution_element_type = self.substitution_element_types[dopant_element_type_index]
            dopant_element_type_map = ':'.join([substitution_element_type, dopant_element_type])
            map_index = self.doping['doping_element_map'].index(dopant_element_type_map)
            insertion_type = self.doping['insertion_type'][map_index]
            if dopant_element_type == 'X':
                max_neighbor_shells = 0
            elif insertion_type == 'pairwise':
                # set for the specific case of S-S pairwise dopant insertion with inter_plane_spacing == 4
                max_neighbor_shells = len(self.relative_energies['doping'][substitution_element_type][map_index]) + 3
            else:
                max_neighbor_shells = self.max_neighbor_shells[substitution_element_type]
            for dopant_site_index in dopant_element_type_site_indices:
                system_shell_based_neighbors[dopant_site_index] = (
                    self.get_shell_based_neighbors(dopant_site_index, max_neighbor_shells,
                                                   self.system_size,
                                                   self.system.system_class_index_list,
                                                   self.system.hop_neighbor_list))
                dopant_site_element_types[dopant_site_index] = dopant_element_type
        return (dopant_site_element_types, system_shell_based_neighbors)

    def get_site_wise_shell_indices(self, dopant_site_element_types,
                                    dopant_site_shell_based_neighbors, prefix_list):
        element_type_index_list = []
        site_indices_list = []
        dopant_element_index_list = []
        site_wise_shell_indices = []
        shell_element_type_list = []
        overlap = 0
        for dopant_element_index, dopant_site_index in enumerate(dopant_site_shell_based_neighbors):
            shell_neighbors = dopant_site_shell_based_neighbors[dopant_site_index]
            element_type_index = self.neighbors.get_quantum_indices(
                                            self.system_size, dopant_site_index)[3]
            element_type = self.material.element_types[element_type_index]
            if element_type_index not in element_type_index_list:
                element_type_index_list.append(element_type_index)
                system_element_index_offset_array = np.repeat(
                    np.arange(0, (self.material.total_elements_per_unit_cell
                                  * self.system.num_cells),
                              self.material.total_elements_per_unit_cell),
                    self.material.n_elements_per_unit_cell[element_type_index])
                site_indices = (
                    np.tile(self.material.n_elements_per_unit_cell[:element_type_index].sum()
                            + np.arange(0, self.material.n_elements_per_unit_cell[element_type_index]),
                            self.system.num_cells)
                    + system_element_index_offset_array)
                site_indices_list.extend([index for index in site_indices])
                num_site_indices = len(site_indices)
                site_wise_shell_indices.extend([self.max_neighbor_shells[element_type] + 1]
                                               * num_site_indices)
                dopant_element_index_list.extend([-1] * num_site_indices)
                shell_element_type_list.extend([element_type] * num_site_indices)
            for shell_index, neighbor_indices in enumerate(shell_neighbors):
                for neighbor_index in neighbor_indices:
                    index = site_indices_list.index(neighbor_index)
                    if shell_index < site_wise_shell_indices[index]:
                        site_wise_shell_indices[index] = shell_index
                        dopant_element_index_list[index] = dopant_element_index
                        if shell_element_type_list[index] != element_type:
                            overlap = 1
                        shell_element_type_list[index] = dopant_site_element_types[dopant_site_index]
        dopant_element_type_index_list = [self.dopant_element_types.index(element_type) for element_type in shell_element_type_list]
        site_wise_shell_indices_array = np.hstack(
                                (np.asarray(site_indices_list)[:, None],
                                 np.asarray(dopant_element_type_index_list)[:, None],
                                 np.asarray(dopant_element_index_list)[:, None],
                                 np.asarray(site_wise_shell_indices)[:, None]))
        if overlap:
            prefix_list.append(
                            'All shell based neighbor sites are independent\n\n')
        else:
            prefix_list.append(
                        'All shell based neighbor sites are NOT independent\n\n')
        return (site_wise_shell_indices_array, prefix_list)

    def generate_initial_occupancy(self, dopant_site_indices):
        """generates initial occupancy list based on species count
        :param species_count:
        :return:
        """
        occupancy = []
        for species_type_index, num_species in enumerate(self.species_count):
            species_type = self.material.species_types[species_type_index]
            if self.doping_active:
                for map_index, dopant_species_type in enumerate(
                                                    self.dopant_species_types):
                    num_dopant_sites = self.doping['num_dopants'][map_index]
                    if (self.doping['site_charge_initiation'][map_index] == 'yes'
                        and dopant_species_type == species_type
                        and num_dopant_sites and num_species):
                        dopant_element_type = self.dopant_element_types[map_index]
                        occupancy.extend(rnd.sample(dopant_site_indices[dopant_element_type], num_species)[:])
                        num_species -= len(dopant_site_indices[dopant_element_type][:num_species])
            if species_type in self.initial_occupancy:
                occupancy.extend([index for index in self.initial_occupancy[species_type]])
                num_species -= len(self.initial_occupancy[species_type])
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

    def base_charge_config(self):
        # generate lattice charge list
        unit_cell_charge_list = np.array(
            [self.material.charge_types[self.ion_charge_type][
                 self.material.element_types[element_type_index]]
             for element_type_index in self.material.element_type_index_list])
        charge_list = np.tile(unit_cell_charge_list, self.system.num_cells)[
                                                                :, np.newaxis]
        return charge_list

    def charge_config(self, occupancy, dopant_site_indices):
        """Returns charge distribution of the current configuration
        :param occupancy:
        :param ion_charge_type:
        :param species_charge_type:
        :return:
        """
        
        charge_list = self.base_charge_config()

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

    def preproduction(self, dst_path, random_seed):
        """Subroutine to setup input files to run the production stage of the simulation
        :param dst_path:
        :param random_seed:
        :return: """
        assert dst_path, 'Please provide the destination path where \
                          simulation output files needs to be saved'
        rnd.seed(random_seed)
        random_seed_list = [rnd.random() for traj_index in range(self.n_traj)]
        for traj_index in range(self.n_traj):
            traj_dir_path = dst_path.joinpath(f'traj{traj_index+1}')
            Path.mkdir(traj_dir_path, parents=True, exist_ok=True)
            random_state_file_path = traj_dir_path.joinpath(f'initial_rnd_state.dump')
            rnd.seed(random_seed_list[traj_index])
            pickle.dump(rnd.getstate(), open(random_state_file_path, 'wb'))

        if 'pairwise' in self.doping['insertion_type']:
            map_index = self.doping['insertion_type'].index('pairwise')
            pairwise_insertion = self.doping['num_dopants'][map_index] != 0
        else:
            pairwise_insertion = 0
        for traj_index in range(self.n_traj):
            prefix_list = []
            traj_dir_path = dst_path.joinpath(f'traj{traj_index+1}')

            if self.doping_active:
                if traj_index == 0:
                    dopant_site_indices_repo = {}
                dopant_site_indices_repo[traj_index] = {}
                if not pairwise_insertion:
                    prefix_list.append(f'Trajectory {traj_index+1}:\n')
                attempt_number = 1
                old_min_shell_separation = [-1] * len(self.doping['num_dopants'])
                while (np.any(old_min_shell_separation < self.doping['min_shell_separation']) and attempt_number <= self.doping['max_attempts']):
                    temp_sub_prefix_list = []
                    temp_dopant_site_indices = self.get_doping_distribution()
                    (temp_sub_prefix_list, new_min_shell_separation) = self.get_doping_analysis(
                                                        temp_dopant_site_indices, temp_sub_prefix_list)
                    if new_min_shell_separation > old_min_shell_separation:
                        unique_flag = 1
                        for traj_dopant_site_indices in dopant_site_indices_repo.values():
                            for i_dopant_element_type, i_dopant_site_indices in traj_dopant_site_indices.items():
                                if ((set(i_dopant_site_indices) == set(temp_dopant_site_indices[i_dopant_element_type])) & (i_dopant_element_type != 'X')):
                                    unique_flag = 0
                                    break
                            if not unique_flag:
                                break
                        if unique_flag:
                            sub_prefix_list = [prefix for prefix in temp_sub_prefix_list]
                            old_min_shell_separation = new_min_shell_separation
                            dopant_site_indices = {}
                            for i_dopant_element_type, i_dopant_site_indices in temp_dopant_site_indices.items():
                                dopant_site_indices[i_dopant_element_type] = [index for index in i_dopant_site_indices]
                    attempt_number += 1
                prefix_list.extend(sub_prefix_list)
                for i_dopant_element_type, i_dopant_site_indices in dopant_site_indices.items():
                    dopant_site_indices_repo[traj_index][i_dopant_element_type] = [index for index in i_dopant_site_indices]
                (dopant_site_element_types, system_shell_based_neighbors) = (
                    self.get_system_shell_based_neighbors(dopant_site_indices))
                (site_wise_shell_indices_array, prefix_list) = (
                    self.get_site_wise_shell_indices(dopant_site_element_types,
                                                     system_shell_based_neighbors,
                                                     prefix_list))
                if pairwise_insertion:
                    output_file_path = dst_path / 'site_indices'
                else:
                    output_file_path = traj_dir_path / 'site_indices'
                np.save(output_file_path, site_wise_shell_indices_array)

                file_name = 'PreProduction'
                prefix = ''.join(prefix_list)
                print_time_elapsed = 0
                if pairwise_insertion:
                    generate_report(self.start_time, dst_path, file_name,
                                    print_time_elapsed, prefix)
                    break
                else:
                    generate_report(self.start_time, traj_dir_path, file_name,
                                    print_time_elapsed, prefix)
        return None

    def do_kmc_steps(self, dst_path, output_data, random_seed, compute_mode):
        """Subroutine to run the KMC simulation by specified number
        of steps
        :param dst_path:
        :return: """
        assert dst_path, 'Please provide the destination path where \
                          simulation output files needs to be saved'

        if compute_mode != 'parallel':
            self.preproduction(dst_path, random_seed)
        num_path_steps_per_traj = int(self.t_final / self.time_interval) + 1
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
            if output_data['unwrapped_traj']['write_every_step']:
                kmc_step_index = 0
            if compute_mode != 'parallel':
                traj_dir_path = dst_path.joinpath(f'traj{traj_index+1}')
                Path.mkdir(traj_dir_path, parents=True, exist_ok=True)
            else:
                traj_dir_path = dst_path

            # Load random state
            random_state_file_path = traj_dir_path.joinpath(f'initial_rnd_state.dump')
            rnd.setstate(pickle.load(open(random_state_file_path, 'rb')))

            # Initialize data arrays
            write_time_data = 0
            for output_data_type, output_attributes in output_data.items():
                if output_attributes['write']:
                    if output_data_type == 'unwrapped_traj':
                        if output_data[output_data_type]['write_every_step']:
                            write_every_step = 1
                            unwrapped_position_array = np.zeros((1, self.total_species * 3))
                        else:
                            write_every_step = 0
                            unwrapped_position_array = np.zeros(
                                    (num_path_steps_per_traj, self.total_species * 3))
                    elif output_data_type == 'time':
                        time_data = [0.0]
                        write_time_data = 1
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

            if self.doping_active:
                self.system_relative_energies = np.copy(self.undoped_system_relative_energies)

                if 'pairwise' in self.doping['insertion_type']:
                    map_index = self.doping['insertion_type'].index('pairwise')
                    pairwise_insertion = self.doping['num_dopants'][map_index] != 0
                else:
                    pairwise_insertion = 0

                # Load doping distribution
                if pairwise_insertion:
                    site_indices_file_path = traj_dir_path.parent / 'site_indices.npy'
                else:
                    site_indices_file_path = traj_dir_path / 'site_indices.npy'
                site_indices_data = np.load(site_indices_file_path)
                site_indices_list = site_indices_data[:, 0]
                dopant_element_type_index_list = site_indices_data[:, 1]
                site_wise_shell_indices = site_indices_data[:, 3]
                dopant_site_indices = {}
                array_indices = np.where(site_wise_shell_indices == 0)[0]
                for array_index in array_indices:
                    site_index = int(site_indices_list[array_index])
                    dopant_element_type = self.dopant_element_types[dopant_element_type_index_list[array_index]]
                    if dopant_element_type in dopant_site_indices:
                        dopant_site_indices[dopant_element_type].append(site_index)
                    else:
                        dopant_site_indices[dopant_element_type] = [site_index]
                
                # update system_relative_energies
                num_site_indices = len(dopant_element_type_index_list)
                for index in range(num_site_indices):
                    site_index = site_indices_list[index]
                    shell_index = site_wise_shell_indices[index]
                    dopant_element_type = self.dopant_element_types[dopant_element_type_index_list[index]]
                    map_index = self.dopant_element_types.index(dopant_element_type)
                    substitution_element_type = self.substitution_element_types[map_index]
                    if shell_index < len(self.relative_energies['doping'][
                                        substitution_element_type][map_index]):
                        self.system_relative_energies[site_index] += (
                            self.relative_energies['doping'][
                                substitution_element_type][map_index][shell_index]
                            * constants.EV2HARTREE)
                occupancy_list = []
            else:
                dopant_site_indices = {}

            current_state_occupancy = self.generate_initial_occupancy(
                                                        dopant_site_indices)
            if self.doping_active:
                occupancy_list.append([index for index in current_state_occupancy])
            current_state_charge_config = self.charge_config(
                                current_state_occupancy, dopant_site_indices)
            current_state_charge_config_prod = np.multiply(
                                    current_state_charge_config.transpose(),
                                    current_state_charge_config)
            current_state_energy = (ewald_neut
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
                if self.doping_active:
                    occupancy_list.append([index for index in current_state_occupancy])
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

                if write_every_step:
                    if kmc_step_index == 0:
                        unwrapped_position_array = np.zeros(
                                                    (1, self.total_species * 3))
                    else:
                        unwrapped_position_array = np.vstack(
                                    (unwrapped_position_array,
                                     unwrapped_position_array[kmc_step_index-1]
                                     + species_displacement_vector_list))
                    kmc_step_index += 1
                    species_displacement_vector_list = np.zeros(
                                                    (1, self.total_species * 3))
                if write_time_data:
                    time_data.append(sim_time)

                # Update data arrays for each path step
                if end_path_index >= start_path_index + 1:
                    if end_path_index >= num_path_steps_per_traj:
                        end_path_index = num_path_steps_per_traj
                    if not write_every_step:
                        unwrapped_position_array[start_path_index:end_path_index] \
                            = (unwrapped_position_array[start_path_index-1]
                               + species_displacement_vector_list)
                    if output_data['energy']['write']:
                        energy_array[start_path_index:end_path_index] = \
                            current_state_energy
                    if not write_every_step:
                        species_displacement_vector_list = np.zeros(
                                                    (1, self.total_species * 3))
                    start_path_index = end_path_index

            # Write output data arrays to disk
            for output_data_type, output_attributes in output_data.items():
                if output_attributes['write']:
                    output_file_name = traj_dir_path.joinpath(
                                                output_attributes['file_name'])
                    with open(output_file_name, 'ab') as output_file:
                        if output_data_type == 'unwrapped_traj':
                            np.save(output_file, unwrapped_position_array)
                        elif output_data_type == 'time':
                            np.save(output_file, np.asarray(time_data))
                        elif output_data_type == 'wrapped_traj':
                            np.save(output_file, wrapped_position_array)
                        elif output_data_type == 'energy':
                            np.save(output_file, energy_array)
                        elif output_data_type == 'delg_0':
                            np.save(output_file, delg_0_array)
                        elif output_data_type == 'potential':
                            np.save(output_file, potential_array)
            if self.doping_active:
                output_file_path = traj_dir_path / 'occupancy.npy'
                occupancy_array = np.asarray(occupancy_list, dtype=int)
                np.save(output_file_path, occupancy_array)

        if self.electric_field_active:
            prefix_list = self.compute_drift_mobility(drift_velocity_array,
                                                      dst_path, prefix_list)

        file_name = 'Run'
        prefix = ''.join(prefix_list)
        print_time_elapsed = 1
        generate_report(self.start_time, dst_path, file_name, print_time_elapsed,
                        prefix)
        return None


class Analysis(object):
    """Post-simulation analysis methods"""
    def __init__(self, material_info, n_dim, species_count, n_traj, t_final,
                 time_interval, msd_t_final, trim_length, temp, repr_time='ns',
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
        self.temp = temp
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

        self.kBT = constants.KB * self.temp / constants.EV2J  # eV
        self.msd_t_final = msd_t_final / self.time_conversion
        self.num_msd_steps_per_traj = int(self.msd_t_final
                                          / self.time_interval) + 1

    def compute_msd(self, dst_path, output_data):
        """Returns the squared displacement of the trajectories
        :param dst_path:
        :return:
        """
        assert dst_path, 'Please provide the destination path where MSD ' \
                         'output files needs to be saved'

        coordinate_file_name = output_data['unwrapped_traj']['file_name']
        for traj_index in range(self.n_traj):
            traj_coordinate_file_path = (dst_path / f'traj{traj_index+1}' / coordinate_file_name)
            if traj_index == 0:
                position_array = np.load(traj_coordinate_file_path)
            else:
                position_array = np.vstack((position_array, np.load(traj_coordinate_file_path)))

        position_array = (
            position_array[
                :self.n_traj * self.num_path_steps_per_traj + 1].reshape(
                (self.n_traj * self.num_path_steps_per_traj,
                 self.total_species, 3))
            * self.dist_conversion)
        sd_array = np.zeros((self.n_traj, self.num_msd_steps_per_traj,
                             self.total_species))
        for traj_index in range(self.n_traj):
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
        species_avg_sd_array = np.zeros((self.n_traj,
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
                    / np.sqrt(self.n_traj))
        file_name = (('%1.2E' % (self.msd_t_final * self.time_conversion))
                     + str(self.repr_time)
                     + (',n_traj: %1.2E' % self.n_traj
                        if self.n_traj != self.n_traj else '')
                     + '_trim=' + str(self.trim_length))
        msd_file_name = ''.join(['MSD_Data_', file_name, '.npy'])
        msd_file_path = dst_path.joinpath(msd_file_name)
        species_types = [species_type
                         for index, species_type in enumerate(
                                            self.material.species_types)
                         if index not in non_existent_species_indices]
        np.save(msd_file_path, msd_data)

        report_file_name = ''.join(['MSD_Analysis',
                             ('_' if file_name else ''), file_name])
        slope_data = np.zeros((self.n_traj, num_existent_species))
        prefix_list = []
        for species_index, species_type in enumerate(species_types):
            for traj_index in range(self.n_traj):
                slope_data[traj_index, species_index], _, _, _, _ = \
                    linregress(msd_data[self.trim_length:-self.trim_length, 0],
                               species_avg_sd_array[traj_index,
                                                    self.trim_length:-self.trim_length,
                                                    species_index])
            slope = np.mean(slope_data[:, species_index])
            species_diff = (slope * constants.ANG2CM ** 2
                            * constants.SEC2NS / (2 * self.n_dim) / self.kBT)
            prefix_list.append(
                        f'Estimated value of {species_type} diffusivity is: {species_diff:.3e} cm2/Vs\n')
            slope_sem = (np.std(slope_data[:, species_index])
                         / np.sqrt(self.n_traj))
            species_diff_sem = (slope_sem * constants.ANG2CM ** 2
                                * constants.SEC2NS / (2 * self.n_dim) / self.kBT)
            prefix_list.append(
                f'Standard error of mean in {species_type} diffusivity is: {species_diff_sem:.3e} cm2/Vs\n')
        prefix = ''.join(prefix_list)
        print_time_elapsed = 1
        generate_report(self.start_time, dst_path, report_file_name,
                        print_time_elapsed, prefix)

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
            species_diff = (slope * constants.ANG2CM ** 2
                            * constants.SEC2NS / (2 * self.n_dim) / self.kBT)
            ax.add_artist(
                AnchoredText('Est. $D_{{%s}}$ = %.3e' % (species_type,
                                                          species_diff)
                             + ' $cm^2/Vs$; $r^2$=%.3e' % (r_value**2),
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
        figure_name = ''.join(['MSD_Plot_', file_name + '.png'])
        figure_path = dst_path.joinpath(figure_name)
        plt.savefig(str(figure_path))
        return None


class ReturnValues(object):
    """dummy class to return objects from methods defined inside
        other classes"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
