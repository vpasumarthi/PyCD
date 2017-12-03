#!/usr/bin/env python
"""
kMC model to run kinetic Monte Carlo simulations and compute mean
square displacement of random walk of charge carriers on 3D lattice
systems
"""
from pathlib import Path
from datetime import datetime
import random as rnd
import itertools
from copy import deepcopy
import pdb

import numpy as np

from PyCT.io import read_poscar


class Material(object):
    """Defines the properties and structure of working material
    :param str name: A string representing the material name
    :param list element_types: list of chemical elements
    :param dict species_to_element_type_map: list of charge carrier
                                                species
    :param unit_cell_coords: positions of all elements in the unit cell
    :type unit_cell_coords: np.array (nx3)
    :param element_type_index_list: list of element types for all
                                    unit cell coordinates
    :type element_type_index_list: np.array (n)
    :param dict charge_types: types of atomic charges considered for
                                the working material
    :param list lattice_parameters: list of three lattice constants in
                                    angstrom and three angles between
                                    them in degrees
    :param float vn: typical frequency for nuclear motion
    :param dict lambda_values: Reorganization energies
    :param dict VAB: Electronic coupling matrix element
    :param dict neighbor_cutoff_dist: List of neighbors and their
                                        respective cutoff distances in
                                        angstrom
    :param float neighbor_cutoff_dist_tol: Tolerance value in angstrom
                                            for neighbor cutoff
                                            distance
    :param str element_type_delimiter: Delimiter between element types
    :param str empty_species_type: name of the empty species type
    :param float epsilon: Dielectric constant of the material

    The additional attributes are:
        * **n_elements_per_unit_cell** (np.array (n)): element-type
                        wise total number of elements in a unit cell
        * **site_list** (list): list of elements that act as sites
        * **element_type_to_species_map** (dict): dictionary of element
                                                    to species mapping
        * **non_empty_species_to_element_type_map** (dict): dictionary
            of species to element mapping with elements excluding
            empty_species_type
        * **hop_element_types** (dict): dictionary of species to
            hopping element types separated by element_type_delimiter
        * **lattice_matrix** (np.array (3x3): lattice cell matrix
    """

    # CONSTANTS
    EPSILON0 = 8.854187817E-12  # Electric constant in F.m-1
    ANG = 1E-10  # Angstrom in m
    KB = 1.38064852E-23  # Boltzmann constant in J/K

    # FUNDAMENTAL ATOMIC UNITS
    # Source: http://physics.nist.gov/cuu/Constants/Table/allascii.txt
    EMASS = 9.10938356E-31  # Electron mass in Kg
    ECHARGE = 1.6021766208E-19  # Elementary charge in C
    HBAR = 1.054571800E-34  # Reduced Planck's constant in J.sec
    KE = 1 / (4 * np.pi * EPSILON0)

    # DERIVED ATOMIC UNITS
    # Bohr radius in m
    BOHR = HBAR**2 / (EMASS * ECHARGE**2 * KE)
    # Hartree in J
    HARTREE = HBAR**2 / (EMASS * BOHR**2)
    AUTIME = HBAR / HARTREE  # sec
    AUTEMPERATURE = HARTREE / KB  # K

    # CONVERSIONS
    EV2J = ECHARGE
    ANG2BOHR = ANG / BOHR
    ANG2UM = 1.00E-04
    J2HARTREE = 1 / HARTREE
    SEC2AUTIME = 1 / AUTIME
    SEC2NS = 1.00E+09
    SEC2PS = 1.00E+12
    SEC2FS = 1.00E+15
    K2AUTEMP = 1 / AUTEMPERATURE

    def __init__(self, material_parameters):

        # Read Input POSCAR
        [self.lattice_matrix, self.element_types,
         self.n_elements_per_unit_cell,
         self.total_elements_per_unit_cell,
         fractional_unit_cell_coords] = (
            read_poscar(material_parameters.input_coord_file_location))
        self.lattice_matrix *= self.ANG2BOHR
        self.n_element_types = len(self.element_types)
        self.element_type_index_list = np.repeat(
                                    np.arange(self.n_element_types),
                                    self.n_elements_per_unit_cell)

        self.name = material_parameters.name
        self.species_types = material_parameters.species_types[:]
        self.num_species_types = len(self.species_types)
        self.species_charge_list = deepcopy(
                            material_parameters.species_charge_list)
        self.species_to_element_type_map = deepcopy(
                    material_parameters.species_to_element_type_map)

        # Initialization
        self.fractional_unit_cell_coords = np.zeros(
                                    fractional_unit_cell_coords.shape)
        self.unit_cell_class_list = []
        start_index = 0
        # Reorder element-wise unit cell coordinates in ascending order
        # of z-coordinate
        for element_type_index in range(self.n_element_types):
            element_fract_unit_cell_coords = (
                                    fractional_unit_cell_coords[
                                        self.element_type_index_list
                                        == element_type_index])
            end_index = (start_index
                        + self.n_elements_per_unit_cell[
                                                element_type_index])
            self.fractional_unit_cell_coords[start_index:end_index] = (
                element_fract_unit_cell_coords[
                    element_fract_unit_cell_coords[:, 2].argsort()])
            element_type = self.element_types[element_type_index]
            self.unit_cell_class_list.extend(
                [material_parameters.class_list[element_type][index]
                 for index
                 in element_fract_unit_cell_coords[:, 2].argsort()])
            start_index = end_index

        self.unit_cell_coords = np.dot(
                self.fractional_unit_cell_coords, self.lattice_matrix)
        self.charge_types = deepcopy(material_parameters.charge_types)

        self.vn = material_parameters.vn / self.SEC2AUTIME
        self.lambda_values = deepcopy(
                                    material_parameters.lambda_values)
        self.lambda_values.update(
                            (x, [y[index] * self.EV2J * self.J2HARTREE
                                 for index in range(len(y))])
                            for x, y in self.lambda_values.items())

        self.VAB = deepcopy(material_parameters.VAB)
        self.VAB.update((x, [y[index] * self.EV2J * self.J2HARTREE
                             for index in range(len(y))])
                        for x, y in self.VAB.items())

        self.neighbor_cutoff_dist = deepcopy(
                            material_parameters.neighbor_cutoff_dist)
        self.neighbor_cutoff_dist.update(
                (x, [(y[index] * self.ANG2BOHR)
                     if y[index] else None for index in range(len(y))])
                for x, y in (self.neighbor_cutoff_dist.items()))
        self.neighbor_cutoff_dist_tol = deepcopy(
                        material_parameters.neighbor_cutoff_dist_tol)
        self.neighbor_cutoff_dist_tol.update(
                (x, [(y[index] * self.ANG2BOHR)
                     if y[index] else None for index in range(len(y))])
                for x, y in (self.neighbor_cutoff_dist_tol.items()))
        self.num_unique_hopping_distances = {
                key: len(value)
                for key, value in (self.neighbor_cutoff_dist.items())}

        self.element_type_delimiter = (
                            material_parameters.element_type_delimiter)
        self.empty_species_type = (
                                material_parameters.empty_species_type)
        self.dielectric_constant = (
                            material_parameters.dielectric_constant)

        self.num_classes = [
                len(set(material_parameters.class_list[element_type]))
                for element_type in self.element_types]
        self.delG0_shift_list = {
            key: [[(value[center_site_class_index][index] * self.EV2J
                    * self.J2HARTREE)
                   for index in range(
                            self.num_unique_hopping_distances[key])]
                  for center_site_class_index in range(len(value))]
            for key, value in (
                        material_parameters.delG0_shift_list.items())}

        site_list = [self.species_to_element_type_map[key]
                     for key in self.species_to_element_type_map
                     if key != self.empty_species_type]
        self.site_list = list(
            set([item for sublist in site_list for item in sublist]))
        self.non_empty_species_to_element_type_map = deepcopy(
                                    self.species_to_element_type_map)
        del self.non_empty_species_to_element_type_map[
                                            self.empty_species_type]

        self.element_type_to_species_map = {}
        for element_type in self.element_types:
            species_list = []
            for species_type_key in (
                    self.non_empty_species_to_element_type_map.keys()):
                if element_type in (
                    self.non_empty_species_to_element_type_map[
                                                    species_type_key]):
                    species_list.append(species_type_key)
            self.element_type_to_species_map[element_type] = (
                                                    species_list[:])

        self.hop_element_types = {
            key: [self.element_type_delimiter.join(comb)
                  for comb in list(itertools.product(
                              self.species_to_element_type_map[key],
                              repeat=2))]
            for key in self.species_to_element_type_map
            if key != self.empty_species_type}

    def generate_sites(self, element_type_indices,
                       cell_size=np.array([1, 1, 1])):
        """Returns system_element_indices and coordinates of
        specified elements in a cell of size *cell_size*

        :param str element_type_indices: element type indices
        :param cell_size: size of the cell
        :type cell_size: np.array (3x1)
        :return: an object with following attributes:

            * **cell_coordinates** (np.array (nx3)):
            * **quantum_index_list** (np.array (nx5)):
            * **system_element_index_list** (np.array (n)):

        :raises ValueError: if the input cell_size is less than
        or equal to 0.
        """
        assert all(size > 0 for size in cell_size), 'Input size \
                                    should always be greater than 0'
        extract_indices = np.in1d(self.element_type_index_list,
                                  element_type_indices).nonzero()[0]
        unit_cell_element_coords = self.unit_cell_coords[
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
        cell_coordinates = np.zeros((num_cells * n_sites_per_unit_cell,
                                     3))
        # Definition format of Quantum Indices
        # quantum_index = [unit_cell_index, element_type_index,
        #                 element_index]
        quantum_index_list = np.zeros(
                    (num_cells * n_sites_per_unit_cell, 5), dtype=int)
        system_element_index_list = np.zeros(
                        num_cells * n_sites_per_unit_cell, dtype=int)
        i_unit_cell = 0
        for x_index in range(cell_size[0]):
            for y_index in range(cell_size[1]):
                for z_index in range(cell_size[2]):
                    start_index = i_unit_cell * n_sites_per_unit_cell
                    end_index = start_index + n_sites_per_unit_cell
                    translation_vector = np.dot(
                                            [x_index, y_index, z_index],
                                            self.lattice_matrix)
                    cell_coordinates[start_index:end_index] = (
                        unit_cell_element_coords + translation_vector)
                    system_element_index_list[
                        start_index:end_index] = (
                                    i_unit_cell * n_sites_per_unit_cell
                                    + unit_cell_element_index_list)
                    quantum_index_list[start_index:end_index] = (
                        np.hstack(
                            (np.tile(
                                np.array([x_index, y_index, z_index]),
                                (n_sites_per_unit_cell, 1)),
                             unit_cell_element_type_index,
                             unit_cell_element_type_element_index_list)
                                  ))
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
        self.start_time = datetime.now()
        self.material = material
        self.system_size = system_size
        self.pbc = pbc[:]

        # total number of unit cells
        self.num_cells = self.system_size.prod()
        self.num_system_elements = (
                        self.num_cells
                        * self.material.total_elements_per_unit_cell)

        # generate all sites in the system
        self.element_type_indices = range(
                                        self.material.n_element_types)
        self.bulk_sites = self.material.generate_sites(
                        self.element_type_indices, self.system_size)

        x_range = range(-1, 2) if self.pbc[0] == 1 else [0]
        y_range = range(-1, 2) if self.pbc[1] == 1 else [0]
        z_range = range(-1, 2) if self.pbc[2] == 1 else [0]
        # Initialization
        self.system_translational_vector_list = np.zeros(
                                                (3**sum(self.pbc), 3))
        index = 0
        for x_offset in x_range:
            for y_offset in y_range:
                for z_offset in z_range:
                    self.system_translational_vector_list[index] = (
                        np.dot(np.multiply(
                            np.array([x_offset, y_offset, z_offset]),
                            system_size),
                               self.material.lattice_matrix))
                    index += 1

    def generate_system_element_index(self, system_size,
                                      quantum_indices):
        """Returns the system_element_index of the element"""
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
        system_element_index = (
            element_index + self.material.n_elements_per_unit_cell[
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

    def generate_quantum_indices(self, system_size,
                                 system_element_index):
        """Returns the quantum indices of the element"""
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
        quantum_indices[4] = (
                        unit_cell_element_index
                        - self.material.n_elements_per_unit_cell[
                                            :quantum_indices[3]].sum())
        n_filled_unit_cells = (
                    (system_element_index - unit_cell_element_index)
                    / self.material.total_elements_per_unit_cell)
        for index in range(3):
            quantum_indices[index] = (n_filled_unit_cells
                                      / system_size[index+1:].prod())
            n_filled_unit_cells -= (quantum_indices[index]
                                    * system_size[index+1:].prod())
        return quantum_indices

    def compute_coordinates(self, system_size, system_element_index):
        """Returns the coordinates in atomic units of the given
            system element index for a given system size"""
        quantum_indices = self.generate_quantum_indices(
                                    system_size, system_element_index)
        unit_cell_translation_vector = np.dot(
                    quantum_indices[:3], self.material.lattice_matrix)
        coordinates = (
                    unit_cell_translation_vector
                    + self.material.unit_cell_coords[
                            quantum_indices[4]
                            + self.material.n_elements_per_unit_cell[
                                        :quantum_indices[3]].sum()])
        return coordinates

    def compute_distance(self, system_size, system_element_index_1,
                         system_element_index_2):
        """Returns the distance in atomic units between the two
            system element indices for a given system size"""
        center_coord = self.compute_coordinates(system_size,
                                                system_element_index_1)
        neighbor_coord = self.compute_coordinates(
                                system_size, system_element_index_2)

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
        """Returns system_element_index_map and distances between
        center sites and its neighbor sites within cutoff distance"""
        neighbor_site_coords = bulk_sites.cell_coordinates[
                                                neighbor_site_indices]
        neighbor_site_system_element_index_list = (
                                bulk_sites.system_element_index_list[
                                                neighbor_site_indices])
        center_site_coords = bulk_sites.cell_coordinates[
                                                center_site_indices]

        neighbor_system_element_indices = np.empty(
                                len(center_site_coords), dtype=object)
        displacement_vector_list = np.empty(len(center_site_coords),
                                            dtype=object)
        num_neighbors = np.array([], dtype=int)

        if cutoff_dist_key == 'Fe:Fe':
            quick_test = 0  # commit reference: 1472bb4
        else:
            quick_test = 0

        for center_site_index, center_coord in enumerate(
                                                center_site_coords):
            i_neighbor_site_index_list = []
            i_displacement_vectors = []
            i_num_neighbors = 0
            if quick_test:
                displacement_list = np.zeros(len(neighbor_site_coords))
            for neighbor_site_index, neighbor_coord in enumerate(
                                                neighbor_site_coords):
                neighbor_image_coords = (
                                self.system_translational_vector_list
                                + neighbor_coord)
                neighbor_image_displacement_vectors = (
                                neighbor_image_coords - center_coord)
                neighbor_image_displacements = np.linalg.norm(
                                neighbor_image_displacement_vectors,
                                axis=1)
                [displacement, image_index] = [
                            np.min(neighbor_image_displacements),
                            np.argmin(neighbor_image_displacements)]
                if quick_test:
                    displacement_list[neighbor_site_index] \
                                                        = displacement
                if (cutoff_dist_limits[0] < displacement
                    <= cutoff_dist_limits[1]):
                    i_neighbor_site_index_list.append(
                                                neighbor_site_index)
                    i_displacement_vectors.append(
                                neighbor_image_displacement_vectors[
                                                        image_index])
                    i_num_neighbors += 1
            neighbor_system_element_indices[center_site_index] = (
                            neighbor_site_system_element_index_list[
                                        i_neighbor_site_index_list])
            displacement_vector_list[center_site_index] = (
                                    np.asarray(i_displacement_vectors))
            num_neighbors = np.append(num_neighbors, i_num_neighbors)
            if quick_test == 1:
                print(np.sort(displacement_list)[:10]
                      / self.material.ANG2BOHR)
                pdb.set_trace()
            elif quick_test == 2:
                for cutoff_dist in range(2, 7):
                    cutoff = cutoff_dist * self.material.ANG2BOHR
                    print(cutoff_dist)
                    print(displacement_list[displacement_list
                                            < cutoff].shape)
                    print(np.unique(np.sort(np.round(
                        displacement_list[displacement_list < cutoff]
                        / self.material.ANG2BOHR, 4))).shape)
                    print(np.unique(np.sort(np.round(
                        displacement_list[displacement_list < cutoff]
                        / self.material.ANG2BOHR, 3))).shape)
                    print(np.unique(np.sort(np.round(
                        displacement_list[displacement_list < cutoff]
                        / self.material.ANG2BOHR, 2))).shape)
                    print(np.unique(np.sort(np.round(
                        displacement_list[displacement_list < cutoff]
                        / self.material.ANG2BOHR, 1))).shape)
                    print(np.unique(np.sort(np.round(
                        displacement_list[displacement_list < cutoff]
                        / self.material.ANG2BOHR, 0))).shape)
                pdb.set_trace()

        return_neighbors = ReturnValues(
            neighbor_system_element_indices=\
                                    neighbor_system_element_indices,
            displacement_vector_list=displacement_vector_list,
            num_neighbors=num_neighbors)
        return return_neighbors

    def generate_cumulative_displacement_list(self, dst_path):
        """Returns cumulative displacement list for the given system
        size printed out to disk"""
        cumulative_displacement_list = np.zeros((
                self.num_system_elements, self.num_system_elements, 3))
        for center_site_index, center_coord in enumerate(
                                    self.bulk_sites.cell_coordinates):
            cumulative_system_translational_vector_list = np.tile(
                                            self.system_translational_vector_list,
                                            (self.num_system_elements, 1, 1))
            cumulative_neighbor_image_coords = (
                cumulative_system_translational_vector_list
                + np.tile(self.bulk_sites.cell_coordinates[
                                                    :, np.newaxis, :],
                          (1, len(
                              self.system_translational_vector_list),
                           1)))
            cumulative_neighbor_image_displacement_vectors = (
                                cumulative_neighbor_image_coords
                                - center_coord)
            cumulative_neighbor_image_displacements = np.linalg.norm(
                        cumulative_neighbor_image_displacement_vectors,
                        axis=2)
            cumulative_displacement_list[center_site_index] = \
                cumulative_neighbor_image_displacement_vectors[
                    np.arange(self.num_system_elements),
                    np.argmin(cumulative_neighbor_image_displacements,
                              axis=1)]
        cumulative_displacement_list_file_path = dst_path.joinpath(
                                    'cumulative_displacement_list.npy')
        np.save(cumulative_displacement_list_file_path,
                cumulative_displacement_list)
        return None

    def generate_neighbor_list(self, dst_path, report=1,
                               local_system_size=np.array([3, 3, 3])):
        """Adds the neighbor list to the system object and
            returns the neighbor list"""
        assert dst_path, \
            'Please provide the path to the parent directory of \
                neighbor list files'
        assert all(size >= 3 for size in local_system_size), \
            'Local system size in all dimensions should always be \
                greater than or equal to 3'

        Path.mkdir(dst_path, parents=True, exist_ok=True)
        hop_neighbor_list_file_path = dst_path.joinpath(
                                            'hop_neighbor_list.npy')

        hop_neighbor_list = {}
        tol_dist = self.material.neighbor_cutoff_dist_tol
        element_types = self.material.element_types[:]

        for cutoff_dist_key in (
                            self.material.neighbor_cutoff_dist.keys()):
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

            for i_cutoff_dist in range(len(cutoff_dist_list)):
                cutoff_dist_limits = (
                        [(cutoff_dist_list[i_cutoff_dist]
                          - tol_dist[cutoff_dist_key][i_cutoff_dist]),
                         (cutoff_dist_list[i_cutoff_dist]
                          + tol_dist[cutoff_dist_key][i_cutoff_dist])])

                neighbor_list_cutoff_dist_key.append(
                    self.hop_neighbor_sites(local_bulk_sites,
                                            center_site_indices,
                                            neighbor_site_indices,
                                            cutoff_dist_limits,
                                            cutoff_dist_key))
            hop_neighbor_list[cutoff_dist_key] = (
                                    neighbor_list_cutoff_dist_key[:])
        np.save(hop_neighbor_list_file_path, hop_neighbor_list)

        if report:
            self.generate_neighbor_list_report(dst_path)
        return None

    def generate_neighbor_list_report(self, dst_path):
        """Generates a neighbor list and prints out a
            report to the output directory"""
        neighbor_list_log_name = 'neighbor_list.log'
        neighbor_list_log_path = dst_path.joinpath(
                                                neighbor_list_log_name)
        report = open(neighbor_list_log_path, 'w')
        end_time = datetime.now()
        time_elapsed = end_time - self.start_time
        report.write(
            'Time elapsed: ' + ('%2d days, ' % time_elapsed.days
                                if time_elapsed.days else '')
            + ('%2d hours' % ((time_elapsed.seconds // 3600) % 24))
            + (', %2d minutes' % ((time_elapsed.seconds // 60) % 60))
            + (', %2d seconds' % (time_elapsed.seconds % 60)))
        report.close()
        return None

    # TODO: remove the function
    def generate_hematite_neighbor_se_indices(self, dst_path,
                                              report=1):
        start_time = datetime.now()
        offset_list = np.array(
            [[[-1, 0, -1], [0, 0, -1], [0, 1, -1], [0, 0, -1]],
             [[-1, -1, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, -1]],
             [[0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0]],
             [[-1, -1, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0]],
             [[0, -1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0]],
             [[-1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0]],
             [[-1, -1, 0], [-1, 0, 0], [0, 0, 0], [0, 0, 0]],
             [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]],
             [[0, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]],
             [[-1, -1, 0], [0, -1, 0], [0, 0, 0], [0, 0, 0]],
             [[0, -1, 1], [0, 0, 1], [1, 0, 1], [0, 0, 1]]])
        element_type_index = 0
        basal_neighbor_element_site_indices = np.array(
                                [11, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 0])
        c_neighbor_element_site_indices = np.array(
                                [9, 4, 11, 6, 1, 8, 3, 10, 5, 0, 7, 2])
        num_basal_neighbors = 3
        num_c_neighbors = 1
        num_neighbors = num_basal_neighbors + num_c_neighbors
        n_elements_per_unit_cell = (
            self.material.n_elements_per_unit_cell[element_type_index])
        neighbor_element_site_indices = np.zeros(
                                    (n_elements_per_unit_cell, 4), int)
        for i_neighbor in range(num_neighbors):
            if i_neighbor < num_basal_neighbors:
                neighbor_element_site_indices[:, i_neighbor] = \
                                    basal_neighbor_element_site_indices
            else:
                neighbor_element_site_indices[:, i_neighbor] = \
                                        c_neighbor_element_site_indices
        system_element_index_offset_array = (
            np.repeat(np.arange(
                        0, (self.material.total_elements_per_unit_cell
                            * self.num_cells),
                        self.material.total_elements_per_unit_cell),
                      self.material.n_elements_per_unit_cell[
                                                element_type_index]))
        center_site_se_indices = (
            np.tile(self.material.n_elements_per_unit_cell[
                                            :element_type_index].sum()
                    + np.arange(
                        0, self.material.n_elements_per_unit_cell[
                                                element_type_index]),
                    self.num_cells)
            + system_element_index_offset_array)
        num_center_site_elements = len(center_site_se_indices)
        neighbor_system_element_indices = np.zeros(
                            (num_center_site_elements, num_neighbors))

        for center_site_index, center_site_se_index in enumerate(
                                            center_site_se_indices):
            center_site_quantum_indices = (
                self.generate_quantum_indices(self.system_size,
                                              center_site_se_index))
            center_site_unit_cell_indices = \
                                        center_site_quantum_indices[:3]
            center_site_element_site_index = \
                                    center_site_quantum_indices[-1:][0]
            for neighbor_index in range(num_neighbors):
                neighbor_unit_cell_indices = (
                    center_site_unit_cell_indices
                    + offset_list[center_site_element_site_index][
                                                    neighbor_index])
                for index, neighbor_unit_cell_index in enumerate(
                                        neighbor_unit_cell_indices):
                    if neighbor_unit_cell_index < 0:
                        neighbor_unit_cell_indices[index] += \
                                                self.system_size[index]
                    elif neighbor_unit_cell_index >= self.system_size[
                                                                index]:
                        neighbor_unit_cell_indices[index] -= \
                                                self.system_size[index]
                    neighbor_quantum_indices = np.hstack((
                        neighbor_unit_cell_indices, element_type_index,
                        neighbor_element_site_indices[
                                center_site_element_site_index][
                                                    neighbor_index]))
                    neighbor_se_index = (
                                    self.generate_system_element_index(
                                            self.system_size,
                                            neighbor_quantum_indices))
                    neighbor_system_element_indices[center_site_index][
                                    neighbor_index] = neighbor_se_index

        file_name = 'neighbor_system_element_indices.npy'
        neighbor_system_element_indices_file_path = dst_path.joinpath(
                                                            file_name)
        np.save(neighbor_system_element_indices_file_path,
                neighbor_system_element_indices)
        if report:
            self.generate_hematite_neighbor_se_indices_report(
                                                dst_path, start_time)
        return None

    def generate_hematite_neighbor_se_indices_report(self, dst_path,
                                                     start_time):
        """Generates a neighbor list and prints out a
            report to the output directory"""
        neighbor_system_element_indices_log_name = \
            'neighbor_system_element_indices.log'
        neighbor_system_element_indices_log_path = dst_path.joinpath(
                            neighbor_system_element_indices_log_name)
        report = open(neighbor_system_element_indices_log_path, 'w')
        end_time = datetime.now()
        time_elapsed = end_time - start_time
        report.write(
            'Time elapsed: ' + ('%2d days, ' % time_elapsed.days
                                if time_elapsed.days else '')
            + ('%2d hours' % ((time_elapsed.seconds // 3600) % 24))
            + (', %2d minutes' % ((time_elapsed.seconds // 60) % 60))
            + (', %2d seconds' % (time_elapsed.seconds % 60)))
        report.close()
        return None

    def generate_species_site_sd_list(self,
                                      center_site_quantum_indices,
                                      dst_path, report=1):
        start_time = datetime.now()
        element_type_index = center_site_quantum_indices[3]
        center_site_se_index = self.generate_system_element_index(
                                        self.system_size,
                                        center_site_quantum_indices)
        system_element_index_offset_array = np.repeat(
            np.arange(0, (self.material.total_elements_per_unit_cell
                          * self.num_cells),
                      self.material.total_elements_per_unit_cell),
            self.material.n_elements_per_unit_cell[element_type_index])
        neighbor_site_se_indices = (
            np.tile(self.material.n_elements_per_unit_cell[
                                            :element_type_index].sum()
                    + np.arange(
                        0, self.material.n_elements_per_unit_cell[
                                                element_type_index]),
                    self.num_cells)
            + system_element_index_offset_array)
        species_site_sd_list = np.zeros(len(neighbor_site_se_indices))
        for neighbor_site_index, neighbor_site_se_index in enumerate(
                                            neighbor_site_se_indices):
            species_site_sd_list[neighbor_site_index] = (
                    self.compute_distance(self.system_size,
                                          center_site_se_index,
                                          neighbor_site_se_index)**2)
        species_site_sd_list /= self.material.ANG2BOHR**2
        file_name = 'species_site_sd_list.npy'
        species_site_sd_list_file_path = dst_path.joinpath(file_name)
        np.save(species_site_sd_list_file_path, species_site_sd_list)
        if report:
            self.generate_species_site_sd_list_report(dst_path,
                                                      start_time)
        return None

    def generate_species_site_sd_list_report(self, dst_path,
                                             start_time):
        """Generates a neighbor list and prints out a
            report to the output directory"""
        species_site_sd_list_log_name = 'species_site_sd_list.log'
        species_site_sd_list_log_path = dst_path.joinpath(
                                        species_site_sd_list_log_name)
        report = open(species_site_sd_list_log_path, 'w')
        end_time = datetime.now()
        time_elapsed = end_time - start_time
        report.write(
            'Time elapsed: ' + ('%2d days, ' % time_elapsed.days
                                if time_elapsed.days else '')
            + ('%2d hours' % ((time_elapsed.seconds // 3600) % 24))
            + (', %2d minutes' % ((time_elapsed.seconds // 60) % 60))
            + (', %2d seconds' % (time_elapsed.seconds % 60)))
        report.close()
        return None

    def generate_transition_prob_matrix(
            self, neighbor_system_element_indices, dst_path, report=1):
        start_time = datetime.now()
        element_type_index = 0
        num_neighbors = len(neighbor_system_element_indices[0])
        num_basal_neighbors = 3
        # num_c_neighbors = 1
        T = 300 * self.material.K2AUTEMP

        hop_element_type = 'Fe:Fe'
        k_list = np.zeros(num_neighbors)
        delG0 = 0
        for neighbor_index in range(num_neighbors):
            if neighbor_index < num_basal_neighbors:
                hop_dist_type = 0
            else:
                hop_dist_type = 1
            lambda_value = self.material.lambda_values[
                                    hop_element_type][hop_dist_type]
            VAB = self.material.VAB[hop_element_type][hop_dist_type]
            delGs = ((lambda_value + delG0) ** 2
                     / (4 * lambda_value)) - VAB
            k_list[neighbor_index] = self.material.vn * np.exp(-delGs
                                                               / T)

        k_total = np.sum(k_list)
        prob_list = k_list / k_total

        system_element_index_offset_array = np.repeat(
            np.arange(0, (self.material.total_elements_per_unit_cell
                          * self.num_cells),
                      self.material.total_elements_per_unit_cell),
            self.material.n_elements_per_unit_cell[element_type_index])
        neighbor_site_se_indices = (
            np.tile(self.material.n_elements_per_unit_cell[
                                            :element_type_index].sum()
                    + np.arange(
                        0, self.material.n_elements_per_unit_cell[
                                                element_type_index]),
                    self.num_cells)
            + system_element_index_offset_array)

        num_element_type_sites = len(neighbor_system_element_indices)
        transition_prob_matrix = np.zeros((num_element_type_sites,
                                           num_element_type_sites))
        for center_site_index in range(num_element_type_sites):
            for neighbor_index in range(num_neighbors):
                neighbor_site_index = np.where(
                    neighbor_site_se_indices
                    == neighbor_system_element_indices[
                            center_site_index][neighbor_index])[0][0]
                transition_prob_matrix[center_site_index][
                    neighbor_site_index] = prob_list[neighbor_index]
        file_name = 'transition_prob_matrix.npy'
        transition_prob_matrix_file_path = dst_path.joinpath(file_name)
        np.save(transition_prob_matrix_file_path,
                transition_prob_matrix)
        if report:
            self.generate_transition_prob_matrix_list_report(dst_path, start_time)
        return None

    def generate_transition_prob_matrix_list_report(self, dst_path,
                                                    start_time):
        """Generates a neighbor list and prints out a report to the
            output directory"""
        transition_prob_matrix_log_name = 'transition_prob_matrix.log'
        transition_prob_matrix_log_path = dst_path.joinpath(
                                    transition_prob_matrix_log_name)
        report = open(transition_prob_matrix_log_path, 'w')
        end_time = datetime.now()
        time_elapsed = end_time - start_time
        report.write(
            'Time elapsed: ' + ('%2d days, ' % time_elapsed.days
                                if time_elapsed.days else '')
            + ('%2d hours' % ((time_elapsed.seconds // 3600) % 24))
            + (', %2d minutes' % ((time_elapsed.seconds // 60) % 60))
            + (', %2d seconds' % (time_elapsed.seconds % 60)))
        report.close()
        return None

    def generate_msd_analytical_data(
                    self, transition_prob_matrix, species_site_sd_list,
                    center_site_quantum_indices, analytical_t_final,
                    analytical_time_interval, dst_path, report=1):
        start_time = datetime.now()

        file_name = '%1.2Ens' % analytical_t_final
        msd_analytical_data_file_name = ('MSD_Analytical_Data_'
                                         + file_name + '.dat')
        msd_analytical_data_file_path = dst_path.joinpath(
                                        msd_analytical_data_file_name)
        open(msd_analytical_data_file_path, 'w').close()

        element_type_index = 0
        num_data_points = int(analytical_t_final
                              / analytical_time_interval) + 1
        msd_data = np.zeros((num_data_points, 2))
        msd_data[:, 0] = np.arange(
                    0, analytical_t_final + analytical_time_interval,
                    analytical_time_interval)

        system_element_index_offset_array = np.repeat(
            np.arange(0, (self.material.total_elements_per_unit_cell
                          * self.num_cells),
                      self.material.total_elements_per_unit_cell),
            self.material.n_elements_per_unit_cell[element_type_index])
        center_site_se_indices = (
            np.tile(self.material.n_elements_per_unit_cell[
                                            :element_type_index].sum()
                    + np.arange(
                        0, self.material.n_elements_per_unit_cell[
                                                element_type_index]),
                    self.num_cells)
            + system_element_index_offset_array)

        center_site_se_index = self.generate_system_element_index(
                        self.system_size, center_site_quantum_indices)
        num_basal_neighbors = 3
        num_c_neighbors = 1
        num_neighbors = num_basal_neighbors + num_c_neighbors
        T = 300 * self.material.K2AUTEMP

        hop_element_type = 'Fe:Fe'
        k_list = np.zeros(num_neighbors)
        delG0 = 0
        for neighbor_index in range(num_neighbors):
            if neighbor_index < num_basal_neighbors:
                hop_dist_type = 0
            else:
                hop_dist_type = 1
            lambda_value = self.material.lambda_values[
                                    hop_element_type][hop_dist_type]
            VAB = self.material.VAB[hop_element_type][hop_dist_type]
            delGs = ((lambda_value + delG0) ** 2
                     / (4 * lambda_value)) - VAB
            k_list[neighbor_index] = self.material.vn * np.exp(-delGs
                                                               / T)

        k_total = np.sum(k_list)
        time_step = (self.material.SEC2NS / k_total
                     / self.material.SEC2AUTIME)

        sim_time = 0
        start_index = 0
        row_index = np.where(center_site_se_indices
                             == center_site_se_index)
        new_transition_prob_matrix = np.copy(transition_prob_matrix)
        with open(msd_analytical_data_file_path, 'a') as \
                                            msd_analytical_data_file:
            np.savetxt(msd_analytical_data_file,
                       msd_data[start_index, :][None, :])
        while True:
            new_transition_prob_matrix = np.dot(
                    new_transition_prob_matrix, transition_prob_matrix)
            sim_time += time_step
            end_index = int(sim_time / analytical_time_interval)
            if end_index >= start_index + 1:
                msd_data[end_index, 1] = np.dot(
                                new_transition_prob_matrix[row_index],
                                species_site_sd_list)
                with open(msd_analytical_data_file_path, 'a') as \
                                            msd_analytical_data_file:
                    np.savetxt(msd_analytical_data_file,
                               msd_data[end_index, :][None, :])
                start_index += 1
                if end_index == num_data_points - 1:
                    break

        if report:
            self.generate_msd_analytical_data_report(file_name,
                                                     dst_path,
                                                     start_time)
        return_msd_data = ReturnValues(msd_data=msd_data)
        return return_msd_data

    def generate_msd_analytical_data_report(self, file_name, dst_path,
                                            start_time):
        """Generates a neighbor list and prints out a report to the
            output directory"""
        msd_analytical_data_log_name = ('MSD_Analytical_Data_'
                                        + file_name + '.log')
        msd_analytical_data_log_path = dst_path.joinpath(
                                        msd_analytical_data_log_name)
        report = open(msd_analytical_data_log_path, 'w')
        end_time = datetime.now()
        time_elapsed = end_time - start_time
        report.write(
            'Time elapsed: ' + ('%2d days, ' % time_elapsed.days
                                if time_elapsed.days else '')
            + ('%2d hours' % ((time_elapsed.seconds // 3600) % 24))
            + (', %2d minutes' % ((time_elapsed.seconds // 60) % 60))
            + (', %2d seconds' % (time_elapsed.seconds % 60)))
        report.close()
        return None


class System(object):
    """defines the system we are working on

    Attributes:
    size: An array (3 x 1) defining the system size in multiple of
    unit cells
    """
    # @profile
    def __init__(self, material_info, material_neighbors,
                 hop_neighbor_list, cumulative_displacement_list,
                 species_count, alpha, n_max, k_max):
        """Return a system object whose size is *size*"""
        self.start_time = datetime.now()

        self.material = material_info
        self.neighbors = material_neighbors
        self.hop_neighbor_list = hop_neighbor_list

        self.pbc = self.neighbors.pbc
        self.species_count = species_count

        # total number of unit cells
        self.system_size = self.neighbors.system_size
        self.num_cells = self.system_size.prod()

        self.cumulative_displacement_list = \
                                        cumulative_displacement_list

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
            np.tile(self.material.unit_cell_class_list, self.num_cells)
            - 1)

        # ewald parameters:
        self.alpha = alpha
        self.n_max = n_max
        self.k_max = k_max

    def generate_random_occupancy(self, species_count):
        """generates initial occupancy list based on species count"""
        occupancy = []
        for species_type_index, num_species in enumerate(
                                                        species_count):
            species_type = self.material.species_types[
                                                    species_type_index]
            species_site_element_list = (
                            self.material.species_to_element_type_map[
                                                        species_type])
            species_site_element_type_index_list = [
                self.material.element_types.index(species_site_element)
                for species_site_element in species_site_element_list]
            species_site_indices = []
            for species_site_element_type_index in (
                                species_site_element_type_index_list):
                system_element_index_offset_array = np.repeat(
                    np.arange(
                        0, (self.material.total_elements_per_unit_cell
                            * self.num_cells),
                        self.material.total_elements_per_unit_cell),
                    self.material.n_elements_per_unit_cell[
                                    species_site_element_type_index])
                site_indices = (
                    np.tile(self.material.n_elements_per_unit_cell[
                                :species_site_element_type_index].sum()
                            + np.arange(
                                0,
                                self.material.n_elements_per_unit_cell[
                                    species_site_element_type_index]),
                            self.num_cells)
                                + system_element_index_offset_array)
                species_site_indices.extend(list(site_indices))
            occupancy.extend(rnd.sample(species_site_indices,
                                        num_species)[:])
        return occupancy

    def charge_config(self, occupancy, ion_charge_type,
                      species_charge_type):
        """Returns charge distribution of the current configuration"""

        # generate lattice charge list
        unit_cell_charge_list = np.array(
            [self.material.charge_types[ion_charge_type][
                    self.material.element_types[element_type_index]]
             for element_type_index in (
                            self.material.element_type_index_list)])
        charge_list = np.tile(unit_cell_charge_list, self.num_cells)[
                                                        :, np.newaxis]

        for species_type_index in range(
                                    self.material.num_species_types):
            start_index = 0 + self.species_count[
                                            :species_type_index].sum()
            end_index = start_index + self.species_count[
                                                    species_type_index]
            center_site_system_element_indices = occupancy[
                                            start_index:end_index][:]
            charge_list[center_site_system_element_indices] += (
                self.material.species_charge_list[species_charge_type][
                                                species_type_index])
        return charge_list

    def ewald_sum_setup(self, outdir=None):
        from scipy.special import erfc
        sqrt_alpha = np.sqrt(self.alpha)
        alpha4 = 4 * self.alpha
        fourier_sum_coeff = (2 * np.pi) / self.system_volume
        precomputed_array = np.zeros((
                                self.neighbors.num_system_elements,
                                self.neighbors.num_system_elements))

        for i in range(-self.n_max, self.n_max+1):
            for j in range(-self.n_max, self.n_max+1):
                for k in range(-self.n_max, self.n_max+1):
                    temp_array = (
                            np.linalg.norm(
                                (self.cumulative_displacement_list
                                 + np.dot(np.array([i, j, k]),
                                          self.translational_matrix)),
                                           axis=2))
                    precomputed_array += erfc(sqrt_alpha
                                              * temp_array) / 2

                    if np.all(np.array([i, j, k]) == 0):
                        for a in range(
                                self.neighbors.num_system_elements):
                            for b in range(
                                self.neighbors.num_system_elements):
                                if a != b:
                                    precomputed_array[a][b] /= (
                                                    temp_array[a][b])
                    else:
                        precomputed_array /= temp_array

        for i in range(-self.k_max, self.k_max+1):
            for j in range(-self.k_max, self.k_max+1):
                for k in range(-self.k_max, self.k_max+1):
                    if not np.all(np.array([i, j, k]) == 0):
                        k_vector = np.dot(
                                        np.array([i, j, k]),
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

        if outdir:
            self.generate_precomputed_array_log_report(outdir)
        return precomputed_array

    def generate_precomputed_array_log_report(self, outdir):
        """Generates an log report of the simulation and outputs
            to the working directory"""
        precomputed_array_log_file_name = 'precomputed_array.log'
        precomputed_array_log_file_path = outdir.joinpath(
                                    precomputed_array_log_file_name)
        report = open(precomputed_array_log_file_path, 'w')
        end_time = datetime.now()
        time_elapsed = end_time - self.start_time
        report.write(
            'Time elapsed: ' + ('%2d days, ' % time_elapsed.days
                                if time_elapsed.days else '')
            + ('%2d hours' % ((time_elapsed.seconds // 3600) % 24))
            + (', %2d minutes' % ((time_elapsed.seconds // 60) % 60))
            + (', %2d seconds' % (time_elapsed.seconds % 60)))
        report.close()
        return None


class Run(object):
    """defines the subroutines for running Kinetic Monte Carlo and
        computing electrostatic interaction energies"""
    def __init__(self, system, precomputed_array, T, ion_charge_type,
                 species_charge_type, n_traj, t_final, time_interval):
        """Returns the PBC condition of the system"""
        self.start_time = datetime.now()

        self.system = system
        self.material = self.system.material
        self.neighbors = self.system.neighbors
        self.precomputed_array = precomputed_array
        self.T = T * self.material.K2AUTEMP
        self.ion_charge_type = ion_charge_type
        self.species_charge_type = species_charge_type
        self.n_traj = int(n_traj)
        self.t_final = t_final * self.material.SEC2AUTIME
        self.time_interval = time_interval * self.material.SEC2AUTIME

        self.system_size = self.system.system_size

        # n_elements_per_unit_cell
        self.headStart_nElementsPerUnitCellCumSum = [
                self.material.n_elements_per_unit_cell[:siteElementTypeIndex].sum()
                for siteElementTypeIndex in self.neighbors.element_type_indices]

        # speciesTypeList
        self.speciesTypeList = [
                        self.material.species_types[index]
                        for index, value in enumerate(self.system.species_count)
                        for _ in range(value)]
        self.speciesTypeIndexList = [
                        index
                        for index, value in enumerate(self.system.species_count)
                        for _ in range(value)]
        self.species_charge_list = [
                self.material.species_charge_list[self.species_charge_type][index]
                for index in self.speciesTypeIndexList]
        self.hopElementTypeList = [
                                self.material.hop_element_types[species_type][0]
                                for species_type in self.speciesTypeList]
        self.lenHopDistTypeList = [
                        len(self.material.neighbor_cutoff_dist[hop_element_type])
                        for hop_element_type in self.hopElementTypeList]
        # number of kinetic processes
        self.nProc = 0
        self.nProcHopElementTypeList = []
        self.nProcSpeciesIndexList = []
        self.nProcSiteElementTypeIndexList = []
        self.nProcLambdaValueList = []
        self.nProcVABList = []
        for hopElementTypeIndex, hop_element_type in enumerate(
                                                    self.hopElementTypeList):
            center_element_type = hop_element_type.split(
                                        self.material.element_type_delimiter)[0]
            species_type_index = self.material.species_types.index(
                self.material.element_type_to_species_map[center_element_type][0])
            center_site_element_type_index = self.material.element_types.index(
                                                            center_element_type)
            for hopDistTypeIndex in range(self.lenHopDistTypeList[
                                                        hopElementTypeIndex]):
                if self.system.species_count[species_type_index] != 0:
                    num_neighbors = self.system.hop_neighbor_list[hop_element_type][
                                                hopDistTypeIndex].num_neighbors
                    self.nProc += num_neighbors[0]
                    self.nProcHopElementTypeList.extend([hop_element_type]
                                                        * num_neighbors[0])
                    self.nProcSpeciesIndexList.extend([hopElementTypeIndex]
                                                      * num_neighbors[0])
                    self.nProcSiteElementTypeIndexList.extend(
                            [center_site_element_type_index] * num_neighbors[0])
                    self.nProcLambdaValueList.extend(
                            [self.material.lambda_values[hop_element_type][
                                                        hopDistTypeIndex]]
                            * num_neighbors[0])
                    self.nProcVABList.extend(
                                    [self.material.VAB[hop_element_type][
                                                        hopDistTypeIndex]]
                                    * num_neighbors[0])

        # system coordinates
        self.systemCoordinates = self.neighbors.bulk_sites.cell_coordinates

        # total number of species
        self.totalSpecies = self.system.species_count.sum()

    def do_kmc_steps(self, outdir, report=1, randomSeed=1):
        """Subroutine to run the KMC simulation by specified number of steps"""
        assert outdir, 'Please provide the destination path where \
                        simulation output files needs to be saved'

        excess = 0
        energy = 1
        unwrappedTrajFileName = outdir.joinpath('unwrappedTraj.dat')
        open(unwrappedTrajFileName, 'wb').close()
        if energy:
            energyTrajFileName = outdir.joinpath('energyTraj.dat')
            open(energyTrajFileName, 'wb').close()

        if excess:
            wrappedTrajFileName = outdir.joinpath('wrappedTraj.dat')
            delG0TrajFileName = outdir.joinpath('delG0Traj.dat')
            potentialTrajFileName = outdir.joinpath('potentialTraj.dat')
            open(wrappedTrajFileName, 'wb').close()
            open(delG0TrajFileName, 'wb').close()
            open(potentialTrajFileName, 'wb').close()

        rnd.seed(randomSeed)
        n_traj = self.n_traj
        numPathStepsPerTraj = int(self.t_final / self.time_interval) + 1
        unwrappedPositionArray = np.zeros((numPathStepsPerTraj,
                                           self.totalSpecies * 3))
        if energy:
            energyArray = np.zeros(numPathStepsPerTraj)

        if excess:
            wrappedPositionArray = np.zeros((numPathStepsPerTraj,
                                             self.totalSpecies * 3))
            delG0Array = np.zeros(self.kmcSteps)
            potentialArray = np.zeros((numPathStepsPerTraj,
                                       self.totalSpecies))
        k_list = np.zeros(self.nProc)
        neighbor_site_system_element_index_list = np.zeros(self.nProc, dtype=int)
        nProcHopDistTypeList = np.zeros(self.nProc, dtype=int)
        rowIndexList = np.zeros(self.nProc, dtype=int)
        neighborIndexList = np.zeros(self.nProc, dtype=int)
        systemCharge = np.dot(
                    self.system.species_count,
                    self.material.species_charge_list[self.species_charge_type])

        ewaldNeut = - (np.pi
                       * (systemCharge**2)
                       / (2 * self.system.system_volume * self.system.alpha))
        precomputed_array = self.precomputed_array
        for _ in range(n_traj):
            currentStateOccupancy = self.system.generate_random_occupancy(
                                                    self.system.species_count)
            currentStateChargeConfig = self.system.charge_config(
                                                        currentStateOccupancy,
                                                        self.ion_charge_type,
                                                        self.species_charge_type)
            currentStateChargeConfigProd = np.multiply(
                                        currentStateChargeConfig.transpose(),
                                        currentStateChargeConfig)
            ewaldSelf = - (np.sqrt(self.system.alpha / np.pi)
                           * np.einsum('ii', currentStateChargeConfigProd))
            currentStateEnergy = (ewaldNeut + ewaldSelf
                                  + np.sum(np.multiply(
                                      currentStateChargeConfigProd,
                                      precomputed_array)))
            startPathIndex = 1
            endPathIndex = startPathIndex + 1
            if energy:
                energyArray[0] = currentStateEnergy
            # TODO: How to deal excess flag?
            # if excess:
            #     # TODO: Avoid using flatten
            #     wrappedPositionArray[pathIndex] = self.systemCoordinates[
            #                                 currentStateOccupancy].flatten()
            speciesDisplacementVectorList = np.zeros((1,
                                                      self.totalSpecies * 3))
            sim_time = 0
            breakFlag = 0
            while True:
                iProc = 0
                delG0List = []
                for speciesIndex, speciesSiteSystemElementIndex in enumerate(
                                                        currentStateOccupancy):
                    # TODO: Avoid re-defining speciesIndex
                    speciesIndex = self.nProcSpeciesIndexList[iProc]
                    hop_element_type = self.nProcHopElementTypeList[iProc]
                    siteElementTypeIndex = self.nProcSiteElementTypeIndexList[
                                                                        iProc]
                    row_index = (speciesSiteSystemElementIndex
                                // self.material.total_elements_per_unit_cell
                                * self.material.n_elements_per_unit_cell[
                                                        siteElementTypeIndex]
                                + speciesSiteSystemElementIndex
                                % self.material.total_elements_per_unit_cell
                                - self.headStart_nElementsPerUnitCellCumSum[
                                                        siteElementTypeIndex])
                    for hop_dist_type in range(self.lenHopDistTypeList[
                                                                speciesIndex]):
                        localNeighborSiteSystemElementIndexList = (
                                self.system.hop_neighbor_list[hop_element_type][
                                    hop_dist_type].neighbor_system_element_indices[
                                                                    row_index])
                        for neighbor_index, neighborSiteSystemElementIndex in \
                                enumerate(
                                    localNeighborSiteSystemElementIndexList):
                            # TODO: Introduce If condition
                            # if neighborSystemElementIndex not in
                            # currentStateOccupancy: commit 898baa8
                            neighbor_site_system_element_index_list[iProc] = \
                                neighborSiteSystemElementIndex
                            nProcHopDistTypeList[iProc] = hop_dist_type
                            rowIndexList[iProc] = row_index
                            neighborIndexList[iProc] = neighbor_index
                            # TODO: Print out a prompt about the assumption;
                            # detailed comment here. <Using species charge to
                            # compute change in energy> May be print log report
                            delG0Ewald = (
                                self.species_charge_list[speciesIndex]
                                * (2
                                   * np.dot(currentStateChargeConfig[:, 0],
                                            (precomputed_array[
                                                neighborSiteSystemElementIndex,
                                                :]
                                             - precomputed_array[
                                                 speciesSiteSystemElementIndex,
                                                 :]))
                                   + self.species_charge_list[speciesIndex]
                                   * (precomputed_array[
                                       speciesSiteSystemElementIndex,
                                       speciesSiteSystemElementIndex]
                                      + precomputed_array[
                                          neighborSiteSystemElementIndex,
                                          neighborSiteSystemElementIndex]
                                      - 2 * precomputed_array[
                                          speciesSiteSystemElementIndex,
                                          neighborSiteSystemElementIndex])))
                            classIndex = (self.system.system_class_index_list[
                                                speciesSiteSystemElementIndex])
                            delG0 = (
                                delG0Ewald
                                + self.material.delG0_shift_list[
                                    self.nProcHopElementTypeList[iProc]][
                                        classIndex][hop_dist_type])
                            delG0List.append(delG0)
                            lambda_value = self.nProcLambdaValueList[iProc]
                            VAB = self.nProcVABList[iProc]
                            delGs = (((lambda_value + delG0) ** 2
                                      / (4 * lambda_value)) - VAB)
                            k_list[iProc] = self.material.vn * np.exp(-delGs
                                                                     / self.T)
                            iProc += 1

                k_total = sum(k_list)
                kCumSum = (k_list / k_total).cumsum()
                rand1 = rnd.random()
                procIndex = np.where(kCumSum > rand1)[0][0]
                rand2 = rnd.random()
                sim_time -= np.log(rand2) / k_total

                # TODO: Address pre-defining excess data arrays
                # if excess:
                #    delG0Array[step] = delG0List[procIndex]
                speciesIndex = self.nProcSpeciesIndexList[procIndex]
                hop_element_type = self.nProcHopElementTypeList[procIndex]
                hop_dist_type = nProcHopDistTypeList[procIndex]
                row_index = rowIndexList[procIndex]
                neighbor_index = neighborIndexList[procIndex]
                oldSiteSystemElementIndex = currentStateOccupancy[speciesIndex]
                newSiteSystemElementIndex = neighbor_site_system_element_index_list[
                                                                    procIndex]
                currentStateOccupancy[speciesIndex] = newSiteSystemElementIndex
                speciesDisplacementVectorList[
                    0, speciesIndex * 3:(speciesIndex + 1) * 3] \
                    += self.system.hop_neighbor_list[
                        hop_element_type][hop_dist_type].displacement_vector_list[
                                                    row_index][neighbor_index]

                currentStateEnergy += delG0List[procIndex]
                currentStateChargeConfig[oldSiteSystemElementIndex] \
                    -= self.species_charge_list[speciesIndex]
                currentStateChargeConfig[newSiteSystemElementIndex] \
                    += self.species_charge_list[speciesIndex]
                endPathIndex = int(sim_time / self.time_interval)
                if endPathIndex >= startPathIndex + 1:
                    if endPathIndex >= numPathStepsPerTraj:
                        endPathIndex = numPathStepsPerTraj
                        breakFlag = 1
                    unwrappedPositionArray[startPathIndex:endPathIndex] \
                        = (unwrappedPositionArray[startPathIndex-1]
                           + speciesDisplacementVectorList)
                    energyArray[startPathIndex:endPathIndex] \
                        = currentStateEnergy
                    speciesDisplacementVectorList \
                        = np.zeros((1, self.totalSpecies * 3))
                    startPathIndex = endPathIndex
                    if breakFlag:
                        break
                    # TODO: Address excess flag
                    # if excess:
                    #     # TODO: Avoid using flatten
                    #     wrappedPositionArray[pathIndex] \
                    #         = self.systemCoordinates[
                    #                         currentStateOccupancy].flatten()
            with open(unwrappedTrajFileName, 'ab') as unwrappedTrajFile:
                np.savetxt(unwrappedTrajFile, unwrappedPositionArray)
            with open(energyTrajFileName, 'ab') as energyTrajFile:
                np.savetxt(energyTrajFile, energyArray)
            if excess:
                with open(wrappedTrajFileName, 'ab') as wrappedTrajFile:
                    np.savetxt(wrappedTrajFile, wrappedPositionArray)
                with open(delG0TrajFileName, 'ab') as delG0TrajFile:
                    np.savetxt(delG0TrajFile, delG0Array)
                with open(potentialTrajFileName, 'ab') as potentialTrajFile:
                    np.savetxt(potentialTrajFile, potentialArray)
        if report:
            self.generateSimulationLogReport(outdir)
        return None

    def generateSimulationLogReport(self, outdir):
        """Generates an log report of the simulation and
            outputs to the working directory"""
        simulationLogFileName = 'Run.log'
        simulationLogFilePath = outdir.joinpath(simulationLogFileName)
        report = open(simulationLogFilePath, 'w')
        end_time = datetime.now()
        time_elapsed = end_time - self.start_time
        report.write('Time elapsed: '
                     + ('%2d days, '
                        % time_elapsed.days if time_elapsed.days else '')
                     + ('%2d hours' % ((time_elapsed.seconds // 3600) % 24))
                     + (', %2d minutes' % ((time_elapsed.seconds // 60) % 60))
                     + (', %2d seconds' % (time_elapsed.seconds % 60)))
        report.close()
        return None


class Analysis(object):
    """Post-simulation analysis methods"""
    def __init__(self, material_info, n_dim, species_count, n_traj, t_final,
                 time_interval, msd_t_final, trim_length, repr_time='ns',
                 repr_dist='Angstrom'):
        """"""
        self.start_time = datetime.now()

        self.material = material_info
        self.n_dim = n_dim
        self.species_count = species_count
        self.totalSpecies = self.species_count.sum()
        self.n_traj = int(n_traj)
        self.t_final = t_final * self.material.SEC2AUTIME
        self.time_interval = time_interval * self.material.SEC2AUTIME
        self.trim_length = trim_length
        self.numPathStepsPerTraj = int(self.t_final / self.time_interval) + 1
        self.repr_time = repr_time
        self.repr_dist = repr_dist

        if repr_time == 'ns':
            self.timeConversion = (self.material.SEC2NS
                                   / self.material.SEC2AUTIME)
        elif repr_time == 'ps':
            self.timeConversion = (self.material.SEC2PS
                                   / self.material.SEC2AUTIME)
        elif repr_time == 'fs':
            self.timeConversion = (self.material.SEC2FS
                                   / self.material.SEC2AUTIME)
        elif repr_time == 's':
            self.timeConversion = 1E+00 / self.material.SEC2AUTIME

        if repr_dist == 'm':
            self.distConversion = self.material.ANG / self.material.ANG2BOHR
        elif repr_dist == 'um':
            self.distConversion = self.material.ANG2UM / self.material.ANG2BOHR
        elif repr_dist == 'angstrom':
            self.distConversion = 1E+00 / self.material.ANG2BOHR

        self.msd_t_final = msd_t_final / self.timeConversion
        self.numMSDStepsPerTraj = int(self.msd_t_final / self.time_interval) + 1

    def compute_msd(self, outdir, report=1):
        """Returns the squared displacement of the trajectories"""
        assert outdir, 'Please provide the destination path where \
                                MSD output files needs to be saved'
        positionArray = np.loadtxt(outdir.joinpath('unwrappedTraj.dat'))
        numTrajRecorded = int(len(positionArray) / self.numPathStepsPerTraj)
        positionArray = (
            positionArray[:numTrajRecorded
                          * self.numPathStepsPerTraj + 1].reshape((
                                  numTrajRecorded * self.numPathStepsPerTraj,
                                  self.totalSpecies, 3))
            * self.distConversion)
        sdArray = np.zeros((numTrajRecorded,
                            self.numMSDStepsPerTraj,
                            self.totalSpecies))
        for trajIndex in range(numTrajRecorded):
            headStart = trajIndex * self.numPathStepsPerTraj
            for time_step in range(1, self.numMSDStepsPerTraj):
                numDisp = self.numPathStepsPerTraj - time_step
                addOn = np.arange(numDisp)
                posDiff = (positionArray[headStart + time_step + addOn]
                           - positionArray[headStart + addOn])
                sdArray[trajIndex, time_step, :] = np.mean(
                            np.einsum('ijk,ijk->ij', posDiff, posDiff), axis=0)
        speciesAvgSDArray = np.zeros((numTrajRecorded,
                                      self.numMSDStepsPerTraj,
                                      self.material.num_species_types
                                      - list(self.species_count).count(0)))
        start_index = 0
        numNonExistentSpecies = 0
        nonExistentSpeciesIndices = []
        for species_type_index in range(self.material.num_species_types):
            if self.species_count[species_type_index] != 0:
                end_index = start_index + self.species_count[species_type_index]
                speciesAvgSDArray[:, :, (species_type_index
                                         - numNonExistentSpecies)] \
                    = np.mean(sdArray[:, :, start_index:end_index], axis=2)
                start_index = end_index
            else:
                numNonExistentSpecies += 1
                nonExistentSpeciesIndices.append(species_type_index)

        msd_data = np.zeros((self.numMSDStepsPerTraj,
                            (self.material.num_species_types
                             + 1 - list(self.species_count).count(0))))
        timeArray = (np.arange(self.numMSDStepsPerTraj)
                     * self.time_interval
                     * self.timeConversion)
        msd_data[:, 0] = timeArray
        msd_data[:, 1:] = np.mean(speciesAvgSDArray, axis=0)
        std_data = np.std(speciesAvgSDArray, axis=0)
        file_name = (('%1.2E' % (self.msd_t_final * self.timeConversion))
                    + str(self.repr_time)
                    + (',n_traj: %1.2E' % numTrajRecorded
                        if numTrajRecorded != self.n_traj else ''))
        msdFileName = 'MSD_Data_' + file_name + '.npy'
        msdFilePath = outdir.joinpath(msdFileName)
        species_types = [
                species_type
                for index, species_type in enumerate(self.material.species_types)
                if index not in nonExistentSpeciesIndices]
        np.save(msdFilePath, msd_data)

        if report:
            self.generateMSDAnalysisLogReport(msd_data, species_types,
                                              file_name, outdir)

        return_msd_data = ReturnValues(msd_data=msd_data,
                                     std_data=std_data,
                                     species_types=species_types,
                                     file_name=file_name)
        return return_msd_data

    def generateMSDAnalysisLogReport(self, msd_data, species_types,
                                     file_name, outdir):
        """Generates an log report of the MSD Analysis and
            outputs to the working directory"""
        msdAnalysisLogFileName = ('MSD_Analysis' + ('_' if file_name else '')
                                  + file_name + '.log')
        msdLogFilePath = outdir.joinpath(msdAnalysisLogFileName)
        report = open(msdLogFilePath, 'w')
        end_time = datetime.now()
        time_elapsed = end_time - self.start_time
        from scipy.stats import linregress
        for speciesIndex, species_type in enumerate(species_types):
            slope, _, _, _, _ = linregress(
                msd_data[self.trim_length:-self.trim_length, 0],
                msd_data[self.trim_length:-self.trim_length, speciesIndex + 1])
            speciesDiff = (slope * self.material.ANG2UM**2
                           * self.material.SEC2NS / (2 * self.n_dim))
            report.write('Estimated value of {:s} diffusivity is: \
                            {:4.3f} um2/s\n'.format(species_type, speciesDiff))
        report.write('Time elapsed: '
                     + ('%2d days, '
                        % time_elapsed.days if time_elapsed.days else '')
                     + ('%2d hours' % ((time_elapsed.seconds // 3600) % 24))
                     + (', %2d minutes' % ((time_elapsed.seconds // 60) % 60))
                     + (', %2d seconds' % (time_elapsed.seconds % 60)))
        report.close()
        return None

    def generate_msd_plot(self, msd_data, std_data, display_error_bars,
                        species_types, file_name, outdir):
        """Returns a line plot of the MSD data"""
        assert outdir, 'Please provide the destination path \
                            where MSD Plot files needs to be saved'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import AnchoredText
        from textwrap import wrap
        fig = plt.figure()
        ax = fig.add_subplot(111)
        from scipy.stats import linregress
        for speciesIndex, species_type in enumerate(species_types):
            ax.plot(msd_data[:, 0], msd_data[:, speciesIndex + 1], 'o',
                    markerfacecolor='blue', markeredgecolor='black',
                    label=species_type)
            if display_error_bars:
                ax.errorbar(msd_data[:, 0], msd_data[:, speciesIndex + 1],
                            yerr=std_data[:, speciesIndex], fmt='o', capsize=3,
                            color='blue', markerfacecolor='none',
                            markeredgecolor='none')
            slope, intercept, rValue, _, _ = linregress(
                msd_data[self.trim_length:-self.trim_length, 0],
                msd_data[self.trim_length:-self.trim_length, speciesIndex + 1])
            speciesDiff = (slope * self.material.ANG2UM**2
                           * self.material.SEC2NS / (2 * self.n_dim))
            ax.add_artist(AnchoredText('Est. $D_{{%s}}$ = %4.3f'
                                       % (species_type, speciesDiff)
                                       + '  ${{\mu}}m^2/s$; $r^2$=%4.3e'
                                       % (rValue**2),
                                       loc=4))
            ax.plot(msd_data[self.trim_length:-self.trim_length, 0], intercept
                    + slope * msd_data[self.trim_length:-self.trim_length, 0],
                    'r', label=species_type+'-fitted')
        ax.set_xlabel('Time (' + self.repr_time + ')')
        ax.set_ylabel('MSD ('
                      + ('$\AA^2$'
                         if self.repr_dist == 'angstrom'
                         else (self.repr_dist + '^2')) + ')')
        figureTitle = 'MSD_' + file_name
        ax.set_title('\n'.join(wrap(figureTitle, 60)))
        plt.legend()
        plt.show()  # temp change
        figureName = ('MSD_Plot_' + file_name + '_Trim='
                      + str(self.trim_length) + '.png')
        figurePath = outdir.joinpath(figureName)
        plt.savefig(str(figurePath))
        return None

    def computeCOCMSD(self, outdir, report=1):
        """Returns the squared displacement of the trajectories"""
        assert outdir, 'Please provide the destination path where \
                                MSD output files needs to be saved'
        numExistentSpecies = 0
        for species_type_index in range(self.material.num_species_types):
            if self.species_count[species_type_index] != 0:
                numExistentSpecies += 1

        positionArray = np.loadtxt(outdir.joinpath('unwrappedTraj.dat'))
        numTrajRecorded = int(len(positionArray) / self.numPathStepsPerTraj)
        positionArray = (
            positionArray[:numTrajRecorded
                          * self.numPathStepsPerTraj + 1].reshape((
                                  numTrajRecorded * self.numPathStepsPerTraj,
                                  self.totalSpecies, 3))
            * self.distConversion)
        cocPositionArray = np.mean(positionArray, axis=1)
        np.savetxt('cocPositionArray.txt', cocPositionArray)
        file_name = 'center_of_charge'
        self.plot_coc_dispvector(cocPositionArray, file_name, outdir)
        cocPositionArray = cocPositionArray[:, np.newaxis, :]
        sdArray = np.zeros((numTrajRecorded,
                            self.numMSDStepsPerTraj,
                            numExistentSpecies))
        for trajIndex in range(numTrajRecorded):
            headStart = trajIndex * self.numPathStepsPerTraj
            for time_step in range(1, self.numMSDStepsPerTraj):
                numDisp = self.numPathStepsPerTraj - time_step
                addOn = np.arange(numDisp)
                posDiff = (cocPositionArray[headStart + time_step + addOn]
                           - cocPositionArray[headStart + addOn])
                sdArray[trajIndex, time_step, :] = np.mean(
                            np.einsum('ijk,ijk->ij', posDiff, posDiff), axis=0)
        speciesAvgSDArray = np.zeros((numTrajRecorded,
                                      self.numMSDStepsPerTraj,
                                      self.material.num_species_types
                                      - list(self.species_count).count(0)))
        start_index = 0
        numNonExistentSpecies = 0
        nonExistentSpeciesIndices = []
        for species_type_index in range(self.material.num_species_types):
            if self.species_count[species_type_index] != 0:
                end_index = start_index + self.species_count[species_type_index]
                speciesAvgSDArray[:, :, (species_type_index
                                         - numNonExistentSpecies)] \
                    = np.mean(sdArray[:, :, start_index:end_index], axis=2)
                start_index = end_index
            else:
                numNonExistentSpecies += 1
                nonExistentSpeciesIndices.append(species_type_index)

        msd_data = np.zeros((self.numMSDStepsPerTraj,
                            (self.material.num_species_types
                             + 1 - list(self.species_count).count(0))))
        timeArray = (np.arange(self.numMSDStepsPerTraj)
                     * self.time_interval
                     * self.timeConversion)
        msd_data[:, 0] = timeArray
        msd_data[:, 1:] = np.mean(speciesAvgSDArray, axis=0)
        std_data = np.std(speciesAvgSDArray, axis=0)
        file_name = (('%1.2E' % (self.msd_t_final * self.timeConversion))
                    + str(self.repr_time)
                    + (',n_traj: %1.2E' % numTrajRecorded
                        if numTrajRecorded != self.n_traj else ''))
        msdFileName = 'COC_MSD_Data_' + file_name + '.npy'
        msdFilePath = outdir.joinpath(msdFileName)
        species_types = [
                species_type
                for index, species_type in enumerate(self.material.species_types)
                if index not in nonExistentSpeciesIndices]
        np.save(msdFilePath, msd_data)

        if report:
            self.generateCOCMSDAnalysisLogReport(msd_data, species_types,
                                                 file_name, outdir)

        return_msd_data = ReturnValues(msd_data=msd_data,
                                     std_data=std_data,
                                     species_types=species_types,
                                     file_name=file_name)
        return return_msd_data

    def plot_coc_dispvector(self, cocPositionArray, file_name, outdir):
        """Returns a line plot of the MSD data"""
        assert outdir, 'Please provide the destination path \
                            where MSD Plot files needs to be saved'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import importlib
        importlib.import_module('mpl_toolkits.mplot3d').Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        numTrajRecorded = int(len(cocPositionArray) / self.numPathStepsPerTraj)
        xmin = ymin = zmin = 10
        xmax = ymax = zmax = -10
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, numTrajRecorded)]
        dispVectorList = np.zeros((numTrajRecorded, 6))
        for trajIndex in range(numTrajRecorded):
            startPos = cocPositionArray[trajIndex * self.numPathStepsPerTraj]
            endPos = cocPositionArray[(trajIndex + 1)
                                      * self.numPathStepsPerTraj - 1]
            dispVectorList[trajIndex, :3] = startPos
            dispVectorList[trajIndex, 3:] = endPos
            posStack = np.vstack((startPos, endPos))
            ax.plot(posStack[:, 0], posStack[:, 1], posStack[:, 2],
                    color=colors[trajIndex])
            xmin = min(xmin, startPos[0], endPos[0])
            ymin = min(ymin, startPos[1], endPos[1])
            zmin = min(zmin, startPos[2], endPos[2])
            xmax = max(xmax, startPos[0], endPos[0])
            ymax = max(ymax, startPos[1], endPos[1])
            zmax = max(zmax, startPos[2], endPos[2])
        np.savetxt('displacement_vector_list.txt', dispVectorList)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([xmin - 0.2 * abs(xmin), xmax + 0.2 * abs(xmax)])
        ax.set_ylim([ymin - 0.2 * abs(ymin), ymax + 0.2 * abs(ymax)])
        ax.set_zlim([zmin - 0.2 * abs(zmin), zmax + 0.2 * abs(zmax)])
        ax.set_title(('trajectory-wise center of charge displacement vectors')
                     + ' \n$N_{{%s}}$=' % ('species') + str(self.totalSpecies))
        plt.show()  # temp change
        figureName = ('COC_DispVectors_' + file_name + '.png')
        figurePath = outdir.joinpath(figureName)
        plt.savefig(figurePath)
        return None

    def generateCOCMSDPlot(self, msd_data, std_data, display_error_bars,
                           species_types, file_name, outdir):
        """Returns a line plot of the MSD data"""
        assert outdir, 'Please provide the destination path \
                            where MSD Plot files needs to be saved'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import AnchoredText
        from textwrap import wrap
        fig = plt.figure()
        ax = fig.add_subplot(111)
        from scipy.stats import linregress
        for speciesIndex, species_type in enumerate(species_types):
            ax.plot(msd_data[:, 0], msd_data[:, speciesIndex + 1], 'o',
                    markerfacecolor='blue', markeredgecolor='black',
                    label=species_type)
            if display_error_bars:
                ax.errorbar(msd_data[:, 0], msd_data[:, speciesIndex + 1],
                            yerr=std_data[:, speciesIndex], fmt='o', capsize=3,
                            color='blue', markerfacecolor='none',
                            markeredgecolor='none')
            slope, intercept, rValue, _, _ = linregress(
                msd_data[self.trim_length:-self.trim_length, 0],
                msd_data[self.trim_length:-self.trim_length, speciesIndex + 1])
            speciesDiff = (slope * self.material.ANG2UM**2
                           * self.material.SEC2NS / (2 * self.n_dim))
            ax.add_artist(AnchoredText('Est. $D_{{%s}}$ = %4.3f'
                                       % (species_type, speciesDiff)
                                       + '  ${{\mu}}m^2/s$; $r^2$=%4.3e'
                                       % (rValue**2),
                                       loc=4))
            ax.plot(msd_data[self.trim_length:-self.trim_length, 0], intercept
                    + slope * msd_data[self.trim_length:-self.trim_length, 0],
                    'r', label=species_type+'-fitted')
        ax.set_xlabel('Time (' + self.repr_time + ')')
        ax.set_ylabel('MSD ('
                      + ('$\AA^2$'
                         if self.repr_dist == 'angstrom'
                         else (self.repr_dist + '^2')) + ')')
        figureTitle = 'MSD_' + file_name
        ax.set_title('\n'.join(wrap(figureTitle, 60)))
        plt.legend()
        plt.show()  # temp change
        figureName = ('COC_MSD_Plot_' + file_name + '_Trim='
                      + str(self.trim_length) + '.png')
        figurePath = outdir.joinpath(figureName)
        plt.savefig(figurePath)
        return None

    def generateCOCMSDAnalysisLogReport(self, msd_data, species_types,
                                        file_name, outdir):
        """Generates an log report of the MSD Analysis and
            outputs to the working directory"""
        msdAnalysisLogFileName = ('COC_MSD_Analysis'
                                  + ('_' if file_name else '')
                                  + file_name + '.log')
        msdLogFilePath = outdir.joinpath(msdAnalysisLogFileName)
        report = open(msdLogFilePath, 'w')
        end_time = datetime.now()
        time_elapsed = end_time - self.start_time
        from scipy.stats import linregress
        for speciesIndex, species_type in enumerate(species_types):
            slope, _, _, _, _ = linregress(
                msd_data[self.trim_length:-self.trim_length, 0],
                msd_data[self.trim_length:-self.trim_length, speciesIndex + 1])
            speciesDiff = (slope * self.material.ANG2UM**2
                           * self.material.SEC2NS / (2 * self.n_dim))
            report.write('Estimated value of {:s} diffusivity is: \
                            {:4.3f} um2/s\n'.format(species_type, speciesDiff))
        report.write('Time elapsed: '
                     + ('%2d days, '
                        % time_elapsed.days if time_elapsed.days else '')
                     + ('%2d hours' % ((time_elapsed.seconds // 3600) % 24))
                     + (', %2d minutes' % ((time_elapsed.seconds // 60) % 60))
                     + (', %2d seconds' % (time_elapsed.seconds % 60)))
        report.close()
        return None

    # TODO: Finish writing the method soon.
    # def displayCollectiveMSDPlot(self, msd_data, species_types,
    #                              file_name, outdir=None):
    #     """Returns a line plot of the MSD data"""
    #     import matplotlib
    #     matplotlib.use('Agg')
    #     import matplotlib.pyplot as plt
    #     from textwrap import wrap
    #     plt.figure()
    #     figNum = 0
    #     numRow = 3
    #     numCol = 2
    #     for iPlot in range(numPlots):
    #         for speciesIndex, species_type in enumerate(species_types):
    #             plt.subplot(numRow, numCol, figNum)
    #             plt.plot(msd_data[:, 0], msd_data[:, speciesIndex + 1],
    #                      label=species_type)
    #             figNum += 1
    #     plt.xlabel('Time (' + self.repr_time + ')')
    #     plt.ylabel('MSD (' + self.repr_dist + '**2)')
    #     figureTitle = 'MSD_' + file_name
    #     plt.title('\n'.join(wrap(figureTitle, 60)))
    #     plt.legend()
    #     if outdir:
    #         figureName = 'MSD_Plot_' + file_name + '.jpg'
    #         figurePath = outdir + directorySeparator + figureName
    #         plt.savefig(figurePath)

    def meanDistance(self, outdir, mean=1, plot=1, report=1):
        """
        Add combType as one of the inputs
        combType = 0  # combType = 0: like-like; 1: like-unlike; 2: both
        if combType == 0:
            numComb = sum(
                [self.species_count[index] * (self.species_count[index] - 1)
                 for index in len(self.species_count)])
        elif combType == 1:
            numComb = np.prod(self.species_count)
        elif combType == 2:
            numComb = (np.prod(self.species_count)
                       + sum([self.species_count[index]
                              * (self.species_count[index] - 1)
                              for index in len(self.species_count)]))
        """
        positionArray = (self.trajectoryData.wrappedPositionArray
                         * self.distConversion)
        numPathStepsPerTraj = int(self.kmcSteps / self.stepInterval) + 1
        # TODO: Currently assuming only electrons exist and coding accordingly.
        # Need to change according to combType
        pbc = [1, 1, 1]  # change to generic
        n_electrons = self.species_count[0]  # change to generic
        x_range = range(-1, 2) if pbc[0] == 1 else [0]
        y_range = range(-1, 2) if pbc[1] == 1 else [0]
        z_range = range(-1, 2) if pbc[2] == 1 else [0]
        # Initialization
        system_translational_vector_list = np.zeros((3**sum(pbc), 3))
        index = 0
        for x_offset in x_range:
            for y_offset in y_range:
                for z_offset in z_range:
                    system_translational_vector_list[index] = np.dot(
                        np.multiply(np.array([x_offset, y_offset, z_offset]),
                                    self.system_size),
                        (self.material.lattice_matrix * self.distConversion))
                    index += 1
        if mean:
            meanDistance = np.zeros((self.n_traj, numPathStepsPerTraj))
        else:
            interDistanceArray = np.zeros((self.n_traj, numPathStepsPerTraj,
                                           n_electrons * (n_electrons - 1) / 2))
        interDistanceList = np.zeros(n_electrons * (n_electrons - 1) / 2)
        for trajIndex in range(self.n_traj):
            headStart = trajIndex * numPathStepsPerTraj
            for step in range(numPathStepsPerTraj):
                index = 0
                for i in range(n_electrons):
                    for j in range(i + 1, n_electrons):
                        neighbor_image_coords = (system_translational_vector_list
                                               + positionArray[
                                                        headStart + step, j])
                        neighbor_image_displacement_vectors = (
                                        neighbor_image_coords
                                        - positionArray[headStart + step, i])
                        neighbor_image_displacements = np.linalg.norm(
                                            neighbor_image_displacement_vectors,
                                            axis=1)
                        displacement = np.min(neighbor_image_displacements)
                        interDistanceList[index] = displacement
                        index += 1
                if mean:
                    meanDistance[trajIndex, step] = np.mean(interDistanceList)
                    meanDistanceOverTraj = np.mean(meanDistance, axis=0)
                else:
                    interDistanceArray[trajIndex, step] = np.copy(
                                                            interDistanceList)

        interDistanceArrayOverTraj = np.mean(interDistanceArray, axis=0)
        kmcSteps = range(0,
                         numPathStepsPerTraj * int(self.stepInterval),
                         int(self.stepInterval))
        if mean:
            meanDistanceArray = np.zeros((numPathStepsPerTraj, 2))
            meanDistanceArray[:, 0] = kmcSteps
            meanDistanceArray[:, 1] = meanDistanceOverTraj
        else:
            interSpeciesDistanceArray = np.zeros((
                                        numPathStepsPerTraj,
                                        n_electrons * (n_electrons - 1) / 2 + 1))
            interSpeciesDistanceArray[:, 0] = kmcSteps
            interSpeciesDistanceArray[:, 1:] = interDistanceArrayOverTraj
        if mean:
            meanDistanceFileName = 'MeanDistanceData.npy'
            meanDistanceFilePath = outdir.joinpath(meanDistanceFileName)
            np.save(meanDistanceFilePath, meanDistanceArray)
        else:
            interSpeciesDistanceFileName = 'InterSpeciesDistance.npy'
            interSpeciesDistanceFilePath = outdir.joinpath(
                                                interSpeciesDistanceFileName)
            np.save(interSpeciesDistanceFilePath, interSpeciesDistanceArray)

        if plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure()
            if mean:
                plt.plot(meanDistanceArray[:, 0], meanDistanceArray[:, 1])
                plt.title('Mean Distance between species \
                            along simulation length')
                plt.xlabel('KMC Step')
                plt.ylabel('Distance (' + self.repr_dist + ')')
                figureName = 'MeanDistanceOverTraj.jpg'
                figurePath = outdir.joinpath(figureName)
                plt.savefig(figurePath)
            else:
                legendList = []
                for i in range(n_electrons):
                    for j in range(i + 1, n_electrons):
                        legendList.append('r_' + str(i) + ':' + str(j))
                lineObjects = plt.plot(interSpeciesDistanceArray[:, 0],
                                       interSpeciesDistanceArray[:, 1:])
                plt.title('Inter-species Distances along simulation length')
                plt.xlabel('KMC Step')
                plt.ylabel('Distance (' + self.repr_dist + ')')
                lgd = plt.legend(lineObjects, legendList, loc='center left',
                                 bbox_to_anchor=(1, 0.5))
                figureName = 'Inter-SpeciesDistance.jpg'
                figurePath = outdir.joinpath(figureName)
                plt.savefig(figurePath, bbox_extra_artists=(lgd,),
                            bbox_inches='tight')
        if report:
            self.generateMeanDisplacementAnalysisLogReport(outdir)
        output = meanDistanceArray if mean else interSpeciesDistanceArray
        return output

    def generateMeanDisplacementAnalysisLogReport(self, outdir):
        """Generates an log report of the MSD Analysis and \
                outputs to the working directory"""
        meanDisplacementAnalysisLogFileName = 'MeanDisplacement_Analysis.log'
        meanDisplacementAnalysisLogFilePath = outdir.joinpath(
                                        meanDisplacementAnalysisLogFileName)
        report = open(meanDisplacementAnalysisLogFilePath, 'w')
        end_time = datetime.now()
        time_elapsed = end_time - self.start_time
        report.write('Time elapsed: '
                     + ('%2d days, '
                        % time_elapsed.days if time_elapsed.days else '')
                     + ('%2d hours' % ((time_elapsed.seconds // 3600) % 24))
                     + (', %2d minutes' % ((time_elapsed.seconds // 60) % 60))
                     + (', %2d seconds' % (time_elapsed.seconds % 60)))
        report.close()
        return None

    def displayWrappedTrajectories(self):
        """ """
        return None

    def displayUnwrappedTrajectories(self):
        """ """
        return None

    def trajectoryToDCD(self):
        """Convert trajectory data and outputs dcd file"""
        return None


class ReturnValues(object):
    """dummy class to return objects from methods \
        defined inside other classes"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
