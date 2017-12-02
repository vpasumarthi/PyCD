#!/usr/bin/env python
"""
kMC model to run kinetic Monte Carlo simulations and compute mean square
displacement of random walk of charge carriers on 3D lattice systems
"""
from pathlib import Path
from datetime import datetime
import random as rnd
import itertools
from copy import deepcopy
import pdb

import numpy as np

from PyCT.io import readPOSCAR


class Material(object):
    """Defines the properties and structure of working material
    :param str name: A string representing the material name
    :param list elementTypes: list of chemical elements
    :param dict species_to_element_type_map: list of charge carrier species
    :param unitcellCoords: positions of all elements in the unit cell
    :type unitcellCoords: np.array (nx3)
    :param elementTypeIndexList: list of element types for all
                                 unit cell coordinates
    :type elementTypeIndexList: np.array (n)
    :param dict charge_types: types of atomic charges considered
                             for the working material
    :param list latticeParameters: list of three lattice constants in angstrom
                                   and three angles between them in degrees
    :param float vn: typical frequency for nuclear motion
    :param dict lambda_values: Reorganization energies
    :param dict VAB: Electronic coupling matrix element
    :param dict neighbor_cutoff_dist: List of neighbors and their respective
                                    cutoff distances in angstrom
    :param float neighbor_cutoff_dist_tol: Tolerance value in angstrom for
                                        neighbor cutoff distance
    :param str element_type_delimiter: Delimiter between element types
    :param str empty_species_type: name of the empty species type
    :param float epsilon: Dielectric constant of the material

    The additional attributes are:
        * **nElementsPerUnitCell** (np.array (n)): element-type wise
                            total number of elements in a unit cell
        * **siteList** (list): list of elements that act as sites
        * **elementTypeToSpeciesMap** (dict): dictionary of element
                                              to species mapping
        * **nonEmptySpeciesToElementTypeMap** (dict): dictionary of species to
                      element mapping with elements excluding empty_species_type
        * **hopElementTypes** (dict): dictionary of species to hopping element
                                      types separated by element_type_delimiter
        * **latticeMatrix** (np.array (3x3): lattice cell matrix
    """
    def __init__(self, material_parameters):
        # CONSTANTS
        self.EPSILON0 = 8.854187817E-12  # Electric constant in F.m-1
        self.ANG = 1E-10  # Angstrom in m
        self.KB = 1.38064852E-23  # Boltzmann constant in J/K

        # FUNDAMENTAL ATOMIC UNITS
        # Source: http://physics.nist.gov/cuu/Constants/Table/allascii.txt
        self.EMASS = 9.10938356E-31  # Electron mass in Kg
        self.ECHARGE = 1.6021766208E-19  # Elementary charge in C
        self.HBAR = 1.054571800E-34  # Reduced Planck's constant in J.sec
        self.KE = 1 / (4 * np.pi * self.EPSILON0)

        # DERIVED ATOMIC UNITS
        # Bohr radius in m
        self.BOHR = self.HBAR**2 / (self.EMASS * self.ECHARGE**2 * self.KE)
        # Hartree in J
        self.HARTREE = self.HBAR**2 / (self.EMASS * self.BOHR**2)
        self.AUTIME = self.HBAR / self.HARTREE  # sec
        self.AUTEMPERATURE = self.HARTREE / self.KB  # K

        # CONVERSIONS
        self.EV2J = self.ECHARGE
        self.ANG2BOHR = self.ANG / self.BOHR
        self.ANG2UM = 1.00E-04
        self.J2HARTREE = 1 / self.HARTREE
        self.SEC2AUTIME = 1 / self.AUTIME
        self.SEC2NS = 1.00E+09
        self.SEC2PS = 1.00E+12
        self.SEC2FS = 1.00E+15
        self.K2AUTEMP = 1 / self.AUTEMPERATURE

        # Read Input POSCAR
        [self.latticeMatrix, self.elementTypes, self.nElementsPerUnitCell,
         self.totalElementsPerUnitCell, fractionalUnitCellCoords] = (
                        readPOSCAR(material_parameters.input_coord_file_location))
        self.latticeMatrix *= self.ANG2BOHR
        self.nElementTypes = len(self.elementTypes)
        self.elementTypeIndexList = np.repeat(np.arange(self.nElementTypes),
                                              self.nElementsPerUnitCell)

        self.name = material_parameters.name
        self.species_types = material_parameters.species_types[:]
        self.numSpeciesTypes = len(self.species_types)
        self.species_charge_list = deepcopy(material_parameters.species_charge_list)
        self.species_to_element_type_map = deepcopy(
                                    material_parameters.species_to_element_type_map)

        # Initialization
        self.fractionalUnitCellCoords = np.zeros(
                                                fractionalUnitCellCoords.shape)
        self.unitcellClassList = []
        startIndex = 0
        # Reorder element-wise unitcell coordinates in ascending order
        # of z-coordinate
        for elementTypeIndex in range(self.nElementTypes):
            elementFractUnitCellCoords = fractionalUnitCellCoords[
                                self.elementTypeIndexList == elementTypeIndex]
            endIndex = startIndex + self.nElementsPerUnitCell[elementTypeIndex]
            self.fractionalUnitCellCoords[startIndex:endIndex] = (
                            elementFractUnitCellCoords[
                                elementFractUnitCellCoords[:, 2].argsort()])
            elementType = self.elementTypes[elementTypeIndex]
            self.unitcellClassList.extend(
                    [material_parameters.class_list[elementType][index]
                     for index in elementFractUnitCellCoords[:, 2].argsort()])
            startIndex = endIndex

        self.unitcellCoords = np.dot(self.fractionalUnitCellCoords,
                                     self.latticeMatrix)
        self.charge_types = deepcopy(material_parameters.charge_types)

        self.vn = material_parameters.vn / self.SEC2AUTIME
        self.lambda_values = deepcopy(material_parameters.lambda_values)
        self.lambda_values.update((x, [y[index] * self.EV2J * self.J2HARTREE
                                      for index in range(len(y))])
                                 for x, y in self.lambda_values.items())

        self.VAB = deepcopy(material_parameters.VAB)
        self.VAB.update((x, [y[index] * self.EV2J * self.J2HARTREE
                             for index in range(len(y))])
                        for x, y in self.VAB.items())

        self.neighbor_cutoff_dist = deepcopy(
                                        material_parameters.neighbor_cutoff_dist)
        self.neighbor_cutoff_dist.update((x, [(y[index] * self.ANG2BOHR)
                                            if y[index] else None
                                            for index in range(len(y))])
                                       for x, y in (
                                           self.neighbor_cutoff_dist.items()))
        self.neighbor_cutoff_dist_tol = deepcopy(
                                    material_parameters.neighbor_cutoff_dist_tol)
        self.neighbor_cutoff_dist_tol.update((x, [(y[index] * self.ANG2BOHR)
                                               if y[index] else None
                                               for index in range(len(y))])
                                          for x, y in (
                                        self.neighbor_cutoff_dist_tol.items()))
        self.numUniqueHoppingDistances = {key: len(value)
                                          for key, value in (
                                              self.neighbor_cutoff_dist.items())}

        self.element_type_delimiter = material_parameters.element_type_delimiter
        self.empty_species_type = material_parameters.empty_species_type
        self.dielectric_constant = material_parameters.dielectric_constant

        self.numClasses = [len(set(material_parameters.class_list[elementType]))
                           for elementType in self.elementTypes]
        self.delG0_shift_list = {key: [[(value[centerSiteClassIndex][index]
                                       * self.EV2J * self.J2HARTREE)
                                      for index in range(
                                          self.numUniqueHoppingDistances[key])]
                                     for centerSiteClassIndex in range(
                                                                len(value))]
                               for key, value in (
                                   material_parameters.delG0_shift_list.items())}

        siteList = [self.species_to_element_type_map[key]
                    for key in self.species_to_element_type_map
                    if key != self.empty_species_type]
        self.siteList = list(
                    set([item for sublist in siteList for item in sublist]))
        self.nonEmptySpeciesToElementTypeMap = deepcopy(
                                                self.species_to_element_type_map)
        del self.nonEmptySpeciesToElementTypeMap[self.empty_species_type]

        self.elementTypeToSpeciesMap = {}
        for elementType in self.elementTypes:
            speciesList = []
            for speciesTypeKey in self.nonEmptySpeciesToElementTypeMap.keys():
                if elementType in (
                        self.nonEmptySpeciesToElementTypeMap[speciesTypeKey]):
                    speciesList.append(speciesTypeKey)
            self.elementTypeToSpeciesMap[elementType] = speciesList[:]

        self.hopElementTypes = {
                    key: [self.element_type_delimiter.join(comb)
                          for comb in list(itertools.product(
                              self.species_to_element_type_map[key], repeat=2))]
                    for key in self.species_to_element_type_map
                    if key != self.empty_species_type}

    def generateSites(self, elementTypeIndices, cellSize=np.array([1, 1, 1])):
        """Returns systemElementIndices and coordinates of specified elements
        in a cell of size *cellSize*

        :param str elementTypeIndices: element type indices
        :param cellSize: size of the cell
        :type cellSize: np.array (3x1)
        :return: an object with following attributes:

            * **cellCoordinates** (np.array (nx3)):
            * **quantumIndexList** (np.array (nx5)):
            * **systemElementIndexList** (np.array (n)):

        :raises ValueError: if the input cellSize is less than or equal to 0.
        """
        assert all(size > 0 for size in cellSize), 'Input size should always \
                                                    be greater than 0'
        extractIndices = np.in1d(self.elementTypeIndexList,
                                 elementTypeIndices).nonzero()[0]
        unitcellElementCoords = self.unitcellCoords[extractIndices]
        numCells = cellSize.prod()
        nSitesPerUnitCell = self.nElementsPerUnitCell[elementTypeIndices].sum()
        unitcellElementIndexList = np.arange(nSitesPerUnitCell)
        unitcellElementTypeIndex = np.reshape(
                np.concatenate((
                    np.asarray([[elementTypeIndex] *
                                self.nElementsPerUnitCell[elementTypeIndex]
                                for elementTypeIndex in elementTypeIndices]))),
                (nSitesPerUnitCell, 1))
        unitCellElementTypeElementIndexList = np.reshape(
                    np.concatenate((
                        [np.arange(self.nElementsPerUnitCell[elementTypeIndex])
                         for elementTypeIndex in elementTypeIndices])),
                    (nSitesPerUnitCell, 1))
        # Initialization
        cellCoordinates = np.zeros((numCells * nSitesPerUnitCell, 3))
        # Definition format of Quantum Indices
        # quantumIndex = [unitCellIndex, elementTypeIndex, elementIndex]
        quantumIndexList = np.zeros((numCells * nSitesPerUnitCell, 5),
                                    dtype=int)
        systemElementIndexList = np.zeros(numCells * nSitesPerUnitCell,
                                          dtype=int)
        iUnitCell = 0
        for xIndex in range(cellSize[0]):
            for yIndex in range(cellSize[1]):
                for zIndex in range(cellSize[2]):
                    startIndex = iUnitCell * nSitesPerUnitCell
                    endIndex = startIndex + nSitesPerUnitCell
                    translationVector = np.dot([xIndex, yIndex, zIndex],
                                               self.latticeMatrix)
                    cellCoordinates[startIndex:endIndex] = (
                                    unitcellElementCoords + translationVector)
                    systemElementIndexList[startIndex:endIndex] = (
                                                iUnitCell * nSitesPerUnitCell
                                                + unitcellElementIndexList)
                    quantumIndexList[startIndex:endIndex] = np.hstack((
                                    np.tile(np.array([xIndex, yIndex, zIndex]),
                                            (nSitesPerUnitCell, 1)),
                                    unitcellElementTypeIndex,
                                    unitCellElementTypeElementIndexList))
                    iUnitCell += 1

        returnSites = ReturnValues(
                                cellCoordinates=cellCoordinates,
                                quantumIndexList=quantumIndexList,
                                systemElementIndexList=systemElementIndexList)
        return returnSites


class Neighbors(object):
    """Returns the neighbor list file
    :param system_size: size of the super cell in terms of number of
                        unit cell in three dimensions
    :type system_size: np.array (3x1)
    """

    def __init__(self, material, system_size, pbc):
        self.startTime = datetime.now()
        self.material = material
        self.system_size = system_size
        self.pbc = pbc[:]

        # total number of unit cells
        self.numCells = self.system_size.prod()
        self.numSystemElements = (self.numCells
                                  * self.material.totalElementsPerUnitCell)

        # generate all sites in the system
        self.elementTypeIndices = range(self.material.nElementTypes)
        self.bulkSites = self.material.generateSites(self.elementTypeIndices,
                                                     self.system_size)

        xRange = range(-1, 2) if self.pbc[0] == 1 else [0]
        yRange = range(-1, 2) if self.pbc[1] == 1 else [0]
        zRange = range(-1, 2) if self.pbc[2] == 1 else [0]
        # Initialization
        self.systemTranslationalVectorList = np.zeros((3**sum(self.pbc), 3))
        index = 0
        for xOffset in xRange:
            for yOffset in yRange:
                for zOffset in zRange:
                    self.systemTranslationalVectorList[index] = np.dot(
                        np.multiply(np.array([xOffset, yOffset, zOffset]),
                                    system_size),
                        self.material.latticeMatrix)
                    index += 1

    def generateSystemElementIndex(self, system_size, quantumIndices):
        """Returns the systemElementIndex of the element"""
        # assert type(system_size) is np.ndarray, \
        #     'Please input system_size as a numpy array'
        # assert type(quantumIndices) is np.ndarray, \
        #     'Please input quantumIndices as a numpy array'
        # assert np.all(system_size > 0), \
        #     'System size should be positive in all dimensions'
        # assert all(quantumIndex >= 0 for quantumIndex in quantumIndices), \
        #     'Quantum Indices cannot be negative'
        # assert quantumIndices[-1] < self.material.nElementsPerUnitCell[
        #                                             quantumIndices[-2]], \
        #                     'Element Index exceed number of elements of the \
        #                     specified element type'
        # assert np.all(quantumIndices[:3] < system_size), \
        #                     'Unit cell indices exceed the given system size'
        unitCellIndex = np.copy(quantumIndices[:3])
        [elementTypeIndex, elementIndex] = quantumIndices[-2:]
        systemElementIndex = elementIndex + self.material.nElementsPerUnitCell[
                                                    :elementTypeIndex].sum()
        n_dim = len(system_size)
        for index in range(n_dim):
            if index == 0:
                systemElementIndex += (self.material.totalElementsPerUnitCell
                                       * unitCellIndex[n_dim-1-index])
            else:
                systemElementIndex += (self.material.totalElementsPerUnitCell
                                       * unitCellIndex[n_dim-1-index]
                                       * system_size[-index:].prod())
        return systemElementIndex

    def generateQuantumIndices(self, system_size, systemElementIndex):
        """Returns the quantum indices of the element"""
        # assert systemElementIndex >= 0, \
        #     'System Element Index cannot be negative'
        # assert systemElementIndex < (
        #     system_size.prod() * self.material.totalElementsPerUnitCell), \
        #     'System Element Index out of range for the given system size'
        quantumIndices = np.zeros(5, dtype=int)  # [0] * 5
        unitcellElementIndex = (systemElementIndex
                                % self.material.totalElementsPerUnitCell)
        quantumIndices[3] = np.where(
                                    self.material.nElementsPerUnitCell.cumsum()
                                    >= (unitcellElementIndex + 1))[0][0]
        quantumIndices[4] = (unitcellElementIndex
                             - self.material.nElementsPerUnitCell[
                                                    :quantumIndices[3]].sum())
        nFilledUnitCells = ((systemElementIndex - unitcellElementIndex)
                            / self.material.totalElementsPerUnitCell)
        for index in range(3):
            quantumIndices[index] = (nFilledUnitCells
                                     / system_size[index+1:].prod())
            nFilledUnitCells -= (quantumIndices[index]
                                 * system_size[index+1:].prod())
        return quantumIndices

    def computeCoordinates(self, system_size, systemElementIndex):
        """Returns the coordinates in atomic units of the given
            system element index for a given system size"""
        quantumIndices = self.generateQuantumIndices(system_size,
                                                     systemElementIndex)
        unitcellTranslationVector = np.dot(quantumIndices[:3],
                                           self.material.latticeMatrix)
        coordinates = (unitcellTranslationVector
                       + self.material.unitcellCoords[
                           quantumIndices[4]
                           + self.material.nElementsPerUnitCell[
                                                :quantumIndices[3]].sum()])
        return coordinates

    def computeDistance(self, system_size, systemElementIndex1,
                        systemElementindex2):
        """Returns the distance in atomic units between the two
            system element indices for a given system size"""
        centerCoord = self.computeCoordinates(system_size, systemElementIndex1)
        neighborCoord = self.computeCoordinates(system_size,
                                                systemElementindex2)

        neighborImageCoords = (self.systemTranslationalVectorList
                               + neighborCoord)
        neighborImageDisplacementVectors = neighborImageCoords - centerCoord
        neighborImageDisplacements = np.linalg.norm(
                                    neighborImageDisplacementVectors, axis=1)
        displacement = np.min(neighborImageDisplacements)
        return displacement

    def hopNeighborSites(self, bulkSites, centerSiteIndices,
                         neighborSiteIndices, cutoffDistLimits, cutoffDistKey):
        """Returns systemElementIndexMap and distances between
            center sites and its neighbor sites within cutoff distance"""
        neighborSiteCoords = bulkSites.cellCoordinates[neighborSiteIndices]
        neighborSiteSystemElementIndexList = bulkSites.systemElementIndexList[
                                                        neighborSiteIndices]
        centerSiteCoords = bulkSites.cellCoordinates[centerSiteIndices]

        neighborSystemElementIndices = np.empty(len(centerSiteCoords),
                                                dtype=object)
        displacementVectorList = np.empty(len(centerSiteCoords), dtype=object)
        numNeighbors = np.array([], dtype=int)

        if cutoffDistKey == 'Fe:Fe':
            quickTest = 0  # commit reference: 1472bb4
        else:
            quickTest = 0

        for centerSiteIndex, centerCoord in enumerate(centerSiteCoords):
            iNeighborSiteIndexList = []
            iDisplacementVectors = []
            iNumNeighbors = 0
            if quickTest:
                displacementList = np.zeros(len(neighborSiteCoords))
            for neighborSiteIndex, neighborCoord in enumerate(
                                                        neighborSiteCoords):
                neighborImageCoords = (self.systemTranslationalVectorList
                                       + neighborCoord)
                neighborImageDisplacementVectors = (neighborImageCoords
                                                    - centerCoord)
                neighborImageDisplacements = np.linalg.norm(
                                            neighborImageDisplacementVectors,
                                            axis=1)
                [displacement, imageIndex] = [
                                        np.min(neighborImageDisplacements),
                                        np.argmin(neighborImageDisplacements)]
                if quickTest:
                    displacementList[neighborSiteIndex] = displacement
                if cutoffDistLimits[0] < displacement <= cutoffDistLimits[1]:
                    iNeighborSiteIndexList.append(neighborSiteIndex)
                    iDisplacementVectors.append(
                                neighborImageDisplacementVectors[imageIndex])
                    iNumNeighbors += 1
            neighborSystemElementIndices[centerSiteIndex] = \
                neighborSiteSystemElementIndexList[iNeighborSiteIndexList]
            displacementVectorList[centerSiteIndex] = \
                np.asarray(iDisplacementVectors)
            numNeighbors = np.append(numNeighbors, iNumNeighbors)
            if quickTest == 1:
                print(np.sort(displacementList)[:10] / self.material.ANG2BOHR)
                pdb.set_trace()
            elif quickTest == 2:
                for cutoffDist in range(2, 7):
                    cutoff = cutoffDist * self.material.ANG2BOHR
                    print(cutoffDist)
                    print(displacementList[displacementList < cutoff].shape)
                    print(np.unique(
                        np.sort(np.round(displacementList[displacementList
                                                          < cutoff]
                                         / self.material.ANG2BOHR, 4))).shape)
                    print(np.unique(
                        np.sort(np.round(displacementList[displacementList
                                                          < cutoff]
                                         / self.material.ANG2BOHR, 3))).shape)
                    print(np.unique(
                        np.sort(np.round(displacementList[displacementList
                                                          < cutoff]
                                         / self.material.ANG2BOHR, 2))).shape)
                    print(np.unique(
                        np.sort(np.round(displacementList[displacementList
                                                          < cutoff]
                                         / self.material.ANG2BOHR, 1))).shape)
                    print(np.unique(
                        np.sort(np.round(displacementList[displacementList
                                                          < cutoff]
                                         / self.material.ANG2BOHR, 0))).shape)
                pdb.set_trace()

        returnNeighbors = ReturnValues(
                    neighborSystemElementIndices=neighborSystemElementIndices,
                    displacementVectorList=displacementVectorList,
                    numNeighbors=numNeighbors)
        return returnNeighbors

    def generate_cumulative_displacement_list(self, dst_path):
        """Returns cumulative displacement list for the given system size
            printed out to disk"""
        cumulative_displacement_list = np.zeros((self.numSystemElements,
                                               self.numSystemElements, 3))
        for centerSiteIndex, centerCoord in enumerate(
                                            self.bulkSites.cellCoordinates):
            cumulativeSystemTranslationalVectorList = np.tile(
                                            self.systemTranslationalVectorList,
                                            (self.numSystemElements, 1, 1))
            cumulativeNeighborImageCoords = (
                    cumulativeSystemTranslationalVectorList
                    + np.tile(self.bulkSites.cellCoordinates[:, np.newaxis, :],
                              (1, len(self.systemTranslationalVectorList), 1)))
            cumulativeNeighborImageDisplacementVectors = (
                                                cumulativeNeighborImageCoords
                                                - centerCoord)
            cumulativeNeighborImageDisplacements = np.linalg.norm(
                                    cumulativeNeighborImageDisplacementVectors,
                                    axis=2)
            cumulative_displacement_list[centerSiteIndex] = \
                cumulativeNeighborImageDisplacementVectors[
                    np.arange(self.numSystemElements),
                    np.argmin(cumulativeNeighborImageDisplacements, axis=1)]
        cumulative_displacement_list_file_path = dst_path.joinpath(
                                        'cumulative_displacement_list.npy')
        np.save(cumulative_displacement_list_file_path,
                cumulative_displacement_list)
        return None

    def generate_neighbor_list(self, dst_path, report=1,
                             localSystemSize=np.array([3, 3, 3])):
        """Adds the neighbor list to the system object and
            returns the neighbor list"""
        assert dst_path, \
            'Please provide the path to the parent directory of \
                neighbor list files'
        assert all(size >= 3 for size in localSystemSize), \
            'Local system size in all dimensions should always be \
                greater than or equal to 3'

        Path.mkdir(dst_path, parents=True, exist_ok=True)
        hopNeighborListFilePath = dst_path.joinpath('hop_neighbor_list.npy')

        hop_neighbor_list = {}
        tolDist = self.material.neighbor_cutoff_dist_tol
        elementTypes = self.material.elementTypes[:]

        for cutoffDistKey in self.material.neighbor_cutoff_dist.keys():
            cutoffDistList = self.material.neighbor_cutoff_dist[cutoffDistKey][:]
            neighborListCutoffDistKey = []
            [centerElementType, _] = cutoffDistKey.split(
                                            self.material.element_type_delimiter)
            centerSiteElementTypeIndex = elementTypes.index(centerElementType)
            localBulkSites = self.material.generateSites(
                                                    self.elementTypeIndices,
                                                    self.system_size)
            systemElementIndexOffsetArray = (
                np.repeat(np.arange(0,
                                    (self.material.totalElementsPerUnitCell
                                     * self.numCells),
                                    self.material.totalElementsPerUnitCell),
                          self.material.nElementsPerUnitCell[
                                                centerSiteElementTypeIndex]))
            centerSiteIndices = neighborSiteIndices = (
                np.tile((self.material.nElementsPerUnitCell[
                                            :centerSiteElementTypeIndex].sum()
                         + np.arange(0, self.material.nElementsPerUnitCell[
                                                centerSiteElementTypeIndex])),
                        self.numCells)
                + systemElementIndexOffsetArray)

            for iCutoffDist in range(len(cutoffDistList)):
                cutoffDistLimits = ([(cutoffDistList[iCutoffDist]
                                      - tolDist[cutoffDistKey][iCutoffDist]),
                                     (cutoffDistList[iCutoffDist]
                                      + tolDist[cutoffDistKey][iCutoffDist])])

                neighborListCutoffDistKey.append(
                    self.hopNeighborSites(localBulkSites, centerSiteIndices,
                                          neighborSiteIndices,
                                          cutoffDistLimits, cutoffDistKey))
            hop_neighbor_list[cutoffDistKey] = neighborListCutoffDistKey[:]
        np.save(hopNeighborListFilePath, hop_neighbor_list)

        if report:
            self.generate_neighbor_listReport(dst_path)
        return None

    def generate_neighbor_listReport(self, dst_path):
        """Generates a neighbor list and prints out a
            report to the output directory"""
        neighborListLogName = 'NeighborList.log'
        neighborListLogPath = dst_path.joinpath(neighborListLogName)
        report = open(neighborListLogPath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - self.startTime
        report.write('Time elapsed: '
                     + ('%2d days, ' % timeElapsed.days
                        if timeElapsed.days else '')
                     + ('%2d hours' % ((timeElapsed.seconds // 3600) % 24))
                     + (', %2d minutes' % ((timeElapsed.seconds // 60) % 60))
                     + (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()
        return None

    # TODO: remove the function
    def generateHematiteNeighborSEIndices(self, dst_path, report=1):
        startTime = datetime.now()
        offsetList = np.array(
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
        elementTypeIndex = 0
        basalNeighborElementSiteIndices = np.array(
                                        [11, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 0])
        cNeighborElementSiteIndices = np.array(
                                        [9, 4, 11, 6, 1, 8, 3, 10, 5, 0, 7, 2])
        numBasalNeighbors = 3
        numCNeighbors = 1
        numNeighbors = numBasalNeighbors + numCNeighbors
        nElementsPerUnitCell = self.material.nElementsPerUnitCell[
                                                            elementTypeIndex]
        neighborElementSiteIndices = np.zeros((nElementsPerUnitCell, 4), int)
        for iNeighbor in range(numNeighbors):
            if iNeighbor < numBasalNeighbors:
                neighborElementSiteIndices[:, iNeighbor] = \
                                                basalNeighborElementSiteIndices
            else:
                neighborElementSiteIndices[:, iNeighbor] = \
                                                    cNeighborElementSiteIndices
        systemElementIndexOffsetArray = (
            np.repeat(np.arange(0,
                                (self.material.totalElementsPerUnitCell
                                 * self.numCells),
                                self.material.totalElementsPerUnitCell),
                      self.material.nElementsPerUnitCell[elementTypeIndex]))
        centerSiteSEIndices = (
            np.tile(self.material.nElementsPerUnitCell[:elementTypeIndex].sum()
                    + np.arange(0,
                                self.material.nElementsPerUnitCell[
                                                            elementTypeIndex]),
                    self.numCells)
            + systemElementIndexOffsetArray)
        numCenterSiteElements = len(centerSiteSEIndices)
        neighborSystemElementIndices = np.zeros((numCenterSiteElements,
                                                 numNeighbors))

        for centerSiteIndex, centerSiteSEIndex in enumerate(
                                                        centerSiteSEIndices):
            centerSiteQuantumIndices = self.generateQuantumIndices(
                                                            self.system_size,
                                                            centerSiteSEIndex)
            centerSiteUnitCellIndices = centerSiteQuantumIndices[:3]
            centerSiteElementSiteIndex = centerSiteQuantumIndices[-1:][0]
            for neighborIndex in range(numNeighbors):
                neighborUnitCellIndices = (
                    centerSiteUnitCellIndices
                    + offsetList[centerSiteElementSiteIndex][neighborIndex])
                for index, neighborUnitCellIndex in enumerate(
                                                    neighborUnitCellIndices):
                    if neighborUnitCellIndex < 0:
                        neighborUnitCellIndices[index] += \
                                                        self.system_size[index]
                    elif neighborUnitCellIndex >= self.system_size[index]:
                        neighborUnitCellIndices[index] -= \
                                                        self.system_size[index]
                    neighborQuantumIndices = np.hstack((
                        neighborUnitCellIndices,
                        elementTypeIndex,
                        neighborElementSiteIndices[centerSiteElementSiteIndex][
                                                            neighborIndex]))
                    neighborSEIndex = self.generateSystemElementIndex(
                                                        self.system_size,
                                                        neighborQuantumIndices)
                    neighborSystemElementIndices[
                            centerSiteIndex][neighborIndex] = neighborSEIndex

        file_name = 'neighborSystemElementIndices.npy'
        neighborSystemElementIndicesFilePath = dst_path.joinpath(file_name)
        np.save(neighborSystemElementIndicesFilePath,
                neighborSystemElementIndices)
        if report:
            self.generateHematiteNeighborSEIndicesReport(dst_path, startTime)
        return None

    def generateHematiteNeighborSEIndicesReport(self, dst_path, startTime):
        """Generates a neighbor list and prints out a
            report to the output directory"""
        neighborSystemElementIndicesLogName = \
            'neighborSystemElementIndices.log'
        neighborSystemElementIndicesLogPath = dst_path.joinpath(
                                        neighborSystemElementIndicesLogName)
        report = open(neighborSystemElementIndicesLogPath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - startTime
        report.write('Time elapsed: '
                     + ('%2d days, '
                        % timeElapsed.days if timeElapsed.days else '')
                     + ('%2d hours' % ((timeElapsed.seconds // 3600) % 24))
                     + (', %2d minutes' % ((timeElapsed.seconds // 60) % 60))
                     + (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()
        return None

    def generateSpeciesSiteSDList(self, centerSiteQuantumIndices,
                                  dst_path, report=1):
        startTime = datetime.now()
        elementTypeIndex = centerSiteQuantumIndices[3]
        centerSiteSEIndex = self.generateSystemElementIndex(
                                                    self.system_size,
                                                    centerSiteQuantumIndices)
        systemElementIndexOffsetArray = np.repeat(
                        np.arange(0,
                                  (self.material.totalElementsPerUnitCell
                                   * self.numCells),
                                  self.material.totalElementsPerUnitCell),
                        self.material.nElementsPerUnitCell[elementTypeIndex])
        neighborSiteSEIndices = (
            np.tile(self.material.nElementsPerUnitCell[:elementTypeIndex].sum()
                    + np.arange(0,
                                self.material.nElementsPerUnitCell[
                                                            elementTypeIndex]),
                    self.numCells)
            + systemElementIndexOffsetArray)
        speciesSiteSDList = np.zeros(len(neighborSiteSEIndices))
        for neighborSiteIndex, neighborSiteSEIndex in enumerate(
                                                        neighborSiteSEIndices):
            speciesSiteSDList[neighborSiteIndex] = self.computeDistance(
                                                        self.system_size,
                                                        centerSiteSEIndex,
                                                        neighborSiteSEIndex)**2
        speciesSiteSDList /= self.material.ANG2BOHR**2
        file_name = 'speciesSiteSDList.npy'
        speciesSiteSDListFilePath = dst_path.joinpath(file_name)
        np.save(speciesSiteSDListFilePath, speciesSiteSDList)
        if report:
            self.generateSpeciesSiteSDListReport(dst_path, startTime)
        return None

    def generateSpeciesSiteSDListReport(self, dst_path, startTime):
        """Generates a neighbor list and prints out a
            report to the output directory"""
        speciesSiteSDListLogName = 'speciesSiteSDList.log'
        speciesSiteSDListLogPath = dst_path.joinpath(speciesSiteSDListLogName)
        report = open(speciesSiteSDListLogPath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - startTime
        report.write('Time elapsed: '
                     + ('%2d days, '
                        % timeElapsed.days if timeElapsed.days else '')
                     + ('%2d hours' % ((timeElapsed.seconds // 3600) % 24))
                     + (', %2d minutes' % ((timeElapsed.seconds // 60) % 60))
                     + (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()
        return None

    def generateTransitionProbMatrix(self, neighborSystemElementIndices,
                                     dst_path, report=1):
        startTime = datetime.now()
        elementTypeIndex = 0
        numNeighbors = len(neighborSystemElementIndices[0])
        numBasalNeighbors = 3
        # numCNeighbors = 1
        T = 300 * self.material.K2AUTEMP

        hopElementType = 'Fe:Fe'
        kList = np.zeros(numNeighbors)
        delG0 = 0
        for neighborIndex in range(numNeighbors):
            if neighborIndex < numBasalNeighbors:
                hopDistType = 0
            else:
                hopDistType = 1
            lambdaValue = self.material.lambda_values[
                                                hopElementType][hopDistType]
            VAB = self.material.VAB[hopElementType][hopDistType]
            delGs = ((lambdaValue + delG0) ** 2 / (4 * lambdaValue)) - VAB
            kList[neighborIndex] = self.material.vn * np.exp(-delGs / T)

        kTotal = np.sum(kList)
        probList = kList / kTotal

        systemElementIndexOffsetArray = np.repeat(
                        np.arange(0,
                                  (self.material.totalElementsPerUnitCell
                                   * self.numCells),
                                  self.material.totalElementsPerUnitCell),
                        self.material.nElementsPerUnitCell[elementTypeIndex])
        neighborSiteSEIndices = (
            np.tile(self.material.nElementsPerUnitCell[:elementTypeIndex].sum()
                    + np.arange(0,
                                self.material.nElementsPerUnitCell[
                                                            elementTypeIndex]),
                    self.numCells)
            + systemElementIndexOffsetArray)

        numElementTypeSites = len(neighborSystemElementIndices)
        transitionProbMatrix = np.zeros((numElementTypeSites,
                                         numElementTypeSites))
        for centerSiteIndex in range(numElementTypeSites):
            for neighborIndex in range(numNeighbors):
                neighborSiteIndex = np.where(
                            neighborSiteSEIndices
                            == neighborSystemElementIndices[centerSiteIndex][
                                                        neighborIndex])[0][0]
                transitionProbMatrix[centerSiteIndex][neighborSiteIndex] = \
                    probList[neighborIndex]
        file_name = 'transitionProbMatrix.npy'
        transitionProbMatrixFilePath = dst_path.joinpath(file_name)
        np.save(transitionProbMatrixFilePath, transitionProbMatrix)
        if report:
            self.generateTransitionProbMatrixListReport(dst_path, startTime)
        return None

    def generateTransitionProbMatrixListReport(self, dst_path, startTime):
        """Generates a neighbor list and prints out a report to the
            output directory"""
        transitionProbMatrixLogName = 'transitionProbMatrix.log'
        transitionProbMatrixLogPath = dst_path.joinpath(
                                                transitionProbMatrixLogName)
        report = open(transitionProbMatrixLogPath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - startTime
        report.write('Time elapsed: '
                     + ('%2d days, '
                        % timeElapsed.days if timeElapsed.days else '')
                     + ('%2d hours' % ((timeElapsed.seconds // 3600) % 24))
                     + (', %2d minutes' % ((timeElapsed.seconds // 60) % 60))
                     + (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()
        return None

    def generateMSDAnalyticalData(self, transitionProbMatrix,
                                  speciesSiteSDList, centerSiteQuantumIndices,
                                  analyticalTFinal, analyticalTimeInterval,
                                  dst_path, report=1):
        startTime = datetime.now()

        file_name = '%1.2Ens' % analyticalTFinal
        MSDAnalyticalDataFileName = 'MSD_Analytical_Data_' + file_name + '.dat'
        MSDAnalyticalDataFilePath = dst_path.joinpath(MSDAnalyticalDataFileName)
        open(MSDAnalyticalDataFilePath, 'w').close()

        elementTypeIndex = 0
        numDataPoints = int(analyticalTFinal / analyticalTimeInterval) + 1
        msd_data = np.zeros((numDataPoints, 2))
        msd_data[:, 0] = np.arange(0,
                                  analyticalTFinal + analyticalTimeInterval,
                                  analyticalTimeInterval)

        systemElementIndexOffsetArray = np.repeat(
                        np.arange(0,
                                  (self.material.totalElementsPerUnitCell
                                   * self.numCells),
                                  self.material.totalElementsPerUnitCell),
                        self.material.nElementsPerUnitCell[elementTypeIndex])
        centerSiteSEIndices = (
            np.tile(self.material.nElementsPerUnitCell[:elementTypeIndex].sum()
                    + np.arange(0,
                                self.material.nElementsPerUnitCell[
                                                            elementTypeIndex]),
                    self.numCells)
            + systemElementIndexOffsetArray)

        centerSiteSEIndex = self.generateSystemElementIndex(
                                                    self.system_size,
                                                    centerSiteQuantumIndices)
        numBasalNeighbors = 3
        numCNeighbors = 1
        numNeighbors = numBasalNeighbors + numCNeighbors
        T = 300 * self.material.K2AUTEMP

        hopElementType = 'Fe:Fe'
        kList = np.zeros(numNeighbors)
        delG0 = 0
        for neighborIndex in range(numNeighbors):
            if neighborIndex < numBasalNeighbors:
                hopDistType = 0
            else:
                hopDistType = 1
            lambdaValue = self.material.lambda_values[hopElementType][
                                                                hopDistType]
            VAB = self.material.VAB[hopElementType][hopDistType]
            delGs = ((lambdaValue + delG0) ** 2 / (4 * lambdaValue)) - VAB
            kList[neighborIndex] = self.material.vn * np.exp(-delGs / T)

        kTotal = np.sum(kList)
        timestep = self.material.SEC2NS / kTotal / self.material.SEC2AUTIME

        simTime = 0
        startIndex = 0
        rowIndex = np.where(centerSiteSEIndices == centerSiteSEIndex)
        newTransitionProbMatrix = np.copy(transitionProbMatrix)
        with open(MSDAnalyticalDataFilePath, 'a') as MSDAnalyticalDataFile:
            np.savetxt(MSDAnalyticalDataFile, msd_data[startIndex, :][None, :])
        while True:
            newTransitionProbMatrix = np.dot(newTransitionProbMatrix,
                                             transitionProbMatrix)
            simTime += timestep
            endIndex = int(simTime / analyticalTimeInterval)
            if endIndex >= startIndex + 1:
                msd_data[endIndex, 1] = np.dot(
                                            newTransitionProbMatrix[rowIndex],
                                            speciesSiteSDList)
                with open(MSDAnalyticalDataFilePath, 'a') as \
                        MSDAnalyticalDataFile:
                    np.savetxt(MSDAnalyticalDataFile,
                               msd_data[endIndex, :][None, :])
                startIndex += 1
                if endIndex == numDataPoints - 1:
                    break

        if report:
            self.generateMSDAnalyticalDataReport(file_name, dst_path, startTime)
        returnMSDData = ReturnValues(msd_data=msd_data)
        return returnMSDData

    def generateMSDAnalyticalDataReport(self, file_name, dst_path, startTime):
        """Generates a neighbor list and prints out a report to the
            output directory"""
        MSDAnalyticalDataLogName = 'MSD_Analytical_Data_' + file_name + '.log'
        MSDAnalyticalDataLogPath = dst_path.joinpath(MSDAnalyticalDataLogName)
        report = open(MSDAnalyticalDataLogPath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - startTime
        report.write('Time elapsed: '
                     + ('%2d days, '
                        % timeElapsed.days if timeElapsed.days else '')
                     + ('%2d hours' % ((timeElapsed.seconds // 3600) % 24))
                     + (', %2d minutes' % ((timeElapsed.seconds // 60) % 60))
                     + (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()
        return None


class System(object):
    """defines the system we are working on

    Attributes:
    size: An array (3 x 1) defining the system size in multiple of unit cells
    """
    # @profile
    def __init__(self, material_info, material_neighbors, hop_neighbor_list,
                 cumulative_displacement_list, species_count, alpha, n_max, k_max):
        """Return a system object whose size is *size*"""
        self.startTime = datetime.now()

        self.material = material_info
        self.neighbors = material_neighbors
        self.hop_neighbor_list = hop_neighbor_list

        self.pbc = self.neighbors.pbc
        self.species_count = species_count

        # total number of unit cells
        self.system_size = self.neighbors.system_size
        self.numCells = self.system_size.prod()

        self.cumulative_displacement_list = cumulative_displacement_list

        # variables for ewald sum
        self.translationalMatrix = np.multiply(self.system_size,
                                               self.material.latticeMatrix)
        self.systemVolume = abs(np.dot(self.translationalMatrix[0],
                                       np.cross(self.translationalMatrix[1],
                                                self.translationalMatrix[2])))
        self.reciprocalLatticeMatrix = (
                        2 * np.pi / self.systemVolume
                        * np.array([np.cross(self.translationalMatrix[1],
                                             self.translationalMatrix[2]),
                                    np.cross(self.translationalMatrix[2],
                                             self.translationalMatrix[0]),
                                    np.cross(self.translationalMatrix[0],
                                             self.translationalMatrix[1])]))
        self.translationalVectorLength = np.linalg.norm(
                                                    self.translationalMatrix,
                                                    axis=1)
        self.reciprocalLatticeVectorLength = np.linalg.norm(
                                                self.reciprocalLatticeMatrix,
                                                axis=1)

        # class list
        self.systemClassIndexList = (
                np.tile(self.material.unitcellClassList, self.numCells) - 1)

        # ewald parameters:
        self.alpha = alpha
        self.n_max = n_max
        self.k_max = k_max

    def generateRandomOccupancy(self, species_count):
        """generates initial occupancy list based on species count"""
        occupancy = []
        for speciesTypeIndex, numSpecies in enumerate(species_count):
            speciesType = self.material.species_types[speciesTypeIndex]
            speciesSiteElementList = self.material.species_to_element_type_map[
                                                                speciesType]
            speciesSiteElementTypeIndexList = [
                        self.material.elementTypes.index(speciesSiteElement)
                        for speciesSiteElement in speciesSiteElementList]
            speciesSiteIndices = []
            for speciesSiteElementTypeIndex in speciesSiteElementTypeIndexList:
                systemElementIndexOffsetArray = np.repeat(
                    np.arange(0,
                              (self.material.totalElementsPerUnitCell
                               * self.numCells),
                              self.material.totalElementsPerUnitCell),
                    self.material.nElementsPerUnitCell[
                                                speciesSiteElementTypeIndex])
                siteIndices = (
                    np.tile(self.material.nElementsPerUnitCell[
                                            :speciesSiteElementTypeIndex].sum()
                            + np.arange(0,
                                        self.material.nElementsPerUnitCell[
                                            speciesSiteElementTypeIndex]),
                            self.numCells)
                    + systemElementIndexOffsetArray)
                speciesSiteIndices.extend(list(siteIndices))
            occupancy.extend(rnd.sample(speciesSiteIndices,
                                        numSpecies)[:])
        return occupancy

    def chargeConfig(self, occupancy, ion_charge_type, species_charge_type):
        """Returns charge distribution of the current configuration"""

        # generate lattice charge list
        unitcellChargeList = np.array(
                [self.material.charge_types[ion_charge_type][
                    self.material.elementTypes[elementTypeIndex]]
                 for elementTypeIndex in self.material.elementTypeIndexList])
        chargeList = np.tile(unitcellChargeList, self.numCells)[:, np.newaxis]

        for speciesTypeIndex in range(self.material.numSpeciesTypes):
            startIndex = 0 + self.species_count[:speciesTypeIndex].sum()
            endIndex = startIndex + self.species_count[speciesTypeIndex]
            centerSiteSystemElementIndices = occupancy[
                                                    startIndex:endIndex][:]
            chargeList[centerSiteSystemElementIndices] += (
                        self.material.species_charge_list[species_charge_type][
                                                            speciesTypeIndex])
        return chargeList

    def ewald_sum_setup(self, outdir=None):
        from scipy.special import erfc
        sqrtalpha = np.sqrt(self.alpha)
        alpha4 = 4 * self.alpha
        fourierSumCoeff = (2 * np.pi) / self.systemVolume
        precomputed_array = np.zeros((self.neighbors.numSystemElements,
                                     self.neighbors.numSystemElements))

        for i in range(-self.n_max, self.n_max+1):
            for j in range(-self.n_max, self.n_max+1):
                for k in range(-self.n_max, self.n_max+1):
                    tempArray = np.linalg.norm(
                                        (self.cumulative_displacement_list
                                         + np.dot(np.array([i, j, k]),
                                                  self.translationalMatrix)),
                                        axis=2)
                    precomputed_array += erfc(sqrtalpha * tempArray) / 2

                    if np.all(np.array([i, j, k]) == 0):
                        for a in range(self.neighbors.numSystemElements):
                            for b in range(self.neighbors.numSystemElements):
                                if a != b:
                                    precomputed_array[a][b] /= tempArray[a][b]
                    else:
                        precomputed_array /= tempArray

        for i in range(-self.k_max, self.k_max+1):
            for j in range(-self.k_max, self.k_max+1):
                for k in range(-self.k_max, self.k_max+1):
                    if not np.all(np.array([i, j, k]) == 0):
                        kVector = np.dot(np.array([i, j, k]),
                                         self.reciprocalLatticeMatrix)
                        kVector2 = np.dot(kVector, kVector)
                        precomputed_array += (
                            fourierSumCoeff
                            * np.exp(-kVector2 / alpha4)
                            * np.cos(np.tensordot(
                                            self.cumulative_displacement_list,
                                            kVector,
                                            axes=([2], [0])))
                            / kVector2)

        precomputed_array /= self.material.dielectric_constant

        if outdir:
            self.generatePreComputedArrayLogReport(outdir)
        return precomputed_array

    def generatePreComputedArrayLogReport(self, outdir):
        """Generates an log report of the simulation and outputs
            to the working directory"""
        precomputedArrayLogFileName = 'precomputed_array.log'
        precomputedArrayLogFilePath = outdir.joinpath(
                                                precomputedArrayLogFileName)
        report = open(precomputedArrayLogFilePath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - self.startTime
        report.write('Time elapsed: '
                     + ('%2d days, '
                        % timeElapsed.days if timeElapsed.days else '')
                     + ('%2d hours' % ((timeElapsed.seconds // 3600) % 24))
                     + (', %2d minutes' % ((timeElapsed.seconds // 60) % 60))
                     + (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()
        return None


class Run(object):
    """defines the subroutines for running Kinetic Monte Carlo and
        computing electrostatic interaction energies"""
    def __init__(self, system, precomputed_array, T, ion_charge_type,
                 species_charge_type, n_traj, t_final, time_interval):
        """Returns the PBC condition of the system"""
        self.startTime = datetime.now()

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

        # nElementsPerUnitCell
        self.headStart_nElementsPerUnitCellCumSum = [
                self.material.nElementsPerUnitCell[:siteElementTypeIndex].sum()
                for siteElementTypeIndex in self.neighbors.elementTypeIndices]

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
                                self.material.hopElementTypes[speciesType][0]
                                for speciesType in self.speciesTypeList]
        self.lenHopDistTypeList = [
                        len(self.material.neighbor_cutoff_dist[hopElementType])
                        for hopElementType in self.hopElementTypeList]
        # number of kinetic processes
        self.nProc = 0
        self.nProcHopElementTypeList = []
        self.nProcSpeciesIndexList = []
        self.nProcSiteElementTypeIndexList = []
        self.nProcLambdaValueList = []
        self.nProcVABList = []
        for hopElementTypeIndex, hopElementType in enumerate(
                                                    self.hopElementTypeList):
            centerElementType = hopElementType.split(
                                        self.material.element_type_delimiter)[0]
            speciesTypeIndex = self.material.species_types.index(
                self.material.elementTypeToSpeciesMap[centerElementType][0])
            centerSiteElementTypeIndex = self.material.elementTypes.index(
                                                            centerElementType)
            for hopDistTypeIndex in range(self.lenHopDistTypeList[
                                                        hopElementTypeIndex]):
                if self.system.species_count[speciesTypeIndex] != 0:
                    numNeighbors = self.system.hop_neighbor_list[hopElementType][
                                                hopDistTypeIndex].numNeighbors
                    self.nProc += numNeighbors[0]
                    self.nProcHopElementTypeList.extend([hopElementType]
                                                        * numNeighbors[0])
                    self.nProcSpeciesIndexList.extend([hopElementTypeIndex]
                                                      * numNeighbors[0])
                    self.nProcSiteElementTypeIndexList.extend(
                            [centerSiteElementTypeIndex] * numNeighbors[0])
                    self.nProcLambdaValueList.extend(
                            [self.material.lambda_values[hopElementType][
                                                        hopDistTypeIndex]]
                            * numNeighbors[0])
                    self.nProcVABList.extend(
                                    [self.material.VAB[hopElementType][
                                                        hopDistTypeIndex]]
                                    * numNeighbors[0])

        # system coordinates
        self.systemCoordinates = self.neighbors.bulkSites.cellCoordinates

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
        kList = np.zeros(self.nProc)
        neighborSiteSystemElementIndexList = np.zeros(self.nProc, dtype=int)
        nProcHopDistTypeList = np.zeros(self.nProc, dtype=int)
        rowIndexList = np.zeros(self.nProc, dtype=int)
        neighborIndexList = np.zeros(self.nProc, dtype=int)
        systemCharge = np.dot(
                    self.system.species_count,
                    self.material.species_charge_list[self.species_charge_type])

        ewaldNeut = - (np.pi
                       * (systemCharge**2)
                       / (2 * self.system.systemVolume * self.system.alpha))
        precomputed_array = self.precomputed_array
        for _ in range(n_traj):
            currentStateOccupancy = self.system.generateRandomOccupancy(
                                                    self.system.species_count)
            currentStateChargeConfig = self.system.chargeConfig(
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
            simTime = 0
            breakFlag = 0
            while True:
                iProc = 0
                delG0List = []
                for speciesIndex, speciesSiteSystemElementIndex in enumerate(
                                                        currentStateOccupancy):
                    # TODO: Avoid re-defining speciesIndex
                    speciesIndex = self.nProcSpeciesIndexList[iProc]
                    hopElementType = self.nProcHopElementTypeList[iProc]
                    siteElementTypeIndex = self.nProcSiteElementTypeIndexList[
                                                                        iProc]
                    rowIndex = (speciesSiteSystemElementIndex
                                // self.material.totalElementsPerUnitCell
                                * self.material.nElementsPerUnitCell[
                                                        siteElementTypeIndex]
                                + speciesSiteSystemElementIndex
                                % self.material.totalElementsPerUnitCell
                                - self.headStart_nElementsPerUnitCellCumSum[
                                                        siteElementTypeIndex])
                    for hopDistType in range(self.lenHopDistTypeList[
                                                                speciesIndex]):
                        localNeighborSiteSystemElementIndexList = (
                                self.system.hop_neighbor_list[hopElementType][
                                    hopDistType].neighborSystemElementIndices[
                                                                    rowIndex])
                        for neighborIndex, neighborSiteSystemElementIndex in \
                                enumerate(
                                    localNeighborSiteSystemElementIndexList):
                            # TODO: Introduce If condition
                            # if neighborSystemElementIndex not in
                            # currentStateOccupancy: commit 898baa8
                            neighborSiteSystemElementIndexList[iProc] = \
                                neighborSiteSystemElementIndex
                            nProcHopDistTypeList[iProc] = hopDistType
                            rowIndexList[iProc] = rowIndex
                            neighborIndexList[iProc] = neighborIndex
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
                            classIndex = (self.system.systemClassIndexList[
                                                speciesSiteSystemElementIndex])
                            delG0 = (
                                delG0Ewald
                                + self.material.delG0_shift_list[
                                    self.nProcHopElementTypeList[iProc]][
                                        classIndex][hopDistType])
                            delG0List.append(delG0)
                            lambdaValue = self.nProcLambdaValueList[iProc]
                            VAB = self.nProcVABList[iProc]
                            delGs = (((lambdaValue + delG0) ** 2
                                      / (4 * lambdaValue)) - VAB)
                            kList[iProc] = self.material.vn * np.exp(-delGs
                                                                     / self.T)
                            iProc += 1

                kTotal = sum(kList)
                kCumSum = (kList / kTotal).cumsum()
                rand1 = rnd.random()
                procIndex = np.where(kCumSum > rand1)[0][0]
                rand2 = rnd.random()
                simTime -= np.log(rand2) / kTotal

                # TODO: Address pre-defining excess data arrays
                # if excess:
                #    delG0Array[step] = delG0List[procIndex]
                speciesIndex = self.nProcSpeciesIndexList[procIndex]
                hopElementType = self.nProcHopElementTypeList[procIndex]
                hopDistType = nProcHopDistTypeList[procIndex]
                rowIndex = rowIndexList[procIndex]
                neighborIndex = neighborIndexList[procIndex]
                oldSiteSystemElementIndex = currentStateOccupancy[speciesIndex]
                newSiteSystemElementIndex = neighborSiteSystemElementIndexList[
                                                                    procIndex]
                currentStateOccupancy[speciesIndex] = newSiteSystemElementIndex
                speciesDisplacementVectorList[
                    0, speciesIndex * 3:(speciesIndex + 1) * 3] \
                    += self.system.hop_neighbor_list[
                        hopElementType][hopDistType].displacementVectorList[
                                                    rowIndex][neighborIndex]

                currentStateEnergy += delG0List[procIndex]
                currentStateChargeConfig[oldSiteSystemElementIndex] \
                    -= self.species_charge_list[speciesIndex]
                currentStateChargeConfig[newSiteSystemElementIndex] \
                    += self.species_charge_list[speciesIndex]
                endPathIndex = int(simTime / self.time_interval)
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
        endTime = datetime.now()
        timeElapsed = endTime - self.startTime
        report.write('Time elapsed: '
                     + ('%2d days, '
                        % timeElapsed.days if timeElapsed.days else '')
                     + ('%2d hours' % ((timeElapsed.seconds // 3600) % 24))
                     + (', %2d minutes' % ((timeElapsed.seconds // 60) % 60))
                     + (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()
        return None


class Analysis(object):
    """Post-simulation analysis methods"""
    def __init__(self, material_info, n_dim, species_count, n_traj, t_final,
                 time_interval, msd_t_final, trim_length, repr_time='ns',
                 repr_dist='Angstrom'):
        """"""
        self.startTime = datetime.now()

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
            for timestep in range(1, self.numMSDStepsPerTraj):
                numDisp = self.numPathStepsPerTraj - timestep
                addOn = np.arange(numDisp)
                posDiff = (positionArray[headStart + timestep + addOn]
                           - positionArray[headStart + addOn])
                sdArray[trajIndex, timestep, :] = np.mean(
                            np.einsum('ijk,ijk->ij', posDiff, posDiff), axis=0)
        speciesAvgSDArray = np.zeros((numTrajRecorded,
                                      self.numMSDStepsPerTraj,
                                      self.material.numSpeciesTypes
                                      - list(self.species_count).count(0)))
        startIndex = 0
        numNonExistentSpecies = 0
        nonExistentSpeciesIndices = []
        for speciesTypeIndex in range(self.material.numSpeciesTypes):
            if self.species_count[speciesTypeIndex] != 0:
                endIndex = startIndex + self.species_count[speciesTypeIndex]
                speciesAvgSDArray[:, :, (speciesTypeIndex
                                         - numNonExistentSpecies)] \
                    = np.mean(sdArray[:, :, startIndex:endIndex], axis=2)
                startIndex = endIndex
            else:
                numNonExistentSpecies += 1
                nonExistentSpeciesIndices.append(speciesTypeIndex)

        msd_data = np.zeros((self.numMSDStepsPerTraj,
                            (self.material.numSpeciesTypes
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
                speciesType
                for index, speciesType in enumerate(self.material.species_types)
                if index not in nonExistentSpeciesIndices]
        np.save(msdFilePath, msd_data)

        if report:
            self.generateMSDAnalysisLogReport(msd_data, species_types,
                                              file_name, outdir)

        returnMSDData = ReturnValues(msd_data=msd_data,
                                     std_data=std_data,
                                     species_types=species_types,
                                     file_name=file_name)
        return returnMSDData

    def generateMSDAnalysisLogReport(self, msd_data, species_types,
                                     file_name, outdir):
        """Generates an log report of the MSD Analysis and
            outputs to the working directory"""
        msdAnalysisLogFileName = ('MSD_Analysis' + ('_' if file_name else '')
                                  + file_name + '.log')
        msdLogFilePath = outdir.joinpath(msdAnalysisLogFileName)
        report = open(msdLogFilePath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - self.startTime
        from scipy.stats import linregress
        for speciesIndex, speciesType in enumerate(species_types):
            slope, _, _, _, _ = linregress(
                msd_data[self.trim_length:-self.trim_length, 0],
                msd_data[self.trim_length:-self.trim_length, speciesIndex + 1])
            speciesDiff = (slope * self.material.ANG2UM**2
                           * self.material.SEC2NS / (2 * self.n_dim))
            report.write('Estimated value of {:s} diffusivity is: \
                            {:4.3f} um2/s\n'.format(speciesType, speciesDiff))
        report.write('Time elapsed: '
                     + ('%2d days, '
                        % timeElapsed.days if timeElapsed.days else '')
                     + ('%2d hours' % ((timeElapsed.seconds // 3600) % 24))
                     + (', %2d minutes' % ((timeElapsed.seconds // 60) % 60))
                     + (', %2d seconds' % (timeElapsed.seconds % 60)))
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
        for speciesIndex, speciesType in enumerate(species_types):
            ax.plot(msd_data[:, 0], msd_data[:, speciesIndex + 1], 'o',
                    markerfacecolor='blue', markeredgecolor='black',
                    label=speciesType)
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
                                       % (speciesType, speciesDiff)
                                       + '  ${{\mu}}m^2/s$; $r^2$=%4.3e'
                                       % (rValue**2),
                                       loc=4))
            ax.plot(msd_data[self.trim_length:-self.trim_length, 0], intercept
                    + slope * msd_data[self.trim_length:-self.trim_length, 0],
                    'r', label=speciesType+'-fitted')
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
        for speciesTypeIndex in range(self.material.numSpeciesTypes):
            if self.species_count[speciesTypeIndex] != 0:
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
            for timestep in range(1, self.numMSDStepsPerTraj):
                numDisp = self.numPathStepsPerTraj - timestep
                addOn = np.arange(numDisp)
                posDiff = (cocPositionArray[headStart + timestep + addOn]
                           - cocPositionArray[headStart + addOn])
                sdArray[trajIndex, timestep, :] = np.mean(
                            np.einsum('ijk,ijk->ij', posDiff, posDiff), axis=0)
        speciesAvgSDArray = np.zeros((numTrajRecorded,
                                      self.numMSDStepsPerTraj,
                                      self.material.numSpeciesTypes
                                      - list(self.species_count).count(0)))
        startIndex = 0
        numNonExistentSpecies = 0
        nonExistentSpeciesIndices = []
        for speciesTypeIndex in range(self.material.numSpeciesTypes):
            if self.species_count[speciesTypeIndex] != 0:
                endIndex = startIndex + self.species_count[speciesTypeIndex]
                speciesAvgSDArray[:, :, (speciesTypeIndex
                                         - numNonExistentSpecies)] \
                    = np.mean(sdArray[:, :, startIndex:endIndex], axis=2)
                startIndex = endIndex
            else:
                numNonExistentSpecies += 1
                nonExistentSpeciesIndices.append(speciesTypeIndex)

        msd_data = np.zeros((self.numMSDStepsPerTraj,
                            (self.material.numSpeciesTypes
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
                speciesType
                for index, speciesType in enumerate(self.material.species_types)
                if index not in nonExistentSpeciesIndices]
        np.save(msdFilePath, msd_data)

        if report:
            self.generateCOCMSDAnalysisLogReport(msd_data, species_types,
                                                 file_name, outdir)

        returnMSDData = ReturnValues(msd_data=msd_data,
                                     std_data=std_data,
                                     species_types=species_types,
                                     file_name=file_name)
        return returnMSDData

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
        for speciesIndex, speciesType in enumerate(species_types):
            ax.plot(msd_data[:, 0], msd_data[:, speciesIndex + 1], 'o',
                    markerfacecolor='blue', markeredgecolor='black',
                    label=speciesType)
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
                                       % (speciesType, speciesDiff)
                                       + '  ${{\mu}}m^2/s$; $r^2$=%4.3e'
                                       % (rValue**2),
                                       loc=4))
            ax.plot(msd_data[self.trim_length:-self.trim_length, 0], intercept
                    + slope * msd_data[self.trim_length:-self.trim_length, 0],
                    'r', label=speciesType+'-fitted')
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
        endTime = datetime.now()
        timeElapsed = endTime - self.startTime
        from scipy.stats import linregress
        for speciesIndex, speciesType in enumerate(species_types):
            slope, _, _, _, _ = linregress(
                msd_data[self.trim_length:-self.trim_length, 0],
                msd_data[self.trim_length:-self.trim_length, speciesIndex + 1])
            speciesDiff = (slope * self.material.ANG2UM**2
                           * self.material.SEC2NS / (2 * self.n_dim))
            report.write('Estimated value of {:s} diffusivity is: \
                            {:4.3f} um2/s\n'.format(speciesType, speciesDiff))
        report.write('Time elapsed: '
                     + ('%2d days, '
                        % timeElapsed.days if timeElapsed.days else '')
                     + ('%2d hours' % ((timeElapsed.seconds // 3600) % 24))
                     + (', %2d minutes' % ((timeElapsed.seconds // 60) % 60))
                     + (', %2d seconds' % (timeElapsed.seconds % 60)))
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
    #         for speciesIndex, speciesType in enumerate(species_types):
    #             plt.subplot(numRow, numCol, figNum)
    #             plt.plot(msd_data[:, 0], msd_data[:, speciesIndex + 1],
    #                      label=speciesType)
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
        xRange = range(-1, 2) if pbc[0] == 1 else [0]
        yRange = range(-1, 2) if pbc[1] == 1 else [0]
        zRange = range(-1, 2) if pbc[2] == 1 else [0]
        # Initialization
        systemTranslationalVectorList = np.zeros((3**sum(pbc), 3))
        index = 0
        for xOffset in xRange:
            for yOffset in yRange:
                for zOffset in zRange:
                    systemTranslationalVectorList[index] = np.dot(
                        np.multiply(np.array([xOffset, yOffset, zOffset]),
                                    self.system_size),
                        (self.material.latticeMatrix * self.distConversion))
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
                        neighborImageCoords = (systemTranslationalVectorList
                                               + positionArray[
                                                        headStart + step, j])
                        neighborImageDisplacementVectors = (
                                        neighborImageCoords
                                        - positionArray[headStart + step, i])
                        neighborImageDisplacements = np.linalg.norm(
                                            neighborImageDisplacementVectors,
                                            axis=1)
                        displacement = np.min(neighborImageDisplacements)
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
        endTime = datetime.now()
        timeElapsed = endTime - self.startTime
        report.write('Time elapsed: '
                     + ('%2d days, '
                        % timeElapsed.days if timeElapsed.days else '')
                     + ('%2d hours' % ((timeElapsed.seconds // 3600) % 24))
                     + (', %2d minutes' % ((timeElapsed.seconds // 60) % 60))
                     + (', %2d seconds' % (timeElapsed.seconds % 60)))
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
