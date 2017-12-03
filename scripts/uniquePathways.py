#!/usr/bin/env python

import os

import numpy as np


def read_poscar(input_file_path):
    lattice_matrix = np.zeros((3, 3))
    lattice_parameter_index = 0
    lattice_parameters_line_range = range(3, 6)
    input_file = open(input_file_path, 'r')
    for line_index, line in enumerate(input_file):
        line_number = line_index + 1
        if line_number in lattice_parameters_line_range:
            lattice_matrix[lattice_parameter_index, :] = np.fromstring(line,
                                                                    sep=' ')
            lattice_parameter_index += 1
        elif line_number == 6:
            element_types = line.split()
        elif line_number == 7:
            n_elements_per_unit_cell = np.fromstring(line, dtype=int, sep=' ')
            total_elements_per_unit_cell = n_elements_per_unit_cell.sum()
            fractional_unit_cell_coords = np.zeros((total_elements_per_unit_cell, 3))
            element_index = 0
        elif line_number > 8 and element_index < total_elements_per_unit_cell:
            fractional_unit_cell_coords[element_index, :] = np.fromstring(line,
                                                                      sep=' ')
            element_index += 1
    input_file.close()
    output = np.array([lattice_matrix, element_types, n_elements_per_unit_cell,
                       fractional_unit_cell_coords], dtype=object)
    return output


def generate_quantum_indices(system_size, system_element_index,
                           n_elements_per_unit_cell):
    """Returns the quantum indices of the element"""
    quantum_indices = np.zeros(5, dtype=int)  # [0] * 5
    total_elements_per_unit_cell = n_elements_per_unit_cell.sum()
    unit_cell_element_index = system_element_index % total_elements_per_unit_cell
    quantum_indices[3] = np.where(np.cumsum(n_elements_per_unit_cell)
                                 >= (unit_cell_element_index + 1))[0][0]
    quantum_indices[4] = (unit_cell_element_index
                         - n_elements_per_unit_cell[:quantum_indices[3]].sum())
    n_filled_unit_cells = ((system_element_index - unit_cell_element_index)
                        / total_elements_per_unit_cell)
    for index in range(3):
        quantum_indices[index] = n_filled_unit_cells / system_size[index+1:].prod()
        n_filled_unit_cells -= quantum_indices[index] * system_size[index+1:].prod()
    return quantum_indices


def generateUniquePathways(inputFileLocation, cutoff_dist_key, neighborCutoff,
                           bridgeCutoff, outdir, pathwayPrec, equivalencyPrec,
                           class_list=[], avoidElementType='',
                           roundLatticeParameters={}, printPathwayList=0,
                           printEquivalency=0, desiredCoordinateParameters={}):
    """ generate unique pathways for the given set of element types"""
    # define input parameters
    [lattice_matrix, element_types, n_elements_per_unit_cell,
     fractional_unit_cell_coords] = read_poscar(inputFileLocation)
    n_element_types = len(element_types)
    total_elements_per_unit_cell = n_elements_per_unit_cell.sum()
    element_type_index_list = np.repeat(np.arange(n_element_types),
                                     n_elements_per_unit_cell)
    [center_element_type, neighborElementType] = cutoff_dist_key.split(':')
    center_site_element_type_index = element_types.index(center_element_type)
    neighborCutoffDistLimits = [0, neighborCutoff]
    bridgeCutoffDistLimits = [0, bridgeCutoff]
    if roundLatticeParameters:
        base = roundLatticeParameters['base']
        prec = roundLatticeParameters['prec']

    # sort element wise coordinates in ascending order of z-coordinate
    start_index = 0
    for element_index in range(n_element_types):
        end_index = start_index + n_elements_per_unit_cell[element_index]
        elementUnitCellCoords = fractional_unit_cell_coords[element_type_index_list
                                                         == element_index]
        fractional_unit_cell_coords[start_index:end_index] = elementUnitCellCoords[
                                        elementUnitCellCoords[:, 2].argsort()]
        start_index = end_index

    # generate array of unit cell translational coordinates
    pbc = np.ones(3, int)
    num_cells = 3**sum(pbc)
    x_range = range(-1, 2) if pbc[0] == 1 else [0]
    y_range = range(-1, 2) if pbc[1] == 1 else [0]
    z_range = range(-1, 2) if pbc[2] == 1 else [0]
    system_size = np.array([len(x_range), len(y_range), len(z_range)])
    unitcellTranslationalCoords = np.zeros((num_cells, 3))  # Initialization
    index = 0
    for x_offset in x_range:
        for y_offset in y_range:
            for z_offset in z_range:
                unitcellTranslationalCoords[index] = np.array([x_offset,
                                                               y_offset,
                                                               z_offset])
                index += 1

    # generate list of element indices to avoid during bridge calculations
    if avoidElementType:
        avoidElementTypeIndex = element_types.index(avoidElementType)
        system_element_index_offset_array = np.repeat(
                            np.arange(0, total_elements_per_unit_cell * num_cells,
                                      total_elements_per_unit_cell),
                            n_elements_per_unit_cell[center_site_element_type_index])
        avoidElementIndices = (
            np.tile(n_elements_per_unit_cell[:avoidElementTypeIndex].sum()
                    + np.arange(0,
                                n_elements_per_unit_cell[avoidElementTypeIndex]),
                    num_cells)
            + system_element_index_offset_array)
    else:
        avoidElementIndices = []

    # extract center site fractional coordinates
    numCenterElements = n_elements_per_unit_cell[center_site_element_type_index]
    center_site_indices = (n_elements_per_unit_cell[
                                            :center_site_element_type_index].sum()
                         + np.arange(numCenterElements))
    centerSiteFractCoords = fractional_unit_cell_coords[center_site_indices]

    # generate fractional coordinates for neighbor sites
    # and all system elements
    neighborSiteFractCoords = np.zeros((num_cells * numCenterElements, 3))
    systemFractCoords = np.zeros((num_cells * total_elements_per_unit_cell, 3))
    for iCell in range(num_cells):
        neighborSiteFractCoords[(iCell * numCenterElements):(
                                        (iCell + 1) * numCenterElements)] \
            = (centerSiteFractCoords + unitcellTranslationalCoords[iCell])
        systemFractCoords[(iCell * total_elements_per_unit_cell):(
                                    (iCell + 1) * total_elements_per_unit_cell)] \
            = (fractional_unit_cell_coords + unitcellTranslationalCoords[iCell])

    # generate bridge neighbor list
    bridgeNeighborList = np.empty(numCenterElements, dtype=object)
    for center_site_index, centerSiteFractCoord in enumerate(
                                                        centerSiteFractCoords):
        iBridgeNeighborList = []
        for neighbor_site_index, neighborSiteFractCoord in enumerate(
                                                            systemFractCoords):
            if neighbor_site_index not in avoidElementIndices:
                latticeDirection = (neighborSiteFractCoord
                                    - centerSiteFractCoord)
                neighborDisplacementVector = np.dot(latticeDirection[None, :],
                                                    lattice_matrix)
                displacement = np.linalg.norm(neighborDisplacementVector)
                if (bridgeCutoffDistLimits[0]
                        < displacement
                        <= bridgeCutoffDistLimits[1]):
                    iBridgeNeighborList.append(neighbor_site_index)
        bridgeNeighborList[center_site_index] = np.asarray(iBridgeNeighborList)

    # initialize class pair list
    if class_list:
        centerSiteClassList = class_list[0]
        neighborSiteClassList = np.tile(class_list[1], num_cells)
        classPairList = np.empty(numCenterElements, dtype=object)

    displacement_vector_list = np.empty(numCenterElements, dtype=object)
    latticeDirectionList = np.empty(numCenterElements, dtype=object)
    displacement_list = np.empty(numCenterElements, dtype=object)
    bridgeList = np.empty(numCenterElements, dtype=object)
    num_neighbors = np.zeros(numCenterElements, dtype=int)
    for center_site_index, centerSiteFractCoord in enumerate(
                                                        centerSiteFractCoords):
        i_displacement_vectors = []
        iLatticeDirectionList = []
        iDisplacements = []
        iBridgeList = []
        if class_list:
            iClassPairList = []
        for neighbor_site_index, neighborSiteFractCoord in enumerate(
                                                    neighborSiteFractCoords):
            latticeDirection = neighborSiteFractCoord - centerSiteFractCoord
            neighborDisplacementVector = np.dot(latticeDirection[None, :],
                                                lattice_matrix)
            displacement = np.linalg.norm(neighborDisplacementVector)
            if (neighborCutoffDistLimits[0]
                    < displacement
                    <= neighborCutoffDistLimits[1]):
                i_displacement_vectors.append(neighborDisplacementVector)
                iDisplacements.append(displacement)
                num_neighbors[center_site_index] += 1
                if roundLatticeParameters:
                    latticeDirection = np.round(
                            (base * np.round((latticeDirection) / base)), prec)
                iLatticeDirectionList.append(latticeDirection)

                # print fractional coordinates in the desired super cell size
                if desiredCoordinateParameters:
                    desiredSystemSize = desiredCoordinateParameters[
                                                        'desiredSystemSize']
                    distList = desiredCoordinateParameters['distList']
                    prec = desiredCoordinateParameters['prec']
                    dist = np.round(displacement, prec)
                    if dist in distList:
                        print dist
                        print ('center class:',
                               centerSiteClassList[center_site_index])
                        print ('neighbor class:',
                               neighborSiteClassList[neighbor_site_index])
                        print ('num of bonds:',
                               len(bridgeNeighborList[center_site_index]))
                        print ('center:',
                               np.round(np.divide(centerSiteFractCoord,
                                                  desiredSystemSize), 3))
                        print ('neighbor:',
                               np.round(np.divide(neighborSiteFractCoord,
                                                  desiredSystemSize), 3))

                # determine class pair list
                if class_list:
                    iClassPairList.append(
                        str(centerSiteClassList[center_site_index])
                        + ':' + str(neighborSiteClassList[neighbor_site_index]))

                # determine bridging species
                bridgeSiteExists = 0
                bridgeSiteType = ''
                for iCenterNeighborSEIndex in bridgeNeighborList[
                                                            center_site_index]:
                    iCenterNeighborFractCoord = systemFractCoords[
                                                        iCenterNeighborSEIndex]
                    bridgelatticeDirection = (neighborSiteFractCoord
                                              - iCenterNeighborFractCoord)
                    bridgeneighborDisplacementVector = np.dot(
                                            bridgelatticeDirection[None, :],
                                            lattice_matrix)
                    bridgedisplacement = np.linalg.norm(
                                            bridgeneighborDisplacementVector)
                    if (bridgeCutoffDistLimits[0]
                            < bridgedisplacement
                            <= bridgeCutoffDistLimits[1]):
                        bridgeSiteExists = 1
                        bridgeSiteIndex = iCenterNeighborSEIndex
                        bridgeSiteQuantumIndices = (
                                generate_quantum_indices(system_size,
                                                       bridgeSiteIndex,
                                                       n_elements_per_unit_cell))
                        bridgeSiteType += (
                                (', ' if bridgeSiteType != '' else '')
                                + element_types[bridgeSiteQuantumIndices[3]])
                if not bridgeSiteExists:
                    bridgeSiteType = 'space'
                iBridgeList.append(bridgeSiteType)

        bridgeList[center_site_index] = np.asarray(iBridgeList)
        displacement_vector_list[center_site_index] = np.asarray(
                                                        i_displacement_vectors)
        latticeDirectionList[center_site_index] = np.asarray(
                                                        iLatticeDirectionList)
        displacement_list[center_site_index] = np.asarray(iDisplacements)
        if class_list:
            classPairList[center_site_index] = np.asarray(iClassPairList)

    # determine irreducible form of lattice directions
    from fractions import gcd
    sortedLatticeDirectionList = np.empty(numCenterElements, dtype=object)
    sortedDisplacementList = np.empty(numCenterElements, dtype=object)
    if class_list:
        sortedClassPairList = np.empty(numCenterElements, dtype=object)
    sortedBridgeList = np.empty(numCenterElements, dtype=object)
    pathwayList = np.empty(numCenterElements, dtype=object)
    for iCenterElementIndex in range(numCenterElements):
        sortedDisplacementList[iCenterElementIndex] = (
                        displacement_list[iCenterElementIndex][
                            displacement_list[iCenterElementIndex].argsort()])
        sortedBridgeList[iCenterElementIndex] = (
                        bridgeList[iCenterElementIndex][
                            displacement_list[iCenterElementIndex].argsort()])
        if class_list:
            sortedClassPairList[iCenterElementIndex] = (
                        classPairList[iCenterElementIndex][
                            displacement_list[iCenterElementIndex].argsort()])
        if roundLatticeParameters:
            latticeDirectionList[iCenterElementIndex] = (
                latticeDirectionList[iCenterElementIndex] / base).astype(int)
            centerSiteLDList = latticeDirectionList[iCenterElementIndex]
            for index in range(num_neighbors[iCenterElementIndex]):
                centerSiteAbsLDList = abs(centerSiteLDList[index])
                nz = np.nonzero(centerSiteAbsLDList)[0]
                nzCenterSiteAbsLDlist = centerSiteAbsLDList[nz]
                if len(nz) == 1:
                    latticeDirectionList[iCenterElementIndex][index] = (
                            centerSiteLDList[index] / centerSiteAbsLDList[nz])
                elif len(nz) == 2:
                    latticeDirectionList[iCenterElementIndex][index] = \
                        centerSiteLDList[index] / gcd(nzCenterSiteAbsLDlist[0],
                                                      nzCenterSiteAbsLDlist[1])
                else:
                    latticeDirectionList[iCenterElementIndex][index] = (
                        centerSiteLDList[index]
                        / gcd(gcd(nzCenterSiteAbsLDlist[0],
                                  nzCenterSiteAbsLDlist[1]),
                              nzCenterSiteAbsLDlist[2]))
        sortedLatticeDirectionList[iCenterElementIndex] = (
                        latticeDirectionList[iCenterElementIndex][
                            displacement_list[iCenterElementIndex].argsort()])

        # print equivalency of all center sites with their
        # respective class reference site
        if printEquivalency:
            if class_list:
                refIndex = (
                        np.argmax(centerSiteClassList
                                  == centerSiteClassList[iCenterElementIndex]))
            else:
                refIndex = 0
            print np.array_equal(
                np.round(sortedDisplacementList[refIndex], equivalencyPrec),
                np.round(sortedDisplacementList[iCenterElementIndex],
                         equivalencyPrec))

        # generate center site pathway list
        if class_list:
            centerSitePathwayList = np.hstack((
                    np.round(sortedLatticeDirectionList[iCenterElementIndex],
                             pathwayPrec),
                    np.round(sortedDisplacementList[iCenterElementIndex],
                             pathwayPrec)[:, None],
                    sortedClassPairList[iCenterElementIndex][:, None],
                    sortedBridgeList[iCenterElementIndex][:, None]))
        else:
            centerSitePathwayList = np.hstack((
                    np.round(sortedLatticeDirectionList[iCenterElementIndex],
                             pathwayPrec),
                    np.round(sortedDisplacementList[iCenterElementIndex],
                             pathwayPrec)[:, None],
                    sortedBridgeList[iCenterElementIndex][:, None]))
        pathwayList[iCenterElementIndex] = centerSitePathwayList

        if printPathwayList:
            np.set_printoptions(suppress=True)
            print centerSitePathwayList

    latticeDirectionListFileName = ('latticeDirectionList_' + center_element_type
                                    + '-' + neighborElementType + '_cutoff='
                                    + str(neighborCutoff))
    displacementListFileName = ('displacementList_' + center_element_type + '-'
                                + neighborElementType + '_cutoff='
                                + str(neighborCutoff))
    pathwayFileName = ('pathwayList_' + center_element_type + '-'
                       + neighborElementType + '_cutoff='
                       + str(neighborCutoff))
    latticeDirectionListFilePath = (os.path.join(outdir,
                                                 latticeDirectionListFileName)
                                    + '.npy')
    displacementListFilePath = (os.path.join(outdir, displacementListFileName)
                                + '.npy')
    pathwayFilePath = os.path.join(outdir, pathwayFileName) + '.npy'
    np.save(latticeDirectionListFilePath, sortedLatticeDirectionList)
    np.save(displacementListFilePath, sortedDisplacementList)
    np.save(pathwayFilePath, pathwayList)
    return
