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


def generateQuantumIndices(system_size, systemElementIndex,
                           n_elements_per_unit_cell):
    """Returns the quantum indices of the element"""
    quantumIndices = np.zeros(5, dtype=int)  # [0] * 5
    total_elements_per_unit_cell = n_elements_per_unit_cell.sum()
    unitcellElementIndex = systemElementIndex % total_elements_per_unit_cell
    quantumIndices[3] = np.where(np.cumsum(n_elements_per_unit_cell)
                                 >= (unitcellElementIndex + 1))[0][0]
    quantumIndices[4] = (unitcellElementIndex
                         - n_elements_per_unit_cell[:quantumIndices[3]].sum())
    nFilledUnitCells = ((systemElementIndex - unitcellElementIndex)
                        / total_elements_per_unit_cell)
    for index in range(3):
        quantumIndices[index] = nFilledUnitCells / system_size[index+1:].prod()
        nFilledUnitCells -= quantumIndices[index] * system_size[index+1:].prod()
    return quantumIndices


def generateUniquePathways(inputFileLocation, cutoffDistKey, neighborCutoff,
                           bridgeCutoff, outdir, pathwayPrec, equivalencyPrec,
                           class_list=[], avoidElementType='',
                           roundLatticeParameters={}, printPathwayList=0,
                           printEquivalency=0, desiredCoordinateParameters={}):
    """ generate unique pathways for the given set of element types"""
    # define input parameters
    [lattice_matrix, element_types, n_elements_per_unit_cell,
     fractional_unit_cell_coords] = read_poscar(inputFileLocation)
    nElementTypes = len(element_types)
    total_elements_per_unit_cell = n_elements_per_unit_cell.sum()
    elementTypeIndexList = np.repeat(np.arange(nElementTypes),
                                     n_elements_per_unit_cell)
    [centerElementType, neighborElementType] = cutoffDistKey.split(':')
    centerSiteElementTypeIndex = element_types.index(centerElementType)
    neighborCutoffDistLimits = [0, neighborCutoff]
    bridgeCutoffDistLimits = [0, bridgeCutoff]
    if roundLatticeParameters:
        base = roundLatticeParameters['base']
        prec = roundLatticeParameters['prec']

    # sort element wise coordinates in ascending order of z-coordinate
    startIndex = 0
    for element_index in range(nElementTypes):
        endIndex = startIndex + n_elements_per_unit_cell[element_index]
        elementUnitCellCoords = fractional_unit_cell_coords[elementTypeIndexList
                                                         == element_index]
        fractional_unit_cell_coords[startIndex:endIndex] = elementUnitCellCoords[
                                        elementUnitCellCoords[:, 2].argsort()]
        startIndex = endIndex

    # generate array of unit cell translational coordinates
    pbc = np.ones(3, int)
    numCells = 3**sum(pbc)
    xRange = range(-1, 2) if pbc[0] == 1 else [0]
    yRange = range(-1, 2) if pbc[1] == 1 else [0]
    zRange = range(-1, 2) if pbc[2] == 1 else [0]
    system_size = np.array([len(xRange), len(yRange), len(zRange)])
    unitcellTranslationalCoords = np.zeros((numCells, 3))  # Initialization
    index = 0
    for xOffset in xRange:
        for yOffset in yRange:
            for zOffset in zRange:
                unitcellTranslationalCoords[index] = np.array([xOffset,
                                                               yOffset,
                                                               zOffset])
                index += 1

    # generate list of element indices to avoid during bridge calculations
    if avoidElementType:
        avoidElementTypeIndex = element_types.index(avoidElementType)
        systemElementIndexOffsetArray = np.repeat(
                            np.arange(0, total_elements_per_unit_cell * numCells,
                                      total_elements_per_unit_cell),
                            n_elements_per_unit_cell[centerSiteElementTypeIndex])
        avoidElementIndices = (
            np.tile(n_elements_per_unit_cell[:avoidElementTypeIndex].sum()
                    + np.arange(0,
                                n_elements_per_unit_cell[avoidElementTypeIndex]),
                    numCells)
            + systemElementIndexOffsetArray)
    else:
        avoidElementIndices = []

    # extract center site fractional coordinates
    numCenterElements = n_elements_per_unit_cell[centerSiteElementTypeIndex]
    centerSiteIndices = (n_elements_per_unit_cell[
                                            :centerSiteElementTypeIndex].sum()
                         + np.arange(numCenterElements))
    centerSiteFractCoords = fractional_unit_cell_coords[centerSiteIndices]

    # generate fractional coordinates for neighbor sites
    # and all system elements
    neighborSiteFractCoords = np.zeros((numCells * numCenterElements, 3))
    systemFractCoords = np.zeros((numCells * total_elements_per_unit_cell, 3))
    for iCell in range(numCells):
        neighborSiteFractCoords[(iCell * numCenterElements):(
                                        (iCell + 1) * numCenterElements)] \
            = (centerSiteFractCoords + unitcellTranslationalCoords[iCell])
        systemFractCoords[(iCell * total_elements_per_unit_cell):(
                                    (iCell + 1) * total_elements_per_unit_cell)] \
            = (fractional_unit_cell_coords + unitcellTranslationalCoords[iCell])

    # generate bridge neighbor list
    bridgeNeighborList = np.empty(numCenterElements, dtype=object)
    for centerSiteIndex, centerSiteFractCoord in enumerate(
                                                        centerSiteFractCoords):
        iBridgeNeighborList = []
        for neighborSiteIndex, neighborSiteFractCoord in enumerate(
                                                            systemFractCoords):
            if neighborSiteIndex not in avoidElementIndices:
                latticeDirection = (neighborSiteFractCoord
                                    - centerSiteFractCoord)
                neighborDisplacementVector = np.dot(latticeDirection[None, :],
                                                    lattice_matrix)
                displacement = np.linalg.norm(neighborDisplacementVector)
                if (bridgeCutoffDistLimits[0]
                        < displacement
                        <= bridgeCutoffDistLimits[1]):
                    iBridgeNeighborList.append(neighborSiteIndex)
        bridgeNeighborList[centerSiteIndex] = np.asarray(iBridgeNeighborList)

    # initialize class pair list
    if class_list:
        centerSiteClassList = class_list[0]
        neighborSiteClassList = np.tile(class_list[1], numCells)
        classPairList = np.empty(numCenterElements, dtype=object)

    displacementVectorList = np.empty(numCenterElements, dtype=object)
    latticeDirectionList = np.empty(numCenterElements, dtype=object)
    displacementList = np.empty(numCenterElements, dtype=object)
    bridgeList = np.empty(numCenterElements, dtype=object)
    numNeighbors = np.zeros(numCenterElements, dtype=int)
    for centerSiteIndex, centerSiteFractCoord in enumerate(
                                                        centerSiteFractCoords):
        iDisplacementVectors = []
        iLatticeDirectionList = []
        iDisplacements = []
        iBridgeList = []
        if class_list:
            iClassPairList = []
        for neighborSiteIndex, neighborSiteFractCoord in enumerate(
                                                    neighborSiteFractCoords):
            latticeDirection = neighborSiteFractCoord - centerSiteFractCoord
            neighborDisplacementVector = np.dot(latticeDirection[None, :],
                                                lattice_matrix)
            displacement = np.linalg.norm(neighborDisplacementVector)
            if (neighborCutoffDistLimits[0]
                    < displacement
                    <= neighborCutoffDistLimits[1]):
                iDisplacementVectors.append(neighborDisplacementVector)
                iDisplacements.append(displacement)
                numNeighbors[centerSiteIndex] += 1
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
                               centerSiteClassList[centerSiteIndex])
                        print ('neighbor class:',
                               neighborSiteClassList[neighborSiteIndex])
                        print ('num of bonds:',
                               len(bridgeNeighborList[centerSiteIndex]))
                        print ('center:',
                               np.round(np.divide(centerSiteFractCoord,
                                                  desiredSystemSize), 3))
                        print ('neighbor:',
                               np.round(np.divide(neighborSiteFractCoord,
                                                  desiredSystemSize), 3))

                # determine class pair list
                if class_list:
                    iClassPairList.append(
                        str(centerSiteClassList[centerSiteIndex])
                        + ':' + str(neighborSiteClassList[neighborSiteIndex]))

                # determine bridging species
                bridgeSiteExists = 0
                bridgeSiteType = ''
                for iCenterNeighborSEIndex in bridgeNeighborList[
                                                            centerSiteIndex]:
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
                                generateQuantumIndices(system_size,
                                                       bridgeSiteIndex,
                                                       n_elements_per_unit_cell))
                        bridgeSiteType += (
                                (', ' if bridgeSiteType != '' else '')
                                + element_types[bridgeSiteQuantumIndices[3]])
                if not bridgeSiteExists:
                    bridgeSiteType = 'space'
                iBridgeList.append(bridgeSiteType)

        bridgeList[centerSiteIndex] = np.asarray(iBridgeList)
        displacementVectorList[centerSiteIndex] = np.asarray(
                                                        iDisplacementVectors)
        latticeDirectionList[centerSiteIndex] = np.asarray(
                                                        iLatticeDirectionList)
        displacementList[centerSiteIndex] = np.asarray(iDisplacements)
        if class_list:
            classPairList[centerSiteIndex] = np.asarray(iClassPairList)

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
                        displacementList[iCenterElementIndex][
                            displacementList[iCenterElementIndex].argsort()])
        sortedBridgeList[iCenterElementIndex] = (
                        bridgeList[iCenterElementIndex][
                            displacementList[iCenterElementIndex].argsort()])
        if class_list:
            sortedClassPairList[iCenterElementIndex] = (
                        classPairList[iCenterElementIndex][
                            displacementList[iCenterElementIndex].argsort()])
        if roundLatticeParameters:
            latticeDirectionList[iCenterElementIndex] = (
                latticeDirectionList[iCenterElementIndex] / base).astype(int)
            centerSiteLDList = latticeDirectionList[iCenterElementIndex]
            for index in range(numNeighbors[iCenterElementIndex]):
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
                            displacementList[iCenterElementIndex].argsort()])

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

    latticeDirectionListFileName = ('latticeDirectionList_' + centerElementType
                                    + '-' + neighborElementType + '_cutoff='
                                    + str(neighborCutoff))
    displacementListFileName = ('displacementList_' + centerElementType + '-'
                                + neighborElementType + '_cutoff='
                                + str(neighborCutoff))
    pathwayFileName = ('pathwayList_' + centerElementType + '-'
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
