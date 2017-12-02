#!/usr/bin/env python

import os

import numpy as np


def readPOSCAR(inputFilePath):
    latticeMatrix = np.zeros((3, 3))
    latticeParameterIndex = 0
    latticeParametersLineRange = range(3, 6)
    inputFile = open(inputFilePath, 'r')
    for lineIndex, line in enumerate(inputFile):
        lineNumber = lineIndex + 1
        if lineNumber in latticeParametersLineRange:
            latticeMatrix[latticeParameterIndex, :] = np.fromstring(line,
                                                                    sep=' ')
            latticeParameterIndex += 1
        elif lineNumber == 6:
            elementTypes = line.split()
        elif lineNumber == 7:
            nElementsPerUnitCell = np.fromstring(line, dtype=int, sep=' ')
            totalElementsPerUnitCell = nElementsPerUnitCell.sum()
            fractionalUnitCellCoords = np.zeros((totalElementsPerUnitCell, 3))
            elementIndex = 0
        elif lineNumber > 8 and elementIndex < totalElementsPerUnitCell:
            fractionalUnitCellCoords[elementIndex, :] = np.fromstring(line,
                                                                      sep=' ')
            elementIndex += 1
    inputFile.close()
    output = np.array([latticeMatrix, elementTypes, nElementsPerUnitCell,
                       fractionalUnitCellCoords], dtype=object)
    return output


def generateQuantumIndices(system_size, systemElementIndex,
                           nElementsPerUnitCell):
    """Returns the quantum indices of the element"""
    quantumIndices = np.zeros(5, dtype=int)  # [0] * 5
    totalElementsPerUnitCell = nElementsPerUnitCell.sum()
    unitcellElementIndex = systemElementIndex % totalElementsPerUnitCell
    quantumIndices[3] = np.where(np.cumsum(nElementsPerUnitCell)
                                 >= (unitcellElementIndex + 1))[0][0]
    quantumIndices[4] = (unitcellElementIndex
                         - nElementsPerUnitCell[:quantumIndices[3]].sum())
    nFilledUnitCells = ((systemElementIndex - unitcellElementIndex)
                        / totalElementsPerUnitCell)
    for index in range(3):
        quantumIndices[index] = nFilledUnitCells / system_size[index+1:].prod()
        nFilledUnitCells -= quantumIndices[index] * system_size[index+1:].prod()
    return quantumIndices


def generateUniquePathways(inputFileLocation, cutoffDistKey, neighborCutoff,
                           bridgeCutoff, outdir, pathwayPrec, equivalencyPrec,
                           classList=[], avoidElementType='',
                           roundLatticeParameters={}, printPathwayList=0,
                           printEquivalency=0, desiredCoordinateParameters={}):
    """ generate unique pathways for the given set of element types"""
    # define input parameters
    [latticeMatrix, elementTypes, nElementsPerUnitCell,
     fractionalUnitCellCoords] = readPOSCAR(inputFileLocation)
    nElementTypes = len(elementTypes)
    totalElementsPerUnitCell = nElementsPerUnitCell.sum()
    elementTypeIndexList = np.repeat(np.arange(nElementTypes),
                                     nElementsPerUnitCell)
    [centerElementType, neighborElementType] = cutoffDistKey.split(':')
    centerSiteElementTypeIndex = elementTypes.index(centerElementType)
    neighborCutoffDistLimits = [0, neighborCutoff]
    bridgeCutoffDistLimits = [0, bridgeCutoff]
    if roundLatticeParameters:
        base = roundLatticeParameters['base']
        prec = roundLatticeParameters['prec']

    # sort element wise coordinates in ascending order of z-coordinate
    startIndex = 0
    for elementIndex in range(nElementTypes):
        endIndex = startIndex + nElementsPerUnitCell[elementIndex]
        elementUnitCellCoords = fractionalUnitCellCoords[elementTypeIndexList
                                                         == elementIndex]
        fractionalUnitCellCoords[startIndex:endIndex] = elementUnitCellCoords[
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
        avoidElementTypeIndex = elementTypes.index(avoidElementType)
        systemElementIndexOffsetArray = np.repeat(
                            np.arange(0, totalElementsPerUnitCell * numCells,
                                      totalElementsPerUnitCell),
                            nElementsPerUnitCell[centerSiteElementTypeIndex])
        avoidElementIndices = (
            np.tile(nElementsPerUnitCell[:avoidElementTypeIndex].sum()
                    + np.arange(0,
                                nElementsPerUnitCell[avoidElementTypeIndex]),
                    numCells)
            + systemElementIndexOffsetArray)
    else:
        avoidElementIndices = []

    # extract center site fractional coordinates
    numCenterElements = nElementsPerUnitCell[centerSiteElementTypeIndex]
    centerSiteIndices = (nElementsPerUnitCell[
                                            :centerSiteElementTypeIndex].sum()
                         + np.arange(numCenterElements))
    centerSiteFractCoords = fractionalUnitCellCoords[centerSiteIndices]

    # generate fractional coordinates for neighbor sites
    # and all system elements
    neighborSiteFractCoords = np.zeros((numCells * numCenterElements, 3))
    systemFractCoords = np.zeros((numCells * totalElementsPerUnitCell, 3))
    for iCell in range(numCells):
        neighborSiteFractCoords[(iCell * numCenterElements):(
                                        (iCell + 1) * numCenterElements)] \
            = (centerSiteFractCoords + unitcellTranslationalCoords[iCell])
        systemFractCoords[(iCell * totalElementsPerUnitCell):(
                                    (iCell + 1) * totalElementsPerUnitCell)] \
            = (fractionalUnitCellCoords + unitcellTranslationalCoords[iCell])

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
                                                    latticeMatrix)
                displacement = np.linalg.norm(neighborDisplacementVector)
                if (bridgeCutoffDistLimits[0]
                        < displacement
                        <= bridgeCutoffDistLimits[1]):
                    iBridgeNeighborList.append(neighborSiteIndex)
        bridgeNeighborList[centerSiteIndex] = np.asarray(iBridgeNeighborList)

    # initialize class pair list
    if classList:
        centerSiteClassList = classList[0]
        neighborSiteClassList = np.tile(classList[1], numCells)
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
        if classList:
            iClassPairList = []
        for neighborSiteIndex, neighborSiteFractCoord in enumerate(
                                                    neighborSiteFractCoords):
            latticeDirection = neighborSiteFractCoord - centerSiteFractCoord
            neighborDisplacementVector = np.dot(latticeDirection[None, :],
                                                latticeMatrix)
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
                if classList:
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
                                            latticeMatrix)
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
                                                       nElementsPerUnitCell))
                        bridgeSiteType += (
                                (', ' if bridgeSiteType != '' else '')
                                + elementTypes[bridgeSiteQuantumIndices[3]])
                if not bridgeSiteExists:
                    bridgeSiteType = 'space'
                iBridgeList.append(bridgeSiteType)

        bridgeList[centerSiteIndex] = np.asarray(iBridgeList)
        displacementVectorList[centerSiteIndex] = np.asarray(
                                                        iDisplacementVectors)
        latticeDirectionList[centerSiteIndex] = np.asarray(
                                                        iLatticeDirectionList)
        displacementList[centerSiteIndex] = np.asarray(iDisplacements)
        if classList:
            classPairList[centerSiteIndex] = np.asarray(iClassPairList)

    # determine irreducible form of lattice directions
    from fractions import gcd
    sortedLatticeDirectionList = np.empty(numCenterElements, dtype=object)
    sortedDisplacementList = np.empty(numCenterElements, dtype=object)
    if classList:
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
        if classList:
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
            if classList:
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
        if classList:
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
