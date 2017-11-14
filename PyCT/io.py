#!/usr/bin/env python

import numpy as np


def readPOSCAR(inputFilePath):
    latticeMatrix = np.zeros((3, 3))
    latticeParameterIndex = 0
    latticeParametersLineRange = range(3, 6)
    inputFile = open(inputFilePath, 'r')
    for lineIndex, line in enumerate(inputFile):
        lineNumber = lineIndex + 1
        if lineNumber in latticeParametersLineRange:
            latticeMatrix[latticeParameterIndex, :] = np.fromstring(
                                                            line, sep=' ')
            latticeParameterIndex += 1
        elif lineNumber == 6:
            elementTypes = line.split()
        elif lineNumber == 7:
            nElementsPerUnitCell = np.fromstring(line, dtype=int, sep=' ')
            totalElementsPerUnitCell = nElementsPerUnitCell.sum()
            fractionalUnitCellCoords = np.zeros((totalElementsPerUnitCell,
                                                 3))
            elementIndex = 0
        elif lineNumber > 8 and elementIndex < totalElementsPerUnitCell:
            fractionalUnitCellCoords[elementIndex, :] = np.fromstring(
                                                            line, sep=' ')
            elementIndex += 1
    inputFile.close()
    POSCAR_INFO = np.array([latticeMatrix, elementTypes,
                            nElementsPerUnitCell, totalElementsPerUnitCell,
                            fractionalUnitCellCoords], dtype=object)
    return POSCAR_INFO
