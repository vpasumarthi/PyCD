#!/usr/bin/env python
"""
kMC model to run kinetic Monte Carlo simulations and compute mean square displacement of 
random walk of charge carriers on 3D lattice systems
"""
import numpy as np
from collections import OrderedDict
import itertools
import random as rnd
from datetime import datetime
import pickle
import os.path
from copy import deepcopy
import platform

directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'

class material(object):
    """Defines the properties and structure of working material
    
    :param str name: A string representing the material name
    :param list elementTypes: list of chemical elements
    :param dict speciesToElementTypeMap: list of charge carrier species
    :param unitcellCoords: positions of all elements in the unit cell
    :type unitcellCoords: np.array (nx3)
    :param elementTypeIndexList: list of element types for all unit cell coordinates
    :type elementTypeIndexList: np.array (n)
    :param dict chargeTypes: types of atomic charges considered for the working material
    :param list latticeParameters: list of three lattice constants in angstrom and three angles between them in degrees
    :param float vn: typical frequency for nuclear motion
    :param dict lambdaValues: Reorganization energies
    :param dict VAB: Electronic coupling matrix element
    :param dict neighborCutoffDist: List of neighbors and their respective cutoff distances in angstrom
    :param float neighborCutoffDistTol: Tolerance value in angstrom for neighbor cutoff distance
    :param str elementTypeDelimiter: Delimiter between element types
    :param str emptySpeciesType: name of the empty species type
    :param str siteIdentifier: suffix to the chargeType to identify site
    :param float epsilon: Dielectric constant of the material
    
    The additional attributes are:
        * **nElementsPerUnitCell** (np.array (n)): element-type wise total number of elements in a unit cell
        * **siteList** (list): list of elements that act as sites
        * **elementTypeToSpeciesMap** (dict): dictionary of element to species mapping
        * **nonEmptySpeciesToElementTypeMap** (dict): dictionary of species to element mapping with elements excluding emptySpeciesType 
        * **hopElementTypes** (dict): dictionary of species to hopping element types separated by elementTypeDelimiter
        * **latticeMatrix** (np.array (3x3): lattice cell matrix
    """ 
    def __init__(self, materialParameters):
        # CONSTANTS
        self.EPSILON0 = 8.854187817E-12 # Electric constant in F.m-1
        self.ANG = 1E-10 # Angstrom in m
        self.KB = 1.38064852E-23 # Boltzmann constant in J/K
        
        # FUNDAMENTAL ATOMIC UNITS (Ref: http://physics.nist.gov/cuu/Constants/Table/allascii.txt)
        self.EMASS = 9.10938356E-31 # Electron mass in Kg
        self.ECHARGE = 1.6021766208E-19 # Elementary charge in C
        self.HBAR =  1.054571800E-34 # Reduced Planck's constant in J.sec
        self.KE = 1 / (4 * np.pi * self.EPSILON0)
        
        # DERIVED ATOMIC UNITS
        self.BOHR = self.HBAR**2 / (self.EMASS * self.ECHARGE**2 * self.KE) # Bohr radius in m
        self.HARTREE = self.HBAR**2 / (self.EMASS * self.BOHR**2) # Hartree in J
        self.AUTIME = self.HBAR / self.HARTREE # sec
        self.AUTEMPERATURE = self.HARTREE / self.KB # K
        
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
        
        # TODO: introduce a method to view the material using ase atoms or other gui module
        self.name = materialParameters.name
        self.elementTypes = materialParameters.elementTypes[:]
        self.speciesTypes = materialParameters.speciesTypes[:]
        self.numSpeciesTypes = len(self.speciesTypes)
        self.speciesChargeList = materialParameters.speciesChargeList[:]
        self.speciesToElementTypeMap = deepcopy(materialParameters.speciesToElementTypeMap)
        self.unitcellCoords = np.zeros((len(materialParameters.unitcellCoords), 3)) # Initialization
        startIndex = 0
        self.nElementTypes = len(self.elementTypes)
        nElementsPerUnitCell = np.zeros(self.nElementTypes, int)
        for elementIndex in range(self.nElementTypes):
            elementUnitCellCoords = materialParameters.unitcellCoords[materialParameters.elementTypeIndexList==elementIndex]
            nElementsPerUnitCell[elementIndex] = len(elementUnitCellCoords)
            endIndex = startIndex + nElementsPerUnitCell[elementIndex]
            self.unitcellCoords[startIndex:endIndex] = elementUnitCellCoords[elementUnitCellCoords[:,2].argsort()]
            startIndex = endIndex
        
        self.unitcellCoords *= self.ANG2BOHR # Unit cell coordinates converted to atomic units
        self.nElementsPerUnitCell = np.copy(nElementsPerUnitCell)
        self.totalElementsPerUnitCell = nElementsPerUnitCell.sum()
        self.elementTypeIndexList = np.sort(materialParameters.elementTypeIndexList.astype(int))
        self.chargeTypes = deepcopy(materialParameters.chargeTypes)
        
        self.latticeParameters = [0] * len(materialParameters.latticeParameters)
        # lattice parameters being converted to atomic units
        for index in range(len(materialParameters.latticeParameters)):
            if index < 3:
                self.latticeParameters[index] = materialParameters.latticeParameters[index] * self.ANG2BOHR
            else:
                self.latticeParameters[index] = materialParameters.latticeParameters[index]
        
        self.vn = materialParameters.vn / self.SEC2AUTIME
        self.lambdaValues = deepcopy(materialParameters.lambdaValues)
        self.lambdaValues.update((x, [y[index] * self.EV2J * self.J2HARTREE for index in range(len(y))]) for x, y in self.lambdaValues.items())

        self.VAB = deepcopy(materialParameters.VAB)
        self.VAB.update((x, [y[index] * self.EV2J * self.J2HARTREE for index in range(len(y))]) for x, y in self.VAB.items())
        
        self.neighborCutoffDist = deepcopy(materialParameters.neighborCutoffDist)
        self.neighborCutoffDist.update((x, [(y[index] * self.ANG2BOHR) if y[index] else None for index in range(len(y))]) for x, y in self.neighborCutoffDist.items())
        self.neighborCutoffDistTol = deepcopy(materialParameters.neighborCutoffDistTol)
        self.neighborCutoffDistTol.update((x, [(y[index] * self.ANG2BOHR) if y[index] else None for index in range(len(y))]) for x, y in self.neighborCutoffDistTol.items())
        
        self.elementTypeDelimiter = materialParameters.elementTypeDelimiter
        self.emptySpeciesType = materialParameters.emptySpeciesType
        self.siteIdentifier = materialParameters.siteIdentifier
        self.dielectricConstant = materialParameters.dielectricConstant
                
        siteList = [self.speciesToElementTypeMap[key] for key in self.speciesToElementTypeMap 
                    if key is not self.emptySpeciesType]
        self.siteList = list(set([item for sublist in siteList for item in sublist]))
        self.nonEmptySpeciesToElementTypeMap = deepcopy(self.speciesToElementTypeMap)
        del self.nonEmptySpeciesToElementTypeMap[self.emptySpeciesType]
        
        self.elementTypeToSpeciesMap = {}
        for elementType in self.elementTypes:
            speciesList = []
            for speciesTypeKey in self.nonEmptySpeciesToElementTypeMap.keys():
                if elementType in self.nonEmptySpeciesToElementTypeMap[speciesTypeKey]:
                    speciesList.append(speciesTypeKey)
            self.elementTypeToSpeciesMap[elementType] = speciesList[:]
        
        self.hopElementTypes = {key: [self.elementTypeDelimiter.join(comb) 
                                      for comb in list(itertools.product(self.speciesToElementTypeMap[key], repeat=2))] 
                                for key in self.speciesToElementTypeMap if key is not self.emptySpeciesType}
        [a, b, c, alpha, beta, gamma] = self.latticeParameters
        self.latticeMatrix = np.array([[ a                , 0                , 0],
                                       [ b * np.cos(gamma), b * np.sin(gamma), 0],
                                       [ c * np.cos(alpha), c * np.cos(beta) , c * np.sqrt(np.sin(alpha)**2 - np.cos(beta)**2)]])
        
    def generateMaterialFile(self, material, materialFileName):
        """ """
        file_material = open(materialFileName, 'w')
        pickle.dump(material, file_material)
        file_material.close()
        pass

    def generateSites(self, elementTypeIndices, cellSize=np.array([1, 1, 1])):
        """Returns systemElementIndices and coordinates of specified elements in a cell of size 
        *cellSize*
            
        :param str elementTypeIndices: element type indices
        :param cellSize: size of the cell
        :type cellSize: np.array (3x1)
        :return: an object with following attributes:
        
            * **cellCoordinates** (np.array (nx3)):  
            * **quantumIndexList** (np.array (nx5)): 
            * **systemElementIndexList** (np.array (n)): 
        
        :raises ValueError: if the input cellSize is less than or equal to 0.
        """
        assert all(size > 0 for size in cellSize), 'Input size should always be greater than 0'
        extractIndices = np.in1d(self.elementTypeIndexList, elementTypeIndices).nonzero()[0]
        unitcellElementCoords = self.unitcellCoords[extractIndices]
        numCells = cellSize.prod()
        nSitesPerUnitCell = self.nElementsPerUnitCell[elementTypeIndices].sum()
        unitcellElementIndexList = np.arange(nSitesPerUnitCell)
        unitcellElementTypeIndex = np.reshape(np.concatenate((np.asarray([[elementTypeIndex] * self.nElementsPerUnitCell[elementTypeIndex] 
                                                                          for elementTypeIndex in elementTypeIndices]))), (nSitesPerUnitCell, 1))
        unitCellElementTypeElementIndexList = np.reshape(np.concatenate(([np.arange(self.nElementsPerUnitCell[elementTypeIndex]) 
                                                                          for elementTypeIndex in elementTypeIndices])), (nSitesPerUnitCell, 1))
        cellCoordinates = np.zeros((numCells * nSitesPerUnitCell, 3)) # Initialization
        # quantumIndex = [unitCellIndex, elementTypeIndex, elementIndex] # Definition format of Quantum Indices
        quantumIndexList = np.zeros((numCells * nSitesPerUnitCell, 5), dtype=int)
        systemElementIndexList = np.zeros(numCells * nSitesPerUnitCell, dtype=int)
        iUnitCell = 0
        for xIndex in range(cellSize[0]):
            for yIndex in range(cellSize[1]):  
                for zIndex in range(cellSize[2]): 
                    startIndex = iUnitCell * nSitesPerUnitCell
                    endIndex = startIndex + nSitesPerUnitCell
                    # TODO: Any reason to use fractional coordinates?
                    unitcellTranslationalCoords = np.dot([xIndex, yIndex, zIndex], self.latticeMatrix)
                    newCellSiteCoords = unitcellElementCoords + unitcellTranslationalCoords
                    cellCoordinates[startIndex:endIndex] = newCellSiteCoords
                    systemElementIndexList[startIndex:endIndex] = iUnitCell * nSitesPerUnitCell + unitcellElementIndexList
                    quantumIndexList[startIndex:endIndex] = np.hstack((np.tile(np.array([xIndex, yIndex, zIndex]), (nSitesPerUnitCell, 1)), unitcellElementTypeIndex, unitCellElementTypeElementIndexList))
                    iUnitCell += 1
        
        returnSites = returnValues()
        returnSites.cellCoordinates = cellCoordinates
        returnSites.quantumIndexList = quantumIndexList
        returnSites.systemElementIndexList = systemElementIndexList
        return returnSites
    
class neighbors(object):
    """Returns the neighbor list file
    :param systemSize: size of the super cell in terms of number of unit cell in three dimensions
    :type systemSize: np.array (3x1)
    """
    
    def __init__(self, material, systemSize=np.array([10, 10, 10]), pbc=[1, 1, 1]):
        self.startTime = datetime.now()
        self.material = material
        self.systemSize = systemSize
        self.pbc = pbc[:]
        
        # total number of unit cells
        self.numCells = self.systemSize.prod()
        self.numSystemElements = self.numCells * self.material.totalElementsPerUnitCell
        
        # generate all sites in the system
        self.elementTypeIndices = range(self.material.nElementTypes)
        self.bulkSites = self.material.generateSites(self.elementTypeIndices, self.systemSize)

        xRange = range(-1, 2) if self.pbc[0] == 1 else [0]
        yRange = range(-1, 2) if self.pbc[1] == 1 else [0]
        zRange = range(-1, 2) if self.pbc[2] == 1 else [0]
        self.unitcellTranslationalCoords = np.zeros((3**sum(self.pbc), 3)) # Initialization
        index = 0
        for xOffset in xRange:
            for yOffset in yRange:
                for zOffset in zRange:
                    self.unitcellTranslationalCoords[index] = np.dot(np.multiply(np.array([xOffset, yOffset, zOffset]), systemSize), self.material.latticeMatrix)
                    index += 1

    def generateNeighborsFile(self, materialNeighbors, neighborsFileName):
        """ """
        file_Neighbors = open(neighborsFileName, 'w')
        pickle.dump(materialNeighbors, file_Neighbors)
        file_Neighbors.close()
        pass
    
    def generateSystemElementIndex(self, systemSize, quantumIndices):
        """Returns the systemElementIndex of the element"""
        #assert type(systemSize) is np.ndarray, 'Please input systemSize as a numpy array'
        #assert type(quantumIndices) is np.ndarray, 'Please input quantumIndices as a numpy array'
        #assert np.all(systemSize > 0), 'System size should be positive in all dimensions'
        #assert all(quantumIndex >= 0 for quantumIndex in quantumIndices), 'Quantum Indices cannot be negative'
        #assert quantumIndices[-1] < self.material.nElementsPerUnitCell[quantumIndices[-2]], 'Element Index exceed number of elements of the specified element type'
        #assert np.all(quantumIndices[:3] < systemSize), 'Unit cell indices exceed the given system size'
        unitCellIndex = np.copy(quantumIndices[:3])
        [elementTypeIndex, elementIndex] = quantumIndices[-2:]
        systemElementIndex = elementIndex + self.material.nElementsPerUnitCell[:elementTypeIndex].sum()
        nDim = len(systemSize)
        for index in range(nDim):
            if index == 0:
                systemElementIndex += self.material.totalElementsPerUnitCell * unitCellIndex[nDim-1-index]
            else:
                systemElementIndex += self.material.totalElementsPerUnitCell * unitCellIndex[nDim-1-index] * systemSize[-index:].prod()
        return systemElementIndex
    
    def generateQuantumIndices(self, systemSize, systemElementIndex):
        """Returns the quantum indices of the element"""
        #assert systemElementIndex >= 0, 'System Element Index cannot be negative'
        #assert systemElementIndex < systemSize.prod() * self.material.totalElementsPerUnitCell, 'System Element Index out of range for the given system size'
        quantumIndices = np.zeros(5, dtype=int)#[0] * 5
        unitcellElementIndex = systemElementIndex % self.material.totalElementsPerUnitCell
        quantumIndices[3] = np.where(self.material.nElementsPerUnitCell.cumsum() >= (unitcellElementIndex + 1))[0][0]
        quantumIndices[4] = unitcellElementIndex - self.material.nElementsPerUnitCell[:quantumIndices[3]].sum()
        nFilledUnitCells = (systemElementIndex - unitcellElementIndex) / self.material.totalElementsPerUnitCell
        for index in range(3):
            quantumIndices[index] = nFilledUnitCells / systemSize[index+1:].prod()
            nFilledUnitCells -= quantumIndices[index] * systemSize[index+1:].prod()
        return quantumIndices
    
    def computeCoordinates(self, systemSize, systemElementIndex):
        """Returns the coordinates in atomic units of the given system element index for a given system size"""
        quantumIndices = self.generateQuantumIndices(systemSize, systemElementIndex)
        unitcellTranslationalCoords = np.dot(quantumIndices[:3], self.material.latticeMatrix)
        coordinates = unitcellTranslationalCoords + self.material.unitcellCoords[quantumIndices[4] + self.material.nElementsPerUnitCell[:quantumIndices[3]].sum()]
        return coordinates
    
    def computeDistance(self, systemSize, systemElementIndex1, systemElementindex):
        """Returns the distance in atomic units between the two system element indices for a given system size"""
        centerCoord = self.computeCoordinates(systemSize, systemElementIndex1)
        neighborCoord = self.computeCoordinates(systemSize, systemElementindex)
        
        neighborImageCoords = self.unitcellTranslationalCoords + neighborCoord
        neighborImageDisplacementVectors = neighborImageCoords - centerCoord
        neighborImageDisplacements = np.linalg.norm(neighborImageDisplacementVectors, axis=1)
        displacement = np.min(neighborImageDisplacements)
        return displacement
    
    def hopNeighborSites(self, bulkSites, centerSiteIndices, neighborSiteIndices, cutoffDistLimits, cutoffDistKey):
        """Returns systemElementIndexMap and distances between center sites and its neighbor sites within cutoff 
        distance"""
        neighborSiteCoords = bulkSites.cellCoordinates[neighborSiteIndices]
        neighborSiteSystemElementIndexList = bulkSites.systemElementIndexList[neighborSiteIndices]
        centerSiteCoords = bulkSites.cellCoordinates[centerSiteIndices]
        centerSiteSystemElementIndexList = bulkSites.systemElementIndexList[centerSiteIndices]
        
        neighborSystemElementIndices = np.empty(len(centerSiteCoords), dtype=object)
        displacementVectorList = np.empty(len(centerSiteCoords), dtype=object)
        numNeighbors = np.array([], dtype=int)

        if cutoffDistKey == 'O:O':
            quickTest = 0 # commit reference: 1472bb4
        else:
            quickTest = 0                

        for centerSiteIndex, centerCoord in enumerate(centerSiteCoords):
            iNeighborSiteIndexList = []
            iDisplacementVectors = []
            iNumNeighbors = 0
            if quickTest:
                displacementList = np.zeros(len(neighborSiteCoords))
            for neighborSiteIndex, neighborCoord in enumerate(neighborSiteCoords):
                neighborImageCoords = self.unitcellTranslationalCoords + neighborCoord
                neighborImageDisplacementVectors = neighborImageCoords - centerCoord
                neighborImageDisplacements = np.linalg.norm(neighborImageDisplacementVectors, axis=1)
                [displacement, imageIndex] = [np.min(neighborImageDisplacements), np.argmin(neighborImageDisplacements)]
                if quickTest:
                    displacementList[neighborSiteIndex] = displacement
                if cutoffDistLimits[0] < displacement <= cutoffDistLimits[1]:
                    iNeighborSiteIndexList.append(neighborSiteIndex)
                    iDisplacementVectors.append(neighborImageDisplacementVectors[imageIndex])
                    iNumNeighbors += 1
            neighborSystemElementIndices[centerSiteIndex] = neighborSiteSystemElementIndexList[iNeighborSiteIndexList]
            displacementVectorList[centerSiteIndex] = np.asarray(iDisplacementVectors)
            numNeighbors = np.append(numNeighbors, iNumNeighbors)
            if quickTest:
#                 print np.sort(displacementList)[:10] / self.material.ANG2BOHR
                for cutoffDist in range(2, 7):
                    cutoff = cutoffDist * self.material.ANG2BOHR
                    print cutoffDist
                    print displacementList[displacementList < cutoff].shape
                    print np.unique(np.sort(np.round(displacementList[displacementList < cutoff] / self.material.ANG2BOHR, 4))).shape
                    print np.unique(np.sort(np.round(displacementList[displacementList < cutoff] / self.material.ANG2BOHR, 3))).shape
                    print np.unique(np.sort(np.round(displacementList[displacementList < cutoff] / self.material.ANG2BOHR, 2))).shape
                    print np.unique(np.sort(np.round(displacementList[displacementList < cutoff] / self.material.ANG2BOHR, 1))).shape
                    print np.unique(np.sort(np.round(displacementList[displacementList < cutoff] / self.material.ANG2BOHR, 0))).shape
                import pdb; pdb.set_trace()
                    
            
        returnNeighbors = returnValues()
        returnNeighbors.neighborSystemElementIndices = neighborSystemElementIndices
        returnNeighbors.displacementVectorList = displacementVectorList
        returnNeighbors.numNeighbors = numNeighbors
        return returnNeighbors
    
    def cumulativeDisplacementList(self, systemSize, dstPath):
        """Returns cumulative displacement list for the given system size printed out to disk"""
        cumulativeDisplacementList = np.zeros((self.numSystemElements, self.numSystemElements, 3))
        for centerSiteIndex, centerCoord in enumerate(self.bulkSites.cellCoordinates):
            cumulativeUnitCellTranslationalCoords = np.tile(self.unitcellTranslationalCoords, (self.numSystemElements, 1, 1))
            cumulativeNeighborImageCoords = cumulativeUnitCellTranslationalCoords + np.tile(self.bulkSites.cellCoordinates[:, np.newaxis, :], (1, len(self.unitcellTranslationalCoords), 1))
            cumulativeNeighborImageDisplacementVectors = cumulativeNeighborImageCoords - centerCoord
            cumulativeNeighborImageDisplacements = np.linalg.norm(cumulativeNeighborImageDisplacementVectors, axis=2)
            cumulativeDisplacementList[centerSiteIndex] = cumulativeNeighborImageDisplacementVectors[np.arange(self.numSystemElements), np.argmin(cumulativeNeighborImageDisplacements, axis=1)]
        return cumulativeDisplacementList
    
    def generateNeighborList(self, neighborListDirPath, generateCumDispList=0, report=1, localSystemSize=np.array([3, 3, 3]), 
                             centerUnitCellIndex=np.array([1, 1, 1])):
        """Adds the neighbor list to the system object and returns the neighbor list"""
        assert neighborListDirPath, 'Please provide the path to the parent directory of neighbor list files'
        assert all(size >= 3 for size in localSystemSize), 'Local system size in all dimensions should always be greater than or equal to 3'
        
        dstPath = neighborListDirPath
        if not os.path.exists(dstPath):
            os.makedirs(dstPath)
        hopNeighborListFilePath = dstPath + directorySeparator + 'hopNeighborList.npy'

        hopNeighborList = {}
        tolDist = self.material.neighborCutoffDistTol
        elementTypes = self.material.elementTypes[:]
        
        for cutoffDistKey in self.material.neighborCutoffDist.keys():
            cutoffDistList = self.material.neighborCutoffDist[cutoffDistKey][:]
            neighborListCutoffDistKey = []
            [centerElementType, neighborElementType] = cutoffDistKey.split(self.material.elementTypeDelimiter)
            centerSiteElementTypeIndex = elementTypes.index(centerElementType) 
            neighborSiteElementTypeIndex = elementTypes.index(neighborElementType)
            localBulkSites = self.material.generateSites(self.elementTypeIndices, 
                                                         self.systemSize)
            systemElementIndexOffsetArray = (np.repeat(np.arange(0, self.material.totalElementsPerUnitCell * self.numCells, self.material.totalElementsPerUnitCell), 
                                                       self.material.nElementsPerUnitCell[centerSiteElementTypeIndex]))
            centerSiteIndices = neighborSiteIndices = (np.tile(self.material.nElementsPerUnitCell[:centerSiteElementTypeIndex].sum() + 
                                                               np.arange(0, self.material.nElementsPerUnitCell[centerSiteElementTypeIndex]), self.numCells) + systemElementIndexOffsetArray)
            
            for iCutoffDist in range(len(cutoffDistList)):
                cutoffDistLimits = [cutoffDistList[iCutoffDist] - tolDist[cutoffDistKey][iCutoffDist], cutoffDistList[iCutoffDist] + tolDist[cutoffDistKey][iCutoffDist]]
                
                neighborListCutoffDistKey.append(self.hopNeighborSites(localBulkSites, centerSiteIndices, 
                                                                       neighborSiteIndices, cutoffDistLimits, cutoffDistKey))
            hopNeighborList[cutoffDistKey] = neighborListCutoffDistKey[:]
        np.save(hopNeighborListFilePath, hopNeighborList)
        
        if generateCumDispList:
            cumulativeDisplacementListFilePath = dstPath + directorySeparator + 'cumulativeDisplacementList.npy'
            cumulativeDisplacementList = self.cumulativeDisplacementList(self.systemSize, dstPath)
            np.save(cumulativeDisplacementListFilePath, cumulativeDisplacementList)

        if report:
            self.generateNeighborListReport(dstPath)

    def generateNeighborListReport(self, dstPath):
        """Generates a neighbor list and prints out a report to the output directory"""
        neighborListLogName = 'NeighborList.log' 
        neighborListLogPath = dstPath + directorySeparator + neighborListLogName
        report = open(neighborListLogPath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - self.startTime
        report.write('Time elapsed: ' + ('%2d days, ' % timeElapsed.days if timeElapsed.days else '') +
                     ('%2d hours' % ((timeElapsed.seconds // 3600) % 24)) + 
                     (', %2d minutes' % ((timeElapsed.seconds // 60) % 60)) + 
                     (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()

    def generateHematiteNeighborSEIndices(self, dstPath, report=1):
        startTime = datetime.now()
        offsetList = np.array([[[-1, 0, -1], [0, 0, -1], [0, 1, -1], [0, 0, -1]],
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
        basalNeighborElementSiteIndices = np.array([11, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 0])
        cNeighborElementSiteIndices = np.array([9, 4, 11, 6, 1, 8, 3, 10, 5, 0, 7, 2])
        numBasalNeighbors = 3
        numCNeighbors = 1
        numNeighbors = numBasalNeighbors + numCNeighbors
        nElementsPerUnitCell = self.material.nElementsPerUnitCell[elementTypeIndex]
        neighborElementSiteIndices = np.zeros((nElementsPerUnitCell, 4), int)
        for iNeighbor in range(numNeighbors):
            if iNeighbor < numBasalNeighbors:
                neighborElementSiteIndices[:, iNeighbor] = basalNeighborElementSiteIndices
            else:
                neighborElementSiteIndices[:, iNeighbor] = cNeighborElementSiteIndices
        localBulkSites = self.material.generateSites(self.elementTypeIndices, self.systemSize)
        systemElementIndexOffsetArray = (np.repeat(np.arange(0, self.material.totalElementsPerUnitCell * self.numCells, self.material.totalElementsPerUnitCell), 
                                                   self.material.nElementsPerUnitCell[elementTypeIndex]))
        centerSiteSEIndices = (np.tile(self.material.nElementsPerUnitCell[:elementTypeIndex].sum() + 
                                     np.arange(0, self.material.nElementsPerUnitCell[elementTypeIndex]), self.numCells) + systemElementIndexOffsetArray)
        numCenterSiteElements = len(centerSiteSEIndices)
        neighborSystemElementIndices = np.zeros((numCenterSiteElements, numNeighbors))
        
        for centerSiteIndex, centerSiteSEIndex in enumerate(centerSiteSEIndices):
            centerSiteQuantumIndices = self.generateQuantumIndices(self.systemSize, centerSiteSEIndex)
            centerSiteUnitCellIndices = centerSiteQuantumIndices[:3]
            centerSiteElementSiteIndex = centerSiteQuantumIndices[-1:][0]
            for neighborIndex in range(numNeighbors):
                neighborUnitCellIndices = centerSiteUnitCellIndices + offsetList[centerSiteElementSiteIndex][neighborIndex]
                for index, neighborUnitCellIndex in enumerate(neighborUnitCellIndices):
                    if neighborUnitCellIndex < 0:
                        neighborUnitCellIndices[index] += self.systemSize[index]
                    elif neighborUnitCellIndex >= self.systemSize[index]:
                        neighborUnitCellIndices[index] -= self.systemSize[index]
                    neighborQuantumIndices = np.hstack((neighborUnitCellIndices, elementTypeIndex, neighborElementSiteIndices[centerSiteElementSiteIndex][neighborIndex]))
                    neighborSEIndex = self.generateSystemElementIndex(self.systemSize, neighborQuantumIndices)
                    neighborSystemElementIndices[centerSiteIndex][neighborIndex] = neighborSEIndex
        
        fileName = 'neighborSystemElementIndices.npy'
        neighborSystemElementIndicesFilePath = dstPath + directorySeparator + fileName
        np.save(neighborSystemElementIndicesFilePath, neighborSystemElementIndices)
        if report:
            self.generateHematiteNeighborSEIndicesReport(dstPath, startTime)
        return

    def generateHematiteNeighborSEIndicesReport(self, dstPath, startTime):
        """Generates a neighbor list and prints out a report to the output directory"""
        neighborSystemElementIndicesLogName = 'neighborSystemElementIndices.log' 
        neighborSystemElementIndicesLogPath = dstPath + directorySeparator + neighborSystemElementIndicesLogName
        report = open(neighborSystemElementIndicesLogPath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - startTime
        report.write('Time elapsed: ' + ('%2d days, ' % timeElapsed.days if timeElapsed.days else '') +
                     ('%2d hours' % ((timeElapsed.seconds // 3600) % 24)) + 
                     (', %2d minutes' % ((timeElapsed.seconds // 60) % 60)) + 
                     (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()

    def generateSpeciesSiteSDList(self, centerSiteQuantumIndices, dstPath, report=1):
        startTime = datetime.now()
        elementTypeIndex = centerSiteQuantumIndices[3]
        centerSiteSEIndex = self.generateSystemElementIndex(self.systemSize, centerSiteQuantumIndices)
        localBulkSites = self.material.generateSites(self.elementTypeIndices, self.systemSize)
        systemElementIndexOffsetArray = (np.repeat(np.arange(0, self.material.totalElementsPerUnitCell * self.numCells, self.material.totalElementsPerUnitCell), 
                                                   self.material.nElementsPerUnitCell[elementTypeIndex]))
        neighborSiteSEIndices = (np.tile(self.material.nElementsPerUnitCell[:elementTypeIndex].sum() + 
                                         np.arange(0, self.material.nElementsPerUnitCell[elementTypeIndex]), self.numCells) + systemElementIndexOffsetArray)
        speciesSiteSDList = np.zeros(len(neighborSiteSEIndices))
        for neighborSiteIndex, neighborSiteSEIndex in enumerate(neighborSiteSEIndices):
            speciesSiteSDList[neighborSiteIndex] = self.computeDistance(self.systemSize, centerSiteSEIndex, neighborSiteSEIndex)**2
        speciesSiteSDList /= self.material.ANG2BOHR**2
        fileName = 'speciesSiteSDList.npy'
        speciesSiteSDListFilePath = dstPath + directorySeparator + fileName
        np.save(speciesSiteSDListFilePath, speciesSiteSDList)
        if report:
            self.generateSpeciesSiteSDListReport(dstPath, startTime)
        return

    def generateSpeciesSiteSDListReport(self, dstPath, startTime):
        """Generates a neighbor list and prints out a report to the output directory"""
        speciesSiteSDListLogName = 'speciesSiteSDList.log' 
        speciesSiteSDListLogPath = dstPath + directorySeparator + speciesSiteSDListLogName
        report = open(speciesSiteSDListLogPath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - startTime
        report.write('Time elapsed: ' + ('%2d days, ' % timeElapsed.days if timeElapsed.days else '') +
                     ('%2d hours' % ((timeElapsed.seconds // 3600) % 24)) + 
                     (', %2d minutes' % ((timeElapsed.seconds // 60) % 60)) + 
                     (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()

    def generateTransitionProbMatrix(self, neighborSystemElementIndices, dstPath, report=1):
        startTime = datetime.now()
        elementTypeIndex = 0
        numNeighbors = len(neighborSystemElementIndices[0])
        numBasalNeighbors = 3
        numCNeighbors = 1
        T = 300 * self.material.K2AUTEMP
        
        hopElementType = 'Fe:Fe'
        kList = np.zeros(numNeighbors)
        delG0 = 0
        for neighborIndex in range(numNeighbors):
            if neighborIndex < numBasalNeighbors:
                hopDistType = 0
            else:
                hopDistType = 1
            lambdaValue = self.material.lambdaValues[hopElementType][hopDistType]
            VAB = self.material.VAB[hopElementType][hopDistType]
            delGs = ((lambdaValue + delG0) ** 2 / (4 * lambdaValue)) - VAB
            kList[neighborIndex] = self.material.vn * np.exp(-delGs / T)
        
        kTotal = np.sum(kList)
        probList = kList / kTotal

        localBulkSites = self.material.generateSites(self.elementTypeIndices, self.systemSize)
        systemElementIndexOffsetArray = (np.repeat(np.arange(0, self.material.totalElementsPerUnitCell * self.numCells, self.material.totalElementsPerUnitCell), 
                                                   self.material.nElementsPerUnitCell[elementTypeIndex]))
        neighborSiteSEIndices = (np.tile(self.material.nElementsPerUnitCell[:elementTypeIndex].sum() + 
                                         np.arange(0, self.material.nElementsPerUnitCell[elementTypeIndex]), self.numCells) + systemElementIndexOffsetArray)
        
        numElementTypeSites = len(neighborSystemElementIndices)
        transitionProbMatrix = np.zeros((numElementTypeSites, numElementTypeSites))
        for centerSiteIndex in range(numElementTypeSites):
            for neighborIndex in range(numNeighbors):
                neighborSiteIndex = np.where(neighborSiteSEIndices == neighborSystemElementIndices[centerSiteIndex][neighborIndex])[0][0]
                transitionProbMatrix[centerSiteIndex][neighborSiteIndex] = probList[neighborIndex]
        fileName = 'transitionProbMatrix.npy'
        transitionProbMatrixFilePath = dstPath + directorySeparator + fileName
        np.save(transitionProbMatrixFilePath, transitionProbMatrix)
        if report:
            self.generateTransitionProbMatrixListReport(dstPath, startTime)
        return
    
    def generateTransitionProbMatrixListReport(self, dstPath, startTime):
        """Generates a neighbor list and prints out a report to the output directory"""
        transitionProbMatrixLogName = 'transitionProbMatrix.log' 
        transitionProbMatrixLogPath = dstPath + directorySeparator + transitionProbMatrixLogName
        report = open(transitionProbMatrixLogPath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - startTime
        report.write('Time elapsed: ' + ('%2d days, ' % timeElapsed.days if timeElapsed.days else '') +
                     ('%2d hours' % ((timeElapsed.seconds // 3600) % 24)) + 
                     (', %2d minutes' % ((timeElapsed.seconds // 60) % 60)) + 
                     (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()
    
    def generateMSDAnalyticalData(self, transitionProbMatrix, speciesSiteSDList, centerSiteQuantumIndices, analyticalTFinal, analyticalTimeInterval, dstPath, report=1):
        startTime = datetime.now()
        
        fileName = '%1.2Ens' % analyticalTFinal
        MSDAnalyticalDataFileName = 'MSD_Analytical_Data_' + fileName + '.dat'
        MSDAnalyticalDataFilePath = dstPath + directorySeparator + MSDAnalyticalDataFileName
        open(MSDAnalyticalDataFilePath, 'w').close()

        elementTypeIndex = 0
        numDataPoints = int(analyticalTFinal / analyticalTimeInterval) + 1
        msdData = np.zeros((numDataPoints, 2))
        msdData[:, 0] = np.arange(0, analyticalTFinal + analyticalTimeInterval, analyticalTimeInterval)

        localBulkSites = self.material.generateSites(self.elementTypeIndices, self.systemSize)
        systemElementIndexOffsetArray = (np.repeat(np.arange(0, self.material.totalElementsPerUnitCell * self.numCells, self.material.totalElementsPerUnitCell), 
                                                   self.material.nElementsPerUnitCell[elementTypeIndex]))
        centerSiteSEIndices = (np.tile(self.material.nElementsPerUnitCell[:elementTypeIndex].sum() + 
                                       np.arange(0, self.material.nElementsPerUnitCell[elementTypeIndex]), self.numCells) + systemElementIndexOffsetArray)
        
        centerSiteSEIndex = self.generateSystemElementIndex(self.systemSize, centerSiteQuantumIndices)
        
        
        
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
            lambdaValue = self.material.lambdaValues[hopElementType][hopDistType]
            VAB = self.material.VAB[hopElementType][hopDistType]
            delGs = ((lambdaValue + delG0) ** 2 / (4 * lambdaValue)) - VAB
            kList[neighborIndex] = self.material.vn * np.exp(-delGs / T)
        
        kTotal = np.sum(kList)        
        timestep = (1 / kTotal) / self.material.SEC2AUTIME * self.material.SEC2NS
        
        simTime = 0
        startIndex = 0
        rowIndex = np.where(centerSiteSEIndices == centerSiteSEIndex)
        newTransitionProbMatrix = np.copy(transitionProbMatrix)
        with open(MSDAnalyticalDataFilePath, 'a') as MSDAnalyticalDataFile:
            np.savetxt(MSDAnalyticalDataFile, msdData[startIndex, :][None, :])
        while True:
            newTransitionProbMatrix = np.dot(newTransitionProbMatrix, transitionProbMatrix)
            simTime += timestep
            endIndex = int(simTime / analyticalTimeInterval)
            if endIndex >= startIndex + 1:
                msdData[endIndex, 1] = np.dot(newTransitionProbMatrix[rowIndex], speciesSiteSDList)
                with open(MSDAnalyticalDataFilePath, 'a') as MSDAnalyticalDataFile:
                    np.savetxt(MSDAnalyticalDataFile, msdData[endIndex, :][None, :])
                startIndex += 1
                if endIndex == numDataPoints - 1:
                    break
        
        if report:
            self.generateMSDAnalyticalDataReport(fileName, dstPath, startTime)
        returnMSDData = returnValues()
        returnMSDData.msdData = msdData
        return returnMSDData
    
    def generateMSDAnalyticalDataReport(self, fileName, dstPath, startTime):
        """Generates a neighbor list and prints out a report to the output directory"""
        MSDAnalyticalDataLogName = 'MSD_Analytical_Data_' + fileName + '.log' 
        MSDAnalyticalDataLogPath = dstPath + directorySeparator + MSDAnalyticalDataLogName
        report = open(MSDAnalyticalDataLogPath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - startTime
        report.write('Time elapsed: ' + ('%2d days, ' % timeElapsed.days if timeElapsed.days else '') +
                     ('%2d hours' % ((timeElapsed.seconds // 3600) % 24)) + 
                     (', %2d minutes' % ((timeElapsed.seconds // 60) % 60)) + 
                     (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()
    
    def generateLatticeDirections(self, cutoffDistKey, cutoff, nDigits, tol, outdir):
        """ generate lattice directions and distances for neighboring atoms"""

        [centerElementType, neighborElementType] = cutoffDistKey.split(self.material.elementTypeDelimiter)
        elementTypes = self.material.elementTypes[:]
        centerSiteElementTypeIndex = elementTypes.index(centerElementType) 
        neighborSiteElementTypeIndex = elementTypes.index(neighborElementType)
        numCenterElements = self.material.nElementsPerUnitCell[centerSiteElementTypeIndex]
        
        cutoffDistLimits = [0, cutoff * self.material.ANG2BOHR]
        
        convArray = np.linalg.inv(self.material.latticeMatrix.T)
        fractionalUnitCellCoords = np.dot(convArray, self.material.unitcellCoords.T).T
        
        numCenterElements = self.material.nElementsPerUnitCell[centerSiteElementTypeIndex]
        numCells = 3**sum(self.pbc)
        xRange = range(-1, 2) if self.pbc[0] == 1 else [0]
        yRange = range(-1, 2) if self.pbc[1] == 1 else [0]
        zRange = range(-1, 2) if self.pbc[2] == 1 else [0]
        unitcellTranslationalCoords = np.zeros((numCells, 3)) # Initialization
        index = 0
        for xOffset in xRange:
            for yOffset in yRange:
                for zOffset in zRange:
                    unitcellTranslationalCoords[index] = np.array([xOffset, yOffset, zOffset])
                    index += 1
        
        centerSiteIndices = self.material.nElementsPerUnitCell[:centerSiteElementTypeIndex].sum() + np.arange(numCenterElements)
        centerSiteFractCoords = fractionalUnitCellCoords[centerSiteIndices]
        neighborSiteFractCoords = np.zeros((numCenterElements * numCells, 3))
        for centerSiteIndex, centerSiteFractCoord in enumerate(centerSiteFractCoords):
             neighborSiteFractCoords[(centerSiteIndex * numCells):((centerSiteIndex+1) * numCells)] = centerSiteFractCoord + unitcellTranslationalCoords

        displacementVectorList = np.empty(numCenterElements, dtype=object)
        latticeDirectionList = np.empty(numCenterElements, dtype=object)
        displacementList = np.empty(numCenterElements, dtype=object)
        for centerSiteIndex, centerSiteFractCoord in enumerate(centerSiteFractCoords):
            iDisplacementVectors = []
            iLatticeDirectionList = []
            iDisplacements = []
            for neighborSiteIndex, neighborSiteFractCoord in enumerate(neighborSiteFractCoords):
                latticeDirection = neighborSiteFractCoord - centerSiteFractCoord
                neighborDisplacementVector = np.dot(latticeDirection[None, :], self.material.latticeMatrix)
                displacement = np.linalg.norm(neighborDisplacementVector)
                if cutoffDistLimits[0] < displacement <= cutoffDistLimits[1]:
                    iDisplacementVectors.append(neighborDisplacementVector)
                    iLatticeDirectionList.append(latticeDirection)
                    iDisplacements.append(displacement)
            displacementVectorList[centerSiteIndex] = np.asarray(iDisplacementVectors)
            latticeDirectionList[centerSiteIndex] = np.asarray(iLatticeDirectionList)
            displacementList[centerSiteIndex] = np.asarray(iDisplacements) / self.material.ANG2BOHR
        
        from fractions import gcd
        for iCenterElementIndex in range(numCenterElements):
            intLDList = np.array(np.round(latticeDirectionList[iCenterElementIndex], nDigits) * 10**nDigits, int)
            latticeDirectionList[iCenterElementIndex] = latticeDirectionList[iCenterElementIndex].astype(int)
            iNumNeighbors = len(intLDList)
            for index in range(iNumNeighbors):
                nz = np.nonzero(intLDList[index])[0]
                if len(nz) == 1:
                    latticeDirectionList[iCenterElementIndex][index] = intLDList[index] / abs(intLDList[index][nz])
                elif len(nz) == 2:
                    two_values = intLDList[index][nz]
                    gcd_value = abs(gcd(two_values[0], two_values[1]))
                    abs_two_values = abs(two_values)
                    max_value = max(abs_two_values)
                    max_value_index = nz[np.argmax(abs_two_values)]
                    min_value = min(abs_two_values)
                    min_value_index = nz[np.argmin(abs_two_values)]
                    multiple = max_value / float(min_value)
                    if abs(abs(multiple) - round(abs(multiple))) < tol:
                        latticeDirectionList[iCenterElementIndex][index][max_value_index] = np.sign(intLDList[index][max_value_index]) * round(abs(multiple))
                        latticeDirectionList[iCenterElementIndex][index][min_value_index] = np.sign(intLDList[index][min_value_index])
                    else:
                        latticeDirectionList[iCenterElementIndex][index] = intLDList[index] / abs(gcd(two_values[0], two_values[1]))
                else:
                    latticeDirectionList[iCenterElementIndex][index] = intLDList[index] / abs(gcd(gcd(intLDList[index][0], intLDList[index][1]), intLDList[index][2]))
        sortedLatticeDirectionList = np.empty(numCenterElements, dtype=object)
        sortedDisplacementList = np.empty(numCenterElements, dtype=object)
        for iCenterElementIndex in range(numCenterElements):
            sortedLatticeDirectionList[iCenterElementIndex] = latticeDirectionList[iCenterElementIndex][displacementList[iCenterElementIndex].argsort()]
            sortedDisplacementList[iCenterElementIndex] = displacementList[iCenterElementIndex][displacementList[iCenterElementIndex].argsort()]
        latticeDirectionListFileName = 'latticeDirectionList_' + centerElementType + '-' + neighborElementType + '_cutoff=' + str(cutoff)
        displacementListFileName = 'displacementList_' + centerElementType + '-' + neighborElementType + '_cutoff=' + str(cutoff)
        latticeDirectionListFilePath = outdir + directorySeparator + latticeDirectionListFileName + '.npy'
        displacementListFilePath = outdir + directorySeparator + displacementListFileName + '.npy'
        np.save(latticeDirectionListFilePath, sortedLatticeDirectionList)
        np.save(displacementListFilePath, sortedDisplacementList)
        return
        
        
class system(object):
    """defines the system we are working on
    
    Attributes:
    size: An array (3 x 1) defining the system size in multiple of unit cells
    """
    #@profile
    def __init__(self, material, neighbors, hopNeighborList, cumulativeDisplacementList, speciesCount, alpha, nmax, kmax):
        """Return a system object whose size is *size*"""
        self.startTime = datetime.now()
        
        self.material = material
        self.neighbors = neighbors
        self.hopNeighborList = hopNeighborList
        
        self.pbc = self.neighbors.pbc
        self.speciesCount = speciesCount
        self.systemCharge = np.dot(speciesCount, self.material.speciesChargeList)
        self.speciesCountCumSum = speciesCount.cumsum()
        
        # total number of unit cells
        self.systemSize = self.neighbors.systemSize
        self.numCells = self.systemSize.prod()
        
        # generate lattice charge list
        unitcellChargeList = np.array([self.material.chargeTypes[self.material.elementTypes[elementTypeIndex]] 
                                       for elementTypeIndex in self.material.elementTypeIndexList])
        self.latticeChargeList = np.tile(unitcellChargeList, self.numCells)
        
        self.cumulativeDisplacementList = cumulativeDisplacementList

        # variables for ewald sum
        self.translationalMatrix = np.multiply(self.systemSize, self.material.latticeMatrix) 
        self.systemVolume = abs(np.dot(self.translationalMatrix[0], np.cross(self.translationalMatrix[1], self.translationalMatrix[2])))
        self.reciprocalLatticeMatrix = 2 * np.pi / self.systemVolume * np.array([np.cross(self.translationalMatrix[1], self.translationalMatrix[2]), 
                                                                                 np.cross(self.translationalMatrix[2], self.translationalMatrix[0]),
                                                                                 np.cross(self.translationalMatrix[0], self.translationalMatrix[1])])
        self.translationalVectorLength = np.linalg.norm(self.translationalMatrix, axis=1)
        self.reciprocalLatticeVectorLength = np.linalg.norm(self.reciprocalLatticeMatrix, axis=1)
        
        # ewald parameters:
        self.alpha = alpha
        self.nmax = nmax
        self.kmax = kmax
    
    def generateRandomOccupancy(self, speciesCount):
        """generates initial occupancy list based on species count"""
        occupancy = []
        for speciesTypeIndex, numSpecies in enumerate(speciesCount):
            centerSiteElementTypeIndex = np.in1d(self.material.elementTypes, self.material.speciesToElementTypeMap[self.material.speciesTypes[speciesTypeIndex]]).nonzero()[0][0]
            systemElementIndexOffsetArray = (np.repeat(np.arange(0, self.material.totalElementsPerUnitCell * self.numCells, self.material.totalElementsPerUnitCell), 
                                                       self.material.nElementsPerUnitCell[centerSiteElementTypeIndex]))
            siteIndices = (np.tile(self.material.nElementsPerUnitCell[:centerSiteElementTypeIndex].sum() + 
                                                               np.arange(0, self.material.nElementsPerUnitCell[centerSiteElementTypeIndex]), self.numCells) + systemElementIndexOffsetArray)
            occupancy.extend(rnd.sample(siteIndices, numSpecies)[:])
        return occupancy
    
    def chargeConfig(self, occupancy):
        """Returns charge distribution of the current configuration"""
        chargeList = np.copy(self.latticeChargeList)
        chargeList = chargeList[:, np.newaxis]
        chargeTypeKeys = self.material.chargeTypes.keys()
        
        siteChargeTypeKeys = [key for key in chargeTypeKeys if key not in self.material.siteList if self.material.siteIdentifier in key]
        for chargeKeyType in siteChargeTypeKeys:
            centerSiteElementType = chargeKeyType.replace(self.material.siteIdentifier,'')
            for speciesType in self.material.elementTypeToSpeciesMap[centerSiteElementType]:
                assert speciesType in self.material.speciesTypes, ('Invalid definition of charge type \'' + str(chargeKeyType) + '\', \'' + 
                                                              str(speciesType) + '\' species does not exist in this configuration') 
                speciesTypeIndex = self.material.speciesTypes.index(speciesType)
                startIndex = 0 + self.speciesCount[:speciesTypeIndex].sum()
                endIndex = self.speciesCountCumSum[speciesTypeIndex]
                centerSiteSystemElementIndices = occupancy[startIndex:endIndex][:]
                chargeList[centerSiteSystemElementIndices] = self.material.chargeTypes[chargeKeyType]
        return chargeList

    def ewaldSumSetup(self, outdir=None):
        from scipy.special import erfc
        sqrtalpha = np.sqrt(self.alpha)
        alpha4 = 4 * self.alpha
        fourierSumCoeff = (2 * np.pi) / self.systemVolume
        precomputedArray = np.zeros((self.neighbors.numSystemElements, self.neighbors.numSystemElements))

        for i in range(-self.nmax, self.nmax+1):
            for j in range(-self.nmax, self.nmax+1):
                for k in range(-self.nmax, self.nmax+1):
                    tempArray = np.linalg.norm(self.cumulativeDisplacementList + np.dot(np.array([i, j, k]), self.translationalMatrix), axis=2)
                    precomputedArray += erfc(sqrtalpha * tempArray) / 2

                    if np.all(np.array([i, j, k])==0):
                        for a in range(self.neighbors.numSystemElements):
                            for b in range(self.neighbors.numSystemElements):
                                if a != b:
                                    precomputedArray[a][b] /= tempArray[a][b]
                    else:
                        precomputedArray /= tempArray
        
        for i in range(-self.kmax, self.kmax+1):
            for j in range(-self.kmax, self.kmax+1):
                for k in range(-self.kmax, self.kmax+1):
                    if not np.all(np.array([i, j, k])==0):
                        kVector = np.dot(np.array([i, j, k]), self.reciprocalLatticeMatrix)
                        kVector2 = np.dot(kVector, kVector)
                        precomputedArray += fourierSumCoeff * np.exp(-kVector2 / alpha4) * np.cos(np.tensordot(self.cumulativeDisplacementList, kVector, axes=([2], [0]))) / kVector2
        
        precomputedArray /= self.material.dielectricConstant
        
        if outdir:
            self.generatePreComputedArrayLogReport(outdir)
        return precomputedArray

    def generatePreComputedArrayLogReport(self, outdir):
        """Generates an log report of the simulation and outputs to the working directory"""
        precomputedArrayLogFileName = 'precompuedArray.log'
        precomputedArrayLogFilePath = outdir + directorySeparator + precomputedArrayLogFileName
        report = open(precomputedArrayLogFilePath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - self.startTime
        report.write('Time elapsed: ' + ('%2d days, ' % timeElapsed.days if timeElapsed.days else '') +
                     ('%2d hours' % ((timeElapsed.seconds // 3600) % 24)) + 
                     (', %2d minutes' % ((timeElapsed.seconds // 60) % 60)) + 
                     (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()
    
class run(object):
    """defines the subroutines for running Kinetic Monte Carlo and computing electrostatic 
    interaction energies"""
    def __init__(self, system, precomputedArray, T, nTraj, tFinal, timeInterval, gui):
        """Returns the PBC condition of the system"""
        self.startTime = datetime.now()

        self.system = system
        self.material = self.system.material
        self.neighbors = self.system.neighbors
        self.precomputedArray = precomputedArray
        self.T = T * self.material.K2AUTEMP
        self.nTraj = int(nTraj)
        self.tFinal = tFinal * self.material.SEC2AUTIME
        self.timeInterval = timeInterval * self.material.SEC2AUTIME
        self.gui = gui

        self.systemSize = self.system.systemSize

        # nElementsPerUnitCell
        self.headStart_nElementsPerUnitCellCumSum = [self.material.nElementsPerUnitCell[:siteElementTypeIndex].sum() for siteElementTypeIndex in self.neighbors.elementTypeIndices]
        
        # speciesTypeList
        self.speciesTypeList = [self.material.speciesTypes[index] for index, value in enumerate(self.system.speciesCount) for i in range(value)]
        self.speciesTypeIndexList = [index for index, value in enumerate(self.system.speciesCount) for iValue in range(value)]
        self.speciesChargeList = [self.material.speciesChargeList[index] for index in self.speciesTypeIndexList]
        self.hopElementTypeList = [self.material.hopElementTypes[speciesType][0] for speciesType in self.speciesTypeList]
        self.lenHopDistTypeList = [len(self.material.neighborCutoffDist[hopElementType]) for hopElementType in self.hopElementTypeList]
        # number of kinetic processes
        self.nProc = 0
        self.nProcHopElementTypeList = []
        self.nProcHopDistTypeList = []
        self.nProcSpeciesIndexList = []
        self.nProcSiteElementTypeIndexList = []
        self.nProcLambdaValueList = []
        self.nProcVABList = []
        for hopElementTypeIndex, hopElementType in enumerate(self.hopElementTypeList):
            centerElementType = hopElementType.split(self.material.elementTypeDelimiter)[0]
            speciesTypeIndex = self.material.speciesTypes.index(self.material.elementTypeToSpeciesMap[centerElementType][0])
            centerSiteElementTypeIndex = self.material.elementTypes.index(centerElementType)
            for hopDistTypeIndex in range(self.lenHopDistTypeList[hopElementTypeIndex]):
                if self.system.speciesCount[speciesTypeIndex] != 0:
                    numNeighbors = np.unique(self.system.hopNeighborList[hopElementType][hopDistTypeIndex].numNeighbors)
                    # TODO: What if it is not equal to 1
                    if len(numNeighbors) == 1:
                        self.nProc += numNeighbors[0]# * self.system.speciesCount[speciesTypeIndex]
                        self.nProcHopElementTypeList.extend([hopElementType] * numNeighbors[0])
                        self.nProcHopDistTypeList.extend([hopDistTypeIndex] * numNeighbors[0])
                        self.nProcSpeciesIndexList.extend([hopElementTypeIndex] * numNeighbors[0])
                        self.nProcSiteElementTypeIndexList.extend([centerSiteElementTypeIndex] * numNeighbors[0])
                        self.nProcLambdaValueList.extend([self.material.lambdaValues[hopElementType][hopDistTypeIndex]] * numNeighbors[0])
                        self.nProcVABList.extend([self.material.VAB[hopElementType][hopDistTypeIndex]] * numNeighbors[0])
        
        # system coordinates
        self.systemCoordinates = self.neighbors.bulkSites.cellCoordinates
        
        # total number of species
        self.totalSpecies = self.system.speciesCount.sum()
    
    def doKMCSteps(self, outdir, report=1, randomSeed=1):
        """Subroutine to run the KMC simulation by specified number of steps"""
        assert outdir, 'Please provide the destination path where simulation output files needs to be saved'
        
        excess = 0
        energy = 1
        unwrappedTrajFileName = outdir + directorySeparator + 'unwrappedTraj.dat'
        open(unwrappedTrajFileName, 'w').close()
        if energy:
            energyTrajFileName = outdir + directorySeparator + 'energyTraj.dat'
            open(energyTrajFileName, 'w').close()
        
        if excess:
            wrappedTrajFileName = outdir + directorySeparator + 'wrappedTraj.dat'
            delG0TrajFileName = outdir + directorySeparator + 'delG0Traj.dat'
            potentialTrajFileName = outdir + directorySeparator + 'potentialTraj.dat'
            open(wrappedTrajFileName, 'w').close()
            open(delG0TrajFileName, 'w').close()
            open(potentialTrajFileName, 'w').close()
        
        rnd.seed(randomSeed)
        nTraj = self.nTraj
        numPathStepsPerTraj = int(self.tFinal / self.timeInterval) + 1
        unwrappedPositionArray = np.zeros(( numPathStepsPerTraj, self.totalSpecies * 3))
        if energy:
            energyArray = np.zeros(( numPathStepsPerTraj ))

        if excess:
            wrappedPositionArray = np.zeros(( numPathStepsPerTraj, self.totalSpecies * 3))
            delG0Array = np.zeros(( self.kmcSteps ))
            potentialArray = np.zeros(( numPathStepsPerTraj, self.totalSpecies))
        kList = np.zeros(self.nProc)
        neighborSiteSystemElementIndexList = np.zeros(self.nProc, dtype=int)
        rowIndexList = np.zeros(self.nProc, dtype=int)
        neighborIndexList = np.zeros(self.nProc, dtype=int)
        
        ewaldNeut = - np.pi * (self.system.systemCharge**2) / (2 * self.system.systemVolume * self.system.alpha)
        precomputedArray = self.precomputedArray
        for trajIndex in range(nTraj):
            currentStateOccupancy = self.system.generateRandomOccupancy(self.system.speciesCount)
            currentStateChargeConfig = self.system.chargeConfig(currentStateOccupancy)
            currentStateChargeConfigProd = np.multiply(currentStateChargeConfig.transpose(), currentStateChargeConfig)
            ewaldSelf = - np.sqrt(self.system.alpha / np.pi) * np.einsum('ii', currentStateChargeConfigProd)
            currentStateEnergy = ewaldNeut + ewaldSelf + np.sum(np.multiply(currentStateChargeConfigProd, precomputedArray))
            startPathIndex = 1
            endPathIndex = startPathIndex + 1
            if energy:
                energyArray[0] = currentStateEnergy
            # TODO: How to deal excess flag?
            #if excess:
                # TODO: Avoid using flatten
            #    wrappedPositionArray[pathIndex] = self.systemCoordinates[currentStateOccupancy].flatten()
            speciesDisplacementVectorList = np.zeros((1, self.totalSpecies * 3))
            simTime = 0
            breakFlag = 0
            while True:
                iProc = 0
                delG0List = []
                for speciesIndex, speciesSiteSystemElementIndex in enumerate(currentStateOccupancy):
                    # TODO: Avoid re-defining speciesIndex
                    speciesIndex = self.nProcSpeciesIndexList[iProc]
                    hopElementType = self.nProcHopElementTypeList[iProc]
                    siteElementTypeIndex = self.nProcSiteElementTypeIndexList[iProc]
                    rowIndex = (speciesSiteSystemElementIndex / self.material.totalElementsPerUnitCell * self.material.nElementsPerUnitCell[siteElementTypeIndex] + 
                                speciesSiteSystemElementIndex % self.material.totalElementsPerUnitCell - self.headStart_nElementsPerUnitCellCumSum[siteElementTypeIndex])
                    for hopDistType in range(self.lenHopDistTypeList[speciesIndex]):
                        localNeighborSiteSystemElementIndexList = self.system.hopNeighborList[hopElementType][hopDistType].neighborSystemElementIndices[rowIndex]
                        for neighborIndex, neighborSiteSystemElementIndex in enumerate(localNeighborSiteSystemElementIndexList):
                            # TODO: Introduce If condition
                            # if neighborSystemElementIndex not in currentStateOccupancy: commit 898baa8
                            neighborSiteSystemElementIndexList[iProc] = neighborSiteSystemElementIndex
                            rowIndexList[iProc] = rowIndex
                            neighborIndexList[iProc] = neighborIndex
                            # TODO: Print out a prompt about the assumption; detailed comment here. <Using species charge to compute change in energy> May be print log report
                            delG0 = (self.speciesChargeList[speciesIndex] 
                                     * (2 * np.dot(currentStateChargeConfig[:, 0], precomputedArray[neighborSiteSystemElementIndex, :] - precomputedArray[speciesSiteSystemElementIndex, :]) 
                                        + self.speciesChargeList[speciesIndex] * (precomputedArray[speciesSiteSystemElementIndex, speciesSiteSystemElementIndex] 
                                                                                  + precomputedArray[neighborSiteSystemElementIndex, neighborSiteSystemElementIndex] 
                                                                                  - 2 * precomputedArray[speciesSiteSystemElementIndex, neighborSiteSystemElementIndex])))
                            delG0List.append(delG0)
                            newStateEnergy = currentStateEnergy + delG0
                            lambdaValue = self.nProcLambdaValueList[iProc]
                            VAB = self.nProcVABList[iProc]
                            delGs = ((lambdaValue + delG0) ** 2 / (4 * lambdaValue)) - VAB
                            kList[iProc] = self.material.vn * np.exp(-delGs / self.T)
                            iProc += 1
                            
                kTotal = sum(kList)
                kCumSum = (kList / kTotal).cumsum()
                rand1 = rnd.random()
                procIndex = np.where(kCumSum > rand1)[0][0]
                rand2 = rnd.random()
                simTime -= np.log(rand2) / kTotal
                
                # TODO: Address pre-defining excess data arrays
                #if excess:
                #    delG0Array[step] = delG0List[procIndex]
                speciesIndex = self.nProcSpeciesIndexList[procIndex]
                hopElementType = self.nProcHopElementTypeList[procIndex]
                hopDistType = self.nProcHopDistTypeList[procIndex]
                rowIndex = rowIndexList[procIndex]
                neighborIndex = neighborIndexList[procIndex]
                oldSiteSystemElementIndex = currentStateOccupancy[speciesIndex]
                newSiteSystemElementIndex = neighborSiteSystemElementIndexList[procIndex]
                currentStateOccupancy[speciesIndex] = newSiteSystemElementIndex
                speciesDisplacementVectorList[0, speciesIndex * 3:(speciesIndex + 1) * 3] += self.system.hopNeighborList[hopElementType][hopDistType].displacementVectorList[rowIndex][neighborIndex]
                
                currentStateEnergy += delG0List[procIndex]
                currentStateChargeConfig[oldSiteSystemElementIndex] -= self.speciesChargeList[speciesIndex]
                currentStateChargeConfig[newSiteSystemElementIndex] += self.speciesChargeList[speciesIndex]
                endPathIndex = int(simTime / self.timeInterval)
                if endPathIndex >= startPathIndex + 1:
                    if endPathIndex >= numPathStepsPerTraj:
                        endPathIndex = numPathStepsPerTraj
                        breakFlag = 1
                    unwrappedPositionArray[startPathIndex:endPathIndex] = unwrappedPositionArray[startPathIndex-1] + speciesDisplacementVectorList
                    energyArray[startPathIndex:endPathIndex] = currentStateEnergy 
                    speciesDisplacementVectorList = np.zeros((1, self.totalSpecies * 3))
                    startPathIndex = endPathIndex
                    if breakFlag:
                        break
                    # TODO: Address excess flag
                    #if excess:
                        # TODO: Avoid using flatten
                    #    wrappedPositionArray[pathIndex] = self.systemCoordinates[currentStateOccupancy].flatten()

            with open(unwrappedTrajFileName, 'a') as unwrappedTrajFile:
                np.savetxt(unwrappedTrajFile, unwrappedPositionArray)
            with open(energyTrajFileName, 'a') as energyTrajFile:
                np.savetxt(energyTrajFile, energyArray)
            if excess:
                with open(wrappedTrajFileName, 'a') as wrappedTrajFile:
                    np.savetxt(wrappedTrajFile, wrappedPositionArray)
                with open(delG0TrajFileName, 'a') as delG0TrajFile:
                    np.savetxt(delG0TrajFile, delG0Array)
                with open(potentialTrajFileName, 'a') as potentialTrajFile:
                    np.savetxt(potentialTrajFile, potentialArray)
        if report:
            self.generateSimulationLogReport(outdir)

    def generateSimulationLogReport(self, outdir):
        """Generates an log report of the simulation and outputs to the working directory"""
        simulationLogFileName = 'Run.log'
        simulationLogFilePath = outdir + directorySeparator + simulationLogFileName
        report = open(simulationLogFilePath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - self.startTime
        report.write('Time elapsed: ' + ('%2d days, ' % timeElapsed.days if timeElapsed.days else '') +
                     ('%2d hours' % ((timeElapsed.seconds // 3600) % 24)) + 
                     (', %2d minutes' % ((timeElapsed.seconds // 60) % 60)) + 
                     (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()
    
class analysis(object):
    """Post-simulation analysis methods"""
    def __init__(self, material, nDim, systemSize, speciesCount, nTraj, tFinal, 
                 timeInterval, msdTFinal, trimLength, reprTime = 'ns', reprDist = 'Angstrom'):
        """"""
        self.startTime = datetime.now()
        self.material = material
        self.nDim = nDim
        self.systemSize = systemSize
        self.speciesCount = speciesCount
        self.totalSpecies = self.speciesCount.sum()
        self.nTraj = int(nTraj)
        self.tFinal = tFinal * self.material.SEC2AUTIME
        self.timeInterval = timeInterval * self.material.SEC2AUTIME
        self.trimLength = trimLength
        self.numPathStepsPerTraj = int(self.tFinal / self.timeInterval) + 1
        self.reprTime = reprTime
        self.reprDist = reprDist
        
        if reprTime == 'ns':
            self.timeConversion = self.material.SEC2NS / self.material.SEC2AUTIME
        elif reprTime == 'ps':
            self.timeConversion = self.material.SEC2PS / self.material.SEC2AUTIME
        elif reprTime == 'fs':
            self.timeConversion = self.material.SEC2FS / self.material.SEC2AUTIME
        elif reprTime == 's':
            self.timeConversion = 1E+00 / self.material.SEC2AUTIME
        
        if reprDist == 'm':
            self.distConversion = self.material.ANG / self.material.ANG2BOHR
        elif reprDist == 'um':
            self.distConversion = self.material.ANG2UM / self.material.ANG2BOHR
        elif reprDist == 'angstrom':
            self.distConversion = 1E+00 / self.material.ANG2BOHR
        
        self.msdTFinal = msdTFinal / self.timeConversion
        self.numMSDStepsPerTraj = int(self.msdTFinal / self.timeInterval) + 1
        
    def computeMSD(self, outdir, report=1):
        """Returns the squared displacement of the trajectories"""
        assert outdir, 'Please provide the destination path where MSD output files needs to be saved'
        positionArray = np.loadtxt(outdir + directorySeparator + 'unwrappedTraj.dat')
        numTrajRecorded = int(len(positionArray) / self.numPathStepsPerTraj)
        positionArray = positionArray[:numTrajRecorded * self.numPathStepsPerTraj + 1].reshape((numTrajRecorded * self.numPathStepsPerTraj, self.totalSpecies, 3)) * self.distConversion
        sdArray = np.zeros((numTrajRecorded, self.numMSDStepsPerTraj, self.totalSpecies))
        for trajIndex in range(numTrajRecorded):
            headStart = trajIndex * self.numPathStepsPerTraj
            for timestep in range(1, self.numMSDStepsPerTraj):
                numDisp = self.numPathStepsPerTraj - timestep
                addOn = np.arange(numDisp)
                posDiff = positionArray[headStart + timestep + addOn] - positionArray[headStart + addOn]
                sdArray[trajIndex, timestep, :] = np.mean(np.einsum('ijk,ijk->ij', posDiff, posDiff), axis=0)
        speciesAvgSDArray = np.zeros((numTrajRecorded, self.numMSDStepsPerTraj, self.material.numSpeciesTypes - list(self.speciesCount).count(0)))
        startIndex = 0
        numNonExistentSpecies = 0
        nonExistentSpeciesIndices = []
        for speciesTypeIndex in range(self.material.numSpeciesTypes):
            if self.speciesCount[speciesTypeIndex] != 0:
                endIndex = startIndex + self.speciesCount[speciesTypeIndex]
                speciesAvgSDArray[:, :, speciesTypeIndex - numNonExistentSpecies] = np.mean(sdArray[:, :, startIndex:endIndex], axis=2)
                startIndex = endIndex
            else:
                numNonExistentSpecies += 1
                nonExistentSpeciesIndices.append(speciesTypeIndex)
        
        msdData = np.zeros((self.numMSDStepsPerTraj, self.material.numSpeciesTypes + 1 - list(self.speciesCount).count(0)))
        timeArray = np.arange(self.numMSDStepsPerTraj) * self.timeInterval * self.timeConversion
        msdData[:, 0] = timeArray
        msdData[:, 1:] = np.mean(speciesAvgSDArray, axis=0)
        stdData = np.std(speciesAvgSDArray, axis=0)
        fileName = (('%1.2E' % (self.msdTFinal * self.timeConversion)) + str(self.reprTime) + 
                    (',nTraj: %1.2E' % numTrajRecorded if numTrajRecorded != self.nTraj else ''))
        msdFileName = 'MSD_Data_' + fileName + '.npy'
        msdFilePath = outdir + directorySeparator + msdFileName
        speciesTypes = [speciesType for index, speciesType in enumerate(self.material.speciesTypes) if index not in nonExistentSpeciesIndices]
        np.save(msdFilePath, msdData)
        
        if report:
            self.generateMSDAnalysisLogReport(msdData, speciesTypes, fileName, outdir)
        
        returnMSDData = returnValues()
        returnMSDData.msdData = msdData
        returnMSDData.stdData = stdData
        returnMSDData.speciesTypes = speciesTypes
        returnMSDData.fileName = fileName
        return returnMSDData

    def generateMSDAnalysisLogReport(self, msdData, speciesTypes, fileName, outdir):
        """Generates an log report of the MSD Analysis and outputs to the working directory"""
        msdAnalysisLogFileName = 'MSD_Analysis' + ('_' if fileName else '') + fileName + '.log'
        msdLogFilePath = outdir + directorySeparator + msdAnalysisLogFileName
        report = open(msdLogFilePath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - self.startTime
        from scipy.stats import linregress
        for speciesIndex, speciesType in enumerate(speciesTypes):
            slope, intercept, rValue, pValue, stdErr = linregress(msdData[self.trimLength:-self.trimLength,0], msdData[self.trimLength:-self.trimLength,speciesIndex + 1])
            speciesDiff = slope * self.material.ANG2UM**2 * self.material.SEC2NS / (2 * self.nDim)
            report.write('Estimated value of {:s} diffusivity is: {:4.3f} um2/s\n'.format(speciesType, speciesDiff))
        report.write('Time elapsed: ' + ('%2d days, ' % timeElapsed.days if timeElapsed.days else '') +
                     ('%2d hours' % ((timeElapsed.seconds // 3600) % 24)) + 
                     (', %2d minutes' % ((timeElapsed.seconds // 60) % 60)) + 
                     (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()

    def generateMSDPlot(self, msdData, stdData, displayErrorBars, speciesTypes, fileName, outdir):
        """Returns a line plot of the MSD data"""
        assert outdir, 'Please provide the destination path where MSD Plot files needs to be saved'
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import AnchoredText
        from textwrap import wrap
        fig = plt.figure()
        ax = fig.add_subplot(111)
        from scipy.stats import linregress
        for speciesIndex, speciesType in enumerate(speciesTypes):
            if displayErrorBars:
                ax.errorbar(msdData[:,0], msdData[:,speciesIndex + 1], yerr=stdData[:,speciesIndex], fmt='o', capsize=3, color='blue', markerfacecolor='blue', markeredgecolor='black', label=speciesType)
            else:
                ax.plot(msdData[:,0], msdData[:,speciesIndex + 1], 'o', markerfacecolor='blue', markeredgecolor='black', label=speciesType)
            slope, intercept, rValue, pValue, stdErr = linregress(msdData[self.trimLength:-self.trimLength,0], msdData[self.trimLength:-self.trimLength,speciesIndex + 1])
            speciesDiff = slope * self.material.ANG2UM**2 * self.material.SEC2NS / (2 * self.nDim)
            ax.add_artist(AnchoredText('Est. $D_{{%s}}$ = %4.3f  ${{\mu}}m^2/s$; $r^2$=%4.3e' % (speciesType, speciesDiff, rValue**2), loc=4))
            ax.plot(msdData[self.trimLength:-self.trimLength,0], intercept + slope * msdData[self.trimLength:-self.trimLength,0], 'r', label=speciesType+'-fitted')
        ax.set_xlabel('Time (' + self.reprTime + ')')
        ax.set_ylabel('MSD (' + ('$\AA^2$' if self.reprDist=='angstrom' else (self.reprDist + '^2')) + ')')
        figureTitle = 'MSD_' + fileName
        ax.set_title('\n'.join(wrap(figureTitle,60)))
        plt.legend()
        plt.show() # Temp change
        figureName = 'MSD_Plot_' + fileName + '_Trim=' + str(self.trimLength) + '.jpg'
        figurePath = outdir + directorySeparator + figureName
        plt.savefig(figurePath)

    '''
    # TODO: Finish writing the method soon.
    def displayCollectiveMSDPlot(self, msdData, speciesTypes, fileName, outdir=None):
        """Returns a line plot of the MSD data"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from textwrap import wrap
        plt.figure()
        figNum = 0
        numRow = 3
        numCol = 2
        for iPlot in range(numPlots):
            #msdData = 
            for speciesIndex, speciesType in enumerate(speciesTypes):
                plt.subplot(numRow, numCol, figNum)
                plt.plot(msdData[:,0], msdData[:,speciesIndex + 1], label=speciesType)
                figNum += 1
        plt.xlabel('Time (' + self.reprTime + ')')
        plt.ylabel('MSD (' + self.reprDist + '**2)')
        figureTitle = 'MSD_' + fileName
        plt.title('\n'.join(wrap(figureTitle,60)))
        plt.legend()
        if outdir:
            figureName = 'MSD_Plot_' + fileName + '.jpg'
            figurePath = outdir + directorySeparator + figureName
            plt.savefig(figurePath)
    '''
    
    def meanDistance(self, outdir, mean=1, plot=1, report=1):
        """
        Add combType as one of the inputs 
        combType = 0: like-like; 1: like-unlike; 2: both
        if combType == 0:
            numComb = sum([self.speciesCount[index] * (self.speciesCount[index] - 1) for index in len(self.speciesCount)])
        elif combType == 1:
            numComb = np.prod(self.speciesCount)
        elif combType == 2:
            numComb = np.prod(self.speciesCount) + sum([self.speciesCount[index] * (self.speciesCount[index] - 1) for index in len(self.speciesCount)])
        """
        positionArray = self.trajectoryData.wrappedPositionArray * self.distConversion
        numPathStepsPerTraj = int(self.kmcSteps / self.stepInterval) + 1
        # TODO: Currently assuming only electrons exist and coding accordingly.
        # Need to change according to combType
        pbc = [1, 1, 1] # change to generic
        nElectrons = self.speciesCount[0] # change to generic
        xRange = range(-1, 2) if pbc[0] == 1 else [0]
        yRange = range(-1, 2) if pbc[1] == 1 else [0]
        zRange = range(-1, 2) if pbc[2] == 1 else [0]
        unitcellTranslationalCoords = np.zeros((3**sum(pbc), 3)) # Initialization
        index = 0
        for xOffset in xRange:
            for yOffset in yRange:
                for zOffset in zRange:
                    unitcellTranslationalCoords[index] = np.dot(np.multiply(np.array([xOffset, yOffset, zOffset]), self.systemSize), (self.material.latticeMatrix * self.distConversion))
                    index += 1
        if mean:
            meanDistance = np.zeros((self.nTraj, numPathStepsPerTraj))
        else:
            interDistanceArray = np.zeros((self.nTraj, numPathStepsPerTraj, nElectrons * (nElectrons - 1) / 2))
        interDistanceList = np.zeros(nElectrons * (nElectrons - 1) / 2)
        for trajIndex in range(self.nTraj):
            headStart = trajIndex * numPathStepsPerTraj
            for step in range(numPathStepsPerTraj):
                index = 0
                for i in range(nElectrons):
                    for j in range(i + 1, nElectrons):
                        neighborImageCoords = unitcellTranslationalCoords + positionArray[headStart + step, j]
                        neighborImageDisplacementVectors = neighborImageCoords - positionArray[headStart + step, i]
                        neighborImageDisplacements = np.linalg.norm(neighborImageDisplacementVectors, axis=1)
                        displacement = np.min(neighborImageDisplacements)
                        interDistanceList[index] = displacement
                        index += 1
                if mean:
                    meanDistance[trajIndex, step] = np.mean(interDistanceList)
                    meanDistanceOverTraj = np.mean(meanDistance, axis=0)
                else:
                    interDistanceArray[trajIndex, step] = np.copy(interDistanceList)
        
        interDistanceArrayOverTraj = np.mean(interDistanceArray, axis=0)
        kmcSteps = range(0, numPathStepsPerTraj * int(self.stepInterval), int(self.stepInterval))
        if mean:
            meanDistanceArray = np.zeros((numPathStepsPerTraj, 2))
            meanDistanceArray[:, 0] = kmcSteps
            meanDistanceArray[:, 1] = meanDistanceOverTraj
        else:
            interSpeciesDistanceArray = np.zeros((numPathStepsPerTraj, nElectrons * (nElectrons - 1) / 2 + 1))
            interSpeciesDistanceArray[:, 0] = kmcSteps
            interSpeciesDistanceArray[:, 1:] = interDistanceArrayOverTraj
        if mean:
            meanDistanceFileName = 'MeanDistanceData.npy'
            meanDistanceFilePath = outdir + directorySeparator + meanDistanceFileName
            np.save(meanDistanceFilePath, meanDistanceArray)
        else:
            interSpeciesDistanceFileName = 'InterSpeciesDistance.npy'
            interSpeciesDistanceFilePath = outdir + directorySeparator + interSpeciesDistanceFileName
            np.save(interSpeciesDistanceFilePath, interSpeciesDistanceArray)
        
        if plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from textwrap import wrap
            plt.figure()
            if mean:
                plt.plot(meanDistanceArray[:, 0], meanDistanceArray[:, 1])
                plt.title('Mean Distance between species along simulation length')
                plt.xlabel('KMC Step')
                plt.ylabel('Distance (' + self.reprDist + ')')
                figureName = 'MeanDistanceOverTraj.jpg'
                figurePath = outdir + directorySeparator + figureName
                plt.savefig(figurePath)
            else:
                legendList = []
                for i in range(nElectrons):
                    for j in range(i + 1, nElectrons):
                        legendList.append('r_' + str(i) + ':' + str(j)) 
                lineObjects = plt.plot(interSpeciesDistanceArray[:, 0], interSpeciesDistanceArray[:, 1:])
                plt.title('Inter-species Distances along simulation length')
                plt.xlabel('KMC Step')
                plt.ylabel('Distance (' + self.reprDist + ')')
                lgd = plt.legend(lineObjects, legendList, loc='center left', bbox_to_anchor=(1, 0.5))
                figureName = 'Inter-SpeciesDistance.jpg'
                figurePath = outdir + directorySeparator + figureName
                plt.savefig(figurePath, bbox_extra_artists=(lgd,), bbox_inches='tight')
        if report:
            self.generateMeanDisplacementAnalysisLogReport(outdir)
        output = meanDistanceArray if mean else interSpeciesDistanceArray
        return output

    def generateMeanDisplacementAnalysisLogReport(self, outdir):
        """Generates an log report of the MSD Analysis and outputs to the working directory"""
        meanDisplacementAnalysisLogFileName = 'MeanDisplacement_Analysis.log'
        meanDisplacementAnalysisLogFilePath = outdir + directorySeparator + meanDisplacementAnalysisLogFileName
        report = open(meanDisplacementAnalysisLogFilePath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - self.startTime
        report.write('Time elapsed: ' + ('%2d days, ' % timeElapsed.days if timeElapsed.days else '') +
                     ('%2d hours' % ((timeElapsed.seconds // 3600) % 24)) + 
                     (', %2d minutes' % ((timeElapsed.seconds // 60) % 60)) + 
                     (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()

    def displayWrappedTrajectories(self):
        """ """
        pass
    
    def displayUnwrappedTrajectories(self):
        """ """
        pass
    
    def trajectoryToDCD(self):
        """Convert trajectory data and outputs dcd file"""
        pass
        
class returnValues(object):
    """dummy class to return objects from methods defined inside other classes"""
    pass
