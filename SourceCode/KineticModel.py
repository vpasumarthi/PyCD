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
        self.J2HARTREE = 1 / self.HARTREE
        self.SEC2AUTIME = 1 / self.AUTIME
        self.K2AUTEMP = 1 / self.AUTEMPERATURE
        
        # TODO: introduce a method to view the material using ase atoms or other gui module
        self.name = materialParameters.name
        self.elementTypes = materialParameters.elementTypes[:]
        self.speciesTypes = materialParameters.speciesTypes[:]
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
        
    def generateMaterialFile(self, material, materialFileName, replaceExistingObjectFiles):
        """ """
        if not os.path.isfile(materialFileName) or replaceExistingObjectFiles:
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
        
        # generate all sites in the system
        elementTypeIndices = range(len(self.material.elementTypes))
        self.bulkSites = self.material.generateSites(elementTypeIndices, self.systemSize)

    def generateNeighborsFile(self, materialNeighbors, neighborsFileName, replaceExistingObjectFiles):
        """ """
        if not os.path.isfile(neighborsFileName) or replaceExistingObjectFiles:
            file_Neighbors = open(neighborsFileName, 'w')
            pickle.dump(materialNeighbors, file_Neighbors)
            file_Neighbors.close()
        pass
    #@profile
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
    #@profile
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
    
    def computeDistance(self, systemSize, systemElementIndex1, systemElementIndex2):
        """Returns the distance in atomic units between the two system element indices for a given system size"""
        centerCoord = self.computeCoordinates(systemSize, systemElementIndex1)
        neighborCoord = self.computeCoordinates(systemSize, systemElementIndex2)
        
        xRange = range(-1, 2) if self.pbc[0] == 1 else [0]
        yRange = range(-1, 2) if self.pbc[1] == 1 else [0]
        zRange = range(-1, 2) if self.pbc[2] == 1 else [0]
        unitcellTranslationalCoords = np.zeros((3**sum(self.pbc), 3)) # Initialization
        index = 0
        for xOffset in xRange:
            for yOffset in yRange:
                for zOffset in zRange:
                    unitcellTranslationalCoords[index] = np.dot(np.multiply(np.array([xOffset, yOffset, zOffset]), systemSize), self.material.latticeMatrix)
                    index += 1
        neighborImageCoords = unitcellTranslationalCoords + neighborCoord
        neighborImageDisplacementVectors = neighborImageCoords - centerCoord
        neighborImageDisplacements = np.linalg.norm(neighborImageDisplacementVectors, axis=1)
        displacement = np.min(neighborImageDisplacements)
        return displacement
    #@profile
    def hopNeighborSites(self, bulkSites, centerSiteIndices, neighborSiteIndices, cutoffDistLimits, cutoffDistKey):
        """Returns systemElementIndexMap and distances between center sites and its neighbor sites within cutoff 
        distance"""
        neighborSiteCoords = bulkSites.cellCoordinates[neighborSiteIndices]
        neighborSiteSystemElementIndexList = bulkSites.systemElementIndexList[neighborSiteIndices]
        neighborSiteQuantumIndexList = bulkSites.quantumIndexList[neighborSiteIndices]
        centerSiteCoords = bulkSites.cellCoordinates[centerSiteIndices]
        centerSiteSystemElementIndexList = bulkSites.systemElementIndexList[centerSiteIndices]
        centerSiteQuantumIndexList = bulkSites.quantumIndexList[centerSiteIndices]
        
        neighborSystemElementIndices = np.empty(len(centerSiteCoords), dtype=object)
        offsetList = np.empty(len(centerSiteCoords), dtype=object)
        neighborElementIndexList = np.empty(len(centerSiteCoords), dtype=object)
        numNeighbors = np.array([], dtype=int)
        displacementVectorList = np.empty(len(centerSiteCoords), dtype=object)
        displacementList = np.empty(len(centerSiteCoords), dtype=object)
        
        # DEBUG: Quick test for number of neighbors: Switch Off Start
        xRange = range(-1, 2) if self.pbc[0] == 1 else [0]
        yRange = range(-1, 2) if self.pbc[1] == 1 else [0]
        zRange = range(-1, 2) if self.pbc[2] == 1 else [0]
        unitcellTranslationalCoords = np.zeros((3**sum(self.pbc), 3)) # Initialization
        index = 0
        for xOffset in xRange:
            for yOffset in yRange:
                for zOffset in zRange:
                    unitcellTranslationalCoords[index] = np.dot(np.multiply(np.array([xOffset, yOffset, zOffset]), self.systemSize), self.material.latticeMatrix)
                    index += 1
        for centerSiteIndex, centerCoord in enumerate(centerSiteCoords):
            iDisplacementVectors = []
            iDisplacements = np.array([])
            iNeighborSiteIndexList = []
            iNumNeighbors = 0
            for neighborSiteIndex, neighborCoord in enumerate(neighborSiteCoords):
                neighborImageCoords = unitcellTranslationalCoords + neighborCoord
                neighborImageDisplacementVectors = neighborImageCoords - centerCoord
                neighborImageDisplacements = np.linalg.norm(neighborImageDisplacementVectors, axis=1)
                [displacement, imageIndex] = [np.min(neighborImageDisplacements), np.argmin(neighborImageDisplacements)]
                if cutoffDistLimits[0] < displacement <= cutoffDistLimits[1]:
                    iNeighborSiteIndexList.append(neighborSiteIndex)
                    iDisplacementVectors.append(neighborImageDisplacementVectors[imageIndex])
                    iDisplacements = np.append(iDisplacements, displacement)
                    iNumNeighbors += 1
            neighborSystemElementIndices[centerSiteIndex] = neighborSiteSystemElementIndexList[iNeighborSiteIndexList]
            offsetList[centerSiteIndex] = neighborSiteQuantumIndexList[iNeighborSiteIndexList, :3] - centerSiteQuantumIndexList[centerSiteIndex, :3]
            neighborElementIndexList[centerSiteIndex] = neighborSiteQuantumIndexList[iNeighborSiteIndexList, 4]
            numNeighbors = np.append(numNeighbors, iNumNeighbors)
            displacementVectorList[centerSiteIndex] = np.asarray(iDisplacementVectors)
            displacementList[centerSiteIndex] = iDisplacements
        # DEBUG: Quick test for number of neighbors: Switch OFF End
        
        # DEBUG: Quick test for number of neighbors: Switch ON Start
        '''
        for centerSiteIndex, centerCoord in enumerate(centerSiteCoords):
            iDisplacementVectors = []
            iDisplacements = np.array([])
            iNeighborSiteIndexList = []
            iNumNeighbors = 0
            for neighborSiteIndex, neighborCoord in enumerate(neighborSiteCoords):
                neighborImageDisplacementVectors = np.array([neighborCoord - centerCoord])
                displacement = np.linalg.norm(neighborImageDisplacementVectors)
                if cutoffDistLimits[0] < displacement <= cutoffDistLimits[1]:
                    iNeighborSiteIndexList.append(neighborSiteIndex)
                    iDisplacementVectors.append(neighborImageDisplacementVectors[0])
                    iDisplacements = np.append(iDisplacements, displacement)
                    iNumNeighbors += 1
            neighborSystemElementIndices[centerSiteIndex] = neighborSiteSystemElementIndexList[iNeighborSiteIndexList]
            offsetList[centerSiteIndex] = neighborSiteQuantumIndexList[iNeighborSiteIndexList, :3] - centerSiteQuantumIndexList[centerSiteIndex, :3]
            neighborElementIndexList[centerSiteIndex] = neighborSiteQuantumIndexList[iNeighborSiteIndexList, 4]
            numNeighbors = np.append(numNeighbors, iNumNeighbors)
            displacementVectorList[centerSiteIndex] = np.asarray(iDisplacementVectors)
            displacementList[centerSiteIndex] = iDisplacements
        '''
        # DEBUG: Quick test for number of neighbors: Switch ON End
            
        systemElementIndexMap = np.empty(2, dtype=object)
        systemElementIndexMap[:] = [centerSiteSystemElementIndexList, neighborSystemElementIndices]
        elementIndexMap = np.empty(2, dtype=object)
        elementIndexMap[:] = [centerSiteQuantumIndexList[:,4], neighborElementIndexList]
        
        returnNeighbors = returnValues()
        returnNeighbors.systemElementIndexMap = systemElementIndexMap
        returnNeighbors.offsetList = offsetList
        returnNeighbors.elementIndexMap = elementIndexMap
        returnNeighbors.numNeighbors = numNeighbors
        returnNeighbors.displacementVectorList = displacementVectorList
        returnNeighbors.displacementList = displacementList
        return returnNeighbors
    
    def electrostaticNeighborSites(self, systemSize, bulkSites, centerSiteIndices, neighborSiteIndices, cutoffDistLimits, cutoffDistKey):
        """Returns systemElementIndexMap and distances between center sites and its 
        neighbor sites within cutoff distance"""
        neighborSiteCoords = bulkSites.cellCoordinates[neighborSiteIndices]
        neighborSiteSystemElementIndexList = bulkSites.systemElementIndexList[neighborSiteIndices]
        neighborSiteQuantumIndexList = bulkSites.quantumIndexList[neighborSiteIndices]
        centerSiteCoords = bulkSites.cellCoordinates[centerSiteIndices]
        centerSiteSystemElementIndexList = bulkSites.systemElementIndexList[centerSiteIndices]
        centerSiteQuantumIndexList = bulkSites.quantumIndexList[centerSiteIndices]
        
        neighborSystemElementIndices = np.empty(len(centerSiteCoords), dtype=object)
        neighborElementTypeIndexList = np.empty(len(centerSiteCoords), dtype=object)
        neighborElementIndexList = np.empty(len(centerSiteCoords), dtype=object)
        numNeighbors = np.array([], dtype=int)
        displacementVectorList = np.empty(len(centerSiteCoords), dtype=object)
        displacementList = np.empty(len(centerSiteCoords), dtype=object)
        
        xRange = range(-1, 2) if self.pbc[0] == 1 else [0]
        yRange = range(-1, 2) if self.pbc[1] == 1 else [0]
        zRange = range(-1, 2) if self.pbc[2] == 1 else [0]
        unitcellTranslationalCoords = np.zeros((3**sum(self.pbc), 3)) # Initialization
        index = 0
        for xOffset in xRange:
            for yOffset in yRange:
                for zOffset in zRange:
                    unitcellTranslationalCoords[index] = np.dot(np.multiply(np.array([xOffset, yOffset, zOffset]), self.systemSize), self.material.latticeMatrix)
                    index += 1
        for centerSiteIndex, centerCoord in enumerate(centerSiteCoords):
            iDisplacementVectors = []
            iDisplacements = np.array([])
            iNeighborSiteIndexList = []
            iNumNeighbors = 0
            for neighborSiteIndex, neighborCoord in enumerate(neighborSiteCoords):
                neighborImageCoords = unitcellTranslationalCoords + neighborCoord
                neighborImageDisplacementVectors = neighborImageCoords - centerCoord
                neighborImageDisplacements = np.linalg.norm(neighborImageDisplacementVectors, axis=1)
                [displacement, imageIndex] = [np.min(neighborImageDisplacements), np.argmin(neighborImageDisplacements)]
                if cutoffDistLimits[0] < displacement <= cutoffDistLimits[1]:
                    iNeighborSiteIndexList.append(neighborSiteIndex)
                    iDisplacementVectors.append(neighborImageDisplacementVectors[imageIndex])
                    iDisplacements = np.append(iDisplacements, displacement)
                    iNumNeighbors += 1
            neighborSystemElementIndices[centerSiteIndex] = np.array(neighborSiteSystemElementIndexList[iNeighborSiteIndexList])
            neighborElementTypeIndexList[centerSiteIndex] = neighborSiteQuantumIndexList[iNeighborSiteIndexList, 3]
            neighborElementIndexList[centerSiteIndex] = neighborSiteQuantumIndexList[iNeighborSiteIndexList, 4]
            numNeighbors = np.append(numNeighbors, iNumNeighbors)
            displacementVectorList[centerSiteIndex] = np.asarray(iDisplacementVectors)
            displacementList[centerSiteIndex] = iDisplacements
            
        systemElementIndexMap = np.empty(2, dtype=object)
        systemElementIndexMap[:] = [centerSiteSystemElementIndexList, neighborSystemElementIndices]
        elementTypeIndexMap = np.empty(2, dtype=object)
        elementTypeIndexMap[:] = [centerSiteQuantumIndexList[:,3], neighborElementTypeIndexList]
        elementIndexMap = np.empty(2, dtype=object)
        elementIndexMap[:] = [centerSiteQuantumIndexList[:,4], neighborElementIndexList]
        
        returnNeighbors = returnValues()
        returnNeighbors.systemElementIndexMap = systemElementIndexMap
        returnNeighbors.elementTypeIndexMap = elementTypeIndexMap
        returnNeighbors.elementIndexMap = elementIndexMap
        returnNeighbors.numNeighbors = numNeighbors
        returnNeighbors.displacementVectorList = displacementVectorList
        returnNeighbors.displacementList = displacementList
        return returnNeighbors

    def extractElectrostaticNeighborSites(self, parentElecNeighborList, cutE):
        """Returns systemElementIndexMap and distances between center sites and its 
        neighbor sites within cutoff distance"""
        parentSystemElementIndexMap = parentElecNeighborList.systemElementIndexMap
        parentElementTypeIndexMap = parentElecNeighborList.elementTypeIndexMap
        parentElementIndexMap = parentElecNeighborList.elementIndexMap
        parentDisplacementVectorList = parentElecNeighborList.displacementVectorList
        parentDisplacementList = parentElecNeighborList.displacementList
        
        numSystemElements = len(parentElecNeighborList.numNeighbors)
        neighborSystemElementIndices = np.empty(numSystemElements, dtype=object)
        neighborElementTypeIndexList = np.empty(numSystemElements, dtype=object)
        neighborElementIndexList = np.empty(numSystemElements, dtype=object)
        numNeighbors = np.empty(numSystemElements, dtype=int)
        displacementVectorList = np.empty(numSystemElements, dtype=object)
        displacementList = np.empty(numSystemElements, dtype=object)
        
        startIndex = 0
        for centerSiteIndex in range(numSystemElements):
            extractIndices = np.where((0 < parentElecNeighborList.displacementList[centerSiteIndex]) & (parentElecNeighborList.displacementList[centerSiteIndex] <= cutE * self.material.ANG2BOHR))
            numNeighbors[centerSiteIndex] = len(extractIndices[0])
            endIndex = startIndex + numNeighbors[centerSiteIndex]
            iNeighborSiteIndexList = extractIndices[0]
            neighborSystemElementIndices[centerSiteIndex] = np.asarray(parentSystemElementIndexMap[1][centerSiteIndex][iNeighborSiteIndexList])
            neighborElementTypeIndexList[centerSiteIndex] = np.asarray(parentElementTypeIndexMap[1][centerSiteIndex][iNeighborSiteIndexList])
            neighborElementIndexList[centerSiteIndex] = np.asarray(parentElementIndexMap[1][centerSiteIndex][iNeighborSiteIndexList])
            displacementVectorList[centerSiteIndex] = np.asarray(parentDisplacementVectorList[centerSiteIndex][iNeighborSiteIndexList])
            displacementList[centerSiteIndex] = np.asarray(parentDisplacementList[centerSiteIndex][iNeighborSiteIndexList])
            startIndex = endIndex
            
        systemElementIndexMap = np.empty(2, dtype=object)
        systemElementIndexMap[:] = [parentSystemElementIndexMap[0], neighborSystemElementIndices]
        elementTypeIndexMap = np.empty(2, dtype=object)
        elementTypeIndexMap[:] = [parentElementTypeIndexMap[0], neighborElementTypeIndexList]
        elementIndexMap = np.empty(2, dtype=object)
        elementIndexMap[:] = [parentElementIndexMap[0], neighborElementIndexList]
        
        returnNeighbors = returnValues()
        returnNeighbors.systemElementIndexMap = systemElementIndexMap
        returnNeighbors.elementTypeIndexMap = elementTypeIndexMap
        returnNeighbors.elementIndexMap = elementIndexMap
        returnNeighbors.numNeighbors = numNeighbors
        returnNeighbors.displacementVectorList = displacementVectorList
        returnNeighbors.displacementList = displacementList
        return returnNeighbors
    #@profile
    def generateNeighborList(self, parent, extract=0, cutE=None, replaceExistingNeighborList=0, outdir=None, report=1, localSystemSize=np.array([3, 3, 3]), 
                                 centerUnitCellIndex=np.array([1, 1, 1])):
        """Adds the neighbor list to the system object and returns the neighbor list"""
        if parent == 1:
            assert not extract, 'Use extract flag only to generate child neighbor list'
            assert not cutE, 'Do not provide a cutoff while generating parent neighbor list'
        else:
            assert cutE, 'Provide a desired cutoff distance in angstroms to generate child neighbor list'
        
        assert outdir, 'Please provide the destination path where neighbor list needs to be saved'
        assert all(size >= 3 for size in localSystemSize), 'Local system size in all dimensions should always be greater than or equal to 3'
        
        # DEBUG: Quick test for number of neighbors: Switch ON Start
        # del self.material.neighborCutoffDist['E']
        # DEBUG: Quick test for number of neighbors: Switch ON End
        
        if extract:
            parentNeighborListFileName = 'ParentNeighborList_SystemSize=' + str(self.systemSize).replace(' ', ',') + '.npy'
            parentNeighborListFilePath = outdir + directorySeparator + parentNeighborListFileName
            parentNeighborList = np.load(parentNeighborListFilePath)[()]
            for iCutE in cutE:
                fileName = 'E' + ('%2.1f' % iCutE)
                ChildNeighborListFileName = 'NeighborList_' + fileName + '.npy'
                ChildNeighborListFilePath = outdir + directorySeparator + ChildNeighborListFileName
                assert (not os.path.isfile(ChildNeighborListFilePath) or replaceExistingNeighborList), 'Requested neighbor list file already exists in the destination folder.'
                neighborList = {}
                for cutoffDistKey in parentNeighborList.keys():
                    if cutoffDistKey is 'E':
                        neighborList[cutoffDistKey] = [self.extractElectrostaticNeighborSites(parentNeighborList['E'][0], iCutE)]
                    else:
                        neighborList[cutoffDistKey] = deepcopy(parentNeighborList[cutoffDistKey])
                np.save(ChildNeighborListFilePath, neighborList)
                if report:
                    self.generateNeighborListReport(parent, outdir, fileName)
        else:
            fileName = 'SystemSize=' + str(self.systemSize).replace(' ', ',') if parent else ('E' + ('' if extract else ('%2.1f' % cutE)))
            neighborListFileName = ('Parent' if parent else '') + 'NeighborList_' + fileName + '.npy'
            neighborListFilePath = outdir + directorySeparator + neighborListFileName
            assert (not os.path.isfile(neighborListFilePath) or replaceExistingNeighborList), 'Requested neighbor list file already exists in the destination folder.'
            neighborList = {}
            tolDist = self.material.neighborCutoffDistTol
            elementTypes = self.material.elementTypes[:]
            for cutoffDistKey in self.material.neighborCutoffDist.keys():
                cutoffDistList = self.material.neighborCutoffDist[cutoffDistKey][:]
                neighborListCutoffDistKey = []
                if cutoffDistKey is 'E':
                    centerSiteIndices = neighborSiteIndices = np.arange(self.numCells * self.material.totalElementsPerUnitCell)
                    cutoffDistLimits = [0, np.inf if parent else cutoffDistList[0]]
                    neighborListCutoffDistKey.append(self.electrostaticNeighborSites(self.systemSize, self.bulkSites, centerSiteIndices, 
                                                                                     neighborSiteIndices, cutoffDistLimits, cutoffDistKey))
                else:
                    [centerElementType, neighborElementType] = cutoffDistKey.split(self.material.elementTypeDelimiter)
                    centerSiteElementTypeIndex = elementTypes.index(centerElementType) 
                    neighborSiteElementTypeIndex = elementTypes.index(neighborElementType)
                    # DEBUG: Quick test for number of neighbors: Switch OFF Start
                    localBulkSites = self.material.generateSites(range(len(self.material.elementTypes)), 
                                                                 self.systemSize)
                    # DEBUG: Quick test for number of neighbors: Switch OFF End
                    # DEBUG: Quick test for number of neighbors: Switch ON Start
                    '''
                    localBulkSites = self.material.generateSites(range(len(self.material.elementTypes)), 
                                                                 localSystemSize)
                    '''
                    systemElementIndexOffsetArray = (np.repeat(np.arange(0, self.material.totalElementsPerUnitCell * self.numCells, self.material.totalElementsPerUnitCell), 
                                                               self.material.nElementsPerUnitCell[centerSiteElementTypeIndex]))
                    # DEBUG: Quick test for number of neighbors: Switch ON Start
                    centerSiteIndices = neighborSiteIndices = (np.tile(self.material.nElementsPerUnitCell[:centerSiteElementTypeIndex].sum() + 
                                                                       np.arange(0, self.material.nElementsPerUnitCell[centerSiteElementTypeIndex]), self.numCells) + systemElementIndexOffsetArray)
                    # DEBUG: Quick test for number of neighbors: Switch ON End
                    # DEBUG: Quick test for number of neighbors: Switch OFF Start
                    '''
                    centerSiteIndices = [self.generateSystemElementIndex(localSystemSize, np.concatenate((centerUnitCellIndex, np.array([centerSiteElementTypeIndex]), np.array([elementIndex])))) 
                                         for elementIndex in range(self.material.nElementsPerUnitCell[centerSiteElementTypeIndex])]
                    neighborSiteIndices = [self.generateSystemElementIndex(localSystemSize, np.array([xSize, ySize, zSize, neighborSiteElementTypeIndex, elementIndex])) 
                                           for xSize in range(localSystemSize[0]) for ySize in range(localSystemSize[1]) 
                                           for zSize in range(localSystemSize[2]) 
                                           for elementIndex in range(self.material.nElementsPerUnitCell[neighborSiteElementTypeIndex])]
                    '''
                    # DEBUG: Quick test for number of neighbors: Switch OFF End
                    for iCutoffDist in range(len(cutoffDistList)):
                        cutoffDistLimits = [cutoffDistList[iCutoffDist] - tolDist[cutoffDistKey][iCutoffDist], cutoffDistList[iCutoffDist] + tolDist[cutoffDistKey][iCutoffDist]]
                        
                        neighborListCutoffDistKey.append(self.hopNeighborSites(localBulkSites, centerSiteIndices, 
                                                                               neighborSiteIndices, cutoffDistLimits, cutoffDistKey))
                neighborList[cutoffDistKey] = neighborListCutoffDistKey[:]
            np.save(neighborListFilePath, neighborList)
            if report:
                self.generateNeighborListReport(parent, outdir, fileName)

    def generateNeighborListReport(self, parent, outdir, fileName):
        """Generates a neighbor list and prints out a report to the output directory"""
        neighborListLogName = ('Parent' if parent else '') + 'NeighborList_' + fileName + '.log' 
        neighborListLogPath = outdir + directorySeparator + neighborListLogName
        report = open(neighborListLogPath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - self.startTime
        report.write('Time elapsed: ' + ('%2d days, ' % timeElapsed.days if timeElapsed.days else '') +
                     ('%2d hours' % ((timeElapsed.seconds // 3600) % 24)) + 
                     (', %2d minutes' % ((timeElapsed.seconds // 60) % 60)) + 
                     (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()
    
class system(object):
    """defines the system we are working on
    
    Attributes:
    size: An array (3 x 1) defining the system size in multiple of unit cells
    """
    def __init__(self, material, neighbors, neighborList, speciesCount):
        """Return a system object whose size is *size*"""
        self.material = material
        self.neighbors = neighbors
        self.neighborList = neighborList
        
        self.pbc = self.neighbors.pbc
        self.speciesCount = speciesCount
        self.speciesCountCumSum = speciesCount.cumsum()
        
        # total number of unit cells
        self.systemSize = self.neighbors.systemSize
        self.numCells = self.systemSize.prod()
        
        # generate lattice charge list
        unitcellChargeList = np.array([self.material.chargeTypes[self.material.elementTypes[elementTypeIndex]] 
                                       for elementTypeIndex in self.material.elementTypeIndexList])
        self.latticeChargeList = np.tile(unitcellChargeList, self.numCells)
        
        # Coefficient Distance List
        self.coeffDistanceList = 1 / (self.material.dielectricConstant * self.neighborList['E'][0].displacementList)
        
    def generateRandomOccupancy(self, speciesCount):
        """generates initial occupancy list based on species count"""
        occupancy = []
        for speciesTypeIndex, numSpecies in enumerate(speciesCount):
            siteElementTypesIndices = np.in1d(self.material.elementTypes, self.material.speciesToElementTypeMap[self.material.speciesTypes[speciesTypeIndex]]).nonzero()[0]
            iSpeciesSystemElementIndices = []
            for iSpecies in range(numSpecies):
                siteElementTypeIndex = rnd.choice(siteElementTypesIndices)
                # Enlist all electron eligible sites, hole eligible sites depending on speciesCount, choose randomly from the list. No need for generateSystemElementIndex
                iSpeciesSiteIndices = np.array([rnd.randint(0, self.systemSize[0]-1), 
                                                rnd.randint(0, self.systemSize[1]-1), 
                                                rnd.randint(0, self.systemSize[2]-1), 
                                                siteElementTypeIndex, 
                                                rnd.randint(0, self.material.nElementsPerUnitCell[siteElementTypeIndex]-1)])
                iSpeciesSystemElementIndex = self.neighbors.generateSystemElementIndex(self.systemSize, iSpeciesSiteIndices)
                if iSpeciesSystemElementIndex in iSpeciesSystemElementIndices:
                    iSpecies -= 1
                else:
                    iSpeciesSystemElementIndices.append(iSpeciesSystemElementIndex)
            occupancy.extend(iSpeciesSystemElementIndices[:])
        return occupancy
    
    def chargeConfig(self, occupancy):
        """Returns charge distribution of the current configuration"""
        chargeList = np.copy(self.latticeChargeList)
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

    def ESPConfig(self, currentStateChargeConfig):
        ESPConfig = deepcopy(self.coeffDistanceList)
        for rowIndex in range(len(currentStateChargeConfig)):
            ESPConfig[rowIndex] *= currentStateChargeConfig[rowIndex]
        return ESPConfig

    def config(self, occupancy):
        """Generates the configuration array for the system"""
        elementTypeIndices = range(len(self.material.elementTypes))
        systemSites = self.material.generateSites(elementTypeIndices, self.systemSize)
        positions = systemSites.cellCoordinates
        systemElementIndexList = systemSites.systemElementIndexList
        chargeList = self.chargeConfig(occupancy)
        
        returnConfig = returnValues()
        returnConfig.positions = positions
        returnConfig.chargeList = chargeList
        returnConfig.systemElementIndexList = systemElementIndexList
        returnConfig.occupancy = occupancy
        return returnConfig

class run(object):
    """defines the subroutines for running Kinetic Monte Carlo and computing electrostatic 
    interaction energies"""
    def __init__(self, system, T, nTraj, kmcSteps, stepInterval, gui):
        """Returns the PBC condition of the system"""
        self.startTime = datetime.now()
        
        self.system = system
        self.material = self.system.material
        self.neighbors = self.system.neighbors
        self.T = T * self.system.material.K2AUTEMP
        self.nTraj = int(nTraj)
        self.kmcSteps = int(kmcSteps)
        self.stepInterval = int(stepInterval)
        self.gui = gui

        self.systemSize = self.system.systemSize

        # import parameters from material class
        self.vn = self.system.material.vn
        self.lambdaValues = self.system.material.lambdaValues
        self.VAB = self.system.material.VAB
        
        # nElementsPerUnitCell
        self.headStart_nElementsPerUnitCellCumSum = [self.system.material.nElementsPerUnitCell[:siteElementTypeIndex].sum() for siteElementTypeIndex in range(self.system.material.nElementTypes)]
        
        # speciesTypeList
        self.speciesTypeList = [self.system.material.speciesTypes[index] for index, value in enumerate(self.system.speciesCount) for i in range(value)]
        self.speciesTypeIndexList = [index for index, value in enumerate(self.system.speciesCount) for iValue in range(value)]
        self.siteElementTypeIndexList = [self.system.material.elementTypes.index(self.system.material.speciesToElementTypeMap[speciesType][0]) 
                                         for speciesType in self.speciesTypeList]
        self.hopElementTypeList = [self.system.material.hopElementTypes[speciesType][0] for speciesType in self.speciesTypeList]
        self.lenHopDistTypeList = [len(self.system.material.neighborCutoffDist[hopElementType]) for hopElementType in self.hopElementTypeList]
        
        unOccupantOccupancy = []
        self.unOccupantChargeConfig = self.system.chargeConfig(unOccupantOccupancy)
        self.occupantChargeConfig = deepcopy(self.unOccupantChargeConfig)
        self.unOccupantESPConfig = self.system.ESPConfig(self.unOccupantChargeConfig)
        self.occupantESPConfig = deepcopy(self.unOccupantESPConfig)
        # number of kinetic processes
        self.nProc = 0
        self.multFactor = np.zeros(len(self.material.speciesTypes))
        for hopElementTypeIndex, hopElementType in enumerate(self.hopElementTypeList):
            centerElementType = hopElementType.split(self.material.elementTypeDelimiter)[0]
            speciesTypeIndex = self.material.speciesTypes.index(self.material.elementTypeToSpeciesMap[centerElementType][0])
            self.multFactor[speciesTypeIndex] = np.true_divide(self.material.chargeTypes[centerElementType + self.material.siteIdentifier], 
                                                               self.material.chargeTypes[centerElementType])
            centerSiteElementTypeIndex = self.material.elementTypes.index(centerElementType)
            systemElementIndexOffsetArray = (np.repeat(np.arange(0, self.material.totalElementsPerUnitCell * self.system.numCells, self.material.totalElementsPerUnitCell), 
                                                       self.material.nElementsPerUnitCell[centerSiteElementTypeIndex]))
            speciesTypeSiteIndices = (np.tile(self.material.nElementsPerUnitCell[:centerSiteElementTypeIndex].sum() + 
                                                               np.arange(0, self.material.nElementsPerUnitCell[centerSiteElementTypeIndex]), self.system.numCells) + systemElementIndexOffsetArray)
            self.occupantChargeConfig[speciesTypeSiteIndices] = self.material.chargeTypes[centerElementType + self.material.siteIdentifier]
            self.occupantESPConfig[speciesTypeSiteIndices] *= self.multFactor[speciesTypeIndex]
            for hopDistTypeIndex in range(self.lenHopDistTypeList[hopElementTypeIndex]):
                if self.system.speciesCount[speciesTypeIndex] != 0:
                    numNeighbors = np.unique(self.system.neighborList[hopElementType][hopDistTypeIndex].numNeighbors)
                    # TODO: What if it is not equal to 1
                    if len(numNeighbors) == 1:
                        self.nProc += numNeighbors[0] * self.system.speciesCount[speciesTypeIndex]
        
        # total number of species
        self.totalSpecies = self.system.speciesCount.sum()

        # Electrostatic interaction neighborlist:
        self.elecNeighborListNeighborSEIndices = self.system.neighborList['E'][0].systemElementIndexMap[1]

    def electrostaticInteractionEnergy(self, occupancy):
        """Subroutine to compute the electrostatic interaction energies"""
        chargeConfig = self.system.chargeConfig(occupancy)
        ESPConfig = self.system.ESPConfig(chargeConfig)
        individualInteractionList = (ESPConfig * chargeConfig[self.elecNeighborListNeighborSEIndices])
        elecIntEnergy = np.sum(np.concatenate(individualInteractionList)) * self.system.material.J2EV # electron-volt
        return elecIntEnergy
        
    def ESPRelativeElectrostaticInteractionEnergy(self, currentStateChargeConfig, newStateChargeConfig, 
                                                  oldSiteSystemElementIndex, newSiteSystemElementIndex):
        """Subroutine to compute the relative electrostatic interaction energies between two states"""
        oldSiteElecIntEnergy = np.sum(self.occupantESPConfig[oldSiteSystemElementIndex] * 
                                      currentStateChargeConfig[self.elecNeighborListNeighborSEIndices[oldSiteSystemElementIndex]])
        oldNeighborSiteElecIntEnergy = np.sum(self.unOccupantESPConfig[newSiteSystemElementIndex] * 
                                              currentStateChargeConfig[self.elecNeighborListNeighborSEIndices[newSiteSystemElementIndex]])
        newSiteElecIntEnergy = np.sum(self.occupantESPConfig[newSiteSystemElementIndex] * 
                                      newStateChargeConfig[self.elecNeighborListNeighborSEIndices[newSiteSystemElementIndex]])
        newNeighborSiteElecIntEnergy = np.sum(self.unOccupantESPConfig[oldSiteSystemElementIndex] * 
                                              newStateChargeConfig[self.elecNeighborListNeighborSEIndices[oldSiteSystemElementIndex]])
        relativeElecEnergy = (newSiteElecIntEnergy + newNeighborSiteElecIntEnergy - 
                              oldSiteElecIntEnergy - oldNeighborSiteElecIntEnergy)
        return relativeElecEnergy

    #@profile
    def doKMCSteps(self, outdir=None, report=1, randomSeed=1):
        """Subroutine to run the KMC simulation by specified number of steps"""
        rnd.seed(randomSeed)
        nTraj = self.nTraj
        kmcSteps = self.kmcSteps
        stepInterval = self.stepInterval
        self.initialOccupancy = self.system.generateRandomOccupancy(self.system.speciesCount)
        currentStateOccupancy = self.initialOccupancy[:]
        numPathStepsPerTraj = int(kmcSteps / stepInterval) + 1
        timeArray = np.zeros(nTraj * numPathStepsPerTraj)
        unwrappedPositionArray = np.zeros(( nTraj * numPathStepsPerTraj, self.totalSpecies, 3))
        # Not necessary for now
        # wrappedPositionArray = np.zeros(( nTraj * numPathStepsPerTraj, self.totalSpecies, 3))
        # speciesDisplacementArray = np.zeros(( nTraj * numPathStepsPerTraj, self.totalSpecies, 3))
        # currentStateConfig = self.system.config(currentStateOccupancy)
        pathIndex = 0
        currentStateChargeConfig = self.system.chargeConfig(currentStateOccupancy)
        newStateChargeConfig = np.copy(currentStateChargeConfig)

        kList = np.zeros(self.nProc)
        neighborSystemElementIndexList = np.zeros(self.nProc, dtype=int)
        speciesIndexList = np.zeros(self.nProc, dtype=int)
        speciesTypeIndexList = np.zeros(self.nProc, dtype=int)
        hopElementTypeList = self.nProc * ['']
        hopDistTypeList = np.zeros(self.nProc, dtype=int)
        rowIndexList = np.zeros(self.nProc, dtype=int)
        neighborIndexList = np.zeros(self.nProc, dtype=int)
        assert 'E' in self.system.material.neighborCutoffDist.keys(), 'Please specify the cutoff distance for electrostatic interactions'
        for dummy in range(nTraj):
            pathIndex += 1
            kmcTime = 0
            speciesDisplacementVectorList = np.zeros((self.totalSpecies, 3))
            for step in range(kmcSteps):
                for speciesIndex, speciesSiteSystemElementIndex in enumerate(currentStateOccupancy):
                    speciesType = self.speciesTypeList[speciesIndex]
                    speciesTypeIndex = self.speciesTypeIndexList[speciesIndex]
                    siteElementTypeIndex = self.siteElementTypeIndexList[speciesIndex]
                    hopElementType = self.hopElementTypeList[speciesIndex]
                    rowIndex = (speciesSiteSystemElementIndex / self.material.totalElementsPerUnitCell * self.material.nElementsPerUnitCell[siteElementTypeIndex] + 
                                speciesSiteSystemElementIndex % self.material.totalElementsPerUnitCell - self.headStart_nElementsPerUnitCellCumSum[siteElementTypeIndex])
                    iNeighborIndex = 0
                    for hopDistType in range(self.lenHopDistTypeList[speciesIndex]):
                        neighborSystemElementIndices = self.system.neighborList[hopElementType][hopDistType].systemElementIndexMap[1][rowIndex]
                        for neighborIndex, neighborSystemElementIndex in enumerate(neighborSystemElementIndices):
                            neighborSystemElementIndexList[iNeighborIndex] = neighborSystemElementIndex
                            speciesIndexList[iNeighborIndex] = speciesIndex
                            speciesTypeIndexList[iNeighborIndex] = speciesTypeIndex
                            hopElementTypeList[iNeighborIndex] = hopElementType
                            hopDistTypeList[iNeighborIndex] = hopDistType
                            rowIndexList[iNeighborIndex] = rowIndex
                            neighborIndexList[iNeighborIndex] = neighborIndex
                            
                            newStateChargeConfig[speciesSiteSystemElementIndex] = self.unOccupantChargeConfig[speciesSiteSystemElementIndex]
                            newStateChargeConfig[neighborSystemElementIndex] = self.occupantChargeConfig[neighborSystemElementIndex]
                            delG0 = self.ESPRelativeElectrostaticInteractionEnergy(currentStateChargeConfig, newStateChargeConfig, 
                                                                                   speciesSiteSystemElementIndex, neighborSystemElementIndex)
                            lambdaValue = self.lambdaValues[hopElementType][hopDistType]
                            VAB = self.VAB[hopElementType][hopDistType]
                            delGs = ((lambdaValue + delG0) ** 2 / (4 * lambdaValue)) - VAB
                            kList[iNeighborIndex] = self.vn * np.exp(-delGs / self.T)
                            newStateChargeConfig[[speciesSiteSystemElementIndex, neighborSystemElementIndex]] = newStateChargeConfig[[neighborSystemElementIndex, speciesSiteSystemElementIndex]]
                            iNeighborIndex += 1
                kTotal = np.sum(kList)
                kCumSum = (kList / kTotal).cumsum()
                rand1 = rnd.random()
                procIndex = np.where(kCumSum > rand1)[0][0]
                rand2 = rnd.random()
                kmcTime -= np.log(rand2) / kTotal
                
                oldSiteSystemElementIndex = currentStateOccupancy[speciesIndexList[procIndex]]
                newSiteSystemElementIndex = neighborSystemElementIndexList[procIndex]
                currentStateOccupancy[speciesIndexList[procIndex]] = newSiteSystemElementIndex
                speciesTypeIndex = speciesTypeIndexList[procIndex]

                currentStateChargeConfig[oldSiteSystemElementIndex] = self.unOccupantChargeConfig[oldSiteSystemElementIndex]
                currentStateChargeConfig[newSiteSystemElementIndex] = self.occupantChargeConfig[newSiteSystemElementIndex]
                
                newStateChargeConfig[oldSiteSystemElementIndex] = self.unOccupantChargeConfig[oldSiteSystemElementIndex]
                newStateChargeConfig[newSiteSystemElementIndex] = self.occupantChargeConfig[newSiteSystemElementIndex]
                
                speciesIndex = speciesIndexList[procIndex]
                hopElementType = hopElementTypeList[procIndex]
                hopDistType = hopDistTypeList[procIndex]
                rowIndex = rowIndexList[procIndex]
                neighborIndex = neighborIndexList[procIndex]
                speciesDisplacementVectorList[speciesIndex] += np.copy(self.system.neighborList[hopElementType][hopDistType].displacementVectorList[rowIndex][neighborIndex])
                if step % stepInterval == 0:
                    speciesSystemElementIndices = np.asarray(currentStateOccupancy)
                    timeArray[pathIndex] = kmcTime
                    unwrappedPositionArray[pathIndex] = unwrappedPositionArray[pathIndex - 1] + speciesDisplacementVectorList
                    speciesDisplacementVectorList = np.zeros((self.totalSpecies, 3))
                    pathIndex += 1
        
        trajectoryData = returnValues()
        trajectoryData.timeArray = timeArray
        trajectoryData.unwrappedPositionArray = unwrappedPositionArray
        
        if outdir:
            trajectoryDataFileName = 'TrajectoryData.npy'
            trajectoryDataFilePath = outdir + directorySeparator + trajectoryDataFileName
            np.save(trajectoryDataFilePath, trajectoryData)
        if report:
            self.generateSimulationLogReport(outdir)
        return trajectoryData

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
    def __init__(self, material, trajectoryData, speciesCount, nTraj, kmcSteps, stepInterval, 
                 nStepsMSD, nDispMSD, binsize, reprTime = 'ns', reprDist = 'Angstrom'):
        """"""
        self.startTime = datetime.now()
        self.material = material
        self.trajectoryData = trajectoryData
        self.speciesCount = speciesCount
        self.nTraj = int(nTraj)
        self.kmcSteps = kmcSteps
        self.stepInterval = stepInterval
        self.nStepsMSD = int(nStepsMSD)
        self.nDispMSD = int(nDispMSD)
        self.binsize = binsize
        self.reprTime = reprTime
        self.reprDist = reprDist
        
        self.timeConversion = (1E+09 if reprTime is 'ns' else 1E+00) / self.material.SEC2AUTIME 
        self.distConversion = (1E-10 if reprDist is 'm' else 1E+00) / self.material.ANG2BOHR        
        
    def computeMSD(self, outdir=None, report=1):
        """Returns the squared displacement of the trajectories"""
        time = self.trajectoryData.timeArray * self.timeConversion
        positionArray = self.trajectoryData.unwrappedPositionArray * self.distConversion
        speciesCount = self.speciesCount
        nSpecies = sum(speciesCount)
        nSpeciesTypes = len(self.material.speciesTypes)
        timeNdisp2 = np.zeros((self.nTraj * (self.nStepsMSD * self.nDispMSD), nSpecies + 1))
        numPathStepsPerTraj = int(self.kmcSteps / self.stepInterval) + 1
        for trajIndex in range(self.nTraj):
            headStart = trajIndex * numPathStepsPerTraj
            for timestep in range(1, self.nStepsMSD + 1):
                for step in range(self.nDispMSD):
                    workingRow = trajIndex * (self.nStepsMSD * self.nDispMSD) + (timestep-1) * self.nDispMSD + step
                    timeNdisp2[workingRow, 0] = time[headStart + step + timestep] - time[headStart + step]
                    timeNdisp2[workingRow, 1:] = np.linalg.norm(positionArray[headStart + step + timestep] - 
                                                                positionArray[headStart + step], axis=1)**2
        timeArrayMSD = timeNdisp2[:, 0]
        minEndTime = np.min(timeArrayMSD[np.arange(self.nStepsMSD * self.nDispMSD - 1, self.nTraj * (self.nStepsMSD * self.nDispMSD), self.nStepsMSD * self.nDispMSD)])  
        bins = np.arange(0, minEndTime, self.binsize)
        nBins = len(bins) - 1
        speciesMSDData = np.zeros((nBins, nSpecies))
        msdHistogram, dummy = np.histogram(timeArrayMSD, bins)
        for iSpecies in range(nSpecies):
            iSpeciesHist, dummy = np.histogram(timeArrayMSD, bins, weights=timeNdisp2[:, iSpecies + 1])
            speciesMSDData[:, iSpecies] = iSpeciesHist / msdHistogram
        msdData = np.zeros((nBins+1, nSpeciesTypes + 1 - list(speciesCount).count(0)))
        msdData[1:, 0] = bins[:-1] + 0.5 * self.binsize
        startIndex = 0
        numNonExistentSpecies = 0
        nonExistentSpeciesIndices = []
        for speciesTypeIndex in range(nSpeciesTypes):
            if speciesCount[speciesTypeIndex] != 0:
                endIndex = startIndex + speciesCount[speciesTypeIndex]
                msdData[1:, speciesTypeIndex + 1 - numNonExistentSpecies] = np.mean(speciesMSDData[:, startIndex:endIndex], axis=1)
                startIndex = endIndex
            else:
                numNonExistentSpecies += 1
                nonExistentSpeciesIndices.append(speciesTypeIndex)
        
        if outdir:
            fileName = (('%1.2E' % self.nStepsMSD) + 'nStepsMSD,' + 
                        ('%1.2E' % self.nDispMSD) + 'nDispMSD')

            msdFileName = 'MSD_Data_' + fileName + '.npy'
            msdFilePath = outdir + directorySeparator + msdFileName
            np.save(msdFilePath, msdData)
        if report:
            self.generateMSDAnalysisLogReport(outdir, fileName)
        returnMSDData = returnValues()
        returnMSDData.msdData = msdData
        returnMSDData.speciesTypes = [speciesType for index, speciesType in enumerate(self.material.speciesTypes) if index not in nonExistentSpeciesIndices]
        returnMSDData.fileName = fileName
        return returnMSDData
    
    def generateMSDAnalysisLogReport(self, outdir, fileName):
        """Generates an log report of the MSD Analysis and outputs to the working directory"""
        msdAnalysisLogFileName = 'MSD_Analysis_' + fileName + '.log'
        msdLogFilePath = outdir + directorySeparator + msdAnalysisLogFileName
        report = open(msdLogFilePath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - self.startTime
        report.write('Time elapsed: ' + ('%2d days, ' % timeElapsed.days if timeElapsed.days else '') +
                     ('%2d hours' % ((timeElapsed.seconds // 3600) % 24)) + 
                     (', %2d minutes' % ((timeElapsed.seconds // 60) % 60)) + 
                     (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()

    def displayMSDPlot(self, msdData, speciesTypes, fileName, outdir=None):
        """Returns a line plot of the MSD data"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from textwrap import wrap
        plt.figure()
        for speciesIndex, speciesType in enumerate(speciesTypes):
            plt.plot(msdData[:,0], msdData[:,speciesIndex + 1], label=speciesType)
            
        plt.xlabel('Time (' + self.reprTime + ')')
        plt.ylabel('MSD (' + self.reprDist + '**2)')
        figureTitle = 'MSD_' + fileName
        plt.title('\n'.join(wrap(figureTitle,60)))
        plt.legend()
        plt.show() # Temp change
        if outdir:
            figureName = 'MSD_Plot_' + fileName + '.jpg'
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
