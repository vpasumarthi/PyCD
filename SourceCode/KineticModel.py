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
        
        self.electrostaticCutoffDistKey = materialParameters.electrostaticCutoffDistKey
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
        self.numSystemElements = self.numCells * self.material.totalElementsPerUnitCell
        
        # generate all sites in the system
        self.elementTypeIndices = range(self.material.nElementTypes)
        self.bulkSites = self.material.generateSites(self.elementTypeIndices, self.systemSize)

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
        centerSiteCoords = bulkSites.cellCoordinates[centerSiteIndices]
        centerSiteSystemElementIndexList = bulkSites.systemElementIndexList[centerSiteIndices]
        
        neighborSystemElementIndices = np.empty(len(centerSiteCoords), dtype=object)
        displacementVectorList = np.empty(len(centerSiteCoords), dtype=object)
        numNeighbors = np.array([], dtype=int)

        quickTest = 0 # commit reference: 1472bb4
        if quickTest:
            for centerSiteIndex, centerCoord in enumerate(centerSiteCoords):
                iDisplacementVectors = []
                iNeighborSiteIndexList = []
                iNumNeighbors = 0
                displacementList = np.zeros(len(neighborSiteCoords))
                for neighborSiteIndex, neighborCoord in enumerate(neighborSiteCoords):
                    neighborImageDisplacementVectors = np.array([neighborCoord - centerCoord])
                    displacement = np.linalg.norm(neighborImageDisplacementVectors)
                    displacementList[neighborSiteIndex] = displacement
                    if cutoffDistLimits[0] < displacement <= cutoffDistLimits[1]:
                        iNeighborSiteIndexList.append(neighborSiteIndex)
                        iDisplacementVectors.append(neighborImageDisplacementVectors[0])
                        iNumNeighbors += 1
                neighborSystemElementIndices[centerSiteIndex] = neighborSiteSystemElementIndexList[iNeighborSiteIndexList]
                numNeighbors = np.append(numNeighbors, iNumNeighbors)
                displacementVectorList[centerSiteIndex] = np.asarray(iDisplacementVectors)
                print np.sort(displacementList) / self.material.ANG2BOHR
                import pdb; pdb.set_trace()
        else:
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
                iNeighborSiteIndexList = []
                iDisplacementVectors = []
                iNumNeighbors = 0
                for neighborSiteIndex, neighborCoord in enumerate(neighborSiteCoords):
                    neighborImageCoords = unitcellTranslationalCoords + neighborCoord
                    neighborImageDisplacementVectors = neighborImageCoords - centerCoord
                    neighborImageDisplacements = np.linalg.norm(neighborImageDisplacementVectors, axis=1)
                    [displacement, imageIndex] = [np.min(neighborImageDisplacements), np.argmin(neighborImageDisplacements)]
                    if cutoffDistLimits[0] < displacement <= cutoffDistLimits[1]:
                        iNeighborSiteIndexList.append(neighborSiteIndex)
                        iDisplacementVectors.append(neighborImageDisplacementVectors[imageIndex])
                        iNumNeighbors += 1
                neighborSystemElementIndices[centerSiteIndex] = neighborSiteSystemElementIndexList[iNeighborSiteIndexList]
                displacementVectorList[centerSiteIndex] = np.asarray(iDisplacementVectors)
                numNeighbors = np.append(numNeighbors, iNumNeighbors)
            
        returnNeighbors = returnValues()
        returnNeighbors.neighborSystemElementIndices = neighborSystemElementIndices
        returnNeighbors.displacementVectorList = displacementVectorList
        returnNeighbors.numNeighbors = numNeighbors
        return returnNeighbors
    
    def electrostaticNeighborSites(self, systemSize, bulkSites, centerSiteIndices, neighborSiteIndices, cutoffDistLimits, cutoffDistKey):
        """Returns systemElementIndexMap and distances between center sites and its 
        neighbor sites within cutoff distance"""
        neighborSiteCoords = bulkSites.cellCoordinates[neighborSiteIndices]
        neighborSiteSystemElementIndexList = bulkSites.systemElementIndexList[neighborSiteIndices]
        centerSiteCoords = bulkSites.cellCoordinates[centerSiteIndices]
        
        numElements = len(centerSiteCoords)
        neighborSystemElementIndexMap = np.empty(numElements, dtype=object)
        displacementList = np.empty(numElements, dtype=object)
        numNeighbors = np.empty(numElements, dtype=int)
        
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
            iNeighborSiteIndexList = []
            iDisplacementList = []
            iNumNeighbors = 0
            for neighborSiteIndex, neighborCoord in enumerate(neighborSiteCoords):
                neighborImageCoords = unitcellTranslationalCoords + neighborCoord
                neighborImageDisplacementVectors = neighborImageCoords - centerCoord
                neighborImageDisplacements = np.linalg.norm(neighborImageDisplacementVectors, axis=1)
                [displacement, imageIndex] = [np.min(neighborImageDisplacements), np.argmin(neighborImageDisplacements)]
                if cutoffDistLimits[0] < displacement <= cutoffDistLimits[1]:
                    iNeighborSiteIndexList.append(neighborSiteIndex)
                    iDisplacementList.append(displacement)
                    iNumNeighbors += 1
            neighborSystemElementIndexMap[centerSiteIndex] = dict(zip(neighborSiteSystemElementIndexList[iNeighborSiteIndexList], range(iNumNeighbors)))
            displacementList[centerSiteIndex] = np.array(iDisplacementList)
            numNeighbors[centerSiteIndex] = iNumNeighbors
            
        returnNeighbors = returnValues()
        returnNeighbors.neighborSystemElementIndexMap = neighborSystemElementIndexMap
        returnNeighbors.displacementList = displacementList
        returnNeighbors.numNeighbors = numNeighbors
        return returnNeighbors

    def extractElectrostaticNeighborSites(self, parentElecNeighborList, cutE):
        """Returns systemElementIndexMap and distances between center sites and its 
        neighbor sites within cutoff distance"""
        from operator import itemgetter
        neighborSystemElementIndexMap = np.empty(self.numSystemElements, dtype=object)
        displacementList = np.empty(self.numSystemElements, dtype=object)
        numNeighbors = np.empty(self.numSystemElements, dtype=int)
        for centerSiteIndex in range(self.numSystemElements):
            columnIndices = np.where((0 < parentElecNeighborList.displacementList[centerSiteIndex]) & (parentElecNeighborList.displacementList[centerSiteIndex] <= cutE * self.material.ANG2BOHR))[0]
            numNeighbors[centerSiteIndex] = len(columnIndices)
            neighborSystemElementIndexMap[centerSiteIndex] = dict(zip(itemgetter(*columnIndices)(parentElecNeighborList.neighborSystemElementIndexMap[centerSiteIndex].keys()), range(numNeighbors[centerSiteIndex])))
            displacementList[centerSiteIndex] = parentElecNeighborList.displacementList[centerSiteIndex][columnIndices]
            
        returnNeighbors = returnValues()
        returnNeighbors.neighborSystemElementIndexMap = neighborSystemElementIndexMap
        returnNeighbors.displacementList = displacementList
        returnNeighbors.numNeighbors = numNeighbors
        return returnNeighbors
    #@profile
    def generateNeighborList(self, parentCutoff, extractCutoff, neighborListDirPath, replaceExistingNeighborList=0, report=1, 
                             localSystemSize=np.array([3, 3, 3]), centerUnitCellIndex=np.array([1, 1, 1])):
        """Adds the neighbor list to the system object and returns the neighbor list"""
        assert parentCutoff >= extractCutoff, 'Cutoff for child neighbor list should be smaller than that of parent neighbor list.'
        assert neighborListDirPath, 'Please provide the path to the parent directory of neighbor list files'
        assert all(size >= 3 for size in localSystemSize), 'Local system size in all dimensions should always be greater than or equal to 3'
        
        quickTest = 0 # commit reference: 1472bb4
        if quickTest:
            del self.material.neighborCutoffDist['E']
        if extractCutoff is None:
            dstPath = neighborListDirPath + directorySeparator + 'E_' + str(parentCutoff)
            if not os.path.exists(dstPath):
                os.makedirs(dstPath)
            hopNeighborListFilePath = dstPath + directorySeparator + 'hopNeighborList.npy'
            assert (not os.path.isfile(hopNeighborListFilePath) or replaceExistingNeighborList), 'Requested neighbor list file already exists in the destination folder.'
            hopNeighborList = {}
            tolDist = self.material.neighborCutoffDistTol
            elementTypes = self.material.elementTypes[:]
            
            for cutoffDistKey in self.material.neighborCutoffDist.keys():
                cutoffDistList = self.material.neighborCutoffDist[cutoffDistKey][:]
                neighborListCutoffDistKey = []
                if cutoffDistKey is not self.material.electrostaticCutoffDistKey:
                    [centerElementType, neighborElementType] = cutoffDistKey.split(self.material.elementTypeDelimiter)
                    centerSiteElementTypeIndex = elementTypes.index(centerElementType) 
                    neighborSiteElementTypeIndex = elementTypes.index(neighborElementType)
                    if quickTest:
                        localBulkSites = self.material.generateSites(self.elementTypeIndices, 
                                                                     localSystemSize)
                        centerSiteIndices = [self.generateSystemElementIndex(localSystemSize, np.concatenate((centerUnitCellIndex, np.array([centerSiteElementTypeIndex]), np.array([elementIndex])))) 
                                             for elementIndex in range(self.material.nElementsPerUnitCell[centerSiteElementTypeIndex])]
                        neighborSiteIndices = [self.generateSystemElementIndex(localSystemSize, np.array([xSize, ySize, zSize, neighborSiteElementTypeIndex, elementIndex])) 
                                               for xSize in range(localSystemSize[0]) for ySize in range(localSystemSize[1]) 
                                               for zSize in range(localSystemSize[2]) 
                                               for elementIndex in range(self.material.nElementsPerUnitCell[neighborSiteElementTypeIndex])]
                    else:
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

            if self.material.electrostaticCutoffDistKey in self.material.neighborCutoffDist.keys():
                elecNeighborListFilePath = dstPath + directorySeparator + 'elecNeighborList.npy'
                assert (not os.path.isfile(elecNeighborListFilePath) or replaceExistingNeighborList), 'Requested neighbor list file already exists in the destination folder.'
                centerSiteIndices = neighborSiteIndices = np.arange(self.numCells * self.material.totalElementsPerUnitCell)
                cutoffDistLimits = [0, parentCutoff]
                elecNeighborList = self.electrostaticNeighborSites(self.systemSize, self.bulkSites, centerSiteIndices, 
                                                                   neighborSiteIndices, cutoffDistLimits, cutoffDistKey)
                np.save(elecNeighborListFilePath, elecNeighborList)
            if report:
                self.generateNeighborListReport(dstPath)
        else:
            from shutil import copy
            dstPath = neighborListDirPath + directorySeparator + 'E_' + str(extractCutoff)
            parentNeighborListDirPath = neighborListDirPath + directorySeparator + 'E_' + str(parentCutoff)
            hopNeighborListFilePath = parentNeighborListDirPath + directorySeparator + 'hopNeighborList.npy'
            if not os.path.exists(dstPath):
                os.makedirs(dstPath)
            copy(hopNeighborListFilePath, dstPath)
            
            parentElecNeighborListFilePath = parentNeighborListDirPath + directorySeparator + 'elecNeighborList.npy'
            parentElecNeighborList = np.load(parentElecNeighborListFilePath)[()]
            childNeighborListFilePath = dstPath + directorySeparator + 'elecNeighborList.npy'
            assert (not os.path.isfile(childNeighborListFilePath) or replaceExistingNeighborList), 'Requested neighbor list file already exists in the destination folder.'
            childNeighborList = self.extractElectrostaticNeighborSites(parentElecNeighborList, extractCutoff)
            np.save(childNeighborListFilePath, childNeighborList)
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
    
class system(object):
    """defines the system we are working on
    
    Attributes:
    size: An array (3 x 1) defining the system size in multiple of unit cells
    """
    def __init__(self, material, neighbors, hopNeighborList, elecNeighborList, speciesCount):
        """Return a system object whose size is *size*"""
        self.material = material
        self.neighbors = neighbors
        self.hopNeighborList = hopNeighborList
        self.elecNeighborList = elecNeighborList
        
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
        
        # inverse of cumulative Distance
        self.inverseCoeffDistanceList = np.empty(self.neighbors.numSystemElements, dtype=object)
        for elementIndex in range(self.neighbors.numSystemElements):
            self.inverseCoeffDistanceList[elementIndex] = 1 / (self.material.dielectricConstant * self.elecNeighborList.displacementList[elementIndex])
        
        # positions of all system elements
        self.systemCartesianCoordinates = self.neighbors.bulkSites.cellCoordinates
        self.systemFractionalCoordinates = np.dot(self.systemCartesianCoordinates, np.linalg.inv(np.multiply(self.systemSize, self.material.latticeMatrix)))
        
        # variables for ewald sum
        self.translationalMatrix = np.multiply(self.systemSize, self.material.latticeMatrix) 
        self.systemVolume = abs(np.dot(self.translationalMatrix[0], np.cross(self.translationalMatrix[1], self.translationalMatrix[2])))
        self.reciprocalLatticeMatrix = 2 * np.pi / self.systemVolume * np.array([np.cross(self.translationalMatrix[1], self.translationalMatrix[2]), 
                                                                                 np.cross(self.translationalMatrix[2], self.translationalMatrix[0]),
                                                                                 np.cross(self.translationalMatrix[0], self.translationalMatrix[1])])
        self.translationalVectorLength = np.linalg.norm(self.translationalMatrix, axis=1)
        self.reciprocalLatticeVectorLength = np.linalg.norm(self.reciprocalLatticeMatrix, axis=1)        
    
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
        ESPConfig = np.zeros(self.neighbors.numSystemElements)
        for elementIndex in range(self.neighbors.numSystemElements):
            neighborIndices = self.elecNeighborList.neighborSystemElementIndexMap[elementIndex].keys()
            ESPConfig[elementIndex] = np.sum(self.inverseCoeffDistanceList[elementIndex] * currentStateChargeConfig[neighborIndices])
        return ESPConfig
    
    def ewaldSum(self, chargeConfig, kmax):
        from scipy.special import erfc
        
        ebsl = 1.00E-16
        
        tpi = 2 * np.pi
        con = self.systemVolume / (4 * np.pi)
        con2 = (4 * np.pi) / self.systemVolume
        gexp = - np.log(ebsl)
        eta = 0.11
        print 'eta value for this calculation: %4.10f' % eta
        
        cccc = np.sqrt(eta / np.pi)
        
        x = np.sum(chargeConfig**2)
        # TODO: Can compute total charge based on speciesCount and their individual charges. Use dot product
        print 'Total charge = %4.10E' % self.systemCharge
        
        ewald = -cccc * x - 4 * np.pi * (self.systemCharge**2) / (self.systemVolume * eta)
        tmax = np.sqrt(2 * gexp / eta)
        seta = np.sqrt(eta) / 2
        
        mmm1 = int(tmax / self.translationalVectorLength[0] + 1.5)
        mmm2 = int(tmax / self.translationalVectorLength[1] + 1.5)
        mmm3 = int(tmax / self.translationalVectorLength[2] + 1.5)
        mmm1 = mmm2 = mmm3 = 0
        print 'lattice summation indices -- %d %d %d' % (mmm1, mmm2, mmm3)
        ewaldReal = 0
        for a in range(self.neighbors.numSystemElements):
            for b in range(self.neighbors.numSystemElements):
                v = np.dot(self.systemFractionalCoordinates[a, :] - self.systemFractionalCoordinates[b, :], self.translationalMatrix)
                prod = chargeConfig[a] * chargeConfig[b]
                for i in range(-mmm1, mmm1+1):
                    for j in range(-mmm2, mmm2+1):
                        for k in range(-mmm3, mmm3+1):
                            if a != b or not np.all(np.array([i, j, k])==0):
                                w = v + np.dot(np.array([i, j, k]), self.translationalMatrix)
                                rmag = np.linalg.norm(w)
                                arg = rmag * seta
                                ewaldReal += prod * erfc(arg) / rmag / self.material.dielectricConstant
        print 'Real space part of the ewald energy in a.u.: %2.8f eV' % (ewaldReal / 2 / self.material.EV2J / self.material.J2HARTREE)
        print 'Electrostatic energy computed from ESPConfig: %2.8f eV' % (np.sum(chargeConfig * self.ESPConfig(chargeConfig)) / 2 / self.material.EV2J / self.material.J2HARTREE)
        ewald += ewaldReal
        mmm1 = kmax
        mmm2 = kmax
        mmm3 = kmax
        print 'Reciprocal lattice summation indices -- %d %d %d' % (mmm1, mmm2, mmm3)
        
        for i in range(-mmm1, mmm1+1):
            for j in range(-mmm2, mmm2+1):
                for k in range(-mmm3, mmm3+1):
                    if not np.all(np.array([i, j, k])==0):
                        w = np.dot(np.array([i, j, k]), self.reciprocalLatticeMatrix)
                        rmag2 = np.dot(w, w)
                        x = con2 * np.exp(-rmag2 / eta) / rmag2
                        for a in range(self.neighbors.numSystemElements):
                            for b in range(self.neighbors.numSystemElements):
                                v = self.systemFractionalCoordinates[a, :] - self.systemFractionalCoordinates[b, :]
                                prod = chargeConfig[a] * chargeConfig[b]
                                arg = tpi * np.dot(np.array([i, j, k]), v)
                                ewald += x * prod * np.cos(arg) / self.material.dielectricConstant
        print 'Ewald energy in Rydbergs: %2.8f' % ewald
    
class run(object):
    """defines the subroutines for running Kinetic Monte Carlo and computing electrostatic 
    interaction energies"""
    def __init__(self, system, T, nTraj, kmcSteps, stepInterval, gui):
        """Returns the PBC condition of the system"""
        self.startTime = datetime.now()

        self.system = system
        self.material = self.system.material
        self.neighbors = self.system.neighbors
        self.T = T * self.material.K2AUTEMP
        self.nTraj = int(nTraj)
        self.kmcSteps = int(kmcSteps)
        self.stepInterval = int(stepInterval)
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
                        self.nProc += numNeighbors[0] * self.system.speciesCount[speciesTypeIndex]
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

    #@profile
    def doKMCSteps(self, outdir, report=1, randomSeed=1):
        """Subroutine to run the KMC simulation by specified number of steps"""
        assert outdir, 'Please provide the destination path where simulation output files needs to be saved'
        
        testEwald = 0
        if testEwald:
            currentStateOccupancy = [0, 660]
            currentStateChargeConfig = self.system.chargeConfig(currentStateOccupancy)
            kmax = 4
            self.system.ewaldSum(currentStateChargeConfig, kmax)
        
        timeDataFileName = outdir + directorySeparator + 'Time.dat'
        unwrappedTrajFileName = outdir + directorySeparator + 'unwrappedTraj.dat'
        open(timeDataFileName, 'w').close()
        open(unwrappedTrajFileName, 'w').close()
        excess = 0
        if excess:
            wrappedTrajFileName = outdir + directorySeparator + 'wrappedTraj.dat'
            energyTrajFileName = outdir + directorySeparator + 'energyTraj.dat'
            delG0TrajFileName = outdir + directorySeparator + 'delG0Traj.dat'
            potentialTrajFileName = outdir + directorySeparator + 'potentialTraj.dat'
            open(wrappedTrajFileName, 'w').close()
            open(energyTrajFileName, 'w').close()
            open(delG0TrajFileName, 'w').close()
            open(potentialTrajFileName, 'w').close()
        rnd.seed(randomSeed)
        nTraj = self.nTraj
        kmcSteps = self.kmcSteps
        stepInterval = self.stepInterval
        numPathStepsPerTraj = int(kmcSteps / stepInterval) + 1
        timeArray = np.zeros(numPathStepsPerTraj)
        unwrappedPositionArray = np.zeros(( numPathStepsPerTraj, self.totalSpecies * 3))
        if excess:
            wrappedPositionArray = np.zeros(( numPathStepsPerTraj, self.totalSpecies * 3))
            energyArray = np.zeros(( numPathStepsPerTraj ))
            delG0Array = np.zeros(( self.kmcSteps ))
            potentialArray = np.zeros(( numPathStepsPerTraj, self.totalSpecies))
        kList = np.zeros(self.nProc)
        neighborSiteSystemElementIndexList = np.zeros(self.nProc, dtype=int)
        rowIndexList = np.zeros(self.nProc, dtype=int)
        neighborIndexList = np.zeros(self.nProc, dtype=int)
        assert 'E' in self.material.neighborCutoffDist.keys(), 'Please specify the cutoff distance for electrostatic interactions'
        for trajIndex in range(nTraj):
            currentStateOccupancy = self.system.generateRandomOccupancy(self.system.speciesCount)
            currentStateChargeConfig = self.system.chargeConfig(currentStateOccupancy)
            currentStateESPConfig = self.system.ESPConfig(currentStateChargeConfig)
            pathIndex = 0
            if excess:
                # TODO: Avoid using flatten
                wrappedPositionArray[pathIndex] = self.systemCoordinates[currentStateOccupancy].flatten()
                energyArray[pathIndex] = np.sum(currentStateChargeConfig * currentStateESPConfig) / 2
                potentialArray[pathIndex] = currentStateESPConfig[currentStateOccupancy]
            pathIndex += 1
            kmcTime = 0
            speciesDisplacementVectorList = np.zeros((1, self.totalSpecies * 3))
            for step in range(kmcSteps):
                iProc = 0
                if excess:
                    delG0List = []
                for speciesIndex, speciesSiteSystemElementIndex in enumerate(currentStateOccupancy):
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
                            columnIndex = self.system.elecNeighborList.neighborSystemElementIndexMap[speciesSiteSystemElementIndex][neighborSiteSystemElementIndex]
                            delG0 = (self.speciesChargeList[speciesIndex] * ((currentStateESPConfig[neighborSiteSystemElementIndex] - currentStateESPConfig[speciesSiteSystemElementIndex]
                                                                              - self.speciesChargeList[speciesIndex] * self.system.inverseCoeffDistanceList[speciesSiteSystemElementIndex][columnIndex])))
                            if excess:
                                delG0List.append(delG0)
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
                kmcTime -= np.log(rand2) / kTotal
                
                if excess:
                    delG0Array[step] = delG0List[procIndex]
                speciesIndex = self.nProcSpeciesIndexList[procIndex]
                hopElementType = self.nProcHopElementTypeList[procIndex]
                hopDistType = self.nProcHopDistTypeList[procIndex]
                rowIndex = rowIndexList[procIndex]
                neighborIndex = neighborIndexList[procIndex]
                oldSiteSystemElementIndex = currentStateOccupancy[speciesIndex]
                newSiteSystemElementIndex = neighborSiteSystemElementIndexList[procIndex]
                currentStateOccupancy[speciesIndex] = newSiteSystemElementIndex
                speciesDisplacementVectorList[0, speciesIndex * 3:(speciesIndex + 1) * 3] += self.system.hopNeighborList[hopElementType][hopDistType].displacementVectorList[rowIndex][neighborIndex]

                oldSiteNeighbors = self.system.elecNeighborList.neighborSystemElementIndexMap[oldSiteSystemElementIndex].keys()
                newSiteNeighbors = self.system.elecNeighborList.neighborSystemElementIndexMap[newSiteSystemElementIndex].keys()
                currentStateESPConfig[oldSiteNeighbors] -= self.speciesChargeList[speciesIndex] * self.system.inverseCoeffDistanceList[oldSiteSystemElementIndex]
                currentStateESPConfig[newSiteNeighbors] += self.speciesChargeList[speciesIndex] * self.system.inverseCoeffDistanceList[newSiteSystemElementIndex]
                currentStateChargeConfig[oldSiteSystemElementIndex] -= self.speciesChargeList[speciesIndex]
                currentStateChargeConfig[newSiteSystemElementIndex] += self.speciesChargeList[speciesIndex]
                if (step + 1) % stepInterval == 0:
                    timeArray[pathIndex] = kmcTime
                    unwrappedPositionArray[pathIndex] = unwrappedPositionArray[pathIndex - 1] + speciesDisplacementVectorList
                    speciesDisplacementVectorList = np.zeros((1, self.totalSpecies * 3))
                    if excess:
                        # TODO: Avoid using flatten
                        wrappedPositionArray[pathIndex] = self.systemCoordinates[currentStateOccupancy].flatten()
                        energyArray[pathIndex] = energyArray[pathIndex - 1] + sum(delG0Array[trajIndex * self.kmcSteps + step + 1 - stepInterval: trajIndex * self.kmcSteps + step + 1])
                        potentialArray[pathIndex] = currentStateESPConfig[currentStateOccupancy]
                    pathIndex += 1

            with open(timeDataFileName, 'a') as timeDataFile:
                np.savetxt(timeDataFile, timeArray)
            with open(unwrappedTrajFileName, 'a') as unwrappedTrajFile:
                np.savetxt(unwrappedTrajFile, unwrappedPositionArray)
            if excess:
                with open(wrappedTrajFileName, 'a') as wrappedTrajFile:
                    np.savetxt(wrappedTrajFile, wrappedPositionArray)
                with open(energyTrajFileName, 'a') as energyTrajFile:
                    np.savetxt(energyTrajFile, energyArray)
                with open(delG0TrajFileName, 'a') as delG0TrajFile:
                    np.savetxt(delG0TrajFile, delG0Array)
                with open(potentialTrajFileName, 'a') as potentialTrajFile:
                    np.savetxt(potentialTrajFile, potentialArray)
        if report:
            self.generateSimulationLogReport(outdir)
        return

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
    def __init__(self, material, speciesCount, nTraj, kmcSteps, stepInterval, 
                 systemSize, nStepsMSD, nDispMSD, binsize, reprTime = 'ns', reprDist = 'Angstrom'):
        """"""
        self.startTime = datetime.now()
        self.material = material
        self.speciesCount = speciesCount
        self.totalSpecies = self.speciesCount.sum()
        self.nTraj = int(nTraj)
        self.kmcSteps = kmcSteps
        self.stepInterval = stepInterval
        self.systemSize = systemSize
        self.nStepsMSD = int(nStepsMSD)
        self.nDispMSD = int(nDispMSD)
        self.binsize = binsize
        self.reprTime = reprTime
        self.reprDist = reprDist
        
        self.timeConversion = (1E+09 if reprTime is 'ns' else 1E+00) / self.material.SEC2AUTIME 
        self.distConversion = (1E-10 if reprDist is 'm' else 1E+00) / self.material.ANG2BOHR        
        
    def computeMSD(self, outdir, report=1):
        """Returns the squared displacement of the trajectories"""
        assert outdir, 'Please provide the destination path where MSD output files needs to be saved'
        numPathStepsPerTraj = int(self.kmcSteps / self.stepInterval) + 1
        time = np.loadtxt(outdir + directorySeparator + 'Time.dat') * self.timeConversion
        numTrajRecorded = int(len(time) / numPathStepsPerTraj)
        positionArray = np.loadtxt(outdir + directorySeparator + 'unwrappedTraj.dat')[:numTrajRecorded * numPathStepsPerTraj + 1].reshape((numTrajRecorded * numPathStepsPerTraj, self.totalSpecies, 3)) * self.distConversion
        speciesCount = self.speciesCount
        nSpecies = sum(speciesCount)
        nSpeciesTypes = len(self.material.speciesTypes)
        timeNdisp2 = np.zeros((numTrajRecorded * (self.nStepsMSD * self.nDispMSD), nSpecies + 1))
        for trajIndex in range(numTrajRecorded):
            headStart = trajIndex * numPathStepsPerTraj
            for timestep in range(1, self.nStepsMSD + 1):
                for step in range(self.nDispMSD):
                    workingRow = trajIndex * (self.nStepsMSD * self.nDispMSD) + (timestep-1) * self.nDispMSD + step
                    timeNdisp2[workingRow, 0] = time[headStart + step + timestep] - time[headStart + step]
                    timeNdisp2[workingRow, 1:] = np.linalg.norm(positionArray[headStart + step + timestep] - 
                                                                positionArray[headStart + step], axis=1)**2
        timeArrayMSD = timeNdisp2[:, 0]
        minEndTime = np.min(timeArrayMSD[np.arange(self.nStepsMSD * self.nDispMSD - 1, numTrajRecorded * (self.nStepsMSD * self.nDispMSD), self.nStepsMSD * self.nDispMSD)])
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
        
        fileName = (('%1.2E' % self.nStepsMSD) + 'nStepsMSD,' + 
                    ('%1.2E' % self.nDispMSD) + 'nDispMSD' + 
                    (',nTraj: %1.2E' % numTrajRecorded if numTrajRecorded != self.nTraj else ''))
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

    def displayMSDPlot(self, msdData, speciesTypes, fileName, outdir):
        """Returns a line plot of the MSD data"""
        assert outdir, 'Please provide the destination path where MSD Plot files needs to be saved'
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
