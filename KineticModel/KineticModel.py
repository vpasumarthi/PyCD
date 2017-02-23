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
#from memory_profiler import profile
import sys
import psutil

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
        # TODO: Is AUPOTENTIAL necessary?
        #self.AUPOTENTIAL = self.HARTREE / self.ECHARGE # V
        
        # CONVERSIONS
        self.EV2J = self.ECHARGE
        self.ANG2BOHR = self.ANG / self.BOHR
        self.J2HARTREE = 1 / self.HARTREE
        self.SEC2AUTIME = 1 / self.AUTIME
        self.K2AUTEMP = 1 / self.AUTEMPERATURE
        
        # TODO: introduce a method to view the material using ase atoms or other gui module
        self.name = materialParameters.name
        self.elementTypes = materialParameters.elementTypes
        self.speciesToElementTypeMap = materialParameters.speciesToElementTypeMap
        self.unitcellCoords = np.zeros((len(materialParameters.unitcellCoords), 3)) # Initialization
        startIndex = 0
        length = len(self.elementTypes)
        nElementsPerUnitCell = np.zeros(length, int)
        for elementIndex in range(length):
            elementUnitCellCoords = materialParameters.unitcellCoords[materialParameters.elementTypeIndexList==elementIndex]
            nElementsPerUnitCell[elementIndex] = len(elementUnitCellCoords)
            endIndex = startIndex + nElementsPerUnitCell[elementIndex]
            self.unitcellCoords[startIndex:endIndex] = elementUnitCellCoords[elementUnitCellCoords[:,2].argsort()]
            startIndex = endIndex
        
        self.unitcellCoords *= self.ANG2BOHR # Unit cell coordinates converted to atomic units
        self.nElementsPerUnitCell = nElementsPerUnitCell
        self.totalElementsPerUnitCell = nElementsPerUnitCell.sum()
        self.elementTypeIndexList = materialParameters.elementTypeIndexList.astype(int)
        self.chargeTypes = materialParameters.chargeTypes
        
        self.latticeParameters = [0] * len(materialParameters.latticeParameters)
        # lattice parameters being converted to atomic units
        for index in range(len(materialParameters.latticeParameters)):
            if index < 3:
                self.latticeParameters[index] = materialParameters.latticeParameters[index] * self.ANG2BOHR
            else:
                self.latticeParameters[index] = materialParameters.latticeParameters[index]
        
        self.vn = materialParameters.vn / self.SEC2AUTIME
        self.lambdaValues = materialParameters.lambdaValues
        self.lambdaValues.update((x, [y[index] * self.EV2J * self.J2HARTREE for index in range(len(y))]) for x, y in self.lambdaValues.items())

        self.VAB = materialParameters.VAB
        self.VAB.update((x, [y[index] * self.EV2J * self.J2HARTREE for index in range(len(y))]) for x, y in self.VAB.items())
        
        self.neighborCutoffDist = materialParameters.neighborCutoffDist
        self.neighborCutoffDist.update((x, [y[index] * self.ANG2BOHR for index in range(len(y))]) for x, y in self.neighborCutoffDist.items())

        self.neighborCutoffDistTol = materialParameters.neighborCutoffDistTol * self.ANG2BOHR
        self.elementTypeDelimiter = materialParameters.elementTypeDelimiter
        self.emptySpeciesType = materialParameters.emptySpeciesType
        self.siteIdentifier = materialParameters.siteIdentifier
        self.dielectricConstant = materialParameters.dielectricConstant
                
        siteList = [self.speciesToElementTypeMap[key] for key in self.speciesToElementTypeMap 
                    if key is not self.emptySpeciesType]
        self.siteList = list(set([item for sublist in siteList for item in sublist]))
        
        nonEmptySpeciesToElementTypeMap = materialParameters.speciesToElementTypeMap.copy()
        del nonEmptySpeciesToElementTypeMap[self.emptySpeciesType]
        self.nonEmptySpeciesToElementTypeMap = nonEmptySpeciesToElementTypeMap
        
        elementTypeToSpeciesMap = {}
        for elementType in self.elementTypes:
            speciesList = []
            for speciesTypeKey in nonEmptySpeciesToElementTypeMap.keys():
                if elementType in nonEmptySpeciesToElementTypeMap[speciesTypeKey]:
                    speciesList.append(speciesTypeKey)
            elementTypeToSpeciesMap[elementType] = speciesList
        self.elementTypeToSpeciesMap = elementTypeToSpeciesMap

        hopElementTypes = {key: [self.elementTypeDelimiter.join(comb) 
                                 for comb in list(itertools.product(self.speciesToElementTypeMap[key], repeat=2))] 
                           for key in self.speciesToElementTypeMap if key is not self.emptySpeciesType}
        self.hopElementTypes = hopElementTypes
        
        [a, b, c, alpha, beta, gamma] = self.latticeParameters
        self.latticeMatrix = np.array([[ a                , 0                , 0],
                                       [ b * np.cos(gamma), b * np.sin(gamma), 0],
                                       [ c * np.cos(alpha), c * np.cos(beta) , c * 
                                        np.sqrt(np.sin(alpha)**2 - np.cos(beta)**2)]])
        
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
        cellCoordinates = np.zeros((numCells * nSitesPerUnitCell, 3))
        # quantumIndex = [unitCellIndex, elementTypeIndex, elementIndex]
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
    
    def generateSystemElementIndex(self, systemSize, quantumIndices):
        """Returns the systemElementIndex of the element"""
        assert 0 not in systemSize, 'System size should be greater than 0 in any dimension'
        assert quantumIndices[-1] < self.nElementsPerUnitCell[quantumIndices[-2]], 'Element Index exceed number of elements of the specified element type'
        assert all(quantumIndex >= 0 for quantumIndex in quantumIndices), 'Quantum Indices cannot be negative'
        unitCellIndex = quantumIndices[:3]
        [elementTypeIndex, elementIndex] = quantumIndices[-2:]
        systemElementIndex = elementIndex + self.nElementsPerUnitCell[:elementTypeIndex].sum()
        nDim = len(systemSize)
        for index in range(nDim):
            if index == 0:
                systemElementIndex += self.totalElementsPerUnitCell * unitCellIndex[nDim-1-index]
            else:
                systemElementIndex += self.totalElementsPerUnitCell * unitCellIndex[nDim-1-index] * systemSize[-index:].prod()
        return systemElementIndex
    
    def generateQuantumIndices(self, systemSize, systemElementIndex):
        """Returns the quantum indices of the element"""
        assert systemElementIndex >= 0, 'System Element Index cannot be negative'
        quantumIndices = [0] * 5
        totalElementsPerUnitCellCumSum = self.nElementsPerUnitCell.cumsum()
        unitcellElementIndex = systemElementIndex % self.totalElementsPerUnitCell
        quantumIndices[3] = np.where(totalElementsPerUnitCellCumSum >= (unitcellElementIndex + 1))[0][0]
        quantumIndices[4] = unitcellElementIndex - self.nElementsPerUnitCell[:quantumIndices[3]].sum()
        nFilledUnitCells = (systemElementIndex - unitcellElementIndex) / self.totalElementsPerUnitCell
        for index in range(3):
            quantumIndices[index] = nFilledUnitCells / systemSize[index+1:].prod()
            nFilledUnitCells -= quantumIndices[index] * systemSize[index+1:].prod()
        return quantumIndices

class neighbors(object):
    """Returns the neighbor list file"""
    
    def __init__(self, material, systemSize=np.array([10, 10, 10]), pbc=[1, 1, 1]):
        self.startTime = datetime.now()
        self.material = material
        self.systemSize = systemSize
        self.pbc = pbc
        
        # total number of unit cells
        self.numCells = self.systemSize.prod()
        
        # generate all sites in the system
        elementTypeIndices = range(len(self.material.elementTypes))
        bulkSites = self.material.generateSites(elementTypeIndices, self.systemSize)
        self.bulkSites = bulkSites
        
    def generateNeighborsFile(self, materialNeighbors, neighborsFileName, replaceExistingObjectFiles):
        """ """
        if not os.path.isfile(neighborsFileName) or replaceExistingObjectFiles:
            file_Neighbors = open(neighborsFileName, 'w')
            pickle.dump(materialNeighbors, file_Neighbors)
            file_Neighbors.close()
        pass
        
    def hopNeighborSites(self, bulkSites, centerSiteIndices, neighborSiteIndices, cutoffDistLimits, cutoffDistKey):
        """Returns systemElementIndexMap and distances between center sites and its neighbor sites within cutoff 
        distance"""
        neighborSiteCoords = bulkSites.cellCoordinates[neighborSiteIndices]
        neighborSiteSystemElementIndexList = bulkSites.systemElementIndexList[neighborSiteIndices]
        neighborSiteQuantumIndexList = bulkSites.quantumIndexList[neighborSiteIndices]
        centerSiteCoords = bulkSites.cellCoordinates[centerSiteIndices]
        centerSiteSystemElementIndexList = bulkSites.systemElementIndexList[centerSiteIndices]
        centerSiteQuantumIndexList = bulkSites.quantumIndexList[centerSiteIndices]
        
        neighborSystemElementIndices = []
        offsetList = [] 
        neighborElementIndexList = []
        numNeighbors = []
        displacementVectorList = []
        displacementList = []
        
        for centerSiteIndex, centerCoord in enumerate(centerSiteCoords):
            iDisplacementVectors = []
            iDisplacements = []
            iNeighborSiteIndexList = []
            for neighborSiteIndex, neighborCoord in enumerate(neighborSiteCoords):
                neighborImageDisplacementVectors = np.array([neighborCoord - centerCoord])
                displacement = np.linalg.norm(neighborImageDisplacementVectors)
                if cutoffDistLimits[0] < displacement <= cutoffDistLimits[1]:
                    iNeighborSiteIndexList.append(neighborSiteIndex)
                    iDisplacementVectors.append(neighborImageDisplacementVectors[0])
                    iDisplacements.append(displacement)
            #print centerSiteIndex, cutoffDistKey, len(iDisplacements)#, sorted(iDisplacements)
            neighborSystemElementIndices.append(np.array(neighborSiteSystemElementIndexList[iNeighborSiteIndexList]))
            offsetList.append(centerSiteQuantumIndexList[centerSiteIndex, :3] - neighborSiteQuantumIndexList[iNeighborSiteIndexList, :3])
            neighborElementIndexList.append(neighborSiteQuantumIndexList[iNeighborSiteIndexList, 4])
            displacementVectorList.append(np.asarray(iDisplacementVectors))
            displacementList.append(iDisplacements)
            
        # TODO: Avoid conversion and initialize the object beforehand
        neighborSystemElementIndices = np.asarray(neighborSystemElementIndices)
        systemElementIndexMap = np.empty(2, dtype=object)
        systemElementIndexMap[:] = [centerSiteSystemElementIndexList, neighborSystemElementIndices]
        offsetList = np.asarray(offsetList)
        neighborElementIndexList = np.asarray(neighborElementIndexList)
        elementIndexMap = np.empty(2, dtype=object)
        elementIndexMap[:] = [centerSiteQuantumIndexList[:,4], neighborElementIndexList]
        numNeighbors = np.asarray(numNeighbors, int)

        returnNeighbors = returnValues()
        returnNeighbors.systemElementIndexMap = systemElementIndexMap
        returnNeighbors.offsetList = offsetList
        returnNeighbors.elementIndexMap = elementIndexMap
        returnNeighbors.numNeighbors = numNeighbors
        # TODO: Avoid conversion and initialize the object beforehand
        returnNeighbors.displacementVectorList = np.asarray(displacementVectorList)
        returnNeighbors.displacementList = np.asarray(displacementList)
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
        
        neighborSystemElementIndices = []
        offsetList = []
        neighborElementTypeIndexList = []
        neighborElementIndexList = []
        numNeighbors = []
        displacementVectorList = []
        displacementList = []
        
        xRange = range(-1, 2) if self.pbc[0] == 1 else [0]
        yRange = range(-1, 2) if self.pbc[1] == 1 else [0]
        zRange = range(-1, 2) if self.pbc[2] == 1 else [0]
        latticeMatrix = self.material.latticeMatrix
        for centerSiteIndex, centerCoord in enumerate(centerSiteCoords):
            iDisplacementVectors = []
            iDisplacements = []
            iNeighborSiteIndexList = []
            iNumNeighbors = 0
            for neighborSiteIndex, neighborCoord in enumerate(neighborSiteCoords):
                neighborImageCoords = np.zeros((3**sum(self.pbc), 3))
                index = 0
                for xOffset in xRange:
                    for yOffset in yRange:
                        for zOffset in zRange:
                            unitcellTranslationalCoords = np.dot(np.multiply(np.array([xOffset, yOffset, zOffset]), self.systemSize), latticeMatrix)
                            neighborImageCoords[index] = neighborCoord + unitcellTranslationalCoords
                            index += 1
                neighborImageDisplacementVectors = neighborImageCoords - centerCoord
                neighborImageDisplacements = np.linalg.norm(neighborImageDisplacementVectors, axis=1)
                [displacement, imageIndex] = [np.min(neighborImageDisplacements), np.argmin(neighborImageDisplacements)]
                if cutoffDistLimits[0] < displacement <= cutoffDistLimits[1]:
                    iNeighborSiteIndexList.append(neighborSiteIndex)
                    iDisplacementVectors.append(neighborImageDisplacementVectors[imageIndex])
                    iDisplacements.append(displacement)
                    iNumNeighbors += 1
            #print centerSiteIndex, cutoffDistKey, len(iDisplacements)#, sorted(iDisplacements)
            neighborSystemElementIndices.append(np.array(neighborSiteSystemElementIndexList[iNeighborSiteIndexList]))
            offsetList.append(centerSiteQuantumIndexList[centerSiteIndex, :3] - neighborSiteQuantumIndexList[iNeighborSiteIndexList, :3])
            neighborElementTypeIndexList.append(neighborSiteQuantumIndexList[iNeighborSiteIndexList, 3])
            neighborElementIndexList.append(neighborSiteQuantumIndexList[iNeighborSiteIndexList, 4])
            displacementVectorList.append(np.asarray(iDisplacementVectors))
            displacementList.append(np.asarray(iDisplacements))
            numNeighbors.append(iNumNeighbors)
            
        # TODO: Avoid conversion and initialize the object beforehand
        neighborSystemElementIndices = np.asarray(neighborSystemElementIndices)
        systemElementIndexMap = np.empty(2, dtype=object)
        systemElementIndexMap[:] = [centerSiteSystemElementIndexList, neighborSystemElementIndices]
        offsetList = np.asarray(offsetList)
        neighborElementIndexList = np.asarray(neighborElementIndexList)
        neighborElementTypeIndexList = np.asarray(neighborElementTypeIndexList)
        elementTypeIndexMap = np.empty(2, dtype=object)
        elementTypeIndexMap[:] = [centerSiteQuantumIndexList[:,3], neighborElementTypeIndexList]
        elementIndexMap = np.empty(2, dtype=object)
        elementIndexMap[:] = [centerSiteQuantumIndexList[:,4], neighborElementIndexList]
        numNeighbors = np.asarray(numNeighbors, int)

        returnNeighbors = returnValues()
        returnNeighbors.systemElementIndexMap = systemElementIndexMap
        returnNeighbors.offsetList = offsetList
        returnNeighbors.elementIndexMap = elementIndexMap
        returnNeighbors.numNeighbors = numNeighbors
        # TODO: Avoid conversion and initialize the object beforehand
        returnNeighbors.displacementVectorList = np.asarray(displacementVectorList)
        returnNeighbors.displacementList = np.asarray(displacementList)
        return returnNeighbors

    def generateNeighborList(self, cutE, replaceExistingNeighborList, outdir=None, report=1, localSystemSize=np.array([3, 3, 3]), 
                             centerUnitCellIndex=np.array([1, 1, 1])):
        """Adds the neighbor list to the system object and returns the neighbor list"""
        assert all(size >= 3 for size in localSystemSize), 'Local system size in all dimensions should always be greater than or equal to 3'
        assert outdir != None, 'Please provide the destination path where neighbor list needs to be saved'

        fileName = 'E' + ('%2.1f' % cutE)
        neighborListFileName = 'NeighborList_' + fileName + '.npy'
        neighborListFilePath = outdir + '/' + neighborListFileName
        if not os.path.isfile(neighborListFilePath) or replaceExistingNeighborList:
            import pdb; pdb.set_trace()
            neighborList = {}
            tolDist = self.material.neighborCutoffDistTol
            elementTypes = self.material.elementTypes
            for cutoffDistKey in self.material.neighborCutoffDist.keys():
                cutoffDistList = self.material.neighborCutoffDist[cutoffDistKey]
                neighborListCutoffDistKey = []
                if cutoffDistKey is not 'E':
                    [centerElementType, neighborElementType] = cutoffDistKey.split(self.material.elementTypeDelimiter)
                    centerSiteElementTypeIndex = elementTypes.index(centerElementType) 
                    neighborSiteElementTypeIndex = elementTypes.index(neighborElementType)
                    localBulkSites = self.material.generateSites(range(len(self.material.elementTypes)), 
                                                                 localSystemSize)
                    centerSiteIndices = [self.material.generateSystemElementIndex(localSystemSize, np.concatenate((centerUnitCellIndex, np.array([centerSiteElementTypeIndex]), np.array([elementIndex])))) 
                                         for elementIndex in range(self.material.nElementsPerUnitCell[centerSiteElementTypeIndex])]
                    neighborSiteIndices = [self.material.generateSystemElementIndex(localSystemSize, np.array([xSize, ySize, zSize, neighborSiteElementTypeIndex, elementIndex])) 
                                           for xSize in range(localSystemSize[0]) for ySize in range(localSystemSize[1]) 
                                           for zSize in range(localSystemSize[2]) 
                                           for elementIndex in range(self.material.nElementsPerUnitCell[neighborSiteElementTypeIndex])]
    
                    for cutoffDist in cutoffDistList:
                        cutoffDistLimits = [cutoffDist-tolDist, cutoffDist+tolDist]
                        neighborListCutoffDistKey.append(self.hopNeighborSites(localBulkSites, centerSiteIndices, 
                                                                               neighborSiteIndices, cutoffDistLimits, cutoffDistKey))
                else:
                    centerSiteIndices = neighborSiteIndices = np.arange(self.numCells * self.material.totalElementsPerUnitCell)
                    cutoffDistLimits = [0, cutoffDistList[0]]
                    neighborListCutoffDistKey.append(self.electrostaticNeighborSites(self.systemSize, self.bulkSites, centerSiteIndices, 
                                                                                     neighborSiteIndices, cutoffDistLimits, cutoffDistKey))
                neighborList[cutoffDistKey] = neighborListCutoffDistKey
            
            np.save(neighborListFilePath, neighborList)
            self.generateNeighborListReport(outdir, fileName)
    
    def generateNeighborListReport(self, outdir, fileName):
        """Generates a neighbor list and prints out a report to the output directory"""
        neighborListLogName = 'NeighborList_' + fileName + '.log'
        neighborListLogPath = outdir + '/' + neighborListLogName
        report = open(neighborListLogPath, 'w')
        endTime = datetime.now()
        timeElapsed = endTime - self.startTime
        report.write('Time elapsed: ' + ('%2d days, ' % timeElapsed.days if timeElapsed.days else '') +
                     ('%2d hours' % ((timeElapsed.seconds // 3600) % 24)) + 
                     (', %2d minutes' % ((timeElapsed.seconds // 60) % 60)) + 
                     (', %2d seconds' % (timeElapsed.seconds % 60)))
        report.close()
    
class initiateSystem(object):
    """ """
    def __init__(self, material, neighbors):
        self.material = material
        self.neighbors = neighbors
        self.systemSize = self.neighbors.systemSize

    def generateRandomOccupancy(self, speciesCount):
        """generates initial occupancy list based on species count"""
        occupancy = OrderedDict()
        for speciesType in speciesCount.keys():
            siteElementTypesIndices = np.in1d(self.material.elementTypes, self.material.speciesToElementTypeMap[speciesType]).nonzero()[0]
            numSpecies = speciesCount[speciesType]
            iSpeciesSystemElementIndices = []
            for iSpecies in range(numSpecies):
                siteElementTypeIndex = rnd.choice(siteElementTypesIndices)
                iSpeciesSiteIndices = np.array([rnd.randint(0, self.systemSize[0]-1), 
                                                rnd.randint(0, self.systemSize[1]-1), 
                                                rnd.randint(0, self.systemSize[2]-1), 
                                                siteElementTypeIndex, 
                                                rnd.randint(0, self.material.nElementsPerUnitCell[siteElementTypeIndex]-1)])
                iSpeciesSystemElementIndex = self.material.generateSystemElementIndex(self.systemSize, iSpeciesSiteIndices)
                if iSpeciesSystemElementIndex in iSpeciesSystemElementIndices:
                    iSpecies -= 1
                else:
                    iSpeciesSystemElementIndices.append(iSpeciesSystemElementIndex)
            occupancy[speciesType] = iSpeciesSystemElementIndices
        return occupancy
    
class system(object):
    """defines the system we are working on
    
    Attributes:
    size: An array (3 x 1) defining the system size in multiple of unit cells
    """
    def __init__(self, material, neighbors, neighborList, occupancy):
        """Return a system object whose size is *size*"""
        self.material = material
        self.neighbors = neighbors
        self.neighborList = neighborList
        self.occupancy = OrderedDict(occupancy)
        
        self.pbc = self.neighbors.pbc
        speciesCount = {key: len(self.occupancy[key]) if key in self.occupancy.keys() else 0 
                        for key in self.material.nonEmptySpeciesToElementTypeMap.keys() 
                        if key is not self.material.emptySpeciesType}
        self.speciesCount = speciesCount
        
        # total number of unit cells
        self.systemSize = self.neighbors.systemSize
        self.numCells = self.systemSize.prod()
        
        # generate lattice charge list
        unitCellChargeList = np.array([self.material.chargeTypes[self.material.elementTypes[elementTypeIndex]] 
                                       for elementTypeIndex in self.material.elementTypeIndexList])
        self.latticeChargeList = np.tile(unitCellChargeList, self.numCells)
        
        # Coefficient Distance List
        self.coeffDistanceList = self.material.KE / (self.material.dielectricConstant * self.neighborList['E'][0].displacementList)
    
    def chargeConfig(self, occupancy):
        """Returns charge distribution of the current configuration"""
        chargeList = self.latticeChargeList
        chargeTypes = self.material.chargeTypes
        chargeTypeKeys = chargeTypes.keys()
        
        shellChargeTypeKeys = [key for key in chargeTypeKeys if self.material.elementTypeDelimiter in key]
        for chargeTypeKey in shellChargeTypeKeys:
            centerSiteElementType = chargeTypeKey.split(self.material.elementTypeDelimiter)[0]
            neighborElementTypeSites = self.neighborList[chargeTypeKey]
            for speciesType in self.material.elementTypeToSpeciesMap[centerSiteElementType]:
                centerSiteSystemElementIndices = self.occupancy[speciesType]
                for centerSiteSystemElementIndex in centerSiteSystemElementIndices:
                    for shellIndex, chargeValue in enumerate(chargeTypes[chargeTypeKey]):
                        neighborOffsetList = neighborElementTypeSites[shellIndex].offsetList
                        neighborElementIndexMap = neighborElementTypeSites[shellIndex].elementIndexMap
                        siteQuantumIndices = self.material.generateQuantumIndices(self.systemSize, centerSiteSystemElementIndex)
                        siteElementIndex = siteQuantumIndices[4]
                        neighborSystemElementIndices = []
                        for neighborIndex in range(len(neighborElementIndexMap[1][siteElementIndex])):
                            neighborUnitCellIndex = [sum(x) for x in zip(siteQuantumIndices[:3], neighborOffsetList[siteElementIndex][neighborIndex])]
                            neighborElementTypeIndex = [self.material.elementTypes.index(chargeTypeKey.split(self.material.elementTypeDelimiter)[1])]
                            neighborElementIndex = [neighborElementIndexMap[1][siteElementIndex][neighborIndex]]
                            neighborQuantumIndices = neighborUnitCellIndex + neighborElementTypeIndex + neighborElementIndex
                            neighborSystemElementIndex = self.material.generateSystemElementIndex(self.systemSize, neighborQuantumIndices)
                            neighborSystemElementIndices.append(neighborSystemElementIndex)
                        chargeList[neighborSystemElementIndices] = chargeValue

        siteChargeTypeKeys = [key for key in chargeTypeKeys if key not in self.material.siteList if self.material.siteIdentifier in key]
        for chargeKeyType in siteChargeTypeKeys:
            centerSiteElementType = chargeKeyType.replace(self.material.siteIdentifier,'')
            for speciesType in self.material.elementTypeToSpeciesMap[centerSiteElementType]:
                assert speciesType in self.occupancy.keys(), ('Invalid definition of charge type \'' + str(chargeKeyType) + '\', \'' + 
                                                              str(speciesType) + '\' species does not exist in this configuration') 
                centerSiteSystemElementIndices = self.occupancy[speciesType]
                chargeList[centerSiteSystemElementIndices] = chargeTypes[chargeKeyType]
        return chargeList

    def ESPConfig(self, currentStateChargeConfig):
        # TODO: Statement that caused memory leak
        #ESPConfig = np.multiply(currentStateChargeConfig[:, np.newaxis], self.coeffDistanceList) # SI units
        ESPConfig = np.copy(self.coeffDistanceList)
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
    def __init__(self, material, system, T, nTraj, kmcSteps, stepInterval, gui):
        """Returns the PBC condition of the system"""
        self.startTime = datetime.now()
        
        self.material = material
        self.system = system
        
        self.T = T
        self.nTraj = int(nTraj)
        self.kmcSteps = int(kmcSteps)
        self.stepInterval = int(stepInterval)
        self.gui = gui
        
        self.systemSize = self.system.systemSize
        
        # import parameters from material class
        self.vn = self.material.vn
        self.lambdaValues = self.material.lambdaValues
        self.VAB = self.material.VAB
        
        # compute number of species existing in the system
        nSpecies = {}
        speciesToElementTypeMap = self.material.speciesToElementTypeMap
        for speciesTypeKey in  speciesToElementTypeMap.keys():
            if speciesTypeKey in self.system.occupancy.keys(): 
                nSpecies[speciesTypeKey] = len(self.system.occupancy[speciesTypeKey])
            elif speciesTypeKey is not self.material.emptySpeciesType:
                nSpecies[speciesTypeKey] = 0
        
        nSpecies[self.material.emptySpeciesType] = (self.material.totalElementsPerUnitCell * self.system.numCells - sum(nSpecies.values()))
        self.nSpecies = nSpecies
        
        # total number of species
        self.totalSpecies = int(sum(self.nSpecies.values()) - self.nSpecies[self.material.emptySpeciesType])
    
        # Electrostatic interaction neighborlist:
        self.elecNeighborListNeighborSEIndices = self.system.neighborList['E'][0].systemElementIndexMap[1]

    def electrostaticInteractionEnergy(self, occupancy):
        """Subroutine to compute the electrostatic interaction energies"""
        chargeConfig = self.system.chargeConfig(occupancy)
        ESPConfig = self.system.ESPConfig(chargeConfig)
        individualInteractionList = (ESPConfig * chargeConfig[self.elecNeighborListNeighborSEIndices])
        elecIntEnergy = np.sum(np.concatenate(individualInteractionList)) * self.material.J2EV # electron-volt
        return elecIntEnergy
        
    def nonESPRelativeElectrostaticInteractionEnergy(self, currentStateChargeConfig, newStateChargeConfig, 
                                                     oldSiteSystemElementIndex, newSiteSystemElementIndex):
        """Subroutine to compute the relative electrostatic interaction energies between two states"""
        #print 'Energies:'
        oldSiteElecIntEnergy = np.sum(currentStateChargeConfig[oldSiteSystemElementIndex] * 
                                      currentStateChargeConfig[self.elecNeighborListNeighborSEIndices[oldSiteSystemElementIndex]] * 
                                      self.system.coeffDistanceList[oldSiteSystemElementIndex])
        #print oldSiteElecIntEnergy
        oldNeighborSiteElecIntEnergy = np.sum(currentStateChargeConfig[newSiteSystemElementIndex] * 
                                              currentStateChargeConfig[self.elecNeighborListNeighborSEIndices[newSiteSystemElementIndex]] * 
                                              self.system.coeffDistanceList[newSiteSystemElementIndex])
        #print oldNeighborSiteElecIntEnergy
        newSiteElecIntEnergy = np.sum(newStateChargeConfig[newSiteSystemElementIndex] * 
                                      newStateChargeConfig[self.elecNeighborListNeighborSEIndices[newSiteSystemElementIndex]] * 
                                      self.system.coeffDistanceList[newSiteSystemElementIndex]) 
        #print newSiteElecIntEnergy
        newNeighborSiteElecIntEnergy = np.sum(newStateChargeConfig[oldSiteSystemElementIndex] * 
                                              newStateChargeConfig[self.elecNeighborListNeighborSEIndices[oldSiteSystemElementIndex]] * 
                                              self.system.coeffDistanceList[oldSiteSystemElementIndex]) 
        #print newNeighborSiteElecIntEnergy
        #TODO: Is absolute necessary?
        relativeElecEnergy = (newSiteElecIntEnergy + newNeighborSiteElecIntEnergy 
                              - oldSiteElecIntEnergy - oldNeighborSiteElecIntEnergy) * self.material.J2EV
        #print relativeElecEnergy
        '''
        print 'Energies:'
        oldSiteCharge2 = (currentStateChargeConfig[oldSiteSystemElementIndex] * 
                          currentStateChargeConfig[self.elecNeighborListNeighborSEIndices[oldSiteSystemElementIndex]])
        oldNeighborSiteCharge2 = (currentStateChargeConfig[newSiteSystemElementIndex] * 
                                  currentStateChargeConfig[self.elecNeighborListNeighborSEIndices[newSiteSystemElementIndex]])
        newSiteCharge2 = (newStateChargeConfig[newSiteSystemElementIndex] * 
                          newStateChargeConfig[self.elecNeighborListNeighborSEIndices[newSiteSystemElementIndex]])
        newNeighborSiteCharge2 = (newStateChargeConfig[oldSiteSystemElementIndex] * 
                                  newStateChargeConfig[self.elecNeighborListNeighborSEIndices[oldSiteSystemElementIndex]])
        relativeOldSiteElecIntEnergy = np.sum((newNeighborSiteCharge2 - oldSiteCharge2) * self.system.coeffDistanceList[oldSiteSystemElementIndex])
        print relativeOldSiteElecIntEnergy
        relativeNewSiteElecIntEnergy = np.sum((newSiteCharge2 - oldNeighborSiteCharge2) * self.system.coeffDistanceList[newSiteSystemElementIndex])
        print relativeNewSiteElecIntEnergy
        #TODO: Is absolute necessary?
        relativeElecEnergy = (relativeOldSiteElecIntEnergy + relativeNewSiteElecIntEnergy) * self.material.J2EV
        print relativeElecEnergy
        '''
        return relativeElecEnergy

    def ESPRelativeElectrostaticInteractionEnergy(self, currentStateESPConfig, newStateESPConfig, 
                                               currentStateChargeConfig, newStateChargeConfig, 
                                               oldSiteSystemElementIndex, newSiteSystemElementIndex):
        """Subroutine to compute the relative electrostatic interaction energies between two states"""
        oldSiteElecIntEnergy = np.sum(currentStateESPConfig[oldSiteSystemElementIndex] * 
                                      currentStateChargeConfig[self.elecNeighborListNeighborSEIndices[oldSiteSystemElementIndex]])
        #print 'ESP:'
        #print oldSiteElecIntEnergy
        oldNeighborSiteElecIntEnergy = np.sum(currentStateESPConfig[newSiteSystemElementIndex] * 
                                              currentStateChargeConfig[self.elecNeighborListNeighborSEIndices[newSiteSystemElementIndex]])
        #print oldNeighborSiteElecIntEnergy
        newSiteElecIntEnergy = np.sum(newStateESPConfig[newSiteSystemElementIndex] * 
                                      newStateChargeConfig[self.elecNeighborListNeighborSEIndices[newSiteSystemElementIndex]])
        #print newSiteElecIntEnergy
        newNeighborSiteElecIntEnergy = np.sum(newStateESPConfig[oldSiteSystemElementIndex] * 
                                              newStateChargeConfig[self.elecNeighborListNeighborSEIndices[oldSiteSystemElementIndex]])
        #print newNeighborSiteElecIntEnergy
        #TODO: Is absolute necessary?
        relativeElecEnergy = abs(newSiteElecIntEnergy + newNeighborSiteElecIntEnergy - 
                                 oldSiteElecIntEnergy - oldNeighborSiteElecIntEnergy) * self.material.J2EV # electron-volt
        #print relativeElecEnergy
        return relativeElecEnergy

    def generateNewStates(self, currentStateOccupancy):
        """generates a list of new occupancy states possible from the current state"""
        neighborList = self.system.neighborList
        newStateOccupancyList = []
        hopElementTypes = []
        hopDistTypes = []
        hoppingSpeciesIndices = []
        speciesDisplacementVectorList = []
        systemElementIndexPairList = []
        newStateOccupancy = deepcopy(currentStateOccupancy)
                                
        cumulativeSpeciesSiteSystemElementIndices = [systemElementIndex for speciesSiteSystemElementIndices in currentStateOccupancy.values() 
                                                 for systemElementIndex in speciesSiteSystemElementIndices]
        for speciesType in currentStateOccupancy.keys():
            for speciesTypeSpeciesIndex, speciesSiteSystemElementIndex in enumerate(currentStateOccupancy[speciesType]):
                speciesIndex = cumulativeSpeciesSiteSystemElementIndices.index(speciesSiteSystemElementIndex)
                for hopElementType in self.material.hopElementTypes[speciesType]:
                    for hopDistTypeIndex in range(len(self.material.neighborCutoffDist[hopElementType])):
                        neighborOffsetList = neighborList[hopElementType][hopDistTypeIndex].offsetList
                        neighborElementIndexMap = neighborList[hopElementType][hopDistTypeIndex].elementIndexMap
                        speciesSiteToNeighborDisplacementVectorList = neighborList[hopElementType][hopDistTypeIndex].displacementVectorList
                        
                        speciesQuantumIndices = self.material.generateQuantumIndices(self.systemSize, speciesSiteSystemElementIndex)
                        speciesSiteElementIndex = speciesQuantumIndices[4]
                        numNeighbors = len(neighborElementIndexMap[1][speciesSiteElementIndex])
                        for neighborIndex in range(numNeighbors):
                            neighborUnitCellIndices = [sum(x) for x in zip(speciesQuantumIndices[:3], neighborOffsetList[speciesSiteElementIndex][neighborIndex])]
                            # TODO: Try to avoid the loop and conditions
                            for index, neighborUnitCellIndex in enumerate(neighborUnitCellIndices):
                                if neighborUnitCellIndex > self.systemSize[index] - 1:
                                    neighborUnitCellIndices[index] -= self.systemSize[index]
                                elif neighborUnitCellIndex < 0:
                                    neighborUnitCellIndices[index] += self.systemSize[index]
                            neighborElementTypeIndex = [self.material.elementTypes.index(hopElementType.split(self.material.elementTypeDelimiter)[1])]
                            neighborElementIndex = [neighborElementIndexMap[1][speciesSiteElementIndex][neighborIndex]]
                            neighborQuantumIndices = neighborUnitCellIndices + neighborElementTypeIndex + neighborElementIndex
                            neighborSystemElementIndex = self.material.generateSystemElementIndex(self.systemSize, neighborQuantumIndices)
                            if neighborSystemElementIndex not in cumulativeSpeciesSiteSystemElementIndices:
                                newStateOccupancy[speciesType][speciesTypeSpeciesIndex] = neighborSystemElementIndex
                                newStateOccupancyList.append(deepcopy(newStateOccupancy))
                                newStateOccupancy[speciesType][speciesTypeSpeciesIndex] = speciesSiteSystemElementIndex
                                hopElementTypes.append(hopElementType)
                                hopDistTypes.append(hopDistTypeIndex)
                                hoppingSpeciesIndices.append(speciesIndex)
                                speciesDisplacementVector = speciesSiteToNeighborDisplacementVectorList[speciesSiteElementIndex][neighborIndex]
                                speciesDisplacementVectorList.append(speciesDisplacementVector)
                                systemElementIndexPairList.append([speciesSiteSystemElementIndex, neighborSystemElementIndex])
                    
        returnNewStates = returnValues()
        returnNewStates.newStateOccupancyList = newStateOccupancyList
        returnNewStates.hopElementTypes = hopElementTypes
        returnNewStates.hopDistTypes = hopDistTypes
        returnNewStates.hoppingSpeciesIndices = hoppingSpeciesIndices
        returnNewStates.speciesDisplacementVectorList = speciesDisplacementVectorList
        returnNewStates.systemElementIndexPairList = systemElementIndexPairList
        return returnNewStates
    
    #@profile
    def doKMCSteps(self, outdir=None, ESPConfig=1, report=1, randomSeed=1):
        """Subroutine to run the KMC simulation by specified number of steps"""
        import pdb; pdb.set_trace()
        rnd.seed(randomSeed)
        nTraj = self.nTraj
        kmcSteps = self.kmcSteps
        stepInterval = self.stepInterval
        currentStateOccupancy = self.system.occupancy
        numPathStepsPerTraj = int(kmcSteps / stepInterval) + 1
        timeArray = np.zeros(nTraj * numPathStepsPerTraj)
        unwrappedPositionArray = np.zeros(( nTraj * numPathStepsPerTraj, self.totalSpecies, 3))
        wrappedPositionArray = np.zeros(( nTraj * numPathStepsPerTraj, self.totalSpecies, 3))
        # TODO: speciesDisplacementArray = [speciesIndex, displacement]
        speciesDisplacementArray = np.zeros(( nTraj * numPathStepsPerTraj, self.totalSpecies, 3))
        pathIndex = 0
        currentStateConfig = self.system.config(currentStateOccupancy)
        currentStateChargeConfig = self.system.chargeConfig(currentStateOccupancy)
        newStateChargeConfig = np.copy(currentStateChargeConfig)
        if ESPConfig:
            currentStateESPConfig = self.system.ESPConfig(currentStateChargeConfig)
            newStateESPConfig = np.copy(currentStateESPConfig)
        shellChargeTypeKeys = [key for key in self.material.chargeTypes.keys() if self.material.elementTypeDelimiter in key]
        shellCharges = 0 if shellChargeTypeKeys==[] else 1
        assert 'E' in self.material.neighborCutoffDist.keys(), 'Please specify the cutoff distance for electrostatic interactions'
        for dummy in range(nTraj):
            pathIndex += 1
            kmcTime = 0
            speciesDisplacementVectorList = np.zeros((self.totalSpecies, 3))
            for step in range(kmcSteps):
                kList = []
                newStates = self.generateNewStates(currentStateOccupancy)
                hopElementTypes = newStates.hopElementTypes
                hopDistTypes = newStates.hopDistTypes
                for newStateIndex in range(len(newStates.hoppingSpeciesIndices)):
                    [oldSiteSystemElementIndex, newSiteSystemElementIndex] = newStates.systemElementIndexPairList[newStateIndex]
                    if shellCharges:
                        newStateOccupancy = newStates.newStateOccupancyList[newStateIndex]
                        newStateChargeConfig = self.system.chargeConfig(newStateOccupancy)
                        newStateESPConfig = self.system.ESPConfig(newStateChargeConfig)
                    else:
                        if ESPConfig:
                            # TODO: Is it necessary to compute multFactor every time?
                            multFactor = np.true_divide(currentStateChargeConfig[[newSiteSystemElementIndex, oldSiteSystemElementIndex]], 
                                                        currentStateChargeConfig[[oldSiteSystemElementIndex, newSiteSystemElementIndex]])
                            # TODO: Statement that didn't work with larger cutoffs
                            #newStateESPConfig[[oldSiteSystemElementIndex, newSiteSystemElementIndex]] *= multFactor[:, np.newaxis]
                            newStateESPConfig[oldSiteSystemElementIndex] *= multFactor[0]
                            newStateESPConfig[newSiteSystemElementIndex] *= multFactor[1]
                        newStateChargeConfig[[oldSiteSystemElementIndex, newSiteSystemElementIndex]] = newStateChargeConfig[[newSiteSystemElementIndex, oldSiteSystemElementIndex]]
                        
                    if ESPConfig:
                        delG0 = self.ESPRelativeElectrostaticInteractionEnergy(currentStateESPConfig, newStateESPConfig, 
                                                                               currentStateChargeConfig, newStateChargeConfig, 
                                                                               oldSiteSystemElementIndex, newSiteSystemElementIndex)
                    else:
                        delG0 = self.nonESPRelativeElectrostaticInteractionEnergy(currentStateChargeConfig, newStateChargeConfig, 
                                                                                  oldSiteSystemElementIndex, newSiteSystemElementIndex)

                    hopElementType = hopElementTypes[newStateIndex]
                    hopDistType = hopDistTypes[newStateIndex]
                    lambdaValue = self.lambdaValues[hopElementType][hopDistType]
                    VAB = self.VAB[hopElementType][hopDistType]
                    delGs = ((lambdaValue + delG0) ** 2 / (4 * lambdaValue)) - VAB
                    kList.append(self.vn * np.exp(-delGs / (self.material.KB * self.T)))
                    if not shellCharges:
                        newStateChargeConfig[[oldSiteSystemElementIndex, newSiteSystemElementIndex]] = newStateChargeConfig[[newSiteSystemElementIndex, oldSiteSystemElementIndex]]
                        if ESPConfig:
                            newStateESPConfig[[oldSiteSystemElementIndex, newSiteSystemElementIndex]] = currentStateESPConfig[[oldSiteSystemElementIndex, newSiteSystemElementIndex]]
                        
                kTotal = np.sum(kList)
                kCumSum = (kList / kTotal).cumsum()
                rand1 = rnd.random()
                procIndex = np.where(kCumSum > rand1)[0][0]
                rand2 = rnd.random()
                kmcTime -= np.log(rand2) / kTotal
                currentStateOccupancy = newStates.newStateOccupancyList[procIndex]
                if shellCharges:
                    currentStateChargeConfig = self.system.chargeConfig(currentStateOccupancy)
                    newStateChargeConfig = self.system.chargeConfig(currentStateOccupancy)
                    currentStateESPConfig = self.system.ESPConfig(currentStateChargeConfig)
                    newStateESPConfig = self.system.ESPConfig(currentStateChargeConfig)
                else:
                    [oldSiteSystemElementIndex, newSiteSystemElementIndex] = newStates.systemElementIndexPairList[procIndex]
                    if ESPConfig:
                        # TODO: Is it necessary to compute multFactor every time?
                        multFactor = np.true_divide(currentStateChargeConfig[[newSiteSystemElementIndex, oldSiteSystemElementIndex]], 
                                                    currentStateChargeConfig[[oldSiteSystemElementIndex, newSiteSystemElementIndex]])
                        # TODO: Statement that didn't work with larger cutoffs
                        #currentStateESPConfig[[oldSiteSystemElementIndex, newSiteSystemElementIndex]] *= multFactor[:, np.newaxis]
                        currentStateESPConfig[oldSiteSystemElementIndex] *= multFactor[0]
                        currentStateESPConfig[newSiteSystemElementIndex] *= multFactor[1]
                        newStateESPConfig[[oldSiteSystemElementIndex, newSiteSystemElementIndex]] = currentStateESPConfig[[oldSiteSystemElementIndex, newSiteSystemElementIndex]]
                    currentStateChargeConfig[[oldSiteSystemElementIndex, newSiteSystemElementIndex]] = currentStateChargeConfig[[newSiteSystemElementIndex, oldSiteSystemElementIndex]]
                    newStateChargeConfig[[oldSiteSystemElementIndex, newSiteSystemElementIndex]] = newStateChargeConfig[[newSiteSystemElementIndex, oldSiteSystemElementIndex]]
                    
                # speciesIndex is different from speciesTypeSpeciesIndex
                speciesIndex = newStates.hoppingSpeciesIndices[procIndex]
                speciesDisplacementVector = newStates.speciesDisplacementVectorList[procIndex]
                speciesDisplacementVectorList[speciesIndex] += speciesDisplacementVector
                # TODO: May not be necessary
                currentStateConfig.chargeList = currentStateChargeConfig
                currentStateConfig.occupancy = currentStateOccupancy
                if step % stepInterval == 0:
                    speciesSystemElementIndices = np.concatenate((currentStateOccupancy.values())).astype(int)
                    timeArray[pathIndex] = kmcTime
                    wrappedPositionArray[pathIndex] = currentStateConfig.positions[speciesSystemElementIndices]
                    speciesDisplacementArray[pathIndex] = speciesDisplacementVectorList
                    unwrappedPositionArray[pathIndex] = unwrappedPositionArray[pathIndex - 1] + speciesDisplacementVectorList
                    speciesDisplacementVectorList = np.zeros((self.totalSpecies, 3))
                    pathIndex += 1
        
        trajectoryData = returnValues()
        trajectoryData.speciesCount = self.system.speciesCount
        trajectoryData.nTraj = nTraj
        trajectoryData.kmcSteps = kmcSteps
        trajectoryData.stepInterval = stepInterval
        trajectoryData.pbc = self.system.pbc
        trajectoryData.systemSize = self.systemSize
        trajectoryData.timeArray = timeArray
        trajectoryData.unwrappedPositionArray = unwrappedPositionArray
        trajectoryData.wrappedPositionArray = wrappedPositionArray
        trajectoryData.speciesDisplacementArray = speciesDisplacementArray
        trajectoryData.EcutoffDist = self.material.neighborCutoffDist['E'][0]
        
        if outdir:
            trajectoryDataFileName = 'TrajectoryData.npy'
            trajectoryDataFilePath = outdir + '/' + trajectoryDataFileName
            np.save(trajectoryDataFilePath, trajectoryData)
        if report:
            self.generateSimulationLogReport(outdir)
        
        return trajectoryData

    def generateSimulationLogReport(self, outdir):
        """Generates an log report of the simulation and outputs to the working directory"""
        simulationLogFileName = 'Run.log'
        simulationLogFilePath = outdir + '/' + simulationLogFileName
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
    def __init__(self, trajectoryData, nStepsMSD, nDispMSD, binsize, reprTime = 'ns', 
                 reprDist = 'Angstrom'):
        '''
        
        '''
        self.startTime = datetime.now()
        
        self.trajectoryData = trajectoryData

        self.nStepsMSD = int(nStepsMSD)
        self.nDispMSD = int(nDispMSD)
        self.binsize = binsize
        self.reprTime = reprTime
        self.reprDist = reprDist
        self.systemSize = self.trajectoryData.systemSize
        self.pbc = self.trajectoryData.pbc
        
        self.timeConversion = 1E+09 if reprTime is 'ns' else 1E+00 
        self.distConversion = 1E-10 if reprDist is 'm' else 1E+00
        
        self.nTraj = self.trajectoryData.nTraj
        self.kmcSteps = self.trajectoryData.kmcSteps
        self.stepInterval = self.trajectoryData.stepInterval
        self.EcutoffDist = self.trajectoryData.EcutoffDist

    def computeMSD(self, timeArray, unwrappedPositionArray, outdir=None, report=1):
        """Returns the squared displacement of the trajectories"""
        time = timeArray * self.timeConversion
        positionArray = unwrappedPositionArray * self.distConversion
        speciesCount = self.trajectoryData.speciesCount
        nSpecies = sum(speciesCount.values())
        speciesTypes = []
        for speciesType in speciesCount.keys():
            if speciesCount[speciesType] is not 0:
                speciesTypes.append(speciesType)
        nSpeciesTypes = len(speciesTypes)
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
        msdData = np.zeros((nBins+1, nSpeciesTypes + 1))
        msdData[1:, 0] = bins[:-1] + 0.5 * self.binsize
        startIndex = 0
        for speciesTypeIndex in range(nSpeciesTypes):
            endIndex = startIndex + speciesCount[speciesTypes[speciesTypeIndex]]
            msdData[1:, speciesTypeIndex + 1] = np.mean(speciesMSDData[:, startIndex:endIndex], axis=1)
            startIndex = endIndex
        
        if outdir:
            fileName = (('%1.2E' % self.nStepsMSD) + 'nStepsMSD,' + 
                        ('%1.2E' % self.nDispMSD) + 'nDispMSD')

            msdFileName = 'MSD_Data_' + fileName + '.npy'
            msdFilePath = outdir + '/' + msdFileName
            np.save(msdFilePath, msdData)
        if report:
            self.generateMSDAnalysisLogReport(outdir, fileName)
        returnMSDData = returnValues()
        returnMSDData.msdData = msdData
        returnMSDData.speciesTypes = speciesTypes
        returnMSDData.fileName = fileName
        return returnMSDData
    
    def generateMSDAnalysisLogReport(self, outdir, fileName):
        """Generates an log report of the MSD Analysis and outputs to the working directory"""
        msdAnalysisLogFileName = 'MSD_Analysis_' + fileName + '.log'
        msdLogFilePath = outdir + '/' + msdAnalysisLogFileName
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
        if outdir:
            figureName = 'MSD_Plot_' + fileName + '.jpg'
            figurePath = outdir + '/' + figureName
            plt.savefig(figurePath)
        
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
