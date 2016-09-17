#!/usr/bin/env python
'''
code to compute msd of random walk of single electron in 3D hematite lattice structure
 switch off PBC, and 
'''
import numpy as np
from collections import OrderedDict
from copy import deepcopy
        
class material(object):
    '''
    defines the structure of working material
    
    Attributes:
        name: A string representing the material name
        elements: list of element symbols
        species_to_sites: dictionary that maps species to sites
        positions: positions of elements in the unit cell
        index: element index of the positions starting from 0
        charge: atomic charges of the elements # first shell atomic charges to be included
        latticeParameters: list of three lattice constants in angstrom and three angles between them in degrees
    '''
    
    def __init__(self, name, elementTypes, speciesTypes, unitcellCoords, elementTypeIndexList, chargeTypes, 
                 latticeParameters, vn, lambdaValues, VAB, neighborCutoffDist, neighborCutoffDistTol, 
                 elementTypeDelimiter, epsilon0):
        '''
        Return an material object whose name is *name* 
        '''
        # TODO: introduce a method to view the material using ase atoms or other gui module
        self.name = name
        self.elementTypes = elementTypes
        self.speciesTypes = speciesTypes
        self.unitcellCoords = np.zeros((len(unitcellCoords), 3))
        startIndex = 0
        length = len(self.elementTypes)
        nElements = np.zeros(length, int)
        for elementIndex in range(length):
            elementUnitCellCoords = unitcellCoords[elementTypeIndexList==elementIndex]
            nElements[elementIndex] = len(elementUnitCellCoords)
            endIndex = startIndex + nElements[elementIndex]
            self.unitcellCoords[startIndex:endIndex] = elementUnitCellCoords[elementUnitCellCoords[:,2].argsort()]
            startIndex = endIndex
        
        # number of elements
        self.nElements = nElements
        self.elementTypeIndexList = elementTypeIndexList.astype(int)
        self.chargeTypes = chargeTypes
        self.latticeParameters = latticeParameters
        self.vn = vn
        self.lambdaValues = lambdaValues
        self.VAB = VAB
        self.neighborCutoffDist = neighborCutoffDist
        self.neighborCutoffDistTol = neighborCutoffDistTol
        self.elementTypeDelimiter = elementTypeDelimiter
        self.epsilon0 = epsilon0
                
        # siteList
        siteList = [self.speciesTypes[key] for key in self.speciesTypes if key is not 'empty']
        self.siteList = siteList
        
        # list of hop element types
        hopElementTypes = {key: self.speciesTypes[key] + self.elementTypeDelimiter + 
                           self.speciesTypes[key] for key in self.speciesTypes 
                           if key is not 'empty'}
        self.hopElementTypes = hopElementTypes
        
        # number of sites present in siteList
        siteElementIndices = [i for i,n in enumerate(elementTypes) if n in siteList]
        nSites = [nElements[i] for i in siteElementIndices]
        self.nSites = np.asarray(nSites, int)
        
        # element - species map
        elementTypeSpeciesMap = {}
        nonEmptySpeciesTypes = speciesTypes.copy()
        del nonEmptySpeciesTypes['empty']
        self.nonEmptySpeciesTypes = nonEmptySpeciesTypes
        for elementType in self.elementTypes:
            speciesList = []
            for speciesTypeKey in nonEmptySpeciesTypes.keys():
                if elementType in nonEmptySpeciesTypes[speciesTypeKey]:
                    speciesList.append(speciesTypeKey)
            elementTypeSpeciesMap[elementType] = speciesList
        self.elementTypeSpeciesMap = elementTypeSpeciesMap

        # lattice cell matrix
        [a, b, c, alpha, beta, gamma] = self.latticeParameters
        latticeMatrix = np.array([[ a                , 0                , 0],
                                  [ b * np.cos(gamma), b * np.sin(gamma), 0],
                                  [ c * np.cos(alpha), c * np.cos(beta) , c * 
                                   np.sqrt(np.sin(alpha)**2 - np.cos(beta)**2)]])
        self.latticeMatrix = latticeMatrix

    def generateSites(self, elementTypeIndices, cellSize=np.array([1, 1, 1])):
        '''
        Returns systemElementIndices and coordinates of specified elements in a cell of size *cellSize*
        '''
        assert all(size > 0 for size in cellSize), 'Input size should always be greater than 0'
        extractIndices = np.in1d(self.elementTypeIndexList, elementTypeIndices).nonzero()[0]
        unitcellElementCoords = self.unitcellCoords[extractIndices]
        numCells = np.prod(cellSize)
        nSitesPerUnitCell = np.sum(self.nElements[elementTypeIndices])
        unitcellElementIndexList = np.arange(nSitesPerUnitCell)
        unitcellElementTypeIndex = np.reshape(np.concatenate((np.asarray([[elementTypeIndex] * self.nElements[elementTypeIndex] 
                                                                          for elementTypeIndex in elementTypeIndices]))), (nSitesPerUnitCell, 1))
        unitCellElementTypeElementIndexList = np.reshape(np.concatenate(([np.arange(self.nElements[elementTypeIndex]) 
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
        '''
        Returns the systemElementIndex of the element
        '''
        assert 0 not in systemSize, 'System size should be greater than 0 in any dimension'
        assert quantumIndices[-1] < self.nElements[quantumIndices[-2]], 'Element Index exceed number of elements of the specified element type'
        assert all(quantumIndex >= 0 for quantumIndex in quantumIndices), 'Quantum Indices cannot be negative'
        unitCellIndex = quantumIndices[:3]
        [elementTypeIndex, elementIndex] = quantumIndices[-2:]
        nElementsPerUnitCell = np.sum(self.nElements)
        systemElementIndex = elementIndex + np.sum(self.nElements[:elementTypeIndex])
        nDim = len(systemSize)
        for index in range(nDim):
            if index == 0:
                systemElementIndex += nElementsPerUnitCell * unitCellIndex[nDim-1-index]
            else:
                systemElementIndex += nElementsPerUnitCell * unitCellIndex[nDim-1-index] * np.prod(systemSize[-index:])
        return systemElementIndex
    
    def generateQuantumIndices(self, systemSize, systemElementIndex):
        '''
        Returns the quantum indices of the element
        '''
        assert systemElementIndex >= 0, 'System Element Index cannot be negative'
        quantumIndices = [0] * 5
        nElementsPerUnitCell = np.sum(self.nElements)
        nElementsPerUnitCellCumSum = np.cumsum(self.nElements)
        unitcellElementIndex = systemElementIndex % nElementsPerUnitCell
        quantumIndices[3] = np.min(np.where(nElementsPerUnitCellCumSum >= (unitcellElementIndex + 1)))
        quantumIndices[4] = unitcellElementIndex - np.sum(self.nElements[:quantumIndices[3]])
        nFilledUnitCells = (systemElementIndex - unitcellElementIndex) / nElementsPerUnitCell
        for index in range(3):
            quantumIndices[index] = nFilledUnitCells / np.prod(systemSize[index+1:])
            nFilledUnitCells -= quantumIndices[index] * np.prod(systemSize[index+1:])
        return quantumIndices
    

class neighbors(object):
    '''
    Returns the neighbor list file
    '''
    
    def __init__(self, material, systemSize=np.array([10, 10, 10]), pbc=[1, 1, 1]):
        '''
        
        '''
        self.material = material
        self.systemSize = systemSize
        self.pbc = pbc
        
        # total number of unit cells
        self.numCells = np.prod(self.systemSize)
        
        # generate all sites in the system
        elementTypeIndices = range(len(self.material.elementTypes))
        bulkSites = self.material.generateSites(elementTypeIndices, self.systemSize)
        self.bulkSites = bulkSites
        
    def hopNeighborSites(self, bulkSites, centerSiteIndices, neighborSiteIndices, cutoffDistLimits, cutoffDistKey):
        '''
        Returns systemElementIndexMap and distances between center sites and its neighbor sites within cutoff 
        distance
        '''
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
        '''
        Returns systemElementIndexMap and distances between center sites and its neighbor sites within cutoff 
        distance
        '''
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
            #print centerSiteIndex, cutoffDistKey, len(iDisplacements)#, sorted(iDisplacements)
            neighborSystemElementIndices.append(np.array(neighborSiteSystemElementIndexList[iNeighborSiteIndexList]))
            offsetList.append(centerSiteQuantumIndexList[centerSiteIndex, :3] - neighborSiteQuantumIndexList[iNeighborSiteIndexList, :3])
            neighborElementTypeIndexList.append(neighborSiteQuantumIndexList[iNeighborSiteIndexList, 3])
            neighborElementIndexList.append(neighborSiteQuantumIndexList[iNeighborSiteIndexList, 4])
            displacementVectorList.append(np.asarray(iDisplacementVectors))
            displacementList.append(iDisplacements)
            
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

    def generateNeighborList(self, localSystemSize = np.array([3, 3, 3]), 
                             centerUnitCellIndex = np.array([1, 1, 1])):
        '''
        Adds the neighbor list to the system object and returns the neighbor list
        '''
        assert all(size >= 3 for size in localSystemSize), 'Local system size in all dimensions should always be greater than or equal to 3'
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
                                     for elementIndex in range(self.material.nElements[centerSiteElementTypeIndex])]
                neighborSiteIndices = [self.material.generateSystemElementIndex(localSystemSize, np.array([xSize, ySize, zSize, neighborSiteElementTypeIndex, elementIndex])) 
                                       for xSize in range(localSystemSize[0]) for ySize in range(localSystemSize[1]) 
                                       for zSize in range(localSystemSize[2]) 
                                       for elementIndex in range(self.material.nElements[neighborSiteElementTypeIndex])]

                for cutoffDist in cutoffDistList:
                    cutoffDistLimits = [cutoffDist-tolDist, cutoffDist+tolDist]
                    neighborListCutoffDistKey.append(self.hopNeighborSites(localBulkSites, centerSiteIndices, 
                                                                           neighborSiteIndices, cutoffDistLimits, cutoffDistKey))
            else:
                centerSiteIndices = neighborSiteIndices = np.arange(self.numCells * np.sum(self.material.nElements))
                cutoffDistLimits = [0, cutoffDistList[0]]
                neighborListCutoffDistKey.append(self.electrostaticNeighborSites(self.systemSize, self.bulkSites, centerSiteIndices, 
                                                                                 neighborSiteIndices, cutoffDistLimits, cutoffDistKey))
            neighborList[cutoffDistKey] = neighborListCutoffDistKey
        return neighborList
    
class system(object):
    '''
    defines the system we are working on
    
    Attributes:
    size: An array (3 x 1) defining the system size in multiple of unit cells
    '''
    
    def __init__(self, material, neighbors, neighborList, occupancy):
        '''
        Return a system object whose size is *size*
        '''
        self.material = material
        self.neighbors = neighbors
        self.neighborList = neighborList
        self.occupancy = OrderedDict(occupancy)
        
        speciesCount = {key: len(self.occupancy[key]) if key in self.occupancy.keys() else 0 
                        for key in self.material.nonEmptySpeciesTypes.keys() 
                        if key is not 'empty'}
        self.speciesCount = speciesCount
        
        # total number of unit cells
        self.systemSize = self.neighbors.systemSize
        self.numCells = np.prod(self.systemSize)
        
        # generate lattice charge list
        unitCellChargeList = np.array([self.material.chargeTypes[self.material.elementTypes[elementTypeIndex]] 
                                       for elementTypeIndex in self.material.elementTypeIndexList])
        self.latticeChargeList = np.tile(unitCellChargeList, self.numCells)
        self.latticeParameters = self.material.latticeParameters
        
    
    def chargeConfig(self, occupancy):
        '''
        Returns charge distribution of the current configuration
        '''
        chargeList = self.latticeChargeList
        chargeTypes = self.material.chargeTypes
        chargeTypeKeys = chargeTypes.keys()
        
        shellChargeTypeKeys = [key for key in chargeTypeKeys if self.material.elementTypeDelimiter in key]
        for chargeTypeKey in shellChargeTypeKeys:
            centerSiteElementType = chargeTypeKey.split(self.material.elementTypeDelimiter)[0]
            neighborElementTypeSites = self.neighborList[chargeTypeKey]
            for speciesType in self.material.elementTypeSpeciesMap[centerSiteElementType]:
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

        siteChargeTypeKeys = [key for key in chargeTypeKeys if key not in self.material.siteList if '0' in key]
        for chargeKeyType in siteChargeTypeKeys:
            centerSiteElementType = chargeKeyType.replace('0','')
            for speciesType in self.material.elementTypeSpeciesMap[centerSiteElementType]:
                assert speciesType in self.occupancy.keys(), ('Invalid definition of charge type \'' + str(chargeKeyType) + '\', \'' + 
                                                              str(speciesType) + '\' species does not exist in this configuration') 
                centerSiteSystemElementIndices = self.occupancy[speciesType]
                chargeList[centerSiteSystemElementIndices] = chargeTypes[chargeKeyType]

        return chargeList

    def config(self, occupancy):
        '''
        Generates the configuration array for the system 
        '''
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
    '''
    defines the subroutines for running Kinetic Monte Carlo and computing electrostatic interaction energies 
    '''
    def __init__(self, material, system, T, nTraj, kmcSteps, stepInterval, gui, kB):
        '''
        Returns the PBC condition of the system
        '''
        self.material = material
        self.system = system
        
        self.T = T
        self.nTraj = int(nTraj)
        self.kmcSteps = int(kmcSteps)
        self.stepInterval = int(stepInterval)
        self.gui = gui
        self.kB = kB
        
        self.systemSize = self.system.systemSize
        
        # import parameters from material class
        self.vn = self.material.vn
        self.lambdaValues = self.material.lambdaValues
        self.VAB = self.material.VAB
        
        # compute number of species existing in the system
        nSpecies = {}
        speciesTypes = self.material.speciesTypes
        for speciesTypeKey in  speciesTypes.keys():
            if speciesTypeKey in self.system.occupancy.keys(): 
                nSpecies[speciesTypeKey] = len(self.system.occupancy[speciesTypeKey])
            elif speciesTypeKey is not 'empty':
                nSpecies[speciesTypeKey] = 0
            
        nSpecies['empty'] = (np.sum(self.material.nElements) * self.system.numCells - np.sum(nSpecies.values()))
        self.nSpecies = nSpecies
        
        # total number of species
        self.totalSpecies = np.sum(self.nSpecies.values()) - self.nSpecies['empty']
    
    def generateDistanceList(self):
        '''
        
        '''
        # Electrostatic interaction neighborlist:
        elecNeighborListSystemElementIndexMap = self.system.neighborList['E'][0].systemElementIndexMap
        self.elecNeighborListSystemElementIndexMap = elecNeighborListSystemElementIndexMap
        
        # Distance List
        distanceList = self.system.neighborList['E'][0].displacementList
        self.distanceList = distanceList
        self.coeffDistanceList = (1/(4 * np.pi * self.material.epsilon0)) * self.distanceList

    def elec(self, occupancy):
        '''
        Subroutine to compute the electrostatic interaction energies
        '''
        configChargeList = self.system.chargeConfig(occupancy)
        elecNeighborCharge2List = deepcopy(self.elecNeighborListSystemElementIndexMap[1])
        for index, centerElementCharge in enumerate(configChargeList):
            elecNeighborCharge2List[index] = centerElementCharge * configChargeList[self.elecNeighborListSystemElementIndexMap[1][index]] 
        individualInteractionList = np.multiply(elecNeighborCharge2List, self.coeffDistanceList)
        elec = np.sum(np.concatenate(individualInteractionList))
        return elec
        
    def delG0(self, positions, currentStateOccupancy, newStateOccupancy):
        '''
        Subroutine to compute the difference in free energies between initial and final states of the system
        '''
        delG0 = self.elec(newStateOccupancy) - self.elec(currentStateOccupancy)
        return delG0
    
    def generateNewStates(self, currentStateOccupancy):
        '''
        generates a list of new occupancy states possible from the current state
        '''
        neighborList = self.system.neighborList
        newStateOccupancyList = []
        hopElementTypes = []
        hopDistTypes = []
        hoppingSpeciesIndices = []
        speciesDisplacementVectorList = []
        
        cumulativeSpeciesSiteSystemElementIndices = [systemElementIndex for speciesSiteSystemElementIndices in currentStateOccupancy.values() 
                                                 for systemElementIndex in speciesSiteSystemElementIndices]
        for speciesType in currentStateOccupancy.keys():
            for speciesTypeSpeciesIndex, speciesSiteSystemElementIndex in enumerate(currentStateOccupancy[speciesType]):
                speciesIndex = cumulativeSpeciesSiteSystemElementIndices.index(speciesSiteSystemElementIndex)
                hopElementType = self.material.hopElementTypes[speciesType]
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
                            newStateOccupancy = deepcopy(currentStateOccupancy)
                            newStateOccupancy[speciesType][speciesTypeSpeciesIndex] = neighborSystemElementIndex
                            newStateOccupancyList.append(newStateOccupancy)
                            hopElementTypes.append(hopElementType)
                            hopDistTypes.append(hopDistTypeIndex)
                            hoppingSpeciesIndices.append(speciesIndex)
                            speciesDisplacementVector = speciesSiteToNeighborDisplacementVectorList[speciesSiteElementIndex][neighborIndex]
                            speciesDisplacementVectorList.append(speciesDisplacementVector)
                    
        returnNewStates = returnValues()
        returnNewStates.newStateOccupancyList = newStateOccupancyList
        returnNewStates.hopElementTypes = hopElementTypes
        returnNewStates.hopDistTypes = hopDistTypes
        returnNewStates.hoppingSpeciesIndices = hoppingSpeciesIndices
        returnNewStates.speciesDisplacementVectorList = speciesDisplacementVectorList
        return returnNewStates

    def doKMCSteps(self, randomSeed = 1):
        '''
        Subroutine to run the kmc simulation by specified number of steps
        '''
        import random as rnd
        rnd.seed(randomSeed)
        nTraj = self.nTraj
        kmcSteps = self.kmcSteps
        stepInterval = self.stepInterval
        currentStateOccupancy = self.system.occupancy
                
        numPathStepsPerTraj = np.floor(kmcSteps / stepInterval) + 1
        timeArray = np.zeros(nTraj * numPathStepsPerTraj)
        unwrappedPositionArray = np.zeros(( nTraj * numPathStepsPerTraj, self.totalSpecies, 3))
        wrappedPositionArray = np.zeros(( nTraj * numPathStepsPerTraj, self.totalSpecies, 3))
        speciesDisplacementArray = np.zeros(( nTraj * numPathStepsPerTraj, self.totalSpecies, 3))
        pathIndex = 0
        speciesSystemElementIndices = np.concatenate((currentStateOccupancy.values()))
        config = self.system.config(currentStateOccupancy)
        assert 'E' in self.material.neighborCutoffDist.keys(), 'Please specify the cutoff distance for electrostatic interactions'
        self.generateDistanceList()
        for dummy in range(nTraj):
            pathIndex += 1
            kmcTime = 0
            speciesDisplacementVectorList = np.zeros((self.totalSpecies, 3))
            for step in range(kmcSteps):
                kList = []
                newStates = self.generateNewStates(currentStateOccupancy)
                hopElementTypes = newStates.hopElementTypes
                hopDistTypes = newStates.hopDistTypes
                for newStateIndex, newStateOccupancy in enumerate(newStates.newStateOccupancyList):
                    delG0 = self.delG0(config, currentStateOccupancy, newStateOccupancy)
                    hopElementType = hopElementTypes[newStateIndex]
                    hopDistType = hopDistTypes[newStateIndex]
                    lambdaValue = self.lambdaValues[hopElementType][hopDistType]
                    VAB = self.VAB[hopElementType][hopDistType]
                    delGs = ((lambdaValue + delG0) ** 2 / (4 * lambdaValue)) - VAB
                    kList.append(self.vn * np.exp(-delGs / (self.kB * self.T)))
                kTotal = np.sum(kList)
                kCumSum = np.cumsum(kList / kTotal)
                rand1 = rnd.random()
                procIndex = np.min(np.where(kCumSum > rand1))
                rand2 = rnd.random()
                kmcTime -= np.log(rand2) / kTotal
                currentStateOccupancy = newStates.newStateOccupancyList[procIndex]
                # species Index is different from speciesTypeSpeciesIndex
                speciesIndex = newStates.hoppingSpeciesIndices[procIndex]
                speciesDisplacementVector = newStates.speciesDisplacementVectorList[procIndex]
                speciesDisplacementVectorList[speciesIndex] += speciesDisplacementVector
                config.chargeList = self.system.chargeConfig(currentStateOccupancy)
                config.occupancy = currentStateOccupancy
                if step % stepInterval == 0:
                    speciesSystemElementIndices = np.concatenate((currentStateOccupancy.values()))
                    timeArray[pathIndex] = kmcTime
                    wrappedPositionArray[pathIndex] = config.positions[speciesSystemElementIndices]
                    speciesDisplacementArray[pathIndex] = speciesDisplacementVectorList
                    unwrappedPositionArray[pathIndex] = unwrappedPositionArray[pathIndex - 1] + speciesDisplacementVectorList
                    speciesDisplacementVectorList = np.zeros((self.totalSpecies, 3))
                    pathIndex += 1
        
        trajectoryData = returnValues()
        trajectoryData.speciesCount = self.system.speciesCount
        trajectoryData.nTraj = nTraj
        trajectoryData.kmcSteps = kmcSteps
        trajectoryData.stepInterval = stepInterval
        trajectoryData.timeArray = timeArray
        trajectoryData.unwrappedPositionArray = unwrappedPositionArray
        trajectoryData.wrappedPositionArray = wrappedPositionArray
        trajectoryData.speciesDisplacementArray = speciesDisplacementArray
        return trajectoryData
        
class analysis(object):
    '''
    Post-simulation analysis methods
    '''
    def __init__(self, trajectoryData, nStepsMSD, nDispMSD, binsize, maxBinSize = 1.0, reprTime = 'ns', reprDist = 'Angstrom'):
        '''
        
        '''
        self.trajectoryData = trajectoryData

        self.nStepsMSD = int(nStepsMSD)
        self.nDispMSD = int(nDispMSD)
        self.binsize = binsize
        self.maxBinSize = maxBinSize
        self.timeConversion = 1E+09 if reprTime is 'ns' else 1E+00 
        self.distConversion = 1E-10 if reprDist is 'm' else 1E+00
        
        self.nTraj = self.trajectoryData.nTraj
        self.kmcSteps = self.trajectoryData.kmcSteps
        self.stepInterval = self.trajectoryData.stepInterval

    def computeMSD(self, timeArray, unwrappedPositionArray):
        '''
        Returns the squared displacement of the trajectories
        '''
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
        numPathStepsPerTraj = np.floor(self.kmcSteps / self.stepInterval) + 1
        for trajIndex in range(self.nTraj):
            for timestep in range(1, self.nStepsMSD + 1):
                for step in range(self.nDispMSD):
                    headStart = trajIndex * (numPathStepsPerTraj + 1)
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
        msdData = np.zeros((nBins, nSpeciesTypes + 1))
        msdData[:, 0] = bins[:-1] + 0.5 * self.binsize
        startIndex = 0
        for speciesTypeIndex in range(nSpeciesTypes):
            endIndex = startIndex + speciesCount[speciesTypes[speciesTypeIndex]]
            msdData[:, speciesTypeIndex + 1] = np.mean(speciesMSDData[:, startIndex:endIndex], axis=1)
            #msdData[:, 1] = np.mean(speciesMSDData, axis=1)
            startIndex = endIndex
        
        returnMSDData = returnValues()
        returnMSDData.msdData = msdData
        returnMSDData.speciesTypes = speciesTypes
        return returnMSDData


class plot(object):
    '''
    class with definitions of plotting funcitons
    '''
    
    def __init__(self, msdAnalysisData):
        '''
        
        '''
        self.msdData = msdAnalysisData.msdData
        self.speciesTypes = msdAnalysisData.speciesTypes

    def displayMSDPlot(self):
        '''
        Returns a line plot of the MSD data
        '''
        import matplotlib.pyplot as plt
        plt.figure(1)
        for speciesIndex, speciesType in enumerate(self.speciesTypes):
            plt.plot(self.msdData[:,0], self.msdData[:,speciesIndex + 1], label=speciesType)
        plt.xlabel('Time (ns)')
        plt.ylabel('MSD (Angstrom**2)')
        plt.legend()
        plt.show()
        #plt.savefig(figname)

class returnValues(object):
    '''
    Returns the values of displacement list and respective coordinates in an object
    '''
    pass
