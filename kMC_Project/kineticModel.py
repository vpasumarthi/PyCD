#!/usr/bin/env python
'''
code to compute msd of random walk of single electron in 3D hematite lattice structure
 switch off PBC, and 
'''
import numpy as np
from collections import OrderedDict
from copy import deepcopy

class modelParameters(object):
    '''
    Definitions of all model parameters that need to be passed on to other classes
    '''
    def __init__(self, T, nTraj, kmcSteps, stepInterval, nStepsMSD, nDispMSD, binsize, maxBinSize, 
                 systemSize=np.array([10, 10, 10]), pbc=[1, 1, 1], gui=0, kB=8.617E-05, reprTime='ns', reprDist='Angstrom'):
        '''
        Definitions of all model parameters
        '''
        # TODO: Is it necessary/better to define these parameters in a dictionary?
        self.T = T
        self.nTraj = int(nTraj)
        self.kmcSteps = int(kmcSteps)
        self.stepInterval = int(stepInterval)
        self.nStepsMSD = int(nStepsMSD)
        self.nDispMSD = int(nDispMSD)
        self.binsize = binsize
        self.maxBinSize = maxBinSize
        self.systemSize = systemSize
        self.pbc = pbc
        self.gui = gui
        self.kB = kB
        self.reprTime = reprTime
        self.reprDist = reprDist
        
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
        for key in speciesTypes:
            assert set(speciesTypes[key]) <= set(elementTypes), 'Specified sites should be a subset of elements'
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
        siteList = []
        for key in self.speciesTypes:
            if key is not 'empty':
                siteList.extend(self.speciesTypes[key])
        siteList = list(set(siteList))
        self.siteList = siteList
        
        # list of hop element types
        hopElementTypes = {}
        for key in self.speciesTypes:
            if key is not 'empty':
                speciesTypeHopElementTypes = []
                for centerElementIndex, centerElementType in enumerate(self.speciesTypes[key]):
                    for neighborElementType in self.speciesTypes[key][centerElementIndex:]:
                        speciesTypeHopElementTypes.append(centerElementType + self.elementTypeDelimiter + 
                                                          neighborElementType)   
                hopElementTypes[key] = speciesTypeHopElementTypes
        self.hopElementTypes = hopElementTypes
        
        # number of sites present in siteList
        siteElementIndices = [i for i,n in enumerate(elementTypes) if n in siteList]
        nSites = [nElements[i] for i in siteElementIndices]
        self.nSites = np.asarray(nSites, int)
        
        # element - species map
        elementTypeSpeciesMap = {}
        nonEmptySpeciesTypes = speciesTypes
        del nonEmptySpeciesTypes['empty']
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
                    # TODO: Shouldn't the product be latticeMatrix * unitCellSize?
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
    
    def __init__(self, modelParameters, material):
        '''
        
        '''
        self.material = material
        self.modelParameters = modelParameters
        
        # total number of unit cells
        self.numCells = np.prod(self.modelParameters.systemSize)
        
        # generate all sites in the system
        elementTypeIndices = range(len(self.material.elementTypes))
        bulkSites = self.material.generateSites(elementTypeIndices, self.modelParameters.systemSize)
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
        
        xRange = range(-1, 2) if self.modelParameters.pbc[0] == 1 else [0]
        yRange = range(-1, 2) if self.modelParameters.pbc[1] == 1 else [0]
        zRange = range(-1, 2) if self.modelParameters.pbc[2] == 1 else [0]
        latticeMatrix = self.material.latticeMatrix
        for centerSiteIndex, centerCoord in enumerate(centerSiteCoords):
            iDisplacementVectors = []
            iDisplacements = []
            iNeighborSiteIndexList = []
            for neighborSiteIndex, neighborCoord in enumerate(neighborSiteCoords):
                neighborImageCoords = np.zeros((3**sum(self.modelParameters.pbc), 3))
                index = 0
                for xOffset in xRange:
                    for yOffset in yRange:
                        for zOffset in zRange:
                            unitcellTranslationalCoords = np.dot(np.multiply(np.array([xOffset, yOffset, zOffset]), self.modelParameters.systemSize), latticeMatrix)
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

    def generateNeighborList(self):
        '''
        Adds the neighbor list to the system object and returns the neighbor list
        '''
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
                localSystemSize = np.array([3, 3, 3])
                localBulkSites = self.material.generateSites(range(len(self.material.elementTypes)), 
                                                             localSystemSize)
                centerSiteIndices = [self.material.generateSystemElementIndex(localSystemSize, np.array([1, 1, 1, centerSiteElementTypeIndex, elementIndex])) 
                                     for elementIndex in range(self.material.nElements[centerSiteElementTypeIndex])]
                neighborSiteIndices = [self.material.generateSystemElementIndex(localSystemSize, np.array([xSize, ySize, zSize, neighborSiteElementTypeIndex, elementIndex])) 
                                       for xSize in range(localSystemSize[0]) for ySize in range(localSystemSize[1]) 
                                       for zSize in range(localSystemSize[2]) 
                                       for elementIndex in range(self.material.nElements[neighborSiteElementTypeIndex])]

                for cutoffDist in cutoffDistList:
                    cutoffDistLimits = [cutoffDist-tolDist, cutoffDist+tolDist]
                    # TODO: include assertions, conditions for systemSizes less than [3, 3, 3]
                    neighborListCutoffDistKey.append(self.hopNeighborSites(localBulkSites, centerSiteIndices, 
                                                                           neighborSiteIndices, cutoffDistLimits, cutoffDistKey))
            else:
                centerSiteIndices = neighborSiteIndices = np.arange(self.numCells * np.sum(self.material.nElements))
                cutoffDistLimits = [0, cutoffDistList[0]]
                neighborListCutoffDistKey.append(self.electrostaticNeighborSites(self.modelParameters.systemSize, self.bulkSites, centerSiteIndices, 
                                                                                 neighborSiteIndices, cutoffDistLimits, cutoffDistKey))
            neighborList[cutoffDistKey] = neighborListCutoffDistKey
        return neighborList
    
class system(object):
    '''
    defines the system we are working on
    
    Attributes:
    size: An array (3 x 1) defining the system size in multiple of unit cells
    '''
    
    def __init__(self, modelParameters, material, occupancy, neighborList):
        '''
        Return a system object whose size is *size*
        '''
        self.modelParameters = modelParameters
        self.material = material
        self.occupancy = OrderedDict(occupancy)
        self.neighborList = neighborList
        
        # total number of unit cells
        self.numCells = np.prod(self.modelParameters.systemSize)
        
        # generate lattice charge list
        unitCellChargeList = np.array([self.material.chargeTypes[self.material.elementTypes[elementTypeIndex]] 
                                       for elementTypeIndex in self.material.elementTypeIndexList])
        self.latticeChargeList = np.tile(unitCellChargeList, self.numCells)
        
        self.latticeParameters = self.material.latticeParameters
        
    
    # TODO: Is it better to shift neighborSites method to material class and add generateNeighborList method to 
    # __init__ function of system class?   
    def chargeConfig(self, occupancy):
        '''
        Returns charge distribution of the current configuration
        '''
        chargeList = self.latticeChargeList
        chargeTypes = self.material.chargeTypes
        chargeTypeKeys = chargeTypes.keys()
        
        shellChargeTypeKeys = [key for key in chargeTypeKeys if key not in self.material.siteList if '0' not in key]
        for chargeTypeKey in shellChargeTypeKeys:
            centerSiteElementType = chargeTypeKey.split(self.material.elementTypeDelimiter)[0]
            neighborElementTypeSites = self.neighborList[chargeTypeKey]
            for speciesType in self.material.elementTypeSpeciesMap[centerSiteElementType]:
                centerSiteSystemElementIndices = self.occupancy[speciesType]
                for centerSiteSystemElementIndex in centerSiteSystemElementIndices:
                    for shellIndex, chargeValue in enumerate(chargeTypes[chargeTypeKey]):
                        neighborOffsetList = neighborElementTypeSites[shellIndex].offsetList
                        neighborElementIndexMap = neighborElementTypeSites[shellIndex].elementIndexMap
                        siteQuantumIndices = self.material.generateQuantumIndices(self.modelParameters.systemSize, centerSiteSystemElementIndex)
                        siteElementIndex = siteQuantumIndices[4]
                        neighborSystemElementIndices = []
                        for neighborIndex in range(len(neighborElementIndexMap[1][siteElementIndex])):
                            neighborUnitCellIndex = [sum(x) for x in zip(siteQuantumIndices[:3], neighborOffsetList[siteElementIndex][neighborIndex])]
                            neighborElementTypeIndex = [self.material.elementTypes.index(chargeTypeKey.split(self.material.elementTypeDelimiter)[1])]
                            neighborElementIndex = [neighborElementIndexMap[1][siteElementIndex][neighborIndex]]
                            neighborQuantumIndices = neighborUnitCellIndex + neighborElementTypeIndex + neighborElementIndex
                            neighborSystemElementIndex = self.material.generateSystemElementIndex(self.modelParameters.systemSize, neighborQuantumIndices)
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
        systemSites = self.material.generateSites(elementTypeIndices, self.modelParameters.systemSize)
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
    def __init__(self, modelParameters, material, system):
        '''
        Returns the PBC condition of the system
        '''
        self.modelParameters = modelParameters
        self.material = material
        self.system = system
        
        # import parameters from modelParameters class
        self.kB = self.modelParameters.kB
        self.T = self.modelParameters.T
        
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
        for speciesType in currentStateOccupancy.keys():
            for speciesIndex, speciesSystemElementIndex in enumerate(currentStateOccupancy[speciesType]):
                for iHopElementType in self.material.hopElementTypes[speciesType]:
                    for hopDistTypeIndex in range(len(self.material.neighborCutoffDist[iHopElementType])):
                        neighborOffsetList = neighborList[iHopElementType][hopDistTypeIndex].offsetList
                        neighborElementIndexMap = neighborList[iHopElementType][hopDistTypeIndex].elementIndexMap
                        speciesQuantumIndices = self.material.generateQuantumIndices(self.modelParameters.systemSize, speciesSystemElementIndex)
                        speciesElementIndex = speciesQuantumIndices[4]
                        neighborSystemElementIndices = []
                        for neighborIndex in range(len(neighborElementIndexMap[1][speciesElementIndex])):
                            neighborUnitCellIndices = [sum(x) for x in zip(speciesQuantumIndices[:3], neighborOffsetList[speciesElementIndex][neighborIndex])]
                            # TODO: Try to avoid the loop and conditions
                            for index, neighborUnitCellIndex in enumerate(neighborUnitCellIndices):
                                if neighborUnitCellIndex > self.modelParameters.systemSize[index] - 1:
                                    neighborUnitCellIndices[index] -= self.modelParameters.systemSize[index]
                                elif neighborUnitCellIndex < 0:
                                    neighborUnitCellIndices[index] += self.modelParameters.systemSize[index]
                            neighborElementTypeIndex = [self.material.elementTypes.index(iHopElementType.split(self.material.elementTypeDelimiter)[1])]
                            neighborElementIndex = [neighborElementIndexMap[1][speciesElementIndex][neighborIndex]]
                            neighborQuantumIndices = neighborUnitCellIndices + neighborElementTypeIndex + neighborElementIndex
                            neighborSystemElementIndex = self.material.generateSystemElementIndex(self.modelParameters.systemSize, neighborQuantumIndices)
                            neighborSystemElementIndices.append(neighborSystemElementIndex)
                        for neighborSystemElementIndex in neighborSystemElementIndices:
                            newStateOccupancy = deepcopy(currentStateOccupancy)
                            newStateOccupancy[speciesType][speciesIndex] = neighborSystemElementIndex
                            newStateOccupancyList.append(newStateOccupancy)
                            hopElementTypes.append(iHopElementType)
                            hopDistTypes.append(hopDistTypeIndex)

        returnNewStates = returnValues()
        returnNewStates.newStateOccupancyList = newStateOccupancyList
        returnNewStates.hopElementTypes = hopElementTypes
        returnNewStates.hopDistTypes = hopDistTypes
        return returnNewStates

    def doKMCSteps(self, randomSeed = 1):
        '''
        Subroutine to run the kmc simulation by specified number of steps
        '''
        # TODO: import neighbor list
        import random as rnd
        rnd.seed(randomSeed)
        nTraj = self.modelParameters.nTraj
        kmcSteps = self.modelParameters.kmcSteps
        stepInterval = self.modelParameters.stepInterval
        currentStateOccupancy = self.system.occupancy
                
        timeArray = np.zeros(nTraj * (np.floor(kmcSteps / stepInterval) + 1))
        positionArray = np.zeros(( nTraj * (np.floor(kmcSteps / stepInterval) + 1), self.totalSpecies, 3))
        pathIndex = 0
        speciesSystemElementIndices = np.concatenate((currentStateOccupancy.values()))
        config = self.system.config(currentStateOccupancy)
        assert 'E' in self.material.neighborCutoffDist.keys(), 'Please specify the cutoff distance for electrostatic interactions'
        self.generateDistanceList()
        for dummy in range(nTraj):
            pathIndex += 1
            kmcTime = 0
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
                config.chargeList = self.system.chargeConfig(currentStateOccupancy)
                config.occupancy = currentStateOccupancy
                if step % stepInterval == 0:
                    # TODO: update position from the displacementvectorlist
                    speciesSystemElementIndices = np.concatenate((currentStateOccupancy.values()))
                    speciesPositionList = config.positions[speciesSystemElementIndices]
                    timeArray[pathIndex] = kmcTime
                    positionArray[pathIndex] = speciesPositionList
                    pathIndex += 1
        timeNpath = np.empty(2, dtype=object)
        timeNpath[:] = [timeArray, positionArray]
        return timeNpath

class analysis(object):
    '''
    Post-simulation analysis methods
    '''
    
    def __init__(self, modelParameters, timeNpath):
        '''
        
        '''
        self.modelParameters = modelParameters
        self.timeNpath = timeNpath
        self.timeConversion = 1E+09 if self.modelParameters.reprTime is 'ns' else 1E+00 
        self.distConversion = 1E-10 if self.modelParameters.reprDist is 'm' else 1E+00
        
        self.nStepsMSD = self.modelParameters.nStepsMSD
        self.nDispMSD = self.modelParameters.nDispMSD
        self.nTraj = self.modelParameters.nTraj
        self.kmcSteps = self.modelParameters.kmcSteps

    def computeMSD(self, timeNpath):
        '''
        Returns the squared displacement of the trajectories
        '''
        time = timeNpath[0] * self.timeConversion
        path = timeNpath[1] * self.distConversion
        nSpecies = len(path[0])
        timeNdisp2 = np.zeros((self.nTraj * (self.nStepsMSD * self.nDispMSD), nSpecies + 1))
        for traj in range(self.nTraj):
            for timestep in range(1, self.nStepsMSD + 1):
                for step in range(self.nDispMSD):
                    headStart = traj * (self.kmcSteps + 1)
                    workingRow = traj * (self.nStepsMSD * self.nDispMSD) + (timestep-1) * self.nDispMSD + step 
                    timeNdisp2[workingRow, 0] = time[headStart + step + timestep] - time[headStart + step]
                    timeNdisp2[workingRow, 1:] = np.linalg.norm(path[headStart + step + timestep] - 
                                                                path[headStart + step], axis=1)**2
        timeArray = timeNdisp2[:, 0]
        minEndTime = np.min(timeArray[np.arange(self.nStepsMSD * self.nDispMSD - 1, self.nTraj * (self.nStepsMSD * self.nDispMSD), self.nStepsMSD * self.nDispMSD)])  
        bins = np.arange(0, minEndTime, self.modelParameters.binsize)
        nBins = len(bins) - 1
        speciesMSDData = np.zeros((nBins, nSpecies))
        msdHistogram, dummy = np.histogram(timeArray, bins)
        for iSpecies in range(nSpecies):
            iSpeciesHist, dummy = np.histogram(timeArray, bins, weights=timeNdisp2[:, iSpecies + 1])
            speciesMSDData[:, iSpecies] = iSpeciesHist / msdHistogram
        msdData = np.zeros((nBins, 2))
        msdData[:, 0] = bins[:-1] + 0.5 * self.modelParameters.binsize
        msdData[:, 1] = np.mean(speciesMSDData, axis=1)
        return msdData
    
class plot(object):
    '''
    class with definitions of plotting funcitons
    '''
    
    def __init__(self, msdData):
        '''
        
        '''
        self.msdData = msdData

    def displayMSDPlot(self):
        '''
        Returns a line plot of the MSD data
        '''
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(self.msdData[:,0], self.msdData[:,1])
        plt.xlabel('Time (ns)')
        plt.ylabel('MSD (Angstrom**2)')
        plt.show()
        #plt.savefig(figname)

class returnValues(object):
    '''
    Returns the values of displacement list and respective coordinates in an object
    '''
    pass
