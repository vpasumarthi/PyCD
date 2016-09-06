#!/usr/bin/env python
'''
code to compute msd of random walk of single electron in 3D hematite lattice structure
 switch off PBC, and 
'''
import numpy as np
from collections import OrderedDict

class modelParameters(object):
    '''
    Definitions of all model parameters that need to be passed on to other classes
    '''
    def __init__(self, T, nTraj, kmcsteps, stepInterval, nsteps_msd, ndisp_msd, binsize, 
                 systemSize=np.array([10, 10, 10]), pbc=1, gui=0, kB=8.617E-05):
        '''
        Definitions of all model parameters
        '''
        # TODO: Is it necessary/better to define these parameters in a dictionary?
        self.T = T
        self.nTraj = int(nTraj)
        self.kmcsteps = int(kmcsteps)
        self.stepInterval = int(stepInterval)
        self.nsteps_msd = int(nsteps_msd)
        self.ndisp_msd = int(ndisp_msd)
        self.binsize = binsize
        self.systemSize = systemSize
        self.pbc = pbc
        self.gui = gui
        self.kB = kB
        
class material(object):
    '''
    defines the structure of working material
    
    Attributes:
        name: A string representing the material name
        elements: list of element symbols
        species_to_sites: dictionary that maps species to sites
        pos: positions of elements in the unit cell
        index: element index of the positions starting from 0
        charge: atomic charges of the elements # first shell atomic charges to be included
        latticeParameters: list of three lattice constants in angstrom and three angles between them in degrees
        # TODO: lattice constants and unit cell matrix may be defined in here
    '''
    
    def __init__(self, name, elementTypes, speciesTypes, unitcellCoords, elementTypeIndexList, chargeTypes, 
                 latticeParameters, vn, lambdaValues, VAB, neighborCutoffDist, neighborCutoffDistTol, 
                 elementTypeDelimiter):
        '''
        Return an material object whose name is *name* 
        '''
        # TODO: introduce a method to view the material using ase atoms or other gui module
        self.name = name
        self.elementTypes = elementTypes
        self.speciesTypes = speciesTypes
        for key in speciesTypes:
            assert set(speciesTypes[key]) <= set(elementTypes), 'Specified sites should be a subset of elements'
        self.unitcellCoords = unitcellCoords
        startIndex = 0
        for elementIndex in range(len(elementTypes)):
            elementUnitCellCoords = unitcellCoords[elementTypeIndexList==elementIndex]
            endIndex = startIndex + len(elementUnitCellCoords)
            self.unitcellCoords[startIndex:endIndex] = elementUnitCellCoords[elementUnitCellCoords[:,2].argsort()]
            startIndex = endIndex
        self.elementTypeIndexList = elementTypeIndexList.astype(int)
        self.chargeTypes = chargeTypes
        self.latticeParameters = latticeParameters
        self.vn = vn
        self.lambdaValues = lambdaValues
        self.VAB = VAB
        self.neighborCutoffDist = neighborCutoffDist
        self.neighborCutoffDistTol = neighborCutoffDistTol
        self.elementTypeDelimiter = elementTypeDelimiter
        
        # number of elements
        length = len(self.elementTypes)
        nElements = np.zeros(length, int)
        for elementIndex in range(length):
            nElements[elementIndex] = np.count_nonzero(self.elementTypeIndexList == elementIndex)
        self.nElements = nElements
        
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
        nSites = np.zeros(len(self.siteList), int)
        for elementTypeIndex, elementType in enumerate(self.elementTypes):
            if elementType in siteList:
                nSites[elementTypeIndex] = len(np.where(self.elementTypeIndexList == elementTypeIndex)[0])
        self.nSites = nSites
        
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
        cell = np.array([[ a                , 0                , 0],
                         [ b * np.cos(gamma), b * np.sin(gamma), 0],
                         [ c * np.cos(alpha), c * np.cos(beta) , c * np.sqrt(np.sin(alpha)**2 - np.cos(beta)**2)]]) # cell matrix
        self.latticeMatrix = cell

    def generateSites(self, elementTypeIndices, cellSize=np.array([1, 1, 1])):
        '''
        Returns systemElementIndices and coordinates of specified elements in a cell of size *cellSize*
        '''
        assert all(size > 0 for size in cellSize), 'Input size should always be greater than 0'
        extractIndices = np.in1d(self.elementTypeIndexList, elementTypeIndices).nonzero()[0]
        unitcellElementCoords = self.unitcellCoords[extractIndices]
        numCells = np.prod(cellSize)
        nSites = len(unitcellElementCoords)
        unitcellElementIndexList = np.arange(nSites)
        unitcellElementTypeIndex = np.reshape(np.concatenate((np.asarray([[elementTypeIndex] * self.nElements[elementTypeIndex] 
                                                                          for elementTypeIndex in elementTypeIndices]))), (nSites, 1))
        unitCellElementTypeElementIndexList = np.reshape(np.concatenate(([np.arange(self.nElements[elementTypeIndex]) 
                                                                          for elementTypeIndex in elementTypeIndices])), (nSites, 1))
        cellCoordinates = np.zeros((numCells * nSites, 3))
        # quantumIndex = [unitCellIndex, elementTypeIndex, elementIndex]
        quantumIndexList = np.zeros((numCells * nSites, 5), dtype=int)
        systemElementIndexList = np.zeros(numCells * nSites, dtype=int)
        iUnitCell = 0
        for xSize in range(cellSize[0]):
            for ySize in range(cellSize[1]):
                for zSize in range(cellSize[2]): 
                    startIndex = iUnitCell * nSites
                    endIndex = startIndex + nSites
                    newCellSiteCoords = unitcellElementCoords + np.dot([xSize, ySize, zSize], self.latticeMatrix)
                    cellCoordinates[startIndex:endIndex, :] = newCellSiteCoords 
                    systemElementIndexList[startIndex:endIndex] = iUnitCell * nSites + unitcellElementIndexList
                    quantumIndexList[startIndex:endIndex] = np.hstack((np.tile(np.array([xSize, ySize, zSize]), (nSites, 1)), unitcellElementTypeIndex, unitCellElementTypeElementIndexList))
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
        unitCellIndex = quantumIndices[:3]
        [elementTypeIndex, elementIndex] = quantumIndices[-2:]
        nElementsPerUnitCell = np.sum(self.nElements)
        systemElementIndex = (elementIndex + np.sum(self.nElements[:elementTypeIndex]) + nElementsPerUnitCell * 
                              (unitCellIndex[2] + unitCellIndex[1] * systemSize[2] + unitCellIndex[0] * 
                               np.prod(systemSize[1:])))
        return systemElementIndex
    
    def generateQuantumIndices(self, systemSize, systemElementIndex):
        '''
        Returns the quantum indices of the element
        '''
        quantumIndices = [0] * 5
        nElementsPerUnitCell = np.sum(self.nElements)
        nElementsPerUnitCellCumSum = np.cumsum(self.nElements)
        unitcellElementIndex = (systemElementIndex + 1) % nElementsPerUnitCell - 1
        quantumIndices[3] = np.min(np.where(nElementsPerUnitCellCumSum >= (unitcellElementIndex + 1)))
        quantumIndices[4] = unitcellElementIndex - np.sum(self.nElements[:quantumIndices[3]])
        nFilledUnitCells = (systemElementIndex - unitcellElementIndex) / nElementsPerUnitCell
        quantumIndices[2] = nFilledUnitCells % systemSize[2] - 1
        quantumIndices[1] = (nFilledUnitCells / systemSize[2]) % systemSize[1]
        quantumIndices[0] = (nFilledUnitCells / np.prod(systemSize[1:])) % systemSize[0]
        return quantumIndices
    
class system(object):
    '''
    defines the system we are working on
    
    Attributes:
    size: An array (3 x 1) defining the system size in multiple of unit cells
    '''
    
    def __init__(self, modelParameters, material, occupancy):
        '''
        Return a system object whose size is *size*
        '''
        self.modelParameters = modelParameters
        self.material = material
        self.occupancy = OrderedDict(occupancy)
        
        # total number of unit cells
        self.numCells = np.prod(self.modelParameters.systemSize)
        
        # generate all sites in the system
        elementTypeIndices = range(len(self.material.elementTypes))
        bulkSites = self.material.generateSites(elementTypeIndices, self.modelParameters.systemSize)
        self.bulkSites = bulkSites
        
        # generate lattice charge list
        unitCellChargeList = np.array([self.material.chargeTypes[self.material.elementTypes[elementTypeIndex]] 
                                       for elementTypeIndex in self.material.elementTypeIndexList])
        self.latticeChargeList = np.tile(unitCellChargeList, self.numCells)
    
    # TODO: Is it better to shift neighborSites method to material class and add generateNeighborList method to 
    # __init__ function of system class?   
    def neighborSites(self, bulkSites, centerSiteIndices, neighborSiteIndices, cutoffDistLimits):
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
                displacementVector = neighborCoord - centerCoord
                if not self.modelParameters.pbc:
                    displacement = np.linalg.norm(displacementVector)
                else:
                    # TODO: Use minimum image convention to compute the distance
                    displacement = np.linalg.norm(displacementVector)
                if cutoffDistLimits[0] < displacement <= cutoffDistLimits[1]:
                    iNeighborSiteIndexList.append(neighborSiteIndex)
                    iDisplacementVectors.append(displacementVector)
                    iDisplacements.append(displacement)
            neighborSystemElementIndices.append(np.array(neighborSiteSystemElementIndexList[iNeighborSiteIndexList]))
            offsetList.append(centerSiteQuantumIndexList[centerSiteIndex, :3] - neighborSiteQuantumIndexList[iNeighborSiteIndexList, :3])
            neighborElementIndexList.append(neighborSiteQuantumIndexList[iNeighborSiteIndexList, 4])
            numNeighbors.append(len(iNeighborSiteIndexList))
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

    def generateNeighborList(self):
        '''
        Adds the neighbor list to the system object and returns the neighbor list
        '''
        neighborList = {}
        tolDist = self.material.neighborCutoffDistTol
        elementTypes = self.material.elementTypes
        systemSize = self.modelParameters.systemSize
        for cutoffDistKey in self.material.neighborCutoffDist.keys():
            cutoffDistList = self.material.neighborCutoffDist[cutoffDistKey]
            neighborListCutoffDistKey = []
            if cutoffDistKey is not 'E':
                [centerElementType, neighborElementType] = cutoffDistKey.split(self.material.elementTypeDelimiter)
                centerSiteElementTypeIndex = elementTypes.index(centerElementType) 
                neighborSiteElementTypeIndex = elementTypes.index(neighborElementType)
                centerSiteIndices = [self.material.generateSystemElementIndex(systemSize, np.array([1, 1, 1, centerSiteElementTypeIndex, elementIndex])) 
                                     for elementIndex in range(self.material.nElements[centerSiteElementTypeIndex])]
                neighborSiteIndices = [self.material.generateSystemElementIndex(systemSize, np.array([xSize, ySize, zSize, neighborSiteElementTypeIndex, elementIndex])) 
                                       for xSize in range(systemSize[0]) for ySize in range(systemSize[1]) 
                                       for zSize in range(systemSize[2]) 
                                       for elementIndex in range(self.material.nElements[neighborSiteElementTypeIndex])]
                for cutoffDist in cutoffDistList:
                    cutoffDistLimits = [cutoffDist-tolDist, cutoffDist+tolDist]
                    # TODO: include assertions, conditions for systemSizes less than [3, 3, 3]
                    neighborListCutoffDistKey.append(self.neighborSites(self.bulkSites, centerSiteIndices, 
                                                                        neighborSiteIndices, cutoffDistLimits))                    
            else:
                centerSiteIndices = neighborSiteIndices = np.arange(self.numCells * np.sum(self.material.nElements))
                cutoffDistLimits = [0, cutoffDistList[0]]
                neighborListCutoffDistKey.append(self.neighborSites(self.bulkSites, centerSiteIndices, 
                                                                    neighborSiteIndices, cutoffDistLimits))
            neighborList[cutoffDistKey] = neighborListCutoffDistKey
        self.neighborList = neighborList
        return neighborList

    def chargeConfig(self, occupancy):
        '''
        Returns charge distribution of the current configuration
        '''
        chargeList = self.latticeChargeList
        chargeTypes = self.material.chargeTypes
        chargeTypeKeys = chargeTypes.keys()
        siteChargeTypeKeys = [key for key in chargeTypeKeys if key not in self.material.siteList if '0' in key]
        for chargeKeyType in siteChargeTypeKeys:
            centerSiteElementType = chargeKeyType.replace('0','')
            for speciesType in self.material.elementTypeSpeciesMap[centerSiteElementType]:
                assert speciesType in self.occupancy.keys(), ('Invalid definition of charge type \'' + str(chargeKeyType) + '\', \'' + 
                                                              str(speciesType) + '\' species does not exist in this configuration') 
                centerSiteSystemElementIndices = self.occupancy[speciesType]
                chargeList[centerSiteSystemElementIndices] = chargeTypes[chargeKeyType]
        
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

    def elec(self, occupancy):
        '''
        Subroutine to compute the electrostatic interaction energies
        '''
        elec = 0
        return elec
        
    def delG0(self, currentStateOccupancy, newStateOccupancy):
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
        hopElementType = []
        hopDistType = []
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
                            neighborUnitCellIndex = [sum(x) for x in zip(speciesQuantumIndices[:3], neighborOffsetList[speciesElementIndex][neighborIndex])]
                            neighborElementTypeIndex = [self.material.elementTypes.index(iHopElementType.split(self.material.elementTypeDelimiter)[1])]
                            neighborElementIndex = [neighborElementIndexMap[1][speciesElementIndex][neighborIndex]]
                            neighborQuantumIndices = neighborUnitCellIndex + neighborElementTypeIndex + neighborElementIndex
                            neighborSystemElementIndex = self.material.generateSystemElementIndex(self.modelParameters.systemSize, neighborQuantumIndices)
                            neighborSystemElementIndices.append(neighborSystemElementIndex)
                        for neighborSystemElementIndex in neighborSystemElementIndices:
                            newStateOccupancy = currentStateOccupancy
                            newStateOccupancy[speciesType][speciesIndex] = neighborSystemElementIndex
                            newStateOccupancyList.append(newStateOccupancy)
                            hopElementType.append(iHopElementType)
                            hopDistType.append(hopDistTypeIndex)

        returnNewStates = returnValues()
        returnNewStates.newStateOccupancyList = newStateOccupancyList
        returnNewStates.hopElementType = hopElementType
        returnNewStates.hopDistType = hopDistType
        return returnNewStates

    def doKMCSteps(self, randomSeed = 1):
        '''
        Subroutine to run the kmc simulation by specified number of steps
        '''
        import random as rnd
        rnd.seed(randomSeed)
        nTraj = self.modelParameters.nTraj
        kmcsteps = self.modelParameters.kmcsteps
        stepInterval = self.modelParameters.stepInterval
        currentStateOccupancy = self.system.occupancy
        
        timeNdisplacement = np.zeros(( nTraj * (np.floor(kmcsteps / stepInterval) + 1), self.totalSpecies + 1))
        pathIndex = 0
        speciesSystemElementIndices = np.concatenate((currentStateOccupancy.values()))
        config = self.system.config(currentStateOccupancy)
        # TODO: may have to use speciesSystemElementIndices.tolist()
        speciesPositionListOld = config.positions[speciesSystemElementIndices]
        for traj in range(nTraj):
            pathIndex += 1
            speciesDisplacementList = np.zeros(self.totalSpecies)
            kmcTime = 0
            for step in range(kmcsteps):
                kList = []
                newStates = self.generateNewStates(currentStateOccupancy)
                for newStateIndex, newStateOccupancy in enumerate(newStates.newStateOccupancyList):
                    hopElementType = newStates.hopElementType[newStateIndex]
                    hopDistType = newStates.hopDistType[newStateIndex]
                    delG0 = self.delG0(currentStateOccupancy, newStateOccupancy)
                    lambdaValue = self.lambdaValues[hopElementType][hopDistType]
                    VAB = self.VAB[hopElementType][hopDistType]
                    delGs = ((lambdaValue + delG0) ** 2 / (4 * lambdaValue)) - VAB
                    kList.append(self.vn * np.exp(-delGs / (self.kB * self.T)))
                kTotal = np.sum(kList)
                kCumSum = np.cumsum(kList / kTotal)
                rand1 = rnd.random()
                procIndex = np.min(np.where(kCumSum > rand1))
                rand2 = rnd.random()
                kmcTime += np.log(rand2) / kTotal
                currentStateOccupancy = newStates.newStateOccupancyList[procIndex]
                config.chargeList = self.system.chargeConfig(currentStateOccupancy)
                config.occupancy = currentStateOccupancy
                if step % stepInterval == 0:
                    speciesSystemElementIndices = np.concatenate((currentStateOccupancy.values()))
                    # TODO: may have to use speciesSystemElementIndices.tolist()
                    speciesPositionListNew = config.positions[speciesSystemElementIndices]
                    speciesDisplacementList = np.linalg.norm(speciesPositionListNew - speciesPositionListOld, axis=1)
                    speciesPositionListOld = speciesPositionListNew
                    timeNdisplacement[pathIndex, :] = np.concatenate((np.array([kmcTime]), speciesDisplacementList))
                    pathIndex += 1
        return timeNdisplacement

class analysis(object):
    '''
    Post-simulation analysis methods
    '''
    
    def __init__(self):
        '''
        
        '''
        
    def compute_sd(self, timeNdisplacement):
        '''
        Subroutine to compute the squared displacement of the trajectories
        '''
        # timeNdisplacement, not timeNpath
        nsteps_msd = modelParameters.nsteps_msd
        ndisp_msd = modelParameters.ndisp_msd
        sec_to_ns = 1E+09
        time = timeNdisplacement[:, 0] * sec_to_ns
        path = timeNdisplacement[:, 1:]
        sd_array = np.zeros((nsteps_msd * ndisp_msd, 2))
        for timestep in range(1, nsteps_msd+1):
                sum_sd_timestep = 0
                for step in range(ndisp_msd):
                        sd_timestep = sum((path[step + timestep] - path[step])**2)
                        sd_array[(timestep-1) * ndisp_msd + step, 0] = time[step + timestep] - time[step]
                        sd_array[(timestep-1) * ndisp_msd + step, 1] = sd_timestep
                        sum_sd_timestep += sd_timestep
        return sd_array
    
    def compute_msd(self, sd_array):
        '''
        subroutine to compute the mean squared displacement
        '''
        nsteps_msd = modelParameters.nsteps_msd
        ndisp_msd = modelParameters.ndisp_msd
        binsize = modelParameters.binsize
        sd_array =  sd_array[sd_array[:, 0].argsort()] # sort sd_traj array by time column
        sd_array[:,0] = np.floor(sd_array[:,0])
        time_final = np.ceil(sd_array[nsteps_msd * ndisp_msd - 1, 0])
        nbins = int(np.ceil(time_final / binsize)) + 1
        time_array = np.arange(0, time_final+1, binsize) + 0.5 * binsize

        msd_array = np.zeros((nbins, 2))
        msd_array[:,0] = time_array
        for ibin in range(nbins):
                msd_array[ibin, 1] = np.mean(sd_array[sd_array[:,0] == ibin, 1])
        return msd_array

    
class plot(object):
    '''
    
    '''
    
    def __init__(self):
        '''
        
        '''
    
    def plot(self, msd):
        '''
        
        '''
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(msd[:,0], msd[:,1])
        plt.xlabel('Time (ns)')
        plt.ylabel('MSD (Angstrom**2)')
        #plt.show()
        #plt.savefig(figname)

class returnValues(object):
    '''
    Returns the values of displacement list and respective coordinates in an object
    '''
    pass
