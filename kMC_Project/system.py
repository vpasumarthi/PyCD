#!/usr/bin/env python
'''
code to compute msd of random walk of single electron in 3D hematite lattice structure
 switch off PBC, and 
'''
import numpy as np
import random as rnd

class modelParameters(object):
    '''
    Definitions of all model parameters that need to be passed on to other classes
    '''
    def __init__(self, T, ntraj, kmcsteps, stepInterval, nsteps_msd, ndisp_msd, binsize, 
                 systemSize=np.array([10, 10, 10]), pbc=1, gui=0, kB=8.617E-05):
        '''
        Definitions of all model parameters
        '''
        # TODO: Is it necessary/better to define these parameters in a dictionary?
        self.T = T
        self.ntraj = ntraj
        self.kmcsteps = kmcsteps
        self.stepInterval = stepInterval
        self.nsteps_msd = nsteps_msd
        self.ndisp_msd = ndisp_msd
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
                 latticeParameters, vn, lambdaValues, VAB, neighborCutoffDist):
        '''
        Return an material object whose name is *name* 
        '''
        # TODO: introduce a method to view the material using ase atoms or other gui module
        self.name = name
        self.elementTypes = elementTypes
        self.species_to_sites = speciesTypes
        for key in speciesTypes:
            assert set(speciesTypes[key]) <= set(elementTypes), 'Specified sites should be a subset of elements'
        self.unitcellCoords = unitcellCoords
        startIndex = 0
        for elementIndex in range(len(elementTypes)):
            elementUnitCellCoords = unitcellCoords[elementTypeIndexList==elementIndex]
            endIndex = startIndex + len(elementUnitCellCoords)
            self.unitcellCoords[startIndex:endIndex] = elementUnitCellCoords[elementUnitCellCoords[:,2].argsort()]
            startIndex = endIndex
        self.elementTypeIndexList = elementTypeIndexList
        self.chargeTypes = chargeTypes
        self.latticeParameters = latticeParameters
        self.vn = vn
        self.lambdaValues = lambdaValues
        self.VAB = VAB
        self.neighborCutoffDist = neighborCutoffDist
   
    def latticeMatrix(self):
        '''
        Returns the lattice cell matrix using lattice parameters
        '''
        [a, b, c, alpha, beta, gamma] = self.latticeParameters
        cell = np.array([[ a                , 0                , 0],
                         [ b * np.cos(gamma), b * np.sin(gamma), 0],
                         [ c * np.cos(alpha), c * np.cos(beta) , c * np.sqrt(np.sin(alpha)**2 - np.cos(beta)**2)]]) # cell matrix
        return cell
    
    def nElements(self):
        '''
        Returns number of elements of each element type in a unit cell
        '''
        length = len(self.elementTypes)
        nElements = np.zeros(length)
        for elementIndex in range(length):
            nElements[elementIndex] = np.count_nonzero(self.elementTypeIndexList == elementIndex)
        return nElements
    '''
    def nSites(self, siteIndex):
        Returns number of sites available in a unit cell
        # TODO: Returns nSites for a siteIndex
        nSites = len(np.where(self.elementTypeIndexList == siteIndex)[0])
        
        SiteList = []
        for key in self.species_to_sites:
            if key != 'empty':
                SiteList.extend(self.species_to_sites[key])
        SiteList = list(set(SiteList))
        nSites = np.zeros(len(self.elementTypes))
        for elementTypeIndex, elementType in enumerate(self.elementTypes):
            if elementType in SiteList:
                nSites[elementTypeIndex] = len(np.where(self.elementTypeIndexList == elementTypeIndex)[0])
        return nSites
    '''
    
    def numLocalNeighborSites(self, siteIndex, neighborCutoffDist, bulksize=np.array([2, 2, 2])):
        '''
        Returns number of neighbor sites available per site
        '''
        sitecoords = self.generate_coords(siteIndex)
        bulkcoords = self.generate_coords(siteIndex, bulksize)
        center = sitecoords[0]
        numLocalNeighborSites = 0
        for neighbor in bulkcoords:
            displacement = neighbor - center
            hop_dist = np.linalg.norm(displacement)
            if 0 < hop_dist <= neighborCutoffDist:
                numLocalNeighborSites += 1
        return numLocalNeighborSites
    '''
    def displacementList(self, siteIndex, bulksize=[2, 2, 2]):
        sitecoords = self.generate_coords(siteIndex)
        bulkcoords = self.generate_coords(siteIndex, bulksize)
        neighborCutoffDist = self.neighborCutoffDist[self.elementTypes[siteIndex]]
        numLocalNeighborSites = self.numLocalNeighborSites(siteIndex, neighborCutoffDist, bulksize)
        totalNeighborSites = numLocalNeighborSites * self.nSites(siteIndex)
        # TODO: shouldn't use any hopdist_critical parameter at all. It should be handled with an offset array for
        # all sites in a unit cell
        hopdist_critical  = (self.hopdist_basal + self.hopdist_c_direction) / 2 

        # Initialization
        local_nn_coords = np.zeros((numLocalNeighborSites, 3))
        local_disp = np.zeros((numLocalNeighborSites, 3))
        nn_coords  = np.zeros((totalNeighborSites, 3)) # nearest neighbor coordinates
        displacementList = np.zeros((totalNeighborSites, 3))
                 
        # fetch all nearest neighbor coordinates
        for site, center in enumerate(sitecoords):
            index = 0
            for nn_coord in bulkcoords:
                displacement = nn_coord - center
                hop_dist = np.linalg.norm(displacement)
                if 0 < hop_dist <= neighborCutoffDist:
                    # TODO: This is very specific to the hematite system expecting only one c-direction hop
                    if hop_dist < hopdist_critical: # hop in c-direction
                        local_nn_coords[-1] = nn_coord
                        local_disp[-1] = displacement
                    else:
                        local_nn_coords[index] = nn_coord
                        local_disp[index] = displacement
                        index += 1
            startIndex = numLocalNeighborSites * site
            endIndex = startIndex + numLocalNeighborSites
            displacementList[startIndex:endIndex] = local_disp
            nn_coords[startIndex:endIndex] = local_nn_coords
        return returnValues(displacementList, nn_coords)
    '''
    
class returnValues(object):
    '''
    Returns the values of displacement list and respective coordinates in an object
    '''
    pass

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
        self.occupancy = occupancy
    
    # TODO: Is it necessary to provide a default value for cellSize?
    def generateCoords(self, elementTypeIndices, cellSize=np.array([1, 1, 1])):
        '''
        Returns systemElementIndices and coordinates of specified elements in a cell of size *cellSize*
        '''
        assert all(size > 0 for size in cellSize), 'Input size should always be greater than 0'
        extractIndices = np.in1d(self.material.elementTypeIndexList, elementTypeIndices).nonzero()[0]
        unitcellElementCoords = self.material.unitcellCoords[extractIndices]
        numCells = np.prod(cellSize)
        nSites = len(unitcellElementCoords)
        unitcellElementIndexList = np.arange(nSites)
        cellCoordinates = np.zeros((numCells * nSites, 3))
        systemElementIndexList = np.zeros(numCells * nSites, dtype=int)
        iUnitCell = 0
        for xSize in range(cellSize[0]):
            for ySize in range(cellSize[1]):
                for zSize in range(cellSize[2]):          
                    startIndex = iUnitCell * nSites
                    endIndex = startIndex + nSites
                    newCellSiteCoords = unitcellElementCoords + np.dot([xSize, ySize, zSize], 
                                                                       self.material.latticeMatrix())
                    cellCoordinates[startIndex:endIndex, :] = newCellSiteCoords 
                    systemElementIndexList[startIndex:endIndex] = iUnitCell * nSites + unitcellElementIndexList 
                    iUnitCell += 1
        returnCoords = returnValues()
        returnCoords.cellCoordinates = cellCoordinates
        returnCoords.systemElementIndexList = systemElementIndexList
        return returnCoords
    
    def neighborSites(self, bulkSiteCoords, bulkSystemElementIndices, centerSiteCoords, centerSystemElementIndices, 
                      cutoffDist):
        '''
        Returns systemElementIndexMap and distances between center sites and its neighbor sites within cutoff 
        distance
        '''
        neighborSystemElementIndices = []
        displacementList = []
        for centerCoord in centerSiteCoords:
            iNeighborSystemElementIndices = []
            iDisplacements = []
            for neighborIndex, neighborCoord in enumerate(bulkSiteCoords):
                displacementVector = neighborCoord - centerCoord
                if not self.modelParameters.pbc:
                    displacement = np.linalg.norm(displacementVector)
                else:
                    # TODO: Use minimum image convention to compute the distance
                    displacement = np.linalg.norm(displacementVector)
                if 0 < displacement <= cutoffDist:
                    iNeighborSystemElementIndices.append(bulkSystemElementIndices[neighborIndex])
                    iDisplacements.append(displacementVector)
            neighborSystemElementIndices.append(iNeighborSystemElementIndices)
            displacementList.append(np.asarray(iDisplacements))
        # TODO: Avoid conversion and initialize the object beforehand
        neighborSystemElementIndices = np.asarray(neighborSystemElementIndices)
        systemElementIndexMap = np.array([centerSystemElementIndices, neighborSystemElementIndices])
        returnNeighbors = returnValues()
        returnNeighbors.systemElementIndexMap = systemElementIndexMap
        # TODO: Avoid conversion and initialize the object beforehand
        returnNeighbors.displacementList = np.asarray(displacementList)
        return returnNeighbors

    def config(self):
        '''
        Generates the configuration array for the system 
        '''
        unitcellCoords = self.material.unitcellCoords
        unitcellElementTypeIndexList = material.elementTypeIndexList
        # Initialization
        positions = np.zeros((np.append(self.size, len(unitcellElementTypeIndexList))))
        charge = positions
        returnConfig = returnValues()
        returnConfig.positions = positions
        returnConfig.charge = charge
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

    def elec(self, occupancy, charge):
        '''
        Subroutine to compute the electrostatic interaction energies
        '''
        elec = 0
        return elec
        
    def delG0(self, occupancy, charge):
        '''
        Subroutine to compute the difference in free energies between initial and final states of the system
        '''
        delG0 = self.elec(occupancy, charge)
        return delG0
    
    def do_kmc_steps(self, occupancy, charge, stepInterval, kmcsteps):
        '''
        Subroutine to run the kmc simulation by specified number of steps
        '''
        T = self.modelParameters.T
        kB = self.modelParameters.kB
        vn = self.material.vn
        lambda_basal = self.material.lambda_basal
        lambda_c_direction = self.material.lambda_c_direction
        VAB_basal = self.material.VAB_basal
        VAB_c_direction = self.material.VAB_c_direction
        N_basal = self.material.N_basal
        N_c_direction = self.material.N_c_direction

        siteList = []
        for key in self.material.species_to_sites:
            if key != 'empty':
                siteList.extend(self.material.species_to_sites[key])
        siteList = list(set(siteList))
        siteIndexList = []
        for site in siteList:
            if site in self.material.elementTypes:
                siteIndexList.append(self.material.elementTypes.index(site))

        displacementMatrix = [material.displacementList(self.material, siteIndex) for siteIndex in siteIndexList] 
        delG0 = self.delG0(occupancy, charge)
        delGs_basal = ((lambda_basal + delG0)**2 / (4 * lambda_basal)) - VAB_basal
        delGs_c_direction = ((lambda_c_direction + delG0)**2 / (4 * lambda_c_direction)) - VAB_c_direction        
        k_basal = vn * np.exp(-delGs_basal / (kB * T))
        k_c_direction = vn * np.exp(-delGs_c_direction / (kB * T))
        nsteps_path = kmcsteps / stepInterval
        N_procs = N_basal + N_c_direction
        k_total = N_basal * k_basal + N_c_direction * k_c_direction
        k = np.array([k_basal, k_basal, k_basal, k_c_direction])
        k_cumsum = np.cumsum(k / k_total)
        
        # TODO: timeNdisplacement, not timeNpath
        timeNpath = np.zeros((nsteps_path+1, 4))
        # TODO: change the logic for layer selection which is specific to the hematite system
        layer = 0 if rnd.random() < 0.5 else 4
        time = 0
        displacement = 0
        for step in range(1, kmcsteps+1):
                rand = rnd.random()
                for i in range(N_procs):
                        if rand <= k_cumsum[i]:
                            # TODO: siteIndex
                            displacement += displacementMatrix[siteIndex].displacementList[layer + i]
                            time -= np.log(rnd.random()) / k_total
                            break
                if step % stepInterval == 0:
                        path_step = step / stepInterval
                        timeNpath[path_step, 0] = timeNpath[path_step-1, 0] + time
                        timeNpath[path_step, 1:] = timeNpath[path_step-1, 1:] + displacement
                        displacement = 0
                        time = 0
                layer = 0 if layer==4 else 4

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