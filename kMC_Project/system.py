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
    def __init__(self, kB, T, ntraj, kmcsteps, stepInterval, nsteps_msd, ndisp_msd, binsize, pbc):
        '''
        Definitions of all model parameters
        '''
        # TODO: Is it necessary/better to define these parameters in a dictionary?
        self.kB = kB
        self.T = T
        self.ntraj = ntraj
        self.kmcsteps = kmcsteps
        self.stepInterval = stepInterval
        self.nsteps_msd = nsteps_msd
        self.ndisp_msd = ndisp_msd
        
        self.binsize = binsize
        self.pbc = pbc
    
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
    
    def __init__(self, name, elementTypes, species_to_sites, unitcellCoords, elementTypeIndexList,
                 charge, latticeParameters, vn, lambda_basal, lambda_c_direction, VAB_basal, VAB_c_direction, 
                 N_basal, N_c_direction, neighborCutoffDist, hopdist_basal, 
                 hopdist_c_direction):
        '''
        Return an material object whose name is *name* 
        '''
        # TODO: introduce a method to view the material using ase atoms or other gui module
        self.name = name
        self.elementTypes = elementTypes
        self.species_to_sites = species_to_sites
        for key in species_to_sites:
            assert set(species_to_sites[key]) <= set(elementTypes), 'Specified sites should be a subset of elements'
        self.unitcellCoords = unitcellCoords
        self.elementTypeIndexList = elementTypeIndexList
        # diff in charge of anions in the first shell different from lattice anions
        # TODO: diff in charge of cations in the first shell from lattice cations
        self.charge = charge
        self.latticeParameters = latticeParameters
        self.vn = vn
        self.lambda_basal = lambda_basal
        self.lambda_c_direction = lambda_c_direction
        self.VAB_basal = VAB_basal
        self.VAB_c_direction = VAB_c_direction
        # TODO: differentiating between basal and c-direction is specific to the system, change basal and c-direction
        # may be the same can be termed as k1, k2, k3, k4 with first three being same rate constants
        self.N_basal = N_basal
        self.N_c_direction = N_c_direction 
        #self.numLocalNeighborSites = numLocalNeighborSites
        self.neighborCutoffDist = neighborCutoffDist
        self.hopdist_basal = hopdist_basal
        self.hopdist_c_direction = hopdist_c_direction
   
    def lattice_matrix(self):
        [a, b, c, alpha, beta, gamma] = self.latticeParameters
        cell = np.array([[ a                , 0                , 0],
                         [ b * np.cos(gamma), b * np.sin(gamma), 0],
                         [ c * np.cos(alpha), c * np.cos(beta) , c * np.sqrt(np.sin(alpha)**2 - np.cos(beta)**2)]]) # cell matrix
        return cell

    def nSites(self, siteIndex):
        '''
        Returns number of sites available in a unit cell
        '''
        # TODO: Returns nSites for a siteIndex
        nSites = len(np.where(self.elementTypeIndexList == siteIndex)[0])
        '''
        SiteList = []
        for key in self.species_to_sites:
            if key != 'empty':
                SiteList.extend(self.species_to_sites[key])
        SiteList = list(set(SiteList))
        nSites = np.zeros(len(self.elementTypes))
        for elementTypeIndex, elementType in enumerate(self.elementTypes):
            if elementType in SiteList:
                nSites[elementTypeIndex] = len(np.where(self.elementTypeIndexList == elementTypeIndex)[0])
        '''
        return nSites

    def generate_coords(self, siteIndex, neighborSize=[1, 1, 1]):
        '''
        Subroutine to generate coordinates of specified number of unit cells around the original cell
        '''
        assert all(element > 0 for element in neighborSize), 'Input size should always be greater than 0'
        unitcellSiteCoords = self.unitcellCoords[self.elementTypeIndexList == siteIndex]
        numCells = (2 * neighborSize[0] - 1) * (2 * neighborSize[1] - 1) * (2 * neighborSize[2] - 1)
        nSites = len(unitcellSiteCoords)
        coords = np.zeros((numCells * nSites, 3))
        startIndex = 0
        endIndex = nSites
        for xSize in range(-neighborSize[0]+1, neighborSize[0]):
            for ySize in range(-neighborSize[1]+1, neighborSize[1]):
                for zSize in range(-neighborSize[2]+1, neighborSize[2]):
                    neighborCellSiteCoords = unitcellSiteCoords + np.dot([xSize, ySize, zSize], 
                                                                         self.lattice_matrix())
                    coords[startIndex:endIndex, :] = neighborCellSiteCoords 
                    startIndex += nSites
                    endIndex += nSites
        return coords

    def numLocalNeighborSites(self, siteIndex, neighborCutoffDist, bulksize=[2, 2, 2]):
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

class returnCoords(object):
    def __init__(self, coords, offset, siteIndexList):
        self.coords = coords
        self.offset = offset
        self.siteIndexList = siteIndexList

class returnValues(object):
    '''
    Returns the values of displacement list and respective coordinates in an object
    '''
    def __init__(self, displacementList, nn_coords):
        self.displacementList = displacementList
        self.nn_coords = nn_coords

class system(object):
    '''
    defines the system we are working on
    
    Attributes:
    size: An array (3 x 1) defining the system size in multiple of unit cells
    '''
    
    def __init__(self, material, occupancy, size=np.array([10, 10, 10])):
        '''
        Return a system object whose size is *size*
        '''
        self.material = material
        self.occupancy = occupancy
        self.size = size

    def config(self):
        '''
        Generates the configuration array for the system 
        '''
        

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
        
    def compute_sd(self, timeNpath):
        '''
        Subroutine to compute the squared displacement of the trajectories
        '''
        nsteps_msd = modelParameters.nsteps_msd
        ndisp_msd = modelParameters.ndisp_msd
        sec_to_ns = 1E+09
        time = timeNpath[:, 0] * sec_to_ns
        path = timeNpath[:, 1:]
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