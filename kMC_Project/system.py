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
    def __init__(self, vn, kB, T, lambda_basal, lambda_c_direction, VAB_basal, VAB_c_direction, N_basal, 
                 N_c_direction, numLocalNeighbors, neighbor_cutoff, hopdist_basal, hopdist_c_direction,
                 nsteps_msd, ndisp_msd, binsize):
        '''
        Definitions of all model parameters
        '''
        # TODO: Is it possible to define these parameters in a dictionary?
        self.vn = vn
        self.kB = kB
        self.T = T
        self.lambda_basal = lambda_basal
        self.lambda_c_direction = lambda_c_direction
        self.VAB_basal = VAB_basal
        self.VAB_c_direction = VAB_c_direction
        # TODO: differentiating between basal and c-direction is specific to the system, change basal and c-direction
        # may be the same can be termed as k1, k2, k3, k4 with first three being same rate constants
        self.N_basal = N_basal
        self.N_c_direction = N_c_direction
        self.numLocalNeighbors = numLocalNeighbors
        self.neighbor_cutoff = neighbor_cutoff
        self.hopdist_basal = hopdist_basal
        self.hopdist_c_direction = hopdist_c_direction
        self.nsteps_msd = nsteps_msd
        self.ndisp_msd = ndisp_msd
        self.binsize = binsize
    
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

    def __init__(self, name, elementTypes, species_to_sites, unitcellCoords, elementTypeIndex, charge, latticeParameters):
        '''
        Return an material object whose name is *name* 
        '''
        self.name = name
        self.elementTypes = elementTypes
        self.species_to_sites = species_to_sites
        for key in species_to_sites:
            assert set(species_to_sites[key]) <= set(elementTypes), 'Specified sites should be a subset of elements'
        self.unitcellCoords = unitcellCoords
        self.elementTypeIndex = elementTypeIndex
        # diff in charge of anions in the first shell different from lattice anions
        # TODO: diff in charge of cations in the first shell from lattice cations
        # TODO: introduce a method to view the material using ase atoms or other gui module
        self.charge = charge
        self.latticeParameters = latticeParameters
        
    def lattice_matrix(self):
        [a, b, c, alpha, beta, gamma] = self.latticeParameters
        cell = np.array([[ a                , 0                , 0],
                         [ b * np.cos(gamma), b * np.sin(gamma), 0],
                         [ c * np.cos(alpha), c * np.cos(beta) , c * np.sqrt(np.sin(alpha)**2 - np.cos(beta)**2)]]) # cell matrix
        return cell
    
    def materialParameters(self):
        numLocalNeighbors = modelParameters.numLocalNeighbors
        # TODO: nsites differs from electron to hole, so even these parameters should be arrays with size equal to 
        # len(species - 1)
        # TODO: Hardcoded value of 1 for index, should be changed accordingly so that it equals the species name
        for i, key in enumerate(self.species_to_sites):
            if key == 'empty':
                empty_index = i
        nsites = []
        for i in range(len(self.species_to_sites)):
            nsites.append()
        tot_neighbors = numLocalNeighbors * nsites
        return tot_neighbors
        
    def displacement_matrix(self, bulksize):
        # Coordinates
        sitecoords = self.generate_coords()
        bulkcoords = self.generate_coords(bulksize)
        numLocalNeighbors = modelParameters.numLocalNeighbors
        tot_neighbors = material.tot_neighbors
        neighbor_cutoff = modelParameters.neighbor_cutoff
        hopdist_basal = modelParameters.hopdist_basal
        hopdist_c_direction = modelParameters.hopdist_c_direction
        hopdist_critical  = (hopdist_basal + hopdist_c_direction) / 2 

        # Initialization
        nn_coords = [0] * tot_neighbors
        local_nn_coords = [0] * numLocalNeighbors
        displacement_matrix = np.zeros((tot_neighbors, 3)) 
        local_disp = np.zeros((numLocalNeighbors, 3)) 
        # fetch all nearest neighbor coordinates
        for site, center in enumerate(sitecoords):
                index = 0 
                for j, nn_coord in enumerate(bulkcoords):
                        displacement = nn_coord.pos - center.pos
                        hop_dist = np.linalg.norm(displacement)
                        if 0 < hop_dist <= neighbor_cutoff:
                                if hop_dist < hopdist_critical: # hop in c-direction
                                        local_nn_coords[numLocalNeighbors-1] = nn_coord
                                        local_disp[numLocalNeighbors-1] = displacement
                                else:
                                        local_nn_coords[index] = nn_coord
                                        local_disp[index] = displacement
                                        index += 1
                start_index = numLocalNeighbors * site
                end_index = start_index + numLocalNeighbors
                displacement_matrix[start_index:end_index] = local_disp
                nn_coords[start_index:end_index] = local_nn_coords        
        return displacement_matrix

class system(material):
    '''
    defines the system we are working on
    
    Attributes:
    size: An array (3 x 1) defining the system size in multiple of unit cells
    '''
    
    def __init__(self, name, elements, species_to_sites, pos, index, charge, latticeParameters, occupancy, size=np.array([10, 10, 10])):
        '''
        Return a system object whose size is *size*
        '''
        super(system, self).__init__(name, elements, species_to_sites, pos, index, charge, latticeParameters)
        self.occupancy = occupancy
        self.size = size
    
    def generate_coords(self, siteIndex, neighborSize=[1, 1, 1]):
        '''
        Subroutine to generate coordinates of specified number of unit cells around the original cell
        '''
        assert all(element > 0 for element in neighborSize), 'Input size should always be greater than 0'
        unitcellSiteCoords = self.unitcellCoords[self.elementTypeIndex == siteIndex]
        numCells = (2 * neighborSize[0] - 1) * (2 * neighborSize[1] - 1) * (2 * neighborSize[2] - 1)
        nsites = len(unitcellSiteCoords)
        coords = np.zeros((numCells * nsites, 3))
        for xIndex, xSize in enumerate(range(-neighborSize[0]+1, neighborSize[0])):
            for yIndex, ySize in enumerate(range(-neighborSize[1]+1, neighborSize[1])):
                for zIndex, zSize in enumerate(range(-neighborSize[2]+1, neighborSize[2])):
                    startIndex = (xIndex + yIndex + zIndex) * nsites
                    endIndex = startIndex + nsites
                    neighborCellSiteCoords = np.multiply(unitcellSiteCoords, [xSize, ySize, zSize])
                    coords[startIndex:endIndex, :] = neighborCellSiteCoords 
        return coords

                
class run(system):
    '''
    defines the subroutines for running Kinetic Monte Carlo and computing electrostatic interaction energies 
    '''
    def __init__(self, pbc=1):
        '''
        Returns the PBC condition of the system
        '''
        self.pbc = pbc

    def elec(self, occupancy, charge):
        '''
        Subroutine to compute the electrostatic interaction energies
        '''
        
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
        vn = modelParameters.vn
        kB = modelParameters.kB
        T = modelParameters.T
        lambda_basal = modelParameters.lambda_basal
        lambda_c_direction = modelParameters.lambda_c_direction
        VAB_basal = modelParameters.VAB_basal
        VAB_c_direction = modelParameters.VAB_c_direction
        N_basal = modelParameters.N_basal
        N_c_direction = modelParameters.N_c_direction
        # TODO: Is it self?
        displacement_matrix = material.displacement_matrix(self) 
        
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
                            # TODO: displacement matrix should be imported from the
                                displacement += displacement_matrix[layer + i]
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