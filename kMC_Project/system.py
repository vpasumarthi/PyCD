#!/usr/bin/env python
'''
code to compute msd of random walk of single electron in 3D hematite lattice structure
 switch off PBC, and 
'''

#import random as rnd
import numpy as np

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

    def __init__(self, name, elements, species_to_sites, pos, index, charge, latticeParameters):
        '''
        Return an material object whose name is *name* 
        '''
        self.name = name
        self.elements = elements
        self.species_to_sites = species_to_sites
        for key in species_to_sites:
            assert set(species_to_sites[key]) <= set(elements), 'Specified sites should be a subset of elements'
        self.pos = pos
        self.index = index
        # diff in charge of anions in the first shell different from lattice anions
        # TODO: diff in charge of cations in the first shell from lattice cations
        # TODO: introduce a method to view the material using ase atoms or other gui module
        self.charge = charge
        self.latticeParameters = latticeParameters
        
    def lattice_matrix(self, latticeParameters):
        [a, b, c, alpha, beta, gamma] = latticeParameters
        cell = np.array([[ a                , 0                , 0],
                         [ b * np.cos(gamma), b * np.sin(gamma), 0],
                         [ c * np.cos(alpha), c * np.cos(beta) , c * np.sqrt(np.sin(alpha)**2 - np.cos(beta)**2)]]) # cell matrix
        return cell
        
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
                
class run(system):
    '''
    
    '''
    
    def __init__(self, pbc=1):
        '''
        
        '''
        self.pbc = pbc
        
    def delG0(self, sites, charge):
        '''
        
        '''
        delG0 = self.elec(sites, charge)
        return delG0
    
    def elec(self, sites, charge):
        '''
        
        '''
        
'''
class analysis(object):
    '''
    
'''
    
    def __init__(self):
        '''
        
'''

    def compute_sd(self, timeNpath):
        #sec_to_ns = 1E+09
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
'''