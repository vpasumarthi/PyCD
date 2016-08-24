#!/usr/bin/env python
'''
code to compute msd of random walk of single electron in 3D hematite lattice structure
 switch off PBC, and 
'''

import random as rnd
import numpy as np
import matplotlib.cm as cm # might be unnecessary

class material(object):
    '''
    defines the material working on
    
    Attributes:
        name: A string representing the material name
        size: An array (3 x 1) defining the system size in multiple of unit cells
        
    '''

    def __init__(self, name, species, sites, occupancy, charge):
        '''
        Return an material object whose name is *name* 
        '''
        self.name = name
        self.species = species
        self.sites = sites
        self.occupancy = occupancy
        self.charge = charge

class system(material):
    '''
    defines the system we are working on
    '''
    
    def __init__(self, size=np.array([10, 10, 10])):
        '''
        Return a system object whose size is *size*
        '''
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
    
    def elec(self, sites, charge):
        '''
        
        '''
        

class analysis(object):
    '''
    
    '''
    
    def __init__(self):
        '''
        
        '''
        
    def msd(self):
        '''
        
        '''
    
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
        plt.plot(msd[:,0], msd[:,1])
        plt.xlabel('Time (ns)')
        plt.ylabel('MSD (Angstrom**2)')
        plt.show()

# Input Variables
x = 9
y = 9
z = 4

PBC = 1
n_traj = int(1e2) # number of trajectories
kmcsteps = int(1e3) # number of kmcsteps
stepInterval = int(1e2) # interval between trajectory is recorded
nsteps_msd = int(5e0) # number of timesteps over which msd computed over

sys_size = np.array([x, y, z]) + np.ones(3) * (0 if PBC else 2)
nsteps_path = kmcsteps / stepInterval
ndisp_msd = nsteps_path - nsteps_msd  # number of displacements used to compute average in the msd calculation, constant for all timesteps

sec_to_ns = 1E+09
outputdatafile = 'msd_randomwalk_3D_hematite_' + ('%1.0e' % ndisp_msd) + 'disp_' + ('%1.0e' % n_traj) + 'traj.txt'
figname = 'msd_randomwalk_3D_hematite_' + ('%1.0e' % ndisp_msd) + 'disp_' + ('%1.0e' % n_traj) + 'traj_newcoord.jpg'

output = np.zeros((nsteps_msd+1,2))

displacement_matrix = np.array([[ -2.8100000000000,  0.7520000000000, -0.6060000000000],
                                [  2.0568913437515,  2.0568913437515, -0.6060000000000],
                                [  0.7520000000000, -2.8100000000000, -0.6060000000000],
                                [  0.0000000000000,  0.0000000000000, -2.9010000000000],
                                [  2.8100000000000, -0.7520000000000,  0.6060000000000],
                                [ -2.0568913437515, -2.0568913437515,  0.6060000000000],
                                [ -0.7520000000000,  2.8100000000000,  0.6060000000000],
                                [  0.0000000000000,  0.0000000000000,  2.9010000000000]]) # Angstroms


lattice_cell = np.array([[  5.03800,   0.00000,   0.00000],
                         [ -2.51900,   4.36304,   0.00000],
                         [ -0.00000,  -0.00000,  13.77200]])

#k_basal = 3.20E+09 / 3 # literature
#k_c_direction = 6.50E+05 # literature

vn = 1.85E+13 # typical frequency for nuclear motion in (1/sec)
kB = 8.617E-05 # boltzmann constant in eV/K
T = 300 # Temperature in K
lambda_basal = 1.74533 # reorganization energy in eV for basal plane
lambda_c_direction = 1.88683 # reorganization energy in eV for c-direction
VAB_basal = 0.184 # electronic coupling matrix element in eV for basal plane
VAB_c_direction = 0.028 # electronic coupling matrix element in eV for c-direction

# here is where the routine for electrostatic interactions is called
delG0 = 0.0 # free energy of reaction in eV, won't be zero for multiple electron system
delGs_basal = ((lambda_basal + delG0)**2 / (4 * lambda_basal)) - VAB_basal
delGs_c_direction = ((lambda_c_direction + delG0)**2 / (4 * lambda_c_direction)) - VAB_c_direction
#k_basal = n_basal * vn * np.exp(-delGs_basal/ (kB* T)) 
#k_c_direction = n_c_direction * vn * np.exp(-delGs_c_direction / (kB * T)) 

k_basal = vn * np.exp(-delGs_basal / (kB * T))
k_c_direction = vn * np.exp(-delGs_c_direction / (kB * T))

N_basal = 3
N_c_direction = 1

N_procs = N_basal + N_c_direction
k_total = N_basal * k_basal + N_c_direction * k_c_direction

k = np.array([k_basal, k_basal, k_basal, k_c_direction])
k_cumsum = np.cumsum(k / k_total)

p_basal = N_basal * k_basal / k_total
p_c_direction = N_c_direction * k_c_direction / k_total


def generate_path(nsteps_path):
        timeNpath = np.zeros((nsteps_path+1, 4))
        layer = 0 if rnd.random() < 0.5 else 4
        time = 0
        displacement = 0
        for step in range(1, kmcsteps+1):
                rand = rnd.random()
                for i in range(N_procs):
                        if rand <= k_cumsum[i]:
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
        return timeNpath

def compute_sd(timeNpath):
        time = timeNpath[:, 0] * sec_to_ns
        path = timeNpath[:, 1:]
        msd = np.zeros(nsteps_msd+1)
        sd_array = np.zeros((nsteps_msd * ndisp_msd, 2))
        for timestep in range(1, nsteps_msd+1):
                sum_sd_timestep = 0
                for step in range(ndisp_msd):
                        sd_timestep = sum((path[step + timestep] - path[step])**2)
                        sd_array[(timestep-1) * ndisp_msd + step, 0] = time[step + timestep] - time[step]
                        sd_array[(timestep-1) * ndisp_msd + step, 1] = sd_timestep
                        sum_sd_timestep += sd_timestep
        return sd_array

def compute_msd(sd_array):
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

binsize = 1 # ns
sum_msd = np.zeros((nsteps_msd+1,2))
colors = iter(cm.rainbow(np.linspace(0, 1, n_traj)))
plt.figure(1)
for traj in range(n_traj):
        timeNpath = generate_path(nsteps_path)
        sd_traj = compute_sd(timeNpath)
        msd_traj = compute_msd(sd_traj)
        len_diff = len(msd_traj) - len(sum_msd)
        if len_diff == 0:
                sum_msd += msd_traj
        elif len_diff < 0:
                sum_msd = sum_msd[:len(msd_traj),:]
                sum_msd += msd_traj
        else:
                sum_msd += msd_traj[:len(sum_msd),:]

msd = np.zeros((len(sum_msd)+1,2))
msd[1:,:] = sum_msd / n_traj


#plt.savefig(figname)