from system import system, material
import numpy as np

name = 'Fe2O3'
elementTypes = ['Fe', 'O']
#sites = ['Fe']
#species = ['electron', 'empty']
species_to_sites = {'electron': ['Fe'], 'empty': ['Fe', 'O'], 'hole': ['O']}
inputFileLocation = "/Users/Viswanath/Box Sync/Visualization/Fe2O3_index_coord.txt"
index_pos = np.loadtxt(inputFileLocation)
unitcellCoords = index_pos[:, 1:]
elementTypeIndexList = index_pos[:,0]
pos_Fe = unitcellCoords[elementTypeIndexList==0, :]
charge = [+1.11, -0.74]
a = 5.038 # lattice constant along x-axis
b = 5.038 # lattice constant along y-axis
c = 13.772 # lattice constant along z-axis
alpha = 90. / 180 * np.pi # interaxial angle between b-c
beta = 90. / 180 * np.pi # lattice angle between a-c
gamma = 120. / 180 * np.pi # lattice angle between a-b
latticeParameters = [a, b, c, alpha, beta, gamma]
size = np.array([15, 15, 15])
occupancy = np.array([0, 1, 2])
hematite = material(name, elementTypes, species_to_sites, unitcellCoords, elementTypeIndexList, charge, latticeParameters)
hematiteSystem = system(hematite, occupancy, size)
#hematite = system(name, elementTypes, species_to_sites, unitcellCoords, elementTypeIndexList, charge, latticeParameters, occupancy, size)
#neighborSize = np.array([2, 1, 1])
#hematite.materialParameters()

vn = 1.85E+13 # typical frequency for nuclear motion in (1/sec)
kB = 8.617E-05 # boltzmann constant in eV/K
T = 300 # Temperature in K
lambda_basal = 1.74533 # reorganization energy in eV for basal plane
lambda_c_direction = 1.88683 # reorganization energy in eV for c-direction
VAB_basal = 0.184 # electronic coupling matrix element in eV for basal plane
VAB_c_direction = 0.028 # electronic coupling matrix element in eV for c-direction
N_basal = 3
N_c_direction = 1 
numLocalNeighborSites = 4
hello = 3
neighbor_cutoff = 3.0
hopdist_basal = 2.0
hopdist_c_direction = 2.5 
nsteps_msd = 1E+02
ndisp_msd = 1E+02
binsize = 1E+01
