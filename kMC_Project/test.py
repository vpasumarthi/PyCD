from system import modelParameters, material, system, run
import numpy as np

kB = 8.617E-05 # boltzmann constant in eV/K
T = 300 # Temperature in K
ntraj = 1E+02
kmcsteps = int(1E+03)
stepInterval = 1E+00
nsteps_msd = 1E+02
ndisp_msd = 1E+02
binsize = 1E+01
pbc = 1

hematiteParameters = modelParameters(kB, T, ntraj, kmcsteps, stepInterval, nsteps_msd, ndisp_msd, binsize, pbc)

name = 'Fe2O3'
elementTypes = ['Fe', 'O']
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
vn = 1.85E+13 # typical frequency for nuclear motion in (1/sec)
lambda_basal = 1.74533 # reorganization energy in eV for basal plane
lambda_c_direction = 1.88683 # reorganization energy in eV for c-direction
VAB_basal = 0.184 # electronic coupling matrix element in eV for basal plane
VAB_c_direction = 0.028 # electronic coupling matrix element in eV for c-direction
N_basal = 3
N_c_direction = 1 
neighborCutoffDist = {'Fe':3.0, 'O': 4.0}
hopdist_basal = 2.971
hopdist_c_direction = 2.901 

hematite = material(name, elementTypes, species_to_sites, unitcellCoords, elementTypeIndexList,
                 charge, latticeParameters, vn, lambda_basal, lambda_c_direction, VAB_basal, VAB_c_direction, 
                 N_basal, N_c_direction, neighborCutoffDist, hopdist_basal, 
                 hopdist_c_direction)

size = np.array([15, 15, 15])
occupancy = np.array([0, 1, 2])

hematiteSystem = system(hematite, occupancy, size)

hematiteRun = run(hematiteParameters, hematite, hematiteSystem)
hematiteRun.do_kmc_steps(occupancy, charge, stepInterval, kmcsteps)
