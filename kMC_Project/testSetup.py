from kineticModel import modelParameters, material, neighbors
import numpy as np

T = 300 # Temperature in K
nTraj = 2E+00
kmcSteps = 1E+02
stepInterval = 1E+00
nStepsMSD = 5E+01
nDispMSD = 5E+01
binsize = 1E+00
maxBinSize = 1 # ns
systemSize = np.array([3, 3, 3])
pbc = [1, 1, 1]
gui = 0
kB = 8.617E-05 # Boltzmann constant in eV/K
reprTime = 'ns'
reprDist = 'Angstrom'

hematiteParameters = modelParameters(T, nTraj, kmcSteps, stepInterval, nStepsMSD, nDispMSD, binsize, maxBinSize, 
                                     systemSize, pbc, gui, kB, reprTime, reprDist)

name = 'Fe2O3'
elementTypes = ['Fe', 'O']
speciesTypes = {'electron': ['Fe'], 'empty': ['Fe', 'O'], 'hole': ['O']}
inputFileLocation = "/Users/Viswanath/Box Sync/Visualization/Fe2O3_index_coord.txt"
index_pos = np.loadtxt(inputFileLocation)
unitcellCoords = index_pos[:, 1:]
elementTypeIndexList = index_pos[:,0]
pos_Fe = unitcellCoords[elementTypeIndexList==0, :]
chargeTypes = {'Fe': +1.11, 'O': -0.74, 'Fe0': +1.5, 'Fe:O': [-0.8, -0.5]}
#chargeTypes = [['Fe', +1.11], ['O', -0.74], ['Fe:O', [-0.8, -0.5]]]
a = 5.038 # lattice constant along x-axis
b = 5.038 # lattice constant along y-axis
c = 13.772 # lattice constant along z-axis
alpha = 90. / 180 * np.pi # interaxial angle between b-c
beta = 90. / 180 * np.pi # lattice angle between a-c
gamma = 120. / 180 * np.pi # lattice angle between a-b
latticeParameters = [a, b, c, alpha, beta, gamma]
vn = 1.85E+13 # typical frequency for nuclear motion in (1/sec)
lambdaValues = {'Fe:Fe': [1.74533, 1.88683]} # reorganization energy in eV for basal plane, c-direction
#lambdaValues = ['Fe:Fe', [1.74533, 1.88683]] # reorganization energy in eV for basal plane, c-direction
VAB = {'Fe:Fe': [0.184, 0.028]} # electronic coupling matrix element in eV for basal plane, c-direction
#VAB = ['Fe:Fe', [0.184, 0.028]] # electronic coupling matrix element in eV for basal plane, c-direction
neighborCutoffDist = {'Fe:Fe': [2.971, 2.901], 'O:O': [2.670, 2.775, 2.887, 3.035], 'Fe:O': [1.946, 2.116], 'E': [5.0]} # Basal: 2.971, C: 2.901
neighborCutoffDistTol = 0.01
elementTypeDelimiter = ':'
# TODO: Value for hematite might differ
epsilon0 = 8.854E-12 # vacuum permittivity in F.m-1

hematite = material(name, elementTypes, speciesTypes, unitcellCoords, elementTypeIndexList, chargeTypes, 
                    latticeParameters, vn, lambdaValues, VAB, neighborCutoffDist, neighborCutoffDistTol, 
                    elementTypeDelimiter, epsilon0)

hematiteNeighborList = neighbors(hematiteParameters, hematite)

neighborList = hematiteNeighborList.generateNeighborList()
np.save('neighborList333.npy', neighborList)
