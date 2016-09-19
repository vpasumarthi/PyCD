from kineticModel import material, neighbors
import numpy as np
import pickle

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
emptySpeciesType = 'empty'
# TODO: Value for hematite might differ
epsilon0 = 8.854E-12 # vacuum permittivity in F.m-1

hematite = material(name, elementTypes, speciesTypes, unitcellCoords, elementTypeIndexList, chargeTypes, 
                    latticeParameters, vn, lambdaValues, VAB, neighborCutoffDist, neighborCutoffDistTol, 
                    elementTypeDelimiter, emptySpeciesType, epsilon0)

print hematite.hopElementTypes
'''
file_hematite = open('file_hematite.obj', 'w')
pickle.dump(hematite, file_hematite)
file_hematite.close()

systemSize = np.array([3, 3, 3])
pbc = np.array([1, 1, 1])
hematiteNeighbors = neighbors(hematite, systemSize, pbc)

file_hematiteNeighbors = open('file_hematiteNeighbors.obj', 'w')
pickle.dump(hematiteNeighbors, file_hematiteNeighbors)
file_hematiteNeighbors.close()

neighborList = hematiteNeighbors.generateNeighborList()
np.save('neighborList333.npy', neighborList)
'''