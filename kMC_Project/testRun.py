from kineticModel import system, run
import numpy as np
import pickle
import time as timer

start_time = timer.time()

file_hematite = open('file_hematite.obj', 'r')
hematite = pickle.load(file_hematite)
file_hematite.close()

file_hematiteNeighbors = open('file_hematiteNeighbors.obj', 'r')
hematiteNeighbors = pickle.load(file_hematiteNeighbors)
file_hematiteNeighbors.close()

# TODO: Automate the choice of sites given number of electron and hole species
elementTypes = ['Fe', 'O']
speciesTypes = {'electron': 'Fe', 'empty': ['Fe', 'O'], 'hole': 'O'}

electronSiteElementTypeIndex = elementTypes.index(speciesTypes['electron'])
electronQuantumIndices = np.array([[1, 1, 1, electronSiteElementTypeIndex, elementSite] for elementSite in np.array([3])])
systemSize = np.array([3, 3, 3])
electronSiteIndices = [hematite.generateSystemElementIndex(systemSize, quantumIndex) 
                       for quantumIndex in electronQuantumIndices]
#occupancy = [['electron', np.asarray(electronSiteIndices, int)]]
occupancy = [['electron', electronSiteIndices]]
#occupancy = [['electron', electronSiteIndices], ['hole', [23, 45]]]
# dictionary neighborList is saved as an numpy array. It can be recovered by calling
# neighborList[()]
neighborList = np.load('neighborList333.npy')
hematiteSystem = system(hematite, hematiteNeighbors, neighborList[()], occupancy)

# TODO: Neighbor List has to be generated automatically within the code.

T = 300 # Temperature in K
nTraj = 1E+00
kmcSteps = 1E+03
stepInterval = 1E+01
gui = 0
kB = 8.617E-05 # Boltzmann constant in eV/K

hematiteRun = run(hematite, hematiteSystem, T, nTraj, kmcSteps, stepInterval, gui, kB)
trajectoryData = hematiteRun.doKMCSteps(randomSeed=2)
np.save('trajectoryData_1electron_PBC_1e03KMCSteps_1e02PathSteps_1Traj.npy', trajectoryData)
print("--- %s seconds ---" % (timer.time() - start_time))