from KineticModel import system, run, initiateSystem
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

initiateHematiteSystem = initiateSystem(hematite, hematiteNeighbors)
speciesCount = {'electron': 1}
initialOccupancy =  initiateHematiteSystem.generateRandomOccupancy(speciesCount)

neighborList = np.load('neighborList333.npy')
hematiteSystem = system(hematite, hematiteNeighbors, neighborList[()], initialOccupancy)

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
