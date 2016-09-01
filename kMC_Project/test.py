from system import modelParameters, material, system, run
import numpy as np

T = 300 # Temperature in K
ntraj = 1E+02
kmcsteps = int(1E+03)
stepInterval = 1E+00
nsteps_msd = 1E+02
ndisp_msd = 1E+02
binsize = 1E+01
systemSize = np.array([10, 10, 10])
pbc = 1
gui = 0
kB = 8.617E-05 # boltzmann constant in eV/K

hematiteParameters = modelParameters(T, ntraj, kmcsteps, stepInterval, nsteps_msd, ndisp_msd, binsize, systemSize, 
                                     pbc, gui, kB)

name = 'Fe2O3'
elementTypes = ['Fe', 'O']
speciesTypes = {'electron': ['Fe'], 'empty': ['Fe', 'O'], 'hole': ['O']}
inputFileLocation = "/Users/Viswanath/Box Sync/Visualization/Fe2O3_index_coord.txt"
index_pos = np.loadtxt(inputFileLocation)
unitcellCoords = index_pos[:, 1:]
elementTypeIndexList = index_pos[:,0]
pos_Fe = unitcellCoords[elementTypeIndexList==0, :]
chargeTypes = {'Fe': +1.11, 'O': -0.74}
a = 5.038 # lattice constant along x-axis
b = 5.038 # lattice constant along y-axis
c = 13.772 # lattice constant along z-axis
alpha = 90. / 180 * np.pi # interaxial angle between b-c
beta = 90. / 180 * np.pi # lattice angle between a-c
gamma = 120. / 180 * np.pi # lattice angle between a-b
latticeParameters = [a, b, c, alpha, beta, gamma]
vn = 1.85E+13 # typical frequency for nuclear motion in (1/sec)
lambdaValues = {'Fe-Fe': [1.74533, 1.88683]} # reorganization energy in eV for basal plane, c-direction
VAB = {'Fe-Fe': [0.184, 0.028]} # electronic coupling matrix element in eV for basal plane, c-direction
neighborCutoffDist = {'Fe-Fe': [2.971, 2.901], 'O-O': [4.0], 'E': [20.0], 'tol': [0.01]} # Basal: 2.971, C: 2.901

hematite = material(name, elementTypes, speciesTypes, unitcellCoords, elementTypeIndexList, chargeTypes, 
                    latticeParameters, vn, lambdaValues, VAB, neighborCutoffDist)

occupancy = np.array([0, 1, 2])
hematiteSystem = system(hematiteParameters, hematite, occupancy)

elementTypeIndices = range(len(elementTypes))
bulkSites = hematiteSystem.generateCoords(elementTypeIndices, systemSize)
bulkSiteCoords = bulkSites.cellCoordinates
bulkSystemElementIndices = bulkSites.systemElementIndexList
nElementSites = len(np.in1d(elementTypeIndexList, elementTypeIndices).nonzero()[0])
startIndex = np.prod(systemSize[1:]) * nElementSites
endIndex = startIndex + nElementSites
centerSiteCoords = bulkSites.cellCoordinates[startIndex:endIndex]
centerSystemElementIndices = bulkSites.systemElementIndexList[startIndex:endIndex]
cutoffDist = 3.0
neighbors = hematiteSystem.neighborSites(bulkSiteCoords, bulkSystemElementIndices, centerSiteCoords, centerSystemElementIndices, cutoffDist)
print neighbors.displacementList

'''
hematiteRun = run(hematiteParameters, hematite, hematiteSystem)
hematiteRun.do_kmc_steps(occupancy, chargeTypes, stepInterval, kmcsteps)
'''