from system import modelParameters, material, system, run
import numpy as np

T = 300 # Temperature in K
nTraj = 2E+00
kmcSteps = 1E+02
stepInterval = 1E+00
nStepsMSD = 5E+01
nDispMSD = 5E+01
binsize = 1E+00
maxBinSize = 1 # ns
systemSize = np.array([9, 9, 4])
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
#neighborCutoffDist = {'Fe:Fe': [2.971, 2.901], 'O:O': [4.0], 'Fe:O': [1.946, 2.116], 'E': [10.0]} # Basal: 2.971, C: 2.901
neighborCutoffDist = {'Fe:Fe': [2.971, 2.901], 'Fe:O': [1.946, 2.116]}#, 'E': [10.0]} # Basal: 2.971, C: 2.901
neighborCutoffDistTol = 0.01
elementTypeDelimiter = ':'
# TODO: Value for hematite might differ
epsilon0 = 8.854E-12 # vacuum permittivity in F.m-1

hematite = material(name, elementTypes, speciesTypes, unitcellCoords, elementTypeIndexList, chargeTypes, 
                    latticeParameters, vn, lambdaValues, VAB, neighborCutoffDist, neighborCutoffDistTol, 
                    elementTypeDelimiter, epsilon0)

electronSiteElementTypeIndex = elementTypes.index(speciesTypes['electron'][0])
# TODO: Automate the choice of sites given number of electron and hole species
electronQuantumIndices = np.array([[1, 1, 1, electronSiteElementTypeIndex, elementSite] for elementSite in np.array([3, 8])])
electronSiteIndices = [hematite.generateSystemElementIndex(systemSize, quantumIndex) 
                       for quantumIndex in electronQuantumIndices]
occupancy = [['electron', np.asarray(electronSiteIndices, int)]]

hematiteSystem = system(hematiteParameters, hematite, occupancy)

localSystemSize = np.array([3, 3, 3])
centerSiteElementTypeIndex = 0
elementIndex = 1
neighborSiteElementTypeIndex = 1
localBulkSites = hematite.generateSites(range(len(elementTypes)), localSystemSize)
centerSiteIndices = [hematite.generateSystemElementIndex(localSystemSize, np.array([1, 1, 1, centerSiteElementTypeIndex, elementIndex]))] 
# for elementIndex in range(hematite.nElements[centerSiteElementTypeIndex])]
print sorted(centerSiteIndices)
neighborSiteIndices = [hematite.generateSystemElementIndex(localSystemSize, np.array([xSize, ySize, zSize, neighborSiteElementTypeIndex, elementIndex])) 
                       for xSize in range(localSystemSize[0]) for ySize in range(localSystemSize[1]) 
                       for zSize in range(localSystemSize[2]) 
                       for elementIndex in range(hematite.nElements[neighborSiteElementTypeIndex])]
print sorted(neighborSiteIndices)
cutoffDistLimits = [-1.1, 120.0]
cutoffDistKey = 'Fe:O'
print hematiteSystem.neighborSites(localBulkSites, centerSiteIndices, neighborSiteIndices, cutoffDistLimits, cutoffDistKey)

'''
# TODO: Neighbor List has to be generated automatically within the code.
hematiteSystem.generateNeighborList()
#print hematiteSystem.neighborList['E'][0].systemElementIndexMap
#print hematiteSystem.config(occupancy)

hematiteRun = run(hematiteParameters, hematite, hematiteSystem)
timeNpath = hematiteRun.doKMCSteps(randomSeed=2)
np.save('timeNpath.npy', timeNpath)
'''