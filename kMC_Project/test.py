from system import modelParameters, material, system, run, analysis
import numpy as np

T = 300 # Temperature in K
nTraj = 2E+00
kmcSteps = 1E+02
stepInterval = 1E+00
nStepsMSD = 5E+00
nDispMSD = 5E+00
binsize = 1E+00
maxBinSize = 1 # ns
systemSize = np.array([3, 3, 3])
pbc = 1
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
neighborCutoffDist = {'Fe:Fe': [2.971, 2.901], 'O:O': [4.0], 'Fe:O': [1.946, 2.116], 'E': [20.0]} # Basal: 2.971, C: 2.901
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
# TODO: Neighbor List has to be generated automatically within the code.
hematiteSystem.generateNeighborList()
#print hematiteSystem.neighborList['E'][0].systemElementIndexMap
#print hematiteSystem.config(occupancy)

hematiteRun = run(hematiteParameters, hematite, hematiteSystem)
timeNpath = hematiteRun.doKMCSteps(randomSeed=2)

hematiteAnalysis = analysis(hematiteParameters, timeNpath)
msdData = hematiteAnalysis.computeMSD(timeNpath)