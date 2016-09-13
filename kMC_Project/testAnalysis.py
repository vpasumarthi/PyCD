from system import modelParameters, analysis, plot
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
pbc = 1
gui = 0
kB = 8.617E-05 # Boltzmann constant in eV/K
reprTime = 'ns'
reprDist = 'Angstrom'

hematiteParameters = modelParameters(T, nTraj, kmcSteps, stepInterval, nStepsMSD, nDispMSD, binsize, maxBinSize, 
                                     systemSize, pbc, gui, kB, reprTime, reprDist)

timeNpath = np.load('timeNpath.npy')
hematiteAnalysis = analysis(hematiteParameters, timeNpath)
msdData = hematiteAnalysis.computeMSD(timeNpath)
#print msdData

hematitePlot = plot(msdData)
hematitePlot.displayMSDPlot()