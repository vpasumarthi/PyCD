from kineticModel import modelParameters, analysis, plot
import numpy as np

T = 300 # Temperature in K
nTraj = 2E+00
kmcSteps = 1E+03
stepInterval = 1E+00
nStepsMSD = 5E+02
nDispMSD = 5E+02
binsize = 5E+00
maxBinSize = 1 # ns
systemSize = np.array([3, 3, 3])
pbc = [1, 1, 1]
gui = 0
kB = 8.617E-05 # Boltzmann constant in eV/K
reprTime = 'ns'
reprDist = 'Angstrom'

hematiteParameters = modelParameters(T, nTraj, kmcSteps, stepInterval, nStepsMSD, nDispMSD, binsize, maxBinSize, 
                                     systemSize, pbc, gui, kB, reprTime, reprDist)

trajectoryData = np.load('trajectoryData_1electron.npy')
hematiteAnalysis = analysis(hematiteParameters, trajectoryData[()])
timeArray = trajectoryData[()].timeArray
unwrappedPositionArray = trajectoryData[()].unwrappedPositionArray
msdData = hematiteAnalysis.computeMSD(timeArray, unwrappedPositionArray)
#print msdData

hematitePlot = plot(msdData)
hematitePlot.displayMSDPlot()