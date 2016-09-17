from kineticModel import analysis, plot
import numpy as np

trajectoryData = np.load('trajectoryData_1electron_PBC_1e03KMCSteps_1e02PathSteps_1Traj.npy')

nStepsMSD = 5E+01
nDispMSD = 5E+01
binsize = 10E+00
maxBinSize = 1 # ns
reprTime = 'ns'
reprDist = 'Angstrom'

hematiteAnalysis = analysis(trajectoryData[()], nStepsMSD, nDispMSD, binsize, maxBinSize, reprTime, reprDist)

timeArray = trajectoryData[()].timeArray
unwrappedPositionArray = trajectoryData[()].unwrappedPositionArray
msdAnalysisData = hematiteAnalysis.computeMSD(timeArray, unwrappedPositionArray)

hematitePlot = plot(msdAnalysisData)
hematitePlot.displayMSDPlot()