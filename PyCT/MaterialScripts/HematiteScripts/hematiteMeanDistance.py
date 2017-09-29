#!/usr/bin/env python

def hematiteMeanDistance(trajectoryDataFileName, shellCharges, cutE, dirPath, systemSize, speciesCount, nTraj, kmcSteps, 
                        stepInterval, nStepsMSD, nDispMSD, binsize, reprTime, reprDist, outdir):

    from PyCT import analysis
    import numpy as np
    import pickle
    import platform
    
    materialName = 'hematite'
    tailName = '_E' + str(cutE) + '.obj'
    directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
    objectFileDirectoryName = 'ObjectFiles'
    objectFileDirPath = dirPath + directorySeparator + objectFileDirectoryName
    materialFileName = objectFileDirPath + directorySeparator + materialName + tailName
    
    file_hematite = open(materialFileName, 'r')
    hematite = pickle.load(file_hematite)
    file_hematite.close()
    
    trajectoryData = np.load(trajectoryDataFileName)[()]
    hematiteAnalysis = analysis(hematite, trajectoryData, speciesCount, nTraj, kmcSteps, stepInterval, 
                                systemSize, nStepsMSD, nDispMSD, binsize, reprTime, reprDist)
    mean = 0
    hematiteAnalysis.meanDistance(outdir, mean, plot=1, report=1)