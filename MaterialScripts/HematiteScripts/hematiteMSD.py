#!/usr/bin/env python

def hematiteMSD(shellCharges, cutE, dirPath, speciesCount, nTraj, kmcSteps, 
                stepInterval, systemSize, nStepsMSD, nDispMSD, binsize, reprTime, reprDist, outdir):

    from KineticModel import analysis
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
    
    hematiteAnalysis = analysis(hematite, speciesCount, nTraj, kmcSteps, stepInterval, 
                                systemSize, nStepsMSD, nDispMSD, binsize, reprTime, reprDist)
    
    msdAnalysisData = hematiteAnalysis.computeMSD(outdir, report=1)
    msdData = msdAnalysisData.msdData
    speciesTypes = msdAnalysisData.speciesTypes
    fileName = msdAnalysisData.fileName
    hematiteAnalysis.displayMSDPlot(msdData, speciesTypes, fileName, outdir)