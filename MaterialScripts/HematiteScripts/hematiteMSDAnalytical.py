#!/usr/bin/env python

def hematiteMSDAnalytical(systemSize, pbc, centerSiteQuantumIndices, analyticalTFinal, analyticalTimeInterval, nDim, 
                          speciesCount, nTraj, tFinal, timeInterval, msdTFinal, trimLength, displayErrorBars, reprTime, reprDist):

    from KineticModel import material, neighbors, analysis
    import numpy as np
    import os
    import platform
    import pickle
    
    # Determine path for system directory    
    cwd = os.path.dirname(os.path.realpath(__file__))
    directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
    nLevelUp = 3 if platform.uname()[0]=='Linux' else 4
    systemDirectoryPath = directorySeparator.join(cwd.split(directorySeparator)[:-nLevelUp] + 
                                                  ['KineticModelSimulations', 'Hematite', ('PBC' if np.all(pbc) else 'NoPBC'), 
                                                   ('SystemSize[' + ','.join(['%i' % systemSize[i] for i in range(len(systemSize))]) + ']')])

    # Determine path for neighbor list directories
    inputFileDirectoryName = 'InputFiles'
    inputFileDirectoryPath = systemDirectoryPath + directorySeparator + inputFileDirectoryName
    
    # Determine path for analytical analysis directories
    analysisDirectoryName = 'AnalysisFiles'
    analysisDirectoryPath = systemDirectoryPath + directorySeparator + analysisDirectoryName
    analyticalAnalysisDirectoryName = 'AnalyticalAnalysis'
    analyticalAnalysisDirectoryPath = analysisDirectoryPath + directorySeparator + analyticalAnalysisDirectoryName
    if not os.path.exists(analyticalAnalysisDirectoryPath):
        os.makedirs(analyticalAnalysisDirectoryPath)
    
    # Build path for material and neighbors object files
    materialName = 'hematite'
    tailName = '.obj'
    directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
    objectFileDirectoryName = 'ObjectFiles'
    objectFileDirPath = inputFileDirectoryPath + directorySeparator + objectFileDirectoryName
    materialFileName = objectFileDirPath + directorySeparator + materialName + tailName
    
    # Load material object
    file_hematite = open(materialFileName, 'r')
    hematite = pickle.load(file_hematite)
    file_hematite.close()
    
    transitionProbMatrixFileName = 'transitionProbMatrix.npy'
    transitionProbMatrixFilePath = inputFileDirectoryPath + directorySeparator + transitionProbMatrixFileName
    transitionProbMatrix = np.load(transitionProbMatrixFilePath)
    hematiteNeighbors = neighbors(hematite, systemSize, pbc)
    speciesSiteSDListFileName = 'speciesSiteSDList.npy'
    speciesSiteSDListFilePath = inputFileDirectoryPath + directorySeparator + speciesSiteSDListFileName
    speciesSiteSDList = np.load(speciesSiteSDListFilePath)
    hematiteNeighbors.generateMSDAnalyticalData(transitionProbMatrix, speciesSiteSDList, centerSiteQuantumIndices, analyticalTFinal, analyticalTimeInterval, analyticalAnalysisDirectoryPath)
    
    hematiteAnalysis = analysis(hematite, nDim, systemSize, speciesCount, nTraj, tFinal, timeInterval, msdTFinal, trimLength, reprTime, reprDist)
    stdData = None
    speciesTypes = ['electron']
    fileName = '%1.2Ens' % analyticalTFinal
    MSDAnalyticalDataFileName = 'MSD_Analytical_Data_' + fileName + '.dat'
    MSDAnalyticalDataFilePath = analyticalAnalysisDirectoryPath + directorySeparator + MSDAnalyticalDataFileName
    msdData = np.loadtxt(MSDAnalyticalDataFilePath)
    
    msdAnalysisData = hematiteAnalysis.generateMSDPlot(msdData, stdData, displayErrorBars, speciesTypes, fileName, analyticalAnalysisDirectoryPath)
