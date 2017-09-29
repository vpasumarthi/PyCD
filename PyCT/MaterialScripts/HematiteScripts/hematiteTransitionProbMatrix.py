#!/usr/bin/env python
def hematiteTransitionProbMatrix(systemSize, pbc, centerSiteQuantumIndices, generateHopNeighborList, generateSpeciesSiteSDList, generateTransitionProbMatrix):
    from hematiteParameters import hematiteParameters
    from PyCT import material, neighbors
    import numpy as np
    import platform
    import os
    import pickle
    
    # Determine path for system directory    
    cwd = os.path.dirname(os.path.realpath(__file__))
    directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
    nLevelUp = 3 if platform.uname()[0]=='Linux' else 4
    systemDirectoryPath = directorySeparator.join(cwd.split(directorySeparator)[:-nLevelUp] + 
                                                  ['PyCTSimulations', 'Hematite', ('PBC' if np.all(pbc) else 'NoPBC'), 
                                                   ('SystemSize[' + ','.join(['%i' % systemSize[i] for i in range(len(systemSize))]) + ']')])

    # Determine path for input files
    inputFileDirectoryName = 'InputFiles'
    inputFileDirectoryPath = systemDirectoryPath + directorySeparator + inputFileDirectoryName
    
    # Build path for material and neighbors object files
    materialName = 'hematite'
    tailName = '.obj'
    objectFileDirectoryName = 'ObjectFiles'
    objectFileDirPath = inputFileDirectoryPath + directorySeparator + objectFileDirectoryName
    if not os.path.exists(objectFileDirPath):
        os.makedirs(objectFileDirPath)
    materialFileName = objectFileDirPath + directorySeparator + materialName + tailName
    neighborsFileName = objectFileDirPath + directorySeparator + materialName + 'Neighbors' + tailName

    # Load material object
    file_hematite = open(materialFileName, 'r')
    hematite = pickle.load(file_hematite)
    file_hematite.close()
    
    # Build neighbors object files
    hematiteNeighbors = neighbors(hematite, systemSize, pbc)
    if generateHopNeighborList:
        hematiteNeighbors.generateHematiteNeighborSEIndices(inputFileDirectoryPath)
    if generateSpeciesSiteSDList:
        hematiteNeighbors.generateSpeciesSiteSDList(centerSiteQuantumIndices, inputFileDirectoryPath)
    if generateTransitionProbMatrix:
        neighborSystemElementIndicesFilePath = inputFileDirectoryPath + directorySeparator + 'neighborSystemElementIndices.npy'
        neighborSystemElementIndices = np.load(neighborSystemElementIndicesFilePath)
        hematiteNeighbors.generateTransitionProbMatrix(neighborSystemElementIndices, inputFileDirectoryPath)