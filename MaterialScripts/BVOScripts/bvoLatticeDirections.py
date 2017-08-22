#!/usr/bin/env python

def bvoLatticeDirections(systemSize, pbc, cutoffDistKey, cutoff, nDigits, nDim):

    from KineticModel import material, neighbors
    import numpy as np
    import os
    import platform
    import pickle
    
    # Determine path for system directory    
    cwd = os.path.dirname(os.path.realpath(__file__))
    directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
    nLevelUp = 3 if platform.uname()[0]=='Linux' else 4
    systemDirectoryPath = directorySeparator.join(cwd.split(directorySeparator)[:-nLevelUp] + 
                                                  ['KineticModelSimulations', 'BVO', ('PBC' if np.all(pbc) else 'NoPBC'), 
                                                   ('SystemSize[' + ','.join(['%i' % systemSize[i] for i in range(len(systemSize))]) + ']')])

    # Determine path for neighbor list directories
    inputFileDirectoryName = 'InputFiles'
    inputFileDirectoryPath = systemDirectoryPath + directorySeparator + inputFileDirectoryName
    
    # Determine path for analytical analysis directories
    analysisDirectoryName = 'AnalysisFiles'
    analysisDirectoryPath = systemDirectoryPath + directorySeparator + analysisDirectoryName
    latticeDirectionsDirectoryName = 'LatticeDirections'
    latticeDirectionsDirectoryPath = analysisDirectoryPath + directorySeparator + latticeDirectionsDirectoryName
    if not os.path.exists(latticeDirectionsDirectoryPath):
        os.makedirs(latticeDirectionsDirectoryPath)
    
    # Build path for material and neighbors object files
    materialName = 'bvo'
    tailName = '.obj'
    directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
    objectFileDirectoryName = 'ObjectFiles'
    objectFileDirPath = inputFileDirectoryPath + directorySeparator + objectFileDirectoryName
    materialFileName = objectFileDirPath + directorySeparator + materialName + tailName
    
    # Load material object
    file_hematite = open(materialFileName, 'r')
    hematite = pickle.load(file_hematite)
    file_hematite.close()
    
    hematiteNeighbors = neighbors(hematite, systemSize, pbc)
    latticeDirectionList = hematiteNeighbors.generateLatticeDirections(cutoffDistKey, cutoff, nDigits, latticeDirectionsDirectoryPath)