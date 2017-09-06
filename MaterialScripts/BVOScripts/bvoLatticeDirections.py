#!/usr/bin/env python

def bvoLatticeDirections(systemSize, pbc, cutoffDistKey, cutoff, base, prec, nDim):

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
    file_material = open(materialFileName, 'r')
    material = pickle.load(file_material)
    file_material.close()
    
    materialNeighbors = neighbors(material, systemSize, pbc)
    latticeDirectionList = materialNeighbors.generateLatticeDirections(cutoffDistKey, cutoff, base, prec, latticeDirectionsDirectoryPath)
