#!/usr/bin/env python
def bvoSetup(systemSize, pbc, parentCutoff, extractCutoff, replaceExistingObjectFiles, replaceExistingNeighborList):
    """Prepare material class object file, neighborlist and saves to the provided destination path"""
    from bvoParameters import bvoParameters
    from KineticModel import material, neighbors
    import numpy as np
    import platform
    import os
    
    # Determine path for system directory    
    cwd = os.path.dirname(os.path.realpath(__file__))
    directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
    nLevelUp = 3 if platform.uname()[0]=='Linux' else 4
    systemDirectoryPath = directorySeparator.join(cwd.split(directorySeparator)[:-nLevelUp] + 
                                                  ['KineticModelSimulations', 'BVO', ('PBC' if np.all(pbc) else 'NoPBC'), 
                                                   ('SystemSize[' + ','.join(['%i' % systemSize[i] for i in range(len(systemSize))]) + ']')])
    # Build path for material and neighbors object files
    materialName = 'BVO'
    cutE = extractCutoff if extractCutoff else parentCutoff
    tailName = '_E' + str(cutE) + '.obj'
    objectFileDirectoryName = 'ObjectFiles'
    objectFileDirPath = systemDirectoryPath + directorySeparator + objectFileDirectoryName
    if not os.path.exists(objectFileDirPath):
        os.makedirs(objectFileDirPath)
    materialFileName = objectFileDirPath + directorySeparator + materialName + tailName
    neighborListDirectoryName = 'NeighborListFiles'
    neighborsFileName = objectFileDirPath + directorySeparator + materialName + 'Neighbors' + tailName
    neighborListDirPath = systemDirectoryPath + directorySeparator + neighborListDirectoryName
    
    # Load BVO parameters and update electrostatic interaction cutoff
    bvoParameters = bvoParameters()
    bvoParameters.neighborCutoffDist['E'] = [cutE]
    
    # Build material object files
    bvo = material(bvoParameters)
    bvo.generateMaterialFile(bvo, materialFileName, replaceExistingObjectFiles)
    
    # Build neighbors object files
    bvoNeighbors = neighbors(bvo, systemSize, pbc)
    bvoNeighbors.generateNeighborsFile(bvoNeighbors, neighborsFileName, replaceExistingObjectFiles)
    bvoNeighbors.generateNeighborList(parentCutoff, extractCutoff, neighborListDirPath, replaceExistingNeighborList)