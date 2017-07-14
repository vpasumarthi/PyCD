#!/usr/bin/env python
def hematiteSetup(systemSize, pbc, replaceExistingObjectFiles, replaceExistingNeighborList):
    """Prepare material class object file, neighborlist and saves to the provided destination path"""
    from hematiteParameters import hematiteParameters
    from KineticModel import material, neighbors
    import numpy as np
    import platform
    import os
    
    # Determine path for system directory    
    cwd = os.path.dirname(os.path.realpath(__file__))
    directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
    nLevelUp = 3 if platform.uname()[0]=='Linux' else 4
    systemDirectoryPath = directorySeparator.join(cwd.split(directorySeparator)[:-nLevelUp] + 
                                                  ['KineticModelSimulations', 'Hematite', ('PBC' if np.all(pbc) else 'NoPBC'), 
                                                   ('SystemSize[' + ','.join(['%i' % systemSize[i] for i in range(len(systemSize))]) + ']')])
    # Build path for material and neighbors object files
    materialName = 'hematite'
    tailName = '.obj'
    objectFileDirectoryName = 'ObjectFiles'
    objectFileDirPath = systemDirectoryPath + directorySeparator + objectFileDirectoryName
    if not os.path.exists(objectFileDirPath):
        os.makedirs(objectFileDirPath)
    materialFileName = objectFileDirPath + directorySeparator + materialName + tailName
    neighborListDirectoryName = 'NeighborListFiles'
    neighborsFileName = objectFileDirPath + directorySeparator + materialName + 'Neighbors' + tailName
    neighborListDirPath = systemDirectoryPath + directorySeparator + neighborListDirectoryName
    
    # Build material object files
    hematite = material(hematiteParameters())
    hematite.generateMaterialFile(hematite, materialFileName, replaceExistingObjectFiles)
    
    # Build neighbors object files
    hematiteNeighbors = neighbors(hematite, systemSize, pbc)
    hematiteNeighbors.generateNeighborsFile(hematiteNeighbors, neighborsFileName, replaceExistingObjectFiles)
    hematiteNeighbors.generateNeighborList(neighborListDirPath, replaceExistingNeighborList)