#!/usr/bin/env python
def hematiteSetup(chargeTypes, cutE, systemSize, pbc, replaceExistingObjectFiles, 
                  parent, extract, replaceExistingNeighborList, outdir):
    """Prepare material class object file, neighborlist and saves to the provided destination path"""
    from hematiteParameters import hematiteParameters
    from KineticModel import material, neighbors
    import platform
    import os
    
    directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
    objectFileDirectoryName = 'ObjectFiles'
    neighborListDirectoryName = 'NeighborListFiles'
    objectFileOutDir = outdir + directorySeparator + objectFileDirectoryName
    if not os.path.exists(objectFileOutDir):
        os.mkdir(objectFileOutDir)
    neighborListOutDir = outdir + directorySeparator + neighborListDirectoryName
    if not os.path.exists(neighborListOutDir):
        os.mkdir(neighborListOutDir)
    
    materialName = 'hematite'
    hematiteParameters = hematiteParameters()
    hematiteParameters.chargeTypes = chargeTypes
    hematiteParameters.neighborCutoffDist['E'] = cutE if extract else [cutE]
    
    cutE = cutE if extract else [cutE]

    for iCutE in cutE:        
        tailName = ('_Parent' if parent else ('_E' + str(iCutE))) + '.obj'
        
        hematite = material(hematiteParameters)
        materialFileName = objectFileOutDir + directorySeparator + materialName + tailName
        hematite.generateMaterialFile(hematite, materialFileName, replaceExistingObjectFiles)
        
        hematiteNeighbors = neighbors(hematite, systemSize, pbc)
        neighborsFileName = objectFileOutDir + directorySeparator + materialName + 'Neighbors' + tailName
        hematiteNeighbors.generateNeighborsFile(hematiteNeighbors, neighborsFileName, replaceExistingObjectFiles)
    hematiteNeighbors.generateNeighborList(parent, neighborListOutDir, extract, cutE if extract else cutE[0], replaceExistingNeighborList)
