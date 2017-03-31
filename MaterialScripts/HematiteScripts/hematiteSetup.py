#!/usr/bin/env python
def hematiteSetup(chargeTypes, cutE, shellCharges, shellChargeTypes, systemSize, pbc, 
                  replaceExistingObjectFiles, parent, extract, replaceExistingNeighborList, outdir):
    """Prepare material class object file, neighborlist and saves to the provided destination path"""
    from hematiteParameters import hematiteParameters
    from KineticModel import material, neighbors
    import platform
    
    directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
    objectFileDirectoryName = 'ObjectFiles'
    neighborListDirectoryName = 'NeighborListFiles'
    objectFileOutDir = outdir + directorySeparator + objectFileDirectoryName
    neighborListOutDir = outdir + directorySeparator + neighborListDirectoryName
    
    materialName = 'hematite'
    hematiteParameters = hematiteParameters()
    hematiteParameters.chargeTypes = chargeTypes
    hematiteParameters.neighborCutoffDist['E'] = cutE if extract else [cutE]
    if not shellCharges:
        for iShellChargeType in shellChargeTypes:
            del hematiteParameters.neighborCutoffDist[iShellChargeType]
    
    cutE = cutE if extract else [cutE]

    for iCutE in cutE:        
        tailName = '_Shell' if shellCharges else '_NoShell' + ('_E' + 'parent' if parent else str(iCutE)) + '.obj'
        
        hematite = material(hematiteParameters)
        materialFileName = objectFileOutDir + directorySeparator + materialName + tailName
        hematite.generateMaterialFile(hematite, materialFileName, replaceExistingObjectFiles)
        
        hematiteNeighbors = neighbors(hematite, systemSize, pbc)
        neighborsFileName = objectFileOutDir + directorySeparator + materialName + 'Neighbors' + tailName
        hematiteNeighbors.generateNeighborsFile(hematiteNeighbors, neighborsFileName, replaceExistingObjectFiles)
    hematiteNeighbors.generateNeighborList(parent, extract, cutE if extract else cutE[0], replaceExistingNeighborList, neighborListOutDir)
