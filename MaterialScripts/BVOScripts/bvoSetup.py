#!/usr/bin/env python
def bvoSetup(systemSize, pbc, replaceExistingObjectFiles, generateHopNeighborList, 
                  generateCumDispList, alpha, nmax, kmax, replaceExistingPrecomputedArray):
    """Prepare material class object file, neighborlist and saves to the provided destination path"""
    from bvoParameters import bvoParameters
    from KineticModel import material, neighbors, system
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

    # Determine path for input files
    inputFileDirectoryName = 'InputFiles'
    inputFileDirectoryPath = systemDirectoryPath + directorySeparator + inputFileDirectoryName
    
    # Build path for material and neighbors object files
    materialName = 'bvo'
    tailName = '.obj'
    objectFileDirectoryName = 'ObjectFiles'
    objectFileDirPath = inputFileDirectoryPath + directorySeparator + objectFileDirectoryName
    if not os.path.exists(objectFileDirPath):
        os.makedirs(objectFileDirPath)
    materialFileName = objectFileDirPath + directorySeparator + materialName + tailName
    neighborsFileName = objectFileDirPath + directorySeparator + materialName + 'Neighbors' + tailName
    
    # Build material object files
    bvo = material(bvoParameters())
    bvo.generateMaterialFile(bvo, materialFileName, replaceExistingObjectFiles)
    
    # Build neighbors object files
    bvoNeighbors = neighbors(bvo, systemSize, pbc)
    bvoNeighbors.generateNeighborsFile(bvoNeighbors, neighborsFileName, replaceExistingObjectFiles)
    bvoNeighbors.generateNeighborList(inputFileDirectoryPath, generateHopNeighborList, generateCumDispList)

    # Build precomputed array and save to disk
    precomputedArrayFilePath = inputFileDirectoryPath + directorySeparator + 'precomputedArray.npy'
    if (not os.path.isfile(precomputedArrayFilePath) or replaceExistingPrecomputedArray):
        # Load input files to instantiate system class
        os.chdir(inputFileDirectoryPath)
        hopNeighborListFileName = inputFileDirectoryPath + directorySeparator + 'hopNeighborList.npy'
        hopNeighborList = np.load(hopNeighborListFileName)[()]
        cumulativeDisplacementListFilePath = inputFileDirectoryPath + directorySeparator + 'cumulativeDisplacementList.npy'
        cumulativeDisplacementList = np.load(cumulativeDisplacementListFilePath)
        
        # Note: speciesCount doesn't effect the precomputedArray, however it is essential to instantiate system class
        # So, any non-zero value of speciesCount will do.
        nElectrons = 1
        nHoles = 0
        speciesCount = np.array([nElectrons, nHoles])

        bvoSystem = system(bvo, bvoNeighbors, hopNeighborList, cumulativeDisplacementList, 
                                speciesCount, alpha, nmax, kmax)
        precomputedArray = bvoSystem.ewaldSumSetup(inputFileDirectoryPath)
        np.save(precomputedArrayFilePath, precomputedArray)
        
        ewaldParametersFilePath = inputFileDirectoryPath + directorySeparator + 'ewaldParameters.npy'
        ewaldParameters = {'alpha': alpha, 'nmax' : nmax, 'kmax' : kmax}
        np.save(ewaldParametersFilePath, ewaldParameters)