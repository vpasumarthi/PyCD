#!/usr/bin/env python
def hematiteSetup(systemSize, pbc, replaceExistingObjectFiles, replaceExistingNeighborList, 
                  alpha, nmax, kmax, replaceExistingPrecomputedArray):
    """Prepare material class object file, neighborlist and saves to the provided destination path"""
    from hematiteParameters import hematiteParameters
    from KineticModel import material, neighbors, system
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
    
    # Determine path for neighbor list directories
    inputFileDirectoryName = 'InputFiles'
    inputFileDirectoryPath = systemDirectoryPath + directorySeparator + inputFileDirectoryName

    # Build material object files
    hematite = material(hematiteParameters())
    hematite.generateMaterialFile(hematite, materialFileName, replaceExistingObjectFiles)
    
    # Build neighbors object files
    hematiteNeighbors = neighbors(hematite, systemSize, pbc)
    hematiteNeighbors.generateNeighborsFile(hematiteNeighbors, neighborsFileName, replaceExistingObjectFiles)
    hematiteNeighbors.generateNeighborList(inputFileDirectoryPath, replaceExistingNeighborList)

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

        hematiteSystem = system(hematite, hematiteNeighbors, hopNeighborList, cumulativeDisplacementList, 
                                speciesCount, alpha, nmax, kmax)
        precomputedArray = hematiteSystem.ewaldSumSetup()
        np.save(precomputedArrayFilePath, precomputedArray)
        
        ewaldParametersFilePath = inputFileDirectoryPath + directorySeparator + 'ewaldParameters.npy'
        ewaldParameters = {'alpha': alpha, 'nmax' : nmax, 'kmax' : kmax}
        np.save(ewaldParametersFilePath, ewaldParameters)