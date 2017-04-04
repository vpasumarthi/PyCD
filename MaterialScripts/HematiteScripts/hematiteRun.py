#!/usr/bin/env python
def hematiteRun(neighborList, shellCharges, cutE, systemDirectoryPath, speciesCount, T, nTraj, kmcSteps, 
                stepInterval, gui, outdir, report, randomSeed):
        
    from KineticModel import system, run, initiateSystem
    import numpy as np
    import pickle
    import platform
    
    materialName = 'hematite'
    tailName = '_Shell' if shellCharges else '_NoShell' + '_E' + str(cutE) + '.obj'
    directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
    objectFileDirectoryName = 'ObjectFiles'
    objectFileDirPath = systemDirectoryPath + directorySeparator + objectFileDirectoryName
    materialFileName = objectFileDirPath + directorySeparator + materialName + tailName
    neighborsFileName = objectFileDirPath + directorySeparator + materialName + 'Neighbors' + tailName
    
    file_hematite = open(materialFileName, 'r')
    hematite = pickle.load(file_hematite)
    file_hematite.close()
    
    file_hematiteNeighbors = open(neighborsFileName, 'r')
    hematiteNeighbors = pickle.load(file_hematiteNeighbors)
    file_hematiteNeighbors.close()
    
    initiateHematiteSystem = initiateSystem(hematite, hematiteNeighbors)
    initialOccupancy =  initiateHematiteSystem.generateRandomOccupancy(speciesCount)
    # TODO: No electron system
    # del initialOccupancy['electron'][0]
    hematiteSystem = system(hematite, hematiteNeighbors, neighborList[()], initialOccupancy, speciesCount)
    hematiteRun = run(hematite, hematiteNeighbors, hematiteSystem, T, nTraj, kmcSteps, stepInterval, gui)
    
    hematiteRun.doKMCSteps(outdir, report, randomSeed)