#!/usr/bin/env python
def bvoRun(systemSize, pbc, Temp, cutE, speciesCount, tFinal, nTraj, stepInterval, 
                   kmcStepCountPrecision, randomSeed, report, overWrite, gui):
    from KineticModel import system, run
    import os
    import platform
    import numpy as np
    import pickle
    
    # Compute number of estimated KMC steps to reach tFinal
    totalSpecies = sum(speciesCount)
    kBasal = 2.58E-08 # au
    kC = 1.57E-11 # au
    SEC2AUTIME = 41341373366343300
    kTotalPerSpecies = 3 * kBasal + kC
    kTotal = kTotalPerSpecies * totalSpecies
    timeStep = 1 / (kTotal * SEC2AUTIME)
    kmcSteps = int(np.ceil(tFinal / timeStep / kmcStepCountPrecision) * kmcStepCountPrecision)

    # Determine path for system directory    
    cwd = os.path.dirname(os.path.realpath(__file__))
    directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
    nLevelUp = 3 if platform.uname()[0]=='Linux' else 4
    systemDirectoryPath = directorySeparator.join(cwd.split(directorySeparator)[:-nLevelUp] + 
                                                  ['KineticModelSimulations', 'BVO', ('PBC' if pbc else 'NoPBC'), 
                                                   ('SystemSize' + str(systemSize).replace(' ', ''))])

    # Change to working directory
    parentDir1 = 'E_' + str(cutE)
    nElectrons = speciesCount[0]
    nHoles = speciesCount[1]
    parentDir2 = str(nElectrons) + ('electron' if nElectrons==1 else 'electrons') + ', ' + str(nHoles) + ('hole' if nHoles==1 else 'holes')
    parentDir3 = str(Temp) + 'K'
    workDir = (('%1.2E' % kmcSteps) + 'Steps,' + ('%1.2E' % (kmcSteps/stepInterval)) + 'PathSteps,' + ('%1.2E' % nTraj) + 'Traj')
    workDir = workDir.replace('+','')
    workDirPath = systemDirectoryPath + directorySeparator + directorySeparator.join([parentDir1, parentDir2, parentDir3, workDir])
    if not os.path.exists(workDirPath):
        os.makedirs(workDirPath)
    os.chdir(workDirPath)

    fileExists = 0
    if os.path.exists('Run.log') and os.path.exists('Time.dat'):
        fileExists = 1
    if not fileExists or overWrite:
        # Build path for material and neighbors object files
        materialName = 'BVO'
        tailName = '_E' + str(cutE) + '.obj'
        directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
        objectFileDirectoryName = 'ObjectFiles'
        objectFileDirPath = systemDirectoryPath + directorySeparator + objectFileDirectoryName
        materialFileName = objectFileDirPath + directorySeparator + materialName + tailName
        neighborsFileName = objectFileDirPath + directorySeparator + materialName + 'Neighbors' + tailName
        
        # Load material object
        materialFile = open(materialFileName, 'r')
        bvo = pickle.load(materialFile)
        materialFile.close()
        
        # Load neighbors object
        neighborsFile = open(neighborsFileName, 'r')
        bvoNeighbors = pickle.load(neighborsFile)
        neighborsFile.close()
        
        # Determine path for neighbor list directories
        neighborListDirectoryName = 'NeighborListFiles'
        neighborListDirectoryPath = systemDirectoryPath + directorySeparator + neighborListDirectoryName + directorySeparator + 'E_' + str(cutE)

        # Load Neighbor List
        os.chdir(neighborListDirectoryPath)
        hopNeighborListFileName = neighborListDirectoryPath + directorySeparator + 'hopNeighborList.npy'
        elecNeighborListFileName =  neighborListDirectoryPath + directorySeparator + 'elecNeighborList.npy'
        hopNeighborList = np.load(hopNeighborListFileName)[()]
        
        # Determine paths for electrostatic neighbor list component files
        neighborSystemElementIndicesFileName = neighborListDirectoryPath + directorySeparator + 'neighborSystemElementIndices.npy'
        displacementListFileName = neighborListDirectoryPath + directorySeparator + 'displacementList.npy'
        numNeighborsFileName = neighborListDirectoryPath + directorySeparator + 'numNeighbors.npy'
        
        # Load electrostatic neighbor list component files
        neighborSystemElementIndices = np.load(neighborSystemElementIndicesFileName)
        displacementList = np.load(displacementListFileName)
        numNeighbors = np.load(numNeighborsFileName)
        
        bvoSystem = system(bvo, bvoNeighbors, hopNeighborList, neighborSystemElementIndices, displacementList, numNeighbors, speciesCount)
        bvoRun = run(bvoSystem, Temp, nTraj, kmcSteps, stepInterval, gui)
        
        bvoRun.doKMCSteps(workDirPath, report, randomSeed)
    else:
        print 'Simulation files already exists in the destination directory'