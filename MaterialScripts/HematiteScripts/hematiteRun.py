#!/usr/bin/env python
def hematiteRun(systemSize, pbc, Temp, speciesCount, tFinal, nTraj, stepInterval, 
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
                                                  ['KineticModelSimulations', 'Hematite', ('PBC' if pbc else 'NoPBC'), 
                                                   ('SystemSize' + str(systemSize).replace(' ', ''))])

    # Change to working directory
    parentDir1 = 'SimulationFiles'
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
        # Determine path for input files
        inputFileDirectoryName = 'InputFiles'
        inputFileDirectoryPath = systemDirectoryPath + directorySeparator + inputFileDirectoryName
        
        # Build path for material and neighbors object files
        materialName = 'hematite'
        tailName = '.obj'
        directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
        objectFileDirectoryName = 'ObjectFiles'
        objectFileDirPath = inputFileDirectoryPath + directorySeparator + objectFileDirectoryName
        materialFileName = objectFileDirPath + directorySeparator + materialName + tailName
        neighborsFileName = objectFileDirPath + directorySeparator + materialName + 'Neighbors' + tailName
        
        # Load material object
        file_hematite = open(materialFileName, 'r')
        hematite = pickle.load(file_hematite)
        file_hematite.close()
        
        # Load neighbors object
        file_hematiteNeighbors = open(neighborsFileName, 'r')
        hematiteNeighbors = pickle.load(file_hematiteNeighbors)
        file_hematiteNeighbors.close()
        
        # Determine path for neighbor list directories
        neighborListDirectoryName = 'NeighborListFiles'
        neighborListDirectoryPath = systemDirectoryPath + directorySeparator + neighborListDirectoryName

        # Load input files to instantiate system class
        os.chdir(inputFileDirectoryPath)
        hopNeighborListFileName = inputFileDirectoryPath + directorySeparator + 'hopNeighborList.npy'
        hopNeighborList = np.load(hopNeighborListFileName)[()]
        cumulativeDisplacementListFilePath = inputFileDirectoryPath + directorySeparator + 'cumulativeDisplacementList.npy'
        cumulativeDisplacementList = np.load(cumulativeDisplacementListFilePath)
        hematiteSystem = system(hematite, hematiteNeighbors, hopNeighborList, cumulativeDisplacementList, speciesCount)
        
        # Load precomputed array, ewald parameters file to instantiate run class
        precomputedArrayFilePath = inputFileDirectoryPath + directorySeparator + 'precomputedArray.npy'
        precomputedArray = np.load(precomputedArrayFilePath)
        ewaldParametersFilePath = inputFileDirectoryPath + directorySeparator + 'ewaldParameters.npy'
        ewaldParameters = np.load(ewaldParametersFilePath)
        hematiteRun = run(hematiteSystem, precomputedArray, ewaldParameters, Temp, nTraj, kmcSteps, stepInterval, gui)
        
        hematiteRun.doKMCSteps(workDirPath, report, randomSeed)
    else:
        print 'Simulation files already exists in the destination directory'