#!/usr/bin/env python

def hematiteMassRun(pbc, systemSize, nTrajList, cutE_List, nSpeciesList, TempList, shellCharges, 
                    kmcStepsList, stepInterval, gui, ESPConfig, report, randomSeed, overWrite):
    
    from hematiteRun import hematiteRun
    import os
    import platform
    import numpy as np
    
    cwd = os.path.dirname(os.path.realpath(__file__))
    directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
    nLevelUp = 4
    neighborListDirectoryName = 'NeighborListFiles'
    systemDirectoryPath = directorySeparator.join(cwd.split(directorySeparator)[:-nLevelUp] + ['KineticModelSimulations', 'Hematite', ('PBC' if pbc else 'NoPBC'), 
                                                       ('SystemSize' + str(systemSize).replace(' ', ''))])
    neighborListDirectoryPath = systemDirectoryPath + directorySeparator + neighborListDirectoryName
    for nTraj in nTrajList:
        for cutE in cutE_List:
            os.chdir(neighborListDirectoryPath)
            neighborListFileName = (neighborListDirectoryPath + directorySeparator + 'NeighborList' + '_E' + str(cutE) + '.npy')
            neighborList = np.load(neighborListFileName)
            os.chdir(systemDirectoryPath)
            parentDir1 = 'E_' + str(cutE)
            #import pdb; pdb.set_trace()
            if not os.path.exists(parentDir1):
                os.mkdir(parentDir1)
            os.chdir(parentDir1)
            parentDir1Path = systemDirectoryPath + directorySeparator + parentDir1
            for speciesIndex in range(len(nSpeciesList[0])):
                nElectrons = nSpeciesList[0][speciesIndex]
                nHoles = nSpeciesList[1][speciesIndex]
                kmcSteps = kmcStepsList[speciesIndex]
                os.chdir(parentDir1Path)
                speciesCount = np.array([nElectrons, nHoles])
                parentDir2 = str(nElectrons) + ('electron' if nElectrons==1 else 'electrons') + ', ' + str(nHoles) + ('hole' if nHoles==1 else 'holes')
                if not os.path.exists(parentDir2):
                    os.mkdir(parentDir2)
                os.chdir(parentDir2)
                parentDir2Path = parentDir1Path + directorySeparator + parentDir2
                for iTemp in TempList:
                    os.chdir(parentDir2Path)
                    parentDir3 = str(iTemp) + 'K'
                    if not os.path.exists(parentDir3):
                        os.mkdir(parentDir3)
                    os.chdir(parentDir3)
                    parentDir3Path = parentDir2Path + directorySeparator + parentDir3
                    workDir = (('%1.2E' % kmcSteps) + 'Steps,' + 
                               ('%1.2E' % (kmcSteps/stepInterval)) + 'PathSteps,' + 
                               ('%1.2E' % nTraj) + 'Traj')
                    workDir = workDir.replace('+','')
                    if not os.path.exists(workDir):
                        os.mkdir(workDir)
                    os.chdir(workDir)
                    outdir = parentDir3Path + directorySeparator + workDir
                    fileExists = 0
                    for fname in os.listdir('.'):
                        if fname.endswith('.log') and fname.startswith('TrajectoryData'):
                            fileExists = 1
                    if not fileExists or overWrite:
                        hematiteRun(neighborList, shellCharges, cutE, systemDirectoryPath, speciesCount, iTemp, nTraj, 
                                    kmcSteps, stepInterval, gui, outdir, ESPConfig, report, randomSeed)