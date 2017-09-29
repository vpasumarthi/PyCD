#!/usr/bin/env python

def hematiteMassMeanDistance(pbc, systemSize, nTrajList, cutE_List, nSpeciesList, TempList, shellCharges, kmcStepsList, 
                             stepInterval, nStepsMSDList, nDispMSDList, binsize, reprTime, reprDist, overWrite):
    
    from hematiteMeanDistance import hematiteMeanDistance
    import os.path
    import platform
    import numpy as np

    cwd = os.path.dirname(os.path.realpath(__file__))
    nLevelUp = 3 if platform.uname()[0]=='Linux' else 4
    directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
    systemDirectoryPath = directorySeparator.join(cwd.split(directorySeparator)[:-nLevelUp] + ['PyCTSimulations', 'Hematite', ('PBC' if pbc else 'NoPBC'), 
                                                       ('SystemSize' + str(systemSize).replace(' ', ''))])
    for nTraj in nTrajList:
        for cutE in cutE_List:
            parentDir1 = 'E_' + str(cutE)
            for speciesIndex in range(len(nSpeciesList[0])):
                kmcSteps = kmcStepsList[speciesIndex]
                nElectrons = nSpeciesList[0][speciesIndex]
                nHoles = nSpeciesList[1][speciesIndex]
                speciesCount = np.array([nElectrons, nHoles])
                parentDir2 = str(nElectrons) + ('electron' if nElectrons==1 else 'electrons') + ', ' + str(nHoles) + ('hole' if nHoles==1 else 'holes')
                for iTemp in TempList:
                    parentDir3 = str(iTemp) + 'K'
                    parentDir = [parentDir1, parentDir2, parentDir3]
                    parentDirPath = systemDirectoryPath + directorySeparator + directorySeparator.join(parentDir)
                    workDir = (('%1.2E' % kmcSteps) + 'Steps,' + 
                               ('%1.2E' % (kmcSteps/stepInterval)) + 'PathSteps,' + 
                               ('%1.2E' % nTraj) + 'Traj')
                    workDir = workDir.replace('+','')
                    outdir = parentDirPath + directorySeparator + workDir
                    os.chdir(outdir)
                    trajectoryDataFileName = ('TrajectoryData.npy')
                    fileExists = 0
                    for fname in os.listdir('.'):
                        if fname.endswith('.jpg'):
                            fileExists = 1
                    if not fileExists or overWrite:
                        nStepsMSD = nStepsMSDList[speciesIndex]
                        nDispMSD = nDispMSDList[speciesIndex]
                        hematiteMeanDistance(trajectoryDataFileName, shellCharges, cutE, systemDirectoryPath, systemSize, speciesCount, 
                                             nTraj, kmcSteps, stepInterval, nStepsMSD, nDispMSD, binsize, reprTime, reprDist, outdir)
