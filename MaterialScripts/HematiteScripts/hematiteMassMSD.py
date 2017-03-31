#!/usr/bin/env python

def hematiteMassMSD(pbc, systemSize, nTrajList, cutE_List, nSpeciesList, TempList, shellCharges, kmcStepsList, 
                    stepInterval, nStepsMSDList, nDispMSDList, binsize, reprTime, reprDist, overWrite):
    
    from hematiteMSD import hematiteMSD
    import os.path
    import platform

    cwd = os.path.dirname(os.path.realpath(__file__))
    directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
    cwd = (cwd + directorySeparator + ('PBC' if pbc else 'NoPBC') + directorySeparator + 
           'SystemSize' + str(systemSize).replace(' ', ''))
    for nTraj in nTrajList:
        dirPath = cwd # directory where neighborList is located
        for cutE in cutE_List:
            parentDir1 = 'E_' + str(cutE)
            for speciesIndex in range(len(nSpeciesList[0])):
                kmcSteps = kmcStepsList[speciesIndex]
                nElectrons = nSpeciesList[0][speciesIndex]
                nHoles = nSpeciesList[1][speciesIndex]
                parentDir2 = str(nElectrons) + ('electron' if nElectrons==1 else 'electrons') + ', ' + str(nHoles) + ('hole' if nHoles==1 else 'holes')
                for iTemp in TempList:
                    parentDir3 = str(iTemp) + 'K'
                    parentDir = [parentDir1, parentDir2, parentDir3]
                    parentDirPath = cwd + directorySeparator + directorySeparator.join(parentDir)
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
                        hematiteMSD(trajectoryDataFileName, shellCharges, cutE, dirPath, nStepsMSD, nDispMSD, binsize, 
                                    reprTime, reprDist, outdir)
