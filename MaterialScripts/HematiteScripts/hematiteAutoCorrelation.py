#!/usr/bin/env python
def hematiteAutoCorrelation(systemSize, pbc, Temp, cutE, speciesCount, tFinal, nTraj, stepInterval, kmcStepCountPrecision, 
                            msdStepCountPrecision, msdTFinal, nDispMin, binsize, reprTime, reprDist, report, overWrite):
    from KineticModel import analysis
    import os
    import platform
    import numpy as np
    import pickle

    # Compute number of estimated KMC steps to reach tFinal
    totalSpecies = sum(speciesCount)
    kBasal = 2.58E-08 # au
    kC = 1.57E-11 # au
    ANG2BOHR = 1.8897261264678997
    SEC2NS = 1.00e+09
    SEC2AUTIME = 4.134137336634339e+16
    kTotalPerSpecies = 3 * kBasal + kC
    kTotal = kTotalPerSpecies * totalSpecies
    timeStep = 1 / (kTotal * SEC2AUTIME)
    kmcSteps = int(np.ceil(tFinal / timeStep / kmcStepCountPrecision) * kmcStepCountPrecision)

    # Compute number of MSD steps and number of displacements    
    trajTimeStep = timeStep * stepInterval
    # TODO: Is it possible to get away with msdStepCountPrecision
    nStepsMSD = int(np.ceil(msdTFinal * 1.00E-09 / trajTimeStep / msdStepCountPrecision) * msdStepCountPrecision)
    nDispMSD = min([nDispMin, kmcSteps / stepInterval / 2])

    # Determine path for system directory    
    cwd = os.path.dirname(os.path.realpath(__file__))
    directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
    nLevelUp = 3 if platform.uname()[0]=='Linux' else 4
    systemDirectoryPath = directorySeparator.join(cwd.split(directorySeparator)[:-nLevelUp] + 
                                                  ['KineticModelSimulations', 'Hematite', ('PBC' if pbc else 'NoPBC'), 
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
        print 'Simulation files do not exist. Aborting.'
    else:
        os.chdir(workDirPath)

        # Build path for material and neighbors object files
        materialName = 'hematite'
        tailName = '_E' + str(cutE) + '.obj'
        directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
        objectFileDirectoryName = 'ObjectFiles'
        objectFileDirPath = systemDirectoryPath + directorySeparator + objectFileDirectoryName
        materialFileName = objectFileDirPath + directorySeparator + materialName + tailName
        
        # Load material object
        file_hematite = open(materialFileName, 'r')
        hematite = pickle.load(file_hematite)
        file_hematite.close()
    
        hematiteAnalysis = analysis(hematite, speciesCount, nTraj, kmcSteps, stepInterval, 
                                    systemSize, nStepsMSD, nDispMSD, binsize, reprTime, reprDist)
        
        msdAnalysisData = hematiteAnalysis.computeAutoMSD(workDirPath, report)
        speciesTypeMeanVelProd = msdAnalysisData.speciesTypeMeanVelProd
        speciesTypes = msdAnalysisData.speciesTypes
        fileName = msdAnalysisData.fileName
        hematiteAnalysis.generateAutoMSDPlot(speciesTypeMeanVelProd, speciesTypes, fileName, workDirPath)