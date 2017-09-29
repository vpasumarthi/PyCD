#!/usr/bin/env python

def hematiteMSD(systemSize, pbc, nDim, Temp, speciesCount, tFinal, nTraj, timeInterval, 
                msdTFinal, trimLength, displayErrorBars, reprTime, reprDist, report, overWrite):

    from KineticModel import analysis
    import numpy as np
    import os
    import platform
    import pickle
    
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
    workDir = (('%1.2E' % tFinal) + 'SEC,' + ('%1.2E' % timeInterval) + 'TimeInterval,' + ('%1.2E' % nTraj) + 'Traj')
    workDirPath = systemDirectoryPath + directorySeparator + directorySeparator.join([parentDir1, parentDir2, parentDir3, workDir])
    if not os.path.exists(workDirPath):
        print 'Simulation files do not exist. Aborting.'
    else:
        os.chdir(workDirPath)

        # Determine path for neighbor list directories
        inputFileDirectoryName = 'InputFiles'
        inputFileDirectoryPath = systemDirectoryPath + directorySeparator + inputFileDirectoryName
    
        # Build path for material and neighbors object files
        materialName = 'hematite'
        tailName = '.obj'
        directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
        objectFileDirectoryName = 'ObjectFiles'
        objectFileDirPath = inputFileDirectoryPath + directorySeparator + objectFileDirectoryName
        materialFileName = objectFileDirPath + directorySeparator + materialName + tailName
        
        # Load material object
        file_hematite = open(materialFileName, 'r')
        hematite = pickle.load(file_hematite)
        file_hematite.close()
    
        hematiteAnalysis = analysis(hematite, nDim, systemSize, speciesCount, nTraj, tFinal, 
                                    timeInterval, msdTFinal, trimLength, reprTime, reprDist)
        
        msdAnalysisData = hematiteAnalysis.computeMSD(workDirPath, report)
        msdData = msdAnalysisData.msdData
        stdData = msdAnalysisData.stdData
        speciesTypes = msdAnalysisData.speciesTypes
        fileName = msdAnalysisData.fileName
        hematiteAnalysis.generateMSDPlot(msdData, stdData, displayErrorBars, speciesTypes, fileName, workDirPath)