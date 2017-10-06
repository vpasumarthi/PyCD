#!/usr/bin/env python

import os
import pickle

from PyCT.core import analysis


def materialMSD(systemDirectoryPath, nDim, Temp, speciesCount,
                tFinal, nTraj, timeInterval, msdTFinal, trimLength,
                displayErrorBars, reprTime, reprDist, report):

    # Change to working directory
    parentDir1 = 'SimulationFiles'
    nElectrons = speciesCount[0]
    nHoles = speciesCount[1]
    parentDir2 = (str(nElectrons)
                  + ('electron' if nElectrons == 1 else 'electrons') + ', '
                  + str(nHoles) + ('hole' if nHoles == 1 else 'holes'))
    parentDir3 = str(Temp) + 'K'
    workDir = (('%1.2E' % tFinal) + 'SEC,' + ('%1.2E' % timeInterval)
               + 'TimeInterval,' + ('%1.2E' % nTraj) + 'Traj')
    workDirPath = os.path.join(systemDirectoryPath, parentDir1, parentDir2,
                               parentDir3, workDir)

    if not os.path.exists(workDirPath):
        print 'Simulation files do not exist. Aborting.'
    else:
        os.chdir(workDirPath)

        # Determine path for input files
        inputFileDirectoryName = 'InputFiles'
        inputFileDirectoryPath = os.path.join(systemDirectoryPath,
                                              inputFileDirectoryName)

        # Build path for material and neighbors object files
        tailName = '.obj'
        objectFileDirectoryName = 'ObjectFiles'
        objectFileDirPath = os.path.join(inputFileDirectoryPath,
                                         objectFileDirectoryName)
        materialFileName = (os.path.join(objectFileDirPath, 'material')
                            + tailName)

        # Load material object
        file_material = open(materialFileName, 'r')
        materialInfo = pickle.load(file_material)
        file_material.close()

        materialAnalysis = analysis(materialInfo, nDim, speciesCount, nTraj,
                                    tFinal, timeInterval, msdTFinal,
                                    trimLength, reprTime, reprDist)

        msdAnalysisData = materialAnalysis.computeMSD(workDirPath, report)
        msdData = msdAnalysisData.msdData
        stdData = msdAnalysisData.stdData
        speciesTypes = msdAnalysisData.speciesTypes
        fileName = msdAnalysisData.fileName
        materialAnalysis.generateMSDPlot(msdData, stdData, displayErrorBars,
                                         speciesTypes, fileName, workDirPath)
