#!/usr/bin/env python

import os
import pickle

from PyCT.core import analysis


def materialMSD(systemDirectoryPath, systemSize, nDim, Temp, speciesCount,
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
        # TODO: Obtain material name in generic sense
        materialName = 'bvo'
        tailName = '.obj'
        objectFileDirectoryName = 'ObjectFiles'
        objectFileDirPath = os.path.join(inputFileDirectoryPath,
                                         objectFileDirectoryName)
        materialFileName = (os.path.join(objectFileDirPath, materialName)
                            + tailName)

        # Load material object
        file_bvo = open(materialFileName, 'r')
        bvo = pickle.load(file_bvo)
        file_bvo.close()

        bvoAnalysis = analysis(bvo, nDim, systemSize, speciesCount, nTraj,
                               tFinal, timeInterval, msdTFinal, trimLength,
                               reprTime, reprDist)

        msdAnalysisData = bvoAnalysis.computeMSD(workDirPath, report)
        msdData = msdAnalysisData.msdData
        stdData = msdAnalysisData.stdData
        speciesTypes = msdAnalysisData.speciesTypes
        fileName = msdAnalysisData.fileName
        bvoAnalysis.generateMSDPlot(msdData, stdData, displayErrorBars,
                                    speciesTypes, fileName, workDirPath)
