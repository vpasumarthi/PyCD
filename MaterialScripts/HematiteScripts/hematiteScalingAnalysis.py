#!/usr/bin/env python

def hematiteScalingAnalysis(simulationParameters, scalingParmeter, outdir, save=0, replaceExistingFiles=0, openFigure=0):
    
    import numpy as np
    import os
    
    simulationParameterNames = ['systemSize', 'cutE', 'nTraj', 'kmcSteps', 'nSpecies']
    inputVarIndex = 0
    for simulationParameterIndex, simulationParameterName in enumerate(simulationParameterNames):
        if scalingParmeter != simulationParameterIndex:
            locals()[simulationParameterName] = simulationParameters[inputVarIndex]
            inputVarIndex += 1 
    
    matchColumnIndices = []
    matchRow = []
    #varList = [varName + ('_List' if varName == 'cutE' else 'List') for varName in simulationParameterNames]
    varSyntax = ['%s', '%1.0f', '%1.0E', '%1.0E', '%d']
    constantParameters = ''
    
    for varNameIndex, varName in enumerate(simulationParameterNames):
        if varName in locals():
            if varNameIndex == 0:
                matchColumnIndices.extend(range(varNameIndex+3))
                matchRow.extend(locals()[varName])
            else:
                matchColumnIndices.append(varNameIndex+2)
                matchRow.append(locals()[varName])
            constantParameters += varName + '=' + (varSyntax[varNameIndex] % locals()[varName]) + ', ' 
        else:
            xVarIndex = varNameIndex
    matchRow = np.asarray(matchRow)
    
    #[systemSize[0], systemSize[1], systemSize[2], cutE, nTraj, kmcSteps, nSpecies, simulationTime, simulationTime/(nTraj * kmcSteps)]
    simulationTimeData = np.load('../simulationTimeData.npy')
    filteredData = simulationTimeData[np.all(simulationTimeData[:, matchColumnIndices] == matchRow, axis=1)]
    if len(filteredData) != 0:
        import matplotlib.pyplot as plt
        y = filteredData[:,-1:]
        x = range(len(y))
        
        if xVarIndex == 0:
            xlabels = np.array(filteredData[:,:3], int)
        else:
            xlabels = filteredData[:, xVarIndex+2]
        labels = []
        for xlabel in xlabels:
            labels.append(str(xlabel).replace(' ', ','))
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.xticks(x, labels)
        ax.plot(y, linestyle='-', marker='o')
        xVar = simulationParameterNames[xVarIndex]
        ax.set_xlabel(xVar)
        ax.set_ylabel('time/step (sec)')
        figureName = 'Scaling Analysis with ' + xVar
        figureTitle = figureName + '\n' + constantParameters[:-2]
        outputFilePath = outdir + '/' + figureName + '; ' + constantParameters[:-2].replace(' ', '') + '.jpg'
        ax.set_title(figureTitle)
        cfa = list(ax.axis())
        cfa[0] -= 1
        cfa[1] += 1
        cfa[2] = 0
        cfa[3] *= 1.2
        ax.axis(cfa)
        if (os.path.isfile(outputFilePath) and replaceExistingFiles) or save:
            plt.savefig(outputFilePath)
            plt.close(fig)
        elif openFigure:
            plt.show()