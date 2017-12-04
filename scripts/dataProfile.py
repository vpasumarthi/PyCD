#!/usr/bin/env python

import os.path
from textwrap import wrap
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt


def diffusionProfile(outdir, systemDirectoryPath, profilingSpeciesTypeIndex,
                     speciesCountList, ion_charge_type, species_charge_type, temp,
                     t_final, time_interval, n_traj, msd_t_final, repr_time):
    profilingSpeciesList = speciesCountList[profilingSpeciesTypeIndex]
    profileLength = len(profilingSpeciesList)
    diffusivityProfileData = np.zeros((profileLength, 2))
    diffusivityProfileData[:, 0] = profilingSpeciesList
    species_type = 'electron' if profilingSpeciesTypeIndex == 0 else 'hole'

    if profilingSpeciesTypeIndex == 0:
        n_holes = speciesCountList[1][0]
    else:
        n_electrons = speciesCountList[0][0]

    parentDir1 = 'SimulationFiles'
    parentDir2 = ('ion_charge_type=' + ion_charge_type
                  + '; species_charge_type=' + species_charge_type)

    file_name = '%1.2E%s' % (msd_t_final, repr_time)
    msdAnalysisLogFileName = ('MSD_Analysis' + ('_' if file_name else '')
                              + file_name + '.log')

    for species_index, nSpecies in enumerate(profilingSpeciesList):
        # Change to working directory
        if profilingSpeciesTypeIndex == 0:
            n_electrons = nSpecies
        else:
            n_holes = nSpecies
        parentDir3 = (str(n_electrons)
                      + ('electron' if n_electrons == 1 else 'electrons') + ', '
                      + str(n_holes) + ('hole' if n_holes == 1 else 'holes'))
        parentDir4 = str(temp) + 'K'
        workDir = (('%1.2E' % t_final) + 'SEC,' + ('%1.2E' % time_interval)
                   + 'TimeInterval,' + ('%1.2E' % n_traj) + 'Traj')
        msdAnalysisLogFilePath = os.path.join(
                    systemDirectoryPath, parentDir1, parentDir2, parentDir3,
                    parentDir4, workDir, msdAnalysisLogFileName)

        with open(msdAnalysisLogFilePath, 'r') as msdAnalysisLogFile:
            firstLine = msdAnalysisLogFile.readline()
        diffusivityProfileData[species_index, 1] = float(firstLine[-13:-6])

    plt.switch_backend('Agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(diffusivityProfileData[:, 0], diffusivityProfileData[:, 1], 'o-',
            color='blue', markerfacecolor='blue', markeredgecolor='black')
    ax.set_xlabel('Number of ' + species_type + 's')
    ax.set_ylabel('Diffusivity (${{\mu}}m^2/s$)')
    figureTitle = ('Diffusion coefficient as a function of number of '
                   + species_type + 's')
    ax.set_title('\n'.join(wrap(figureTitle, 60)))
    filename = (str(species_type) + 'DiffusionProfile_' + ion_charge_type[0]
                + species_charge_type[0] + '_' + str(profilingSpeciesList[0])
                + '-' + str(profilingSpeciesList[-1]))
    figureName = filename + '.png'
    figurePath = os.path.join(outdir, figureName)
    plt.savefig(figurePath)

    dataFileName = filename + '.txt'
    dataFilePath = os.path.join(outdir, dataFileName)
    np.savetxt(dataFilePath, diffusivityProfileData)


def runtimeProfile(outdir, systemDirectoryPath, profilingSpeciesTypeIndex,
                   speciesCountList, ion_charge_type, species_charge_type, temp,
                   t_final, time_interval, n_traj):
    profilingSpeciesList = speciesCountList[profilingSpeciesTypeIndex]
    profileLength = len(profilingSpeciesList)
    elapsedSecondsData = np.zeros((profileLength, 2))
    elapsedSecondsData[:, 0] = profilingSpeciesList
    species_type = 'electron' if profilingSpeciesTypeIndex == 0 else 'hole'

    if profilingSpeciesTypeIndex == 0:
        n_holes = speciesCountList[1][0]
    else:
        n_electrons = speciesCountList[0][0]

    parentDir1 = 'SimulationFiles'
    parentDir2 = ('ion_charge_type=' + ion_charge_type
                  + '; species_charge_type=' + species_charge_type)
    runLogFileName = 'Run.log'

    for species_index, nSpecies in enumerate(profilingSpeciesList):
        # Change to working directory
        if profilingSpeciesTypeIndex == 0:
            n_electrons = nSpecies
        else:
            n_holes = nSpecies
        parentDir3 = (str(n_electrons)
                      + ('electron' if n_electrons == 1 else 'electrons') + ', '
                      + str(n_holes) + ('hole' if n_holes == 1 else 'holes'))
        parentDir4 = str(temp) + 'K'
        workDir = (('%1.2E' % t_final) + 'SEC,' + ('%1.2E' % time_interval)
                   + 'TimeInterval,' + ('%1.2E' % n_traj) + 'Traj')
        runLogFilePath = os.path.join(
                    systemDirectoryPath, parentDir1, parentDir2, parentDir3,
                    parentDir4, workDir, runLogFileName)

        with open(runLogFilePath, 'r') as runLogFile:
            firstLine = runLogFile.readline()
        if 'days' in firstLine:
            numDays = float(firstLine[14:16])
            numHours = float(firstLine[23:25])
            numMinutes = float(firstLine[33:35])
            numSeconds = float(firstLine[45:47])
        else:
            numDays = 0
            numHours = float(firstLine[14:16])
            numMinutes = float(firstLine[24:26])
            numSeconds = float(firstLine[36:38])
        elapsedTime = timedelta(days=numDays, hours=numHours,
                                minutes=numMinutes, seconds=numSeconds)
        elapsedSecondsData[species_index, 1] = int(elapsedTime.total_seconds())

    plt.switch_backend('Agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(elapsedSecondsData[:, 0], elapsedSecondsData[:, 1], 'o-',
            color='blue', markerfacecolor='blue', markeredgecolor='black')
    ax.set_xlabel('Number of ' + species_type + 's')
    ax.set_ylabel('Run Time (sec)')
    figureTitle = ('Simulation run time as a function of number of '
                   + species_type + 's')
    ax.set_title('\n'.join(wrap(figureTitle, 60)))
    filename = (str(species_type) + 'RunTimeProfile_' + ion_charge_type[0]
                + species_charge_type[0] + '_' + str(profilingSpeciesList[0])
                + '-' + str(profilingSpeciesList[-1]))
    figureName = filename + '.png'
    figurePath = os.path.join(outdir, figureName)
    plt.savefig(figurePath)

    dataFileName = filename + '.txt'
    dataFilePath = os.path.join(outdir, dataFileName)
    np.savetxt(dataFilePath, elapsedSecondsData)
