#!/usr/bin/env python
import numpy as np
import os
import platform

directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
inputCoordinatesDirectoryName= 'InputCoordinates'
inputCoordinateFileName = 'BVO_index_coord_Experimental.txt'
inputCoordinateSourceType = 1 # 0: AFlow; 1: ms-BVO; 2: 9013437

class bvoParameters(object):
    
    def __init__(self):
        self.name = 'BVO'
        self.speciesTypes = ['electron', 'hole']
        self.speciesChargeList = [-1, +1]
        self.speciesToElementTypeMap = {'electron': ['V'], 'empty': ['Bi', 'O', 'V'], 'hole': ['O']}
        cwd = os.path.dirname(os.path.realpath(__file__))
        # Input coordinates source: http://www.crystallography.net/cod/9013437.html
        # Source article DOI: 10.1016/0025-5408(72)90227-9
        inputFileLocation = cwd + directorySeparator + inputCoordinatesDirectoryName + directorySeparator + inputCoordinateFileName
        index_pos = np.loadtxt(inputFileLocation)
        self.unitcellCoords = index_pos[:, 1:] 
        self.elementTypeIndexList = index_pos[:,0]
        self.chargeTypes = {'Bi': +3.0, 'O': -2.0, 'V': +5.0, 'V0': +4.0} # multiples of elementary charge
        if inputCoordinateSourceType == 0:
            # AFlow
            self.elementTypes = ['Bi', 'O', 'V']
            a = 5.1893210411 # lattice constant along x-axis
            b = 5.1893210411 # lattice constant along y-axis
            c = 11.7714366913 # lattice constant along z-axis
            self.neighborCutoffDist = {'V:V': [3.92335, 5.18932], 'O:O': [2.808, 2.815, 2.984, 3.002, 3.117]}
            self.neighborCutoffDistTol = {'V:V': [0.002, 0.002], 'O:O': [0.002, 0.002, 0.002, 0.002, 0.002]}
            alpha = 90. / 180 * np.pi # interaxial angle between b-c
            beta = 90. / 180 * np.pi # lattice angle between a-c
            gamma = 90. / 180 * np.pi # lattice angle between a-b
        elif inputCoordinateSourceType == 1:
            # ms-BVO
            self.elementTypes = ['O', 'V', 'Bi']
            a = 5.19560 # lattice constant along x-axis
            b = 5.09350 # lattice constant along y-axis
            c = 11.70450 # lattice constant along z-axis
            self.neighborCutoffDist = {'V:V': [3.8474, 3.9447, 5.0935, 5.1956], 'O:O': [2.7450, 2.8600, 2.9723, 3.0076, 3.0473]} # 'O:O': [2.794, 2.829, 2.953, 3.009, 3.047, 11.704]
            self.neighborCutoffDistTol = {'V:V': [0.002, 0.002, 0.002, 0.002], 'O:O': [0.002, 0.002, 0.002, 0.002, 0.002]}
            alpha = 90. / 180 * np.pi # interaxial angle between b-c
            beta = 90. / 180 * np.pi # lattice angle between a-c
            gamma = 90.383 / 180 * np.pi # lattice angle between a-b
        elif inputCoordinateSourceType == 2:
            # 9013437
            self.elementTypes = ['Bi', 'V', 'O']
            a = 5.193500 # lattice constant along x-axis
            b = 5.089800 # lattice constant along y-axis
            c = 11.69720 # lattice constant along z-axis
            self.neighborCutoffDist = {'V:V': [3.8242, 3.9656, 5.0898, 5.1935], 'O:O': [2.7945, 2.8077, 2.8296, 3.0028, 3.1179]}
            self.neighborCutoffDistTol = {'V:V': [0.002, 0.002, 0.002, 0.002], 'O:O': [0.002, 0.002, 0.002, 0.002, 0.002]}
            alpha = 90. / 180 * np.pi # interaxial angle between b-c
            beta = 90. / 180 * np.pi # lattice angle between a-c
            gamma = 90.387 / 180 * np.pi # lattice angle between a-b
            

        self.latticeParameters = [a, b, c, alpha, beta, gamma]
        # typical frequency for nuclear motion in (1/sec)
        self.vn = 1.00E+13 # source: Liu et al., PCCP 2015. DOI: 10.1039/C5CP04299B
        self.lambdaValues = {'V:V': [1.4768, 1.4964, 1.4652, 1.4932], 'O:O': [1.4668, 0.6756, 0.9952, 1.4072, 2.3464]} # reorganization energy in eV
        self.VAB = {'V:V': [0.000, 0.000, 0.000, 0.000], 'O:O': [0.000, 0.000, 0.000, 0.000, 0.000]} # electronic coupling matrix element in eV
        self.elementTypeDelimiter = ':' 
        self.emptySpeciesType = 'empty'
        self.siteIdentifier = '0'
        self.dielectricConstant = 38.299 # Estimated dielectric constant computed using Clausius-Mosotti equation
                                         # Shannon et al. JAP 1993. DOI: 10.1063/1.353856