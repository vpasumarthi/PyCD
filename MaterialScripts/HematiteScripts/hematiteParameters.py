#!/usr/bin/env python
import numpy as np
import os
import platform

directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
inputCoordinatesDirectoryName= 'InputCoordinates'
inputCoordinateFileName = 'Fe2O3_RelaxCell_Trial03_index_coord.txt'

class hematiteParameters(object):
    
    def __init__(self):
        self.name = 'Fe2O3'
        self.elementTypes = ['Fe', 'O']
        self.speciesToElementTypeMap = {'electron': ['Fe'], 'empty': ['Fe', 'O'], 'hole': ['O']}
        cwd = os.path.dirname(os.path.realpath(__file__))
        inputFileLocation = cwd + directorySeparator + inputCoordinatesDirectoryName + directorySeparator + inputCoordinateFileName
        index_pos = np.loadtxt(inputFileLocation)
        self.unitcellCoords = index_pos[:, 1:] 
        self.elementTypeIndexList = index_pos[:,0]
        self.chargeTypes = {'Fe': +3, 'O': -2, 'Fe0': +2} # multiples of elementary charge
        #a = 5.038 # lattice constant along x-axis
        #b = 5.038 # lattice constant along y-axis
        #c = 13.772 # lattice constant along z-axis
        a = 4.783 # lattice constant along x-axis # RelaxCell_Trial03
        b = 4.783 # lattice constant along y-axis # RelaxCell_Trial03
        c = 13.376 # lattice constant along z-axis # RelaxCell_Trial03
        alpha = 90. / 180 * np.pi # interaxial angle between b-c
        beta = 90. / 180 * np.pi # lattice angle between a-c
        gamma = 120. / 180 * np.pi # lattice angle between a-b
        self.latticeParameters = [a, b, c, alpha, beta, gamma]
        self.vn = 1.85E+13 # typical frequency for nuclear motion in (1/sec)
        self.lambdaValues = {'Fe:Fe': [1.74533, 1.88683]} # reorganization energy in eV for basal plane, c-direction
        self.VAB = {'Fe:Fe': [0.184, 0.028]} # electronic coupling matrix element in eV for basal plane, c-direction
        # self.neighborCutoffDist = {'Fe:Fe': [2.971, 2.901], 'O:O': [2.670, 2.775, 2.887, 3.035], 'Fe:O': [1.946, 2.116]} # Basal: 2.971, C: 2.901 # Crystallography Coordinates
        #self.neighborCutoffDist = {'Fe:Fe': [2.938, 2.76], 'O:O': [2.836], 'Fe:O': [2.064]} # Fixed Cell
        #self.neighborCutoffDistTol = {'Fe:Fe': [0.033, 0.097], 'O:O': [0.237], 'Fe:O': [0.126]} # Fixed Cell
        #self.neighborCutoffDist = {'Fe:Fe': [2.931, 2.659], 'O:O': [2.870], 'Fe:O': [2.024]} # Fixed Cell_Trial03
        #self.neighborCutoffDistTol = {'Fe:Fe': [0.002, 0.002], 'O:O': [0.123], 'Fe:O': [0.047]} # Fixed Cell_Trial03
        self.neighborCutoffDist = {'Fe:Fe': [2.783, 2.569], 'O:O': [2.722], 'Fe:O': [1.936]} # Relax Cell_Trial03
        self.neighborCutoffDistTol = {'Fe:Fe': [0.003, 0.002], 'O:O': [0.128], 'Fe:O': [0.041]} # Relax Cell_Trial03
        #self.neighborCutoffDist = {'Fe:Fe': [2.780, 2.572], 'O:O': [3.000], 'Fe:O': [2.000]} # Relax Cell
        #self.neighborCutoffDistTol = {'Fe:Fe': [0.156, 0.014], 'O:O': [0.500], 'Fe:O': [0.500]} # Relax Cell
        self.elementTypeDelimiter = ':' 
        self.emptySpeciesType = 'empty'
        self.siteIdentifier = '0'
        self.dielectricConstant = 25.0 # Rosso, K. M., et al. (2003). The Journal of Chemical Physics 118(14): 6455-6466.
