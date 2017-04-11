#!/usr/bin/env python
import numpy as np
import os
import platform

directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
inputCoordinatesDirectoryName= 'InputCoordinates'
inputCoordinateFileName = 'Fe2O3_index_coord.txt'

class hematiteParameters(object):
    
    def __init__(self):
        self.name = 'Fe2O3'
        self.elementTypes = ['Fe', 'O']
        self.speciesTypes = ['electron', 'hole']
        self.speciesChargeList = [-1, +1]
        self.speciesToElementTypeMap = {'electron': ['Fe'], 'empty': ['Fe', 'O'], 'hole': ['O']}
        cwd = os.path.dirname(os.path.realpath(__file__))
        inputFileLocation = cwd + directorySeparator + inputCoordinatesDirectoryName + directorySeparator + inputCoordinateFileName
        index_pos = np.loadtxt(inputFileLocation)
        self.unitcellCoords = index_pos[:, 1:] 
        self.elementTypeIndexList = index_pos[:,0]
        self.chargeTypes = {'Fe': +3, 'O': -2, 'Fe0': +2} # multiples of elementary charge
        a = 5.0180397814313995 # lattice constant along x-axis # RelaxCell_Trial08
        b = 5.0180397814313995 # lattice constant along y-axis # RelaxCell_Trial08
        c = 13.8742926905499075 # lattice constant along z-axis # RelaxCell_Trial08
        alpha = 90. / 180 * np.pi # interaxial angle between b-c
        beta = 90. / 180 * np.pi # lattice angle between a-c
        gamma = 120. / 180 * np.pi # lattice angle between a-b
        self.latticeParameters = [a, b, c, alpha, beta, gamma]
        self.vn = 1.85E+13 # typical frequency for nuclear motion in (1/sec)
        self.lambdaValues = {'Fe:Fe': [1.74533, 1.88683]} # reorganization energy in eV for basal plane, c-direction
        self.VAB = {'Fe:Fe': [0.184, 0.028]} # electronic coupling matrix element in eV for basal plane, c-direction
        self.neighborCutoffDist = {'Fe:Fe': [2.963, 2.930], 'O:O': [2.847], 'Fe:O': [2.033]} # Relax Cell_Trial08
        self.neighborCutoffDistTol = {'Fe:Fe': [0.002, 0.002], 'O:O': [0.163], 'Fe:O': [0.103]} # Relax Cell_Trial08
        self.elementTypeDelimiter = ':' 
        self.emptySpeciesType = 'empty'
        self.siteIdentifier = '0'
        self.dielectricConstant = 25.0 # Rosso, K. M., et al. (2003). The Journal of Chemical Physics 118(14): 6455-6466.
