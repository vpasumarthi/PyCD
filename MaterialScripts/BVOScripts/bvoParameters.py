#!/usr/bin/env python
import numpy as np
import os
import platform

directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
inputCoordinatesDirectoryName= 'InputCoordinates'
inputCoorType = 0 # 0:Experimental; 1: Relaxed
inputCoordinateFileName = 'BVO_index_coord_' + ('Experimental' if inputCoorType is 0 else 'relaxed') + '.txt'

class bvoParameters(object):
    
    def __init__(self):
        self.name = 'BVO'
        self.elementTypes = ['Bi', 'O', 'V']
        self.speciesTypes = ['electron', 'hole']
        self.speciesChargeList = [-1, +1]
        self.speciesToElementTypeMap = {'electron': ['V'], 'empty': ['Bi', 'O', 'V'], 'hole': ['O']}
        cwd = os.path.dirname(os.path.realpath(__file__))
        inputFileLocation = cwd + directorySeparator + inputCoordinatesDirectoryName + directorySeparator + inputCoordinateFileName
        index_pos = np.loadtxt(inputFileLocation)
        self.unitcellCoords = index_pos[:, 1:] 
        self.elementTypeIndexList = index_pos[:,0]
        self.chargeTypes = {'Bi': +3.0, 'O': -2.0, 'V': +5.0, 'V0': +4.0} # multiples of elementary charge
        if inputCoorType:
            a = 5.0180397814313995 # lattice constant along x-axis # RelaxCell_Trial08
            b = 5.0180397814313995 # lattice constant along y-axis # RelaxCell_Trial08
            c = 13.8742926905499075 # lattice constant along z-axis # RelaxCell_Trial08
            self.neighborCutoffDist = {'V:V': [2.963, 2.930], 'O:O': [2.847], 'V:O': [2.033]} # Relax Cell_Trial08
            self.neighborCutoffDistTol = {'V:V': [0.002, 0.002], 'O:O': [0.163], 'V:O': [0.103]} # Relax Cell_Trial08
        else:
            a = 5.193500 # lattice constant along x-axis # Experimental
            b = 5.089800 # lattice constant along y-axis # Experimental
            c = 11.69720 # lattice constant along z-axis # Experimental
            self.neighborCutoffDist = {'V:V': [3.824, 3.966, 5.090, 5.194]} # Basal: 2.971, C: 2.901 # Experimental
            self.neighborCutoffDistTol = {'V:V': [0.002, 0.002, 0.002, 0.002]} # Experimental
        alpha = 90. / 180 * np.pi # interaxial angle between b-c
        beta = 90. / 180 * np.pi # lattice angle between a-c
        gamma = 90.387 / 180 * np.pi # lattice angle between a-b
        self.latticeParameters = [a, b, c, alpha, beta, gamma]
        self.vn = 1.85E+13 # typical frequency for nuclear motion in (1/sec)
        self.lambdaValues = {'V:V': [1.74533, 1.88683]} # reorganization energy in eV for basal plane, c-direction
        self.VAB = {'V:V': [0.184, 0.028]} # electronic coupling matrix element in eV for basal plane, c-direction
        self.electrostaticCutoffDistKey = 'E'
        self.elementTypeDelimiter = ':' 
        self.emptySpeciesType = 'empty'
        self.siteIdentifier = '0'
        self.dielectricConstant = 9.355 # Direction-averaged value of the static dielectric constant of the potential 
                                        # model used in KMC paper. Dielectric constant tensor obtained from METADISE.
                                        # KMC paper: Kerisit et al., The Journal of Chemical Physics 127, 124706 (2007); 
                                        # DOI: 10.1063/1.2768522
