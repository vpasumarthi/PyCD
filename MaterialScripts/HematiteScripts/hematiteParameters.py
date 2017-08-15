#!/usr/bin/env python
import numpy as np
import os
import platform

directorySeparator = '\\' if platform.uname()[0]=='Windows' else '/'
inputCoordinatesDirectoryName= 'InputCoordinates'
inputCoordinateFileName = 'Fe2O3_index_coord_Experimental.txt'

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
        self.chargeTypes = {'Fe': +3.0, 'O': -2.0, 'Fe0': +2.0} # multiples of elementary charge
        a = 5.0380000000000003 # lattice constant along x-axis # Experimental
        b = 5.0380000000000003 # lattice constant along y-axis # Experimental
        c = 13.7720000000000002 # lattice constant along z-axis # Experimental
        self.neighborCutoffDist = {'Fe:Fe': [2.971, 2.901]} # Basal: 2.971, C: 2.901 # Experimental
        self.neighborCutoffDistTol = {'Fe:Fe': [0.002, 0.002]} # Experimental
        alpha = 90. / 180 * np.pi # interaxial angle between b-c
        beta = 90. / 180 * np.pi # lattice angle between a-c
        gamma = 120. / 180 * np.pi # lattice angle between a-b
        self.latticeParameters = [a, b, c, alpha, beta, gamma]
        # typical frequency for nuclear motion in (1/sec)
        self.vn = 1.85E+13 # source: Kerisit et al., GCA 2006. DOI: 10.1016/j.gca.2005.12.021
        self.lambdaValues = {'Fe:Fe': [1.74533, 1.88683]} # reorganization energy in eV for basal plane, c-direction
        self.VAB = {'Fe:Fe': [0.184, 0.028]} # electronic coupling matrix element in eV for basal plane, c-direction
        self.elementTypeDelimiter = ':' 
        self.emptySpeciesType = 'empty'
        self.siteIdentifier = '0'
        self.dielectricConstant = 9.355 # Direction-averaged value of the static dielectric constant of the potential 
                                        # model used in KMC paper. Dielectric constant tensor obtained from METADISE.
                                        # KMC paper: Kerisit et al., The Journal of Chemical Physics 127, 124706 (2007); 
                                        # DOI: 10.1063/1.2768522
