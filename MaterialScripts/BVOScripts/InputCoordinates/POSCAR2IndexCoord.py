#!/usr/bin/env python

import numpy as np

inputData = np.loadtxt('POSCAR', skiprows=8)
indexLength = np.array([4, 16, 4])
indices = np.arange(len(indexLength))
output = np.zeros((len(inputData), 4))
output[:, 0] = np.repeat(indices, indexLength)
output[:, 1:] = inputData
np.savetxt('BVO_index_coord_Experimental.txt', output, fmt='%d %+26.16f %+26.16f %+26.16f')
