#!/usr/bin/env python

import numpy as np

inputData = np.loadtxt('InputPOSCAR')
indexLength = np.array([12, 18])
indices = np.arange(len(indexLength))
output = np.zeros((len(inputData), 4))
output[:, 0] = np.repeat(indices, indexLength)
output[:, 1:] = inputData
np.savetxt('output.txt', output, fmt='%d %+26.16f %+26.16f %+26.16f')