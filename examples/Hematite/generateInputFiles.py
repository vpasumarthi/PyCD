#!/usr/bin/env python

from pathlib import Path

import numpy as np

from PyCT.materialSetup import materialSetup

# Input parameters:
system_size = np.array([2, 2, 1])
pbc = np.array([1, 1, 1])
generate_hop_neighbor_list = 1
generate_cum_disp_list = 1
generate_precomputed_array = 1
systemDirectoryPath = Path.cwd()
input_file_directory_name = 'InputFiles'
input_directory_path = systemDirectoryPath.joinpath(input_file_directory_name)

materialSetup(input_directory_path, system_size, pbc, generate_hop_neighbor_list,
              generate_cum_disp_list, generate_precomputed_array)
