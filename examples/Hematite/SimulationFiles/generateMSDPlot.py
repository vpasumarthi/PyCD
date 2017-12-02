#!/usr/bin/env python

from pathlib import Path

from PyCT.material_msd import material_msd

dst_path = Path.cwd()
material_msd(dst_path)
