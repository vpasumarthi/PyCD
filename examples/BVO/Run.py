#!/usr/bin/env python

from pathlib import Path

from PyCD.material_run import material_run

dst_path = Path.cwd()
material_run(dst_path)
