#!/usr/bin/env python

from pathlib import Path

from PyCT.materialMSD import materialMSD

dst_path = Path.cwd()
materialMSD(dst_path)
