#!/usr/bin/env python

from pathlib import Path

from PyCT.materialRun import materialRun

dstPath = Path.cwd()
materialRun(dstPath)
