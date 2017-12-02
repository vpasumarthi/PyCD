#!/usr/bin/env python

from pathlib import Path

from PyCT.materialMSD import materialMSD

dstPath = Path.cwd()
materialMSD(dstPath)
