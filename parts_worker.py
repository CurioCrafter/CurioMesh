# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Andrew Rainsberger and CurioMesh contributors

"""Run by a separate background Blender: measures parts of an Auto Segments job, so several cores work at once.

Loads job.pkl from the folder it is given, then measures each part number it reads from its standard input and writes
part_<k>.pkl. Quits when its input closes: when the job stops, or the Blender that started it is gone. See parallel.py.
"""

import importlib
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))
importlib.import_module(os.path.basename(here) + ".parallel").worker_main(sys.argv)
