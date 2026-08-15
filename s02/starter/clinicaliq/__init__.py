"""
ClinicalIQ package -- Session 2: Multi-Turn Memory (US-02)
==========================================================

This file runs automatically when Python imports the clinicaliq package.
Use it to set up the environment before any other module loads.
"""
import os

os.environ.setdefault("HF_HUB_VERBOSITY", "error")

from dotenv import load_dotenv
load_dotenv()
