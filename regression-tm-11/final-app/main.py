"""
Main entrypoint for California Housing Price Prediction application.
Directly invokes app.py logic when executed with Streamlit.
"""

import os
import sys

# Ensure current directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# Import and execute app
from app import *

