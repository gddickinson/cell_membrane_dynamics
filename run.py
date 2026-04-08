#!/usr/bin/env python3
"""
Convenience script to start the application.

DEPRECATED: Use main.py directly instead.
    python main.py
"""

import sys
import os
import warnings

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.warn(
    "run.py is deprecated. Use 'python main.py' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import main function and run application
from main import main

if __name__ == "__main__":
    main()
