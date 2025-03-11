#!/usr/bin/env python3
# run.py - Convenience script to start the application

import sys
import os

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import main function and run application
from main import main

if __name__ == "__main__":
    main()