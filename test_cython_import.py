import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

try:
    from reversi.cython import board as cython_board
    print("Successfully imported Cython board module")
    print(f"Module location: {cython_board.__file__}")
except ImportError as e:
    print(f"Failed to import Cython board: {e}")
    print("Python path:", sys.path)
