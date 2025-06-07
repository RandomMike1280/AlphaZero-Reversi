"""
Cython implementation wrapper for the Board class.
"""
import sys
import os

# Add the parent directory to Python path to find the reversi package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    # Try to import the Cython implementation
    from reversi.cython import Board as CythonBoard
    BOARD_IMPL = 'cython'
    print("Using Cython optimized implementation")
except ImportError as e:
    # Fall back to pure Python implementation
    print(f"Cython import failed: {e}, falling back to Python implementation")
    from .board import Board as PythonBoard
    BOARD_IMPL = 'python'
    
    # Create an alias for the Python implementation
    class CythonBoard(PythonBoard):
        pass
