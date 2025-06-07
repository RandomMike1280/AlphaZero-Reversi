import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

try:
    from reversi.cython import Board as CythonBoard
    print("Successfully imported Cython board module")
    
    # Print all attributes of the Board class
    print("\nBoard class attributes:")
    board = CythonBoard()
    for attr in dir(board):
        if not attr.startswith('__'):  # Skip private attributes
            print(f"- {attr}")
    
    # Try to get the current state
    print("\nTrying to access board state:")
    try:
        print(f"Black pieces: {bin(board.black) if hasattr(board, 'black') else 'N/A'}")
        print(f"White pieces: {bin(board.white) if hasattr(board, 'white') else 'N/A'}")
        print(f"Current player: {board.current_player if hasattr(board, 'current_player') else 'N/A'}")
    except Exception as e:
        print(f"Error accessing board state: {e}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
