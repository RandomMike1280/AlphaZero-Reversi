import sys
import os
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

try:
    # Try to import the Cython implementation directly
    from reversi.cython import Board as CythonBoard
    print("Successfully imported Cython board module")
    
    # Test the Cython implementation
    print("\nTesting Cython implementation:")
    board = CythonBoard()
    
    # Print initial board state
    print("Initial board state:")
    print(board.get_board_state())
    
    # Get and make some moves
    print("\nMaking some moves:")
    for _ in range(5):
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            print("No valid moves, passing")
            board.make_move(-1, -1)
        else:
            row, col = valid_moves[0]
            print(f"Making move at ({row}, {col})")
            board.make_move(row, col)
        
        # Print board state after move
        print("\nBoard state:")
        print(board.get_board_state())
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
