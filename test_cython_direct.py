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
    
    # Test performance
    start_time = time.time()
    num_moves = 100000
    
    for _ in range(num_moves):
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            board.make_move(-1, -1)
        else:
            row, col = valid_moves[0]
            board.make_move(row, col)
            
        if board.game_over:
            board = CythonBoard()
    
    elapsed = time.time() - start_time
    moves_per_sec = num_moves / elapsed if elapsed > 0 else float('inf')
    
    print(f"Processed {num_moves} moves in {elapsed:.2f} seconds")
    print(f"{moves_per_sec:.2f} moves per second")
    print("Final board:")
    print(board)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
