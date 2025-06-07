import sys
import os
import time
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

def test_cython_performance():
    try:
        from reversi.cython import Board as CythonBoard
        print("Testing Cython implementation performance")
        
        board = CythonBoard()
        num_moves = 100000
        move_count = 0
        consecutive_passes = 0
        
        start_time = time.time()
        
        for _ in range(num_moves):
            valid_moves = board.get_valid_moves()
            
            if not valid_moves:
                board.make_move(-1, -1)  # Pass
                consecutive_passes += 1
                
                # If both players passed in a row, game is over
                if consecutive_passes >= 2:
                    board = CythonBoard()
                    consecutive_passes = 0
            else:
                consecutive_passes = 0
                # Make first valid move
                row, col = valid_moves[0]
                board.make_move(row, col)
                move_count += 1
        
        elapsed = time.time() - start_time
        moves_per_sec = num_moves / elapsed if elapsed > 0 else float('inf')
        
        print(f"Processed {num_moves} moves in {elapsed:.2f} seconds")
        print(f"{moves_per_sec:.2f} moves per second")
        print(f"Actual moves made: {move_count}")
        print("Final board state:")
        print(board.get_board_state())
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cython_performance()
